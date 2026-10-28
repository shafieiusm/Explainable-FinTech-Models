"""Optimisation and training (Section 3.11, Eq. 59–65).

A custom ``tf.GradientTape`` loop is used so that the fairness regulariser
(Eq. 50–51) and the optional adversarial debiasing term (Eq. 52) can be
trained jointly with the task loss:

    L = L_BCE(y, ŷ)  +  λ · |P(ŷ | A=1) − P(ŷ | A=0)|  [+ β · L_adv]

Adam (Eq. 61–65) is used with early stopping and a reduce-on-plateau
learning-rate schedule.
"""
from __future__ import annotations

import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow import keras

from .fairness import bias_regularizer, build_adversary, gradient_reversal
from .model import predict_proba


def weighted_bce(y, p, pos_weight):
    """Binary cross-entropy (Eq. 59) with an optional weight on the minority class."""
    eps = 1e-7
    p = tf.clip_by_value(p, eps, 1.0 - eps)
    loss = -(pos_weight * y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p))
    return tf.reduce_mean(loss)


class Trainer:
    def __init__(self, model: keras.Model, train_cfg: dict, fair_cfg: dict | None = None, verbose: int = 1):
        self.model = model
        self.cfg = train_cfg
        fair_cfg = fair_cfg or {}
        self.verbose = verbose
        self.lambda_bias = float(fair_cfg.get("lambda_bias", 0.0)) if fair_cfg.get("enabled", False) else 0.0
        self.use_adv = bool(fair_cfg.get("enabled", False) and fair_cfg.get("adversarial", False))
        self.adv_weight = float(fair_cfg.get("adv_weight", 0.1))
        self.adversary = build_adversary() if self.use_adv else None
        self.optimizer = keras.optimizers.Adam(learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
                                               beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        self.pos_weight = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self.history: list[dict] = []

    # ---------------------------------------------------------------- steps
    def _variables(self):
        v = list(self.model.trainable_variables)
        if self.adversary is not None:
            v += list(self.adversary.trainable_variables)
        return v

    @tf.function
    def _train_step(self, x, y, a):
        with tf.GradientTape() as tape:
            p, _ = self.model(x, training=True)
            p = tf.reshape(p, [-1])
            task = weighted_bce(y, p, self.pos_weight)
            r_bias = bias_regularizer(p, a)
            loss = task + self.lambda_bias * r_bias                       # Eq. 50
            if self.adversary is not None:                                # Eq. 52
                a_hat = tf.reshape(self.adversary(tf.reshape(gradient_reversal(p), [-1, 1]), training=True), [-1])
                adv = weighted_bce(a, a_hat, 1.0)
                loss = loss + self.adv_weight * adv
            loss = loss + tf.add_n(self.model.losses) if self.model.losses else loss
        variables = self._variables()
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))             # Eq. 60–65
        return task, r_bias

    # ----------------------------------------------------------------- fit
    def fit(self, X_tr, y_tr, a_tr, X_va, y_va, a_va=None):
        bs = int(self.cfg.get("batch_size", 128))
        epochs = int(self.cfg.get("epochs", 100))
        patience = int(self.cfg.get("early_stopping_patience", 10))
        lr_patience = int(self.cfg.get("lr_patience", 4))
        lr_factor = float(self.cfg.get("lr_factor", 0.5))

        if self.cfg.get("class_weighting", True):
            pos = max(float(y_tr.mean()), 1e-6)
            self.pos_weight.assign((1.0 - pos) / pos)

        ds = (tf.data.Dataset.from_tensor_slices((X_tr.astype("float32"), y_tr.astype("float32"),
                                                  a_tr.astype("float32")))
              .shuffle(min(len(X_tr), 100_000), reshuffle_each_iteration=True)
              .batch(bs, drop_remainder=False)
              .prefetch(tf.data.AUTOTUNE))

        try:  # create optimizer slots outside tf.function
            self.optimizer.build(self._variables())
        except Exception:  # older Keras optimisers build lazily
            pass

        best_auc, best_weights, wait, lr_wait = -np.inf, self.model.get_weights(), 0, 0
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            task_sum, bias_sum, nb = 0.0, 0.0, 0
            for xb, yb, ab in ds:
                t, r = self._train_step(xb, yb, ab)
                task_sum += float(t)
                bias_sum += float(r)
                nb += 1
            p_va = predict_proba(self.model, X_va)
            val_auc = roc_auc_score(y_va, p_va)
            val_loss = float(weighted_bce(tf.constant(y_va, tf.float32), tf.constant(p_va, tf.float32), 1.0))
            rec = {"epoch": epoch, "train_loss": task_sum / nb, "train_bias": bias_sum / nb,
                   "val_loss": val_loss, "val_auc": float(val_auc),
                   "lr": float(tf.convert_to_tensor(self.optimizer.learning_rate).numpy())}
            self.history.append(rec)
            if self.verbose:
                print(f"epoch {epoch:3d} | loss {rec['train_loss']:.4f} | R_bias {rec['train_bias']:.4f} | "
                      f"val_loss {val_loss:.4f} | val_AUC {val_auc:.4f} | lr {rec['lr']:.2e} | {time.time() - t0:.1f}s")

            if val_auc > best_auc + 1e-4:
                best_auc, best_weights, wait, lr_wait = val_auc, self.model.get_weights(), 0, 0
            else:
                wait += 1
                lr_wait += 1
                if lr_wait >= lr_patience:  # dynamic learning-rate scheduling
                    self.optimizer.learning_rate.assign(rec["lr"] * lr_factor)
                    lr_wait = 0
                if wait >= patience:
                    if self.verbose:
                        print(f"early stopping at epoch {epoch}; best val AUC {best_auc:.4f}")
                    break
        self.model.set_weights(best_weights)
        return self.history
