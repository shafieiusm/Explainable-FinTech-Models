"""Fairness-aware optimisation (Section 3.9, Eq. 48–52).

NumPy metrics are importable without TensorFlow; the differentiable losses
import TensorFlow lazily.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Metrics (NumPy)
# --------------------------------------------------------------------------- #
def demographic_parity_difference(y_pred: np.ndarray, a: np.ndarray) -> float:
    """|P(Ŷ=1 | A=1) − P(Ŷ=1 | A=0)|  — 0 means Eq. 48 holds exactly."""
    y_pred, a = np.asarray(y_pred), np.asarray(a).astype(bool)
    if a.all() or (~a).all():
        return 0.0
    return float(abs(y_pred[a].mean() - y_pred[~a].mean()))


def disparate_impact_ratio(y_pred: np.ndarray, a: np.ndarray) -> float:
    """min(rate_0, rate_1) / max(rate_0, rate_1); the "80 % rule" asks for ≥ 0.8."""
    y_pred, a = np.asarray(y_pred), np.asarray(a).astype(bool)
    r1, r0 = y_pred[a].mean(), y_pred[~a].mean()
    hi = max(r0, r1)
    return float(min(r0, r1) / hi) if hi > 0 else 1.0


def equalized_odds_difference(y_true: np.ndarray, y_pred: np.ndarray, a: np.ndarray) -> float:
    """max over y∈{0,1} of |P(Ŷ=1 | Y=y, A=1) − P(Ŷ=1 | Y=y, A=0)|  (Eq. 49)."""
    y_true, y_pred, a = np.asarray(y_true), np.asarray(y_pred), np.asarray(a).astype(bool)
    gaps = []
    for y in (0, 1):
        m1, m0 = (y_true == y) & a, (y_true == y) & ~a
        if m1.any() and m0.any():
            gaps.append(abs(y_pred[m1].mean() - y_pred[m0].mean()))
    return float(max(gaps)) if gaps else 0.0


def group_accuracy_gap(y_true: np.ndarray, y_pred: np.ndarray, a: np.ndarray) -> float:
    y_true, y_pred, a = np.asarray(y_true), np.asarray(y_pred), np.asarray(a).astype(bool)
    acc1 = (y_true[a] == y_pred[a]).mean()
    acc0 = (y_true[~a] == y_pred[~a]).mean()
    return float(abs(acc1 - acc0))


def fairness_report(y_true, y_pred, a) -> dict:
    return {
        "demographic_parity_difference": demographic_parity_difference(y_pred, a),
        "equalized_odds_difference": equalized_odds_difference(y_true, y_pred, a),
        "disparate_impact_ratio": disparate_impact_ratio(y_pred, a),
        "group_accuracy_gap": group_accuracy_gap(y_true, y_pred, a),
    }


# --------------------------------------------------------------------------- #
# Differentiable regularisers (TensorFlow)
# --------------------------------------------------------------------------- #
def bias_regularizer(p, a):
    """R_bias = |P1 − P2|  (Eq. 51) computed on soft predictions within a batch."""
    import tensorflow as tf

    a = tf.cast(a, p.dtype)
    eps = tf.constant(1e-6, p.dtype)
    p1 = tf.reduce_sum(p * a) / (tf.reduce_sum(a) + eps)
    p0 = tf.reduce_sum(p * (1.0 - a)) / (tf.reduce_sum(1.0 - a) + eps)
    return tf.abs(p1 - p0)


def gradient_reversal(x, scale: float = 1.0):
    """Identity in the forward pass, −scale·grad in the backward pass.

    Used for adversarial debiasing: the adversary D tries to recover the
    protected attribute from the risk score while the predictor G learns to
    prevent it, i.e. min_G max_D L(G, D)  (Eq. 52).
    """
    import tensorflow as tf

    @tf.custom_gradient
    def _grl(t):
        def grad(dy):
            return -scale * dy
        return tf.identity(t), grad

    return _grl(x)


def build_adversary():
    from tensorflow import keras

    adv = keras.Sequential([
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ], name="fairness_adversary")
    adv.build((None, 1))
    return adv
