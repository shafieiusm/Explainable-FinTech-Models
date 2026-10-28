"""Explainable-AI module (Section 3.8, Eq. 43–47).

* SHAP values φ_i (Eq. 43) and global importance FI_j = mean|φ_ij| (Eq. 44)
* LIME local surrogate g(z) = φ0 + Σ φ_i z_i (Eq. 45)
* Integrated gradients IG_i(x) (Eq. 46)
* Attention explainability score AES_i = α_i / Σ α_j (Eq. 47)
* Permutation importance and an ensemble ranking (Figure 4)

SHAP, LIME and permutation importance only need a ``predict_fn`` mapping an
(n, d) array to P(default); integrated gradients and attention need the
Keras model.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .preprocessing import aggregate_by_group

PredictFn = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------- SHAP
def shap_values(predict_fn: PredictFn, X_background: np.ndarray, X_explain: np.ndarray,
                nsamples: int | str = "auto", seed: int = 0) -> np.ndarray:
    """Kernel SHAP approximation of Eq. 43.  Returns an (n, d) array."""
    import shap

    bg = X_background
    if len(bg) > 100:
        bg = shap.kmeans(bg, 50)
    explainer = shap.KernelExplainer(lambda z: predict_fn(np.asarray(z, dtype=np.float32)), bg)
    np.random.seed(seed)  # KernelExplainer samples coalitions with numpy's global RNG
    vals = explainer.shap_values(X_explain, nsamples=nsamples, silent=True)
    vals = vals[0] if isinstance(vals, list) else vals
    vals = np.asarray(vals)
    if vals.ndim == 3:  # (n, d, outputs)
        vals = vals[..., -1]
    return vals


def global_importance(phi: np.ndarray, groups: list[str] | None = None) -> pd.Series:
    """FI_j = (1/N) Σ_i |φ_ij|  (Eq. 44).

    If ``groups`` is given, one-hot columns are first summed per sample
    (SHAP values are additive), then the absolute mean is taken.
    """
    phi = np.asarray(phi)
    if groups is not None:
        df = pd.DataFrame(phi, columns=groups)
        df = df.T.groupby(level=0, sort=False).sum().T
        return df.abs().mean().sort_values(ascending=False)
    return pd.Series(np.abs(phi).mean(0)).sort_values(ascending=False)


# --------------------------------------------------------------------- LIME
def lime_explanation(predict_fn: PredictFn, X_train: np.ndarray, x: np.ndarray, feature_names: list[str],
                     num_features: int = 6, seed: int = 0) -> list[tuple[str, float]]:
    """Local surrogate explanation for one borrower (Eq. 45).

    Positive weights push towards *default*; negative weights towards
    *non-default*.
    """
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=X_train, feature_names=feature_names, class_names=["non-default", "default"],
        mode="classification", discretize_continuous=True, random_state=seed,
    )

    def proba(z):
        p = predict_fn(np.asarray(z, dtype=np.float32))
        return np.column_stack([1 - p, p])

    exp = explainer.explain_instance(x, proba, num_features=num_features, labels=(1,))
    return [(name, float(w)) for name, w in exp.as_list(label=1)]


# ------------------------------------------------------ integrated gradients
def integrated_gradients(model, X: np.ndarray, baseline: np.ndarray | None = None, steps: int = 50,
                         batch_size: int = 256) -> np.ndarray:
    """IG_i(x) = (x_i − x'_i) ∫₀¹ ∂F(x' + α(x − x'))/∂x_i dα  (Eq. 46).

    The integral is approximated with the trapezoidal rule.  The default
    baseline x' is the all-zeros vector, i.e. the training mean after
    z-scoring.
    """
    import tensorflow as tf

    X = np.asarray(X, dtype=np.float32)
    base = np.zeros(X.shape[1], np.float32) if baseline is None else np.asarray(baseline, np.float32)
    alphas = tf.reshape(tf.linspace(0.0, 1.0, steps + 1), [-1, 1, 1])
    out = np.zeros_like(X)
    for i in range(0, len(X), batch_size):
        xb = tf.constant(X[i : i + batch_size])
        b = tf.constant(base)[None, None, :]
        path = b + alphas * (xb[None] - b)                       # (steps+1, B, d)
        flat = tf.reshape(path, [-1, X.shape[1]])
        with tf.GradientTape() as tape:
            tape.watch(flat)
            p, _ = model(flat, training=False)
        g = tf.reshape(tape.gradient(p, flat), path.shape)
        avg = tf.reduce_mean((g[:-1] + g[1:]) / 2.0, axis=0)     # trapezoid
        out[i : i + batch_size] = ((xb - b[0]) * avg).numpy()
    return out


# ----------------------------------------------------------------- attention
def attention_scores(model, X: np.ndarray, groups: list[str] | None = None) -> pd.Series:
    """AES_i = α_i / Σ_j α_j  (Eq. 47), averaged over borrowers."""
    from .model import predict_attention

    alpha = predict_attention(model, X).mean(0)
    s = aggregate_by_group(alpha, groups) if groups is not None else pd.Series(alpha)
    return (s / s.sum()).sort_values(ascending=False)


# ---------------------------------------------------------------- permutation
def permutation_importance(predict_fn: PredictFn, X: np.ndarray, y: np.ndarray, groups: list[str],
                           n_repeats: int = 3, seed: int = 0) -> pd.Series:
    """Drop in AUC-ROC when all columns of one original attribute are shuffled."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, predict_fn(X))
    groups_arr = np.asarray(groups)
    res = {}
    for g in dict.fromkeys(groups):
        cols = np.where(groups_arr == g)[0]
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, cols] = X[perm][:, cols]
            drops.append(base - roc_auc_score(y, predict_fn(Xp)))
        res[g] = max(float(np.mean(drops)), 0.0)
    return pd.Series(res).sort_values(ascending=False)


def ensemble_importance(*scores: pd.Series) -> pd.Series:
    """Average of several importance vectors after normalising each to sum 1."""
    norm = [s.clip(lower=0) / s.clip(lower=0).sum() for s in scores if s is not None and s.sum() > 0]
    df = pd.concat(norm, axis=1).fillna(0.0)
    out = df.mean(axis=1)
    return (out / out.sum()).sort_values(ascending=False)
