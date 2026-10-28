"""Bayesian posterior approximation via Monte-Carlo dropout (Eq. 40).

Keeping dropout active at inference and averaging T stochastic forward passes
approximates the predictive posterior P(y | x, D) = ∫ P(y | x, θ) P(θ | D) dθ
(Gal & Ghahramani, 2016).  Batch-normalisation layers are frozen so that they
keep using their population statistics.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras


def mc_dropout_predict(model: keras.Model, X: np.ndarray, T: int = 30, batch_size: int = 2048):
    """Return (mean PD, std PD, 95 % interval) over ``T`` stochastic passes."""
    bn_layers = [l for l in model.layers if isinstance(l, keras.layers.BatchNormalization)]
    states = [l.trainable for l in bn_layers]
    for l in bn_layers:
        l.trainable = False  # BN in inference mode even when training=True
    try:
        draws = np.zeros((T, len(X)), dtype=np.float32)
        for t in range(T):
            for i in range(0, len(X), batch_size):
                xb = tf.convert_to_tensor(X[i : i + batch_size], tf.float32)
                p, _ = model(xb, training=True)
                draws[t, i : i + batch_size] = p.numpy().reshape(-1)
    finally:
        for l, s in zip(bn_layers, states):
            l.trainable = s
    return {
        "mean": draws.mean(0),
        "std": draws.std(0),
        "lower": np.percentile(draws, 2.5, axis=0),
        "upper": np.percentile(draws, 97.5, axis=0),
    }
