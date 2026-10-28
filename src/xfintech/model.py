"""Attention-based Bi-LSTM residual credit-risk network (Sections 3.3–3.7).

Architecture (Figure 1)::

    x ∈ R^d ──► Dense(128)+ReLU ─► Dense(64)+ReLU ─► BN ─► Dropout ───────────────┐   Eq. 13–18
     │                                                                             │
     └─► FeatureTokenizer (e_j = x_j W_j + b_j) ─► Multi-Head Self-Attention ─►    │   Eq. 10, 19–24
          Add & LayerNorm ─► Bi-LSTM (64 ⟷ 64) ─► Attention pooling (α, c) ────────┤   Eq. 27–33, 25–26
                                                                                   ▼
                                         Concat ─► Dense ─► [Residual block]×2 ─► Dropout   Eq. 34–36
                                                                                   ▼
                                                     z = Wᵀh + b ─► σ(z) = P(default | x)   Eq. 37–38

The model returns ``[p_default (B, 1), feature_attention (B, d)]``; ``feature_attention``
(B, d) holds the attention weights α used by the attention explainability
score (Eq. 47).
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# --------------------------------------------------------------------------- #
# Custom layers
# --------------------------------------------------------------------------- #
@keras.utils.register_keras_serializable(package="xfintech")
class FeatureTokenizer(layers.Layer):
    """Turns each scalar feature into a d_token-dimensional embedding (Eq. 10).

    Output: (batch, n_features, d_token) — the borrower "sequence" consumed by
    self-attention and the Bi-LSTM.
    """

    def __init__(self, n_features: int, d_token: int, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.d_token = d_token

    def build(self, input_shape):
        self.w = self.add_weight(name="w", shape=(self.n_features, self.d_token), initializer="glorot_uniform")
        self.b = self.add_weight(name="b", shape=(self.n_features, self.d_token), initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        return tf.expand_dims(x, -1) * self.w + self.b

    def get_config(self):
        return {**super().get_config(), "n_features": self.n_features, "d_token": self.d_token}


@keras.utils.register_keras_serializable(package="xfintech")
class AttentionPooling(layers.Layer):
    """Additive attention: α_i = softmax(e_i) (Eq. 25), c = Σ α_i h_i (Eq. 26)."""

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.proj = layers.Dense(units, activation="tanh")
        self.score = layers.Dense(1, use_bias=False)

    def call(self, h):
        e = tf.squeeze(self.score(self.proj(h)), axis=-1)  # (B, T)
        alpha = tf.nn.softmax(e, axis=-1)
        context = tf.reduce_sum(h * tf.expand_dims(alpha, -1), axis=1)
        return context, alpha

    def get_config(self):
        return {**super().get_config(), "units": self.units}


@keras.utils.register_keras_serializable(package="xfintech")
class ResidualBlock(layers.Layer):
    """H(x) = LN(x + F(x)),  F(x) = W2 σ(W1 x + b1) + b2   (Eq. 34–36, Figure 2)."""

    def __init__(self, units: int, dropout: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout
        self.fc1 = layers.Dense(units, activation="relu")
        self.drop = layers.Dropout(dropout)
        self.fc2 = layers.Dense(units)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=None):
        f = self.fc2(self.drop(self.fc1(x), training=training))
        return self.norm(x + f)

    def get_config(self):
        return {**super().get_config(), "units": self.units, "dropout": self.dropout_rate}


@keras.utils.register_keras_serializable(package="xfintech")
class UniformAttention(layers.Layer):
    """Uniform weights 1/T — stands in for α when attention is ablated."""

    def call(self, t):
        if len(t.shape) == 3:
            t = t[..., 0]
        return tf.ones_like(t) / tf.cast(tf.shape(t)[1], t.dtype)


# --------------------------------------------------------------------------- #
# Model builder
# --------------------------------------------------------------------------- #
def build_model(n_features: int, cfg: dict) -> keras.Model:
    """Build the Explainable FinTech network.

    ``cfg`` is the ``model`` section of the YAML config.  The flags
    ``use_attention``, ``use_lstm`` and ``use_residual`` switch the components
    on/off for the ablation study (Table 2).
    """
    dense_units = cfg.get("dense_units", [128, 64])
    dropout = cfg.get("dropout", 0.3)
    d_token = cfg.get("d_token", 32)
    heads = cfg.get("attention_heads", 8)
    lstm_units = cfg.get("lstm_units", 64)
    res_units = cfg.get("residual_units", 128)
    n_blocks = cfg.get("residual_blocks", 2)
    use_att = cfg.get("use_attention", True)
    use_lstm = cfg.get("use_lstm", True)
    use_res = cfg.get("use_residual", True)

    x_in = keras.Input(shape=(n_features,), name="borrower_features")  # X ∈ R^{N×d}  (Eq. 12)

    # ---- deep credit-scoring branch (Eq. 13–18)
    h = x_in
    for i, u in enumerate(dense_units):
        h = layers.Dense(u, activation="relu", name=f"credit_dense_{i + 1}")(h)
    h = layers.BatchNormalization(name="batch_norm")(h)
    h = layers.Dropout(dropout, name="dropout_dense")(h)
    branches = [h]

    # ---- token branch: self-attention + Bi-LSTM + attention pooling
    if use_att or use_lstm:
        seq = FeatureTokenizer(n_features, d_token, name="feature_tokenizer")(x_in)
        if use_att:
            mha = layers.MultiHeadAttention(num_heads=heads, key_dim=max(4, d_token // heads) * 2,
                                            dropout=0.1, name="multi_head_attention")       # Eq. 19–24
            att = mha(seq, seq)
            seq = layers.LayerNormalization(epsilon=1e-6, name="attn_norm")(layers.Add(name="attn_residual")([seq, att]))
        if use_lstm:
            seq = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True),
                                       name="bi_lstm")(seq)                                  # Eq. 27–33
        if use_att:
            ctx, alpha = AttentionPooling(64, name="attention_pooling")(seq)                # Eq. 25–26
        else:
            ctx = layers.GlobalAveragePooling1D(name="mean_pooling")(seq)
            alpha = UniformAttention(name="uniform_attention")(seq)
        branches.append(ctx)
    else:
        alpha = UniformAttention(name="uniform_attention")(x_in)

    z = layers.Concatenate(name="fusion")(branches) if len(branches) > 1 else branches[0]
    z = layers.Dense(res_units, activation="relu", name="fusion_dense")(z)

    # ---- deep residual credit-risk learning (Eq. 34–36)
    if use_res:
        for b in range(n_blocks):
            z = ResidualBlock(res_units, dropout, name=f"residual_block_{b + 1}")(z)
    else:
        for b in range(n_blocks):  # same depth, no shortcut / LN
            z = layers.Dense(res_units, activation="relu", name=f"plain_block_{b + 1}")(z)
    z = layers.Dropout(dropout, name="dropout_head")(z)

    # ---- probabilistic credit-risk estimation (Eq. 37–38)
    logit = layers.Dense(1, name="logit")(z)
    p = layers.Activation("sigmoid", name="p_default")(logit)  # shape (B, 1)

    alpha = layers.Activation("linear", name="feature_attention")(alpha)
    return keras.Model(inputs=x_in, outputs=[p, alpha], name="ExplainableFinTechNet")


def predict_proba(model: keras.Model, X, batch_size: int = 2048):
    """Return P(default) as a 1-D numpy array."""
    import numpy as np

    out = []
    for i in range(0, len(X), batch_size):
        p, _ = model(tf.convert_to_tensor(X[i : i + batch_size], dtype=tf.float32), training=False)
        out.append(p.numpy().reshape(-1))
    return np.concatenate(out) if out else np.zeros(0)


def predict_attention(model: keras.Model, X, batch_size: int = 2048):
    """Return the feature-attention matrix α (n, d)."""
    import numpy as np

    out = []
    for i in range(0, len(X), batch_size):
        _, a = model(tf.convert_to_tensor(X[i : i + batch_size], dtype=tf.float32), training=False)
        out.append(a.numpy())
    return np.concatenate(out)
