"""Physics-conditioned causal TCN for probabilistic multi-horizon forecasting.

The model predicts ordered P10/P50/P90 quantiles for the next H samples.
No future samples or full-series decomposition are used by this module.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="wave_fess")
class OrderedQuantiles(layers.Layer):
    """Map unconstrained logits to 0 <= q10 <= q50 <= q90 <= 1."""

    def call(self, inputs):
        low_logit, mid_logit, high_logit = tf.unstack(inputs, axis=-1)
        q10 = tf.nn.sigmoid(low_logit)
        q50 = q10 + (1.0 - q10) * tf.nn.sigmoid(mid_logit)
        q90 = q50 + (1.0 - q50) * tf.nn.sigmoid(high_logit)
        return tf.stack([q10, q50, q90], axis=-1)


@keras.utils.register_keras_serializable(package="wave_fess")
class FiLM(layers.Layer):
    """Feature-wise linear modulation from a physical context vector."""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.projection = layers.Dense(2 * self.channels)

    def call(self, inputs):
        features, context = inputs
        params = self.projection(context)
        gamma, beta = tf.split(params, 2, axis=-1)
        gamma = gamma[:, None, :]
        beta = beta[:, None, :]
        return features * (1.0 + gamma) + beta

    def get_config(self):
        cfg = super().get_config()
        cfg.update(channels=self.channels)
        return cfg


@keras.utils.register_keras_serializable(package="wave_fess")
class LastTimeStep(layers.Layer):
    def call(self, inputs):
        return inputs[:, -1, :]


@keras.utils.register_keras_serializable(package="wave_fess")
class ZeroLike(layers.Layer):
    def call(self, inputs):
        return tf.zeros_like(inputs)


@keras.utils.register_keras_serializable(package="wave_fess")
class ControlAwareQuantileLoss(keras.losses.Loss):
    """Quantile loss with first-step, ramp, peak, and bound penalties."""

    def __init__(
        self,
        first_step_weight=2.0,
        ramp_weight=0.30,
        peak_weight=0.50,
        peak_threshold=0.75,
        physics_weight=0.10,
        name="control_aware_quantile_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.first_step_weight = float(first_step_weight)
        self.ramp_weight = float(ramp_weight)
        self.peak_weight = float(peak_weight)
        self.peak_threshold = float(peak_threshold)
        self.physics_weight = float(physics_weight)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        quantiles = tf.constant([0.10, 0.50, 0.90], dtype=y_pred.dtype)
        error = y_true[..., None] - y_pred
        pinball = tf.maximum(quantiles * error, (quantiles - 1.0) * error)

        weights = tf.ones_like(y_true[0], dtype=y_pred.dtype)
        first = tf.reshape(tf.cast(self.first_step_weight, y_pred.dtype), (1,))
        weights = tf.concat([first, weights[1:]], axis=0)

        peak_mask = tf.cast(y_true >= self.peak_threshold, y_pred.dtype)
        sample_weight = 1.0 + self.peak_weight * peak_mask
        quantile_term = tf.reduce_mean(pinball * weights[None, :, None] * sample_weight[..., None])

        median = y_pred[..., 1]
        true_ramp = y_true[:, 1:] - y_true[:, :-1]
        pred_ramp = median[:, 1:] - median[:, :-1]
        ramp_term = tf.reduce_mean(tf.abs(true_ramp - pred_ramp))

        # OrderedQuantiles already enforces these bounds. Keeping the penalty
        # makes the physical requirement explicit and protects custom heads.
        lower_violation = tf.nn.relu(-y_pred)
        upper_violation = tf.nn.relu(y_pred - 1.0)
        physics_term = tf.reduce_mean(lower_violation + upper_violation)

        return quantile_term + self.ramp_weight * ramp_term + self.physics_weight * physics_term

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            first_step_weight=self.first_step_weight,
            ramp_weight=self.ramp_weight,
            peak_weight=self.peak_weight,
            peak_threshold=self.peak_threshold,
            physics_weight=self.physics_weight,
        )
        return cfg


def _film(x, context, channels, name):
    return FiLM(channels, name=f"{name}_film")([x, context])


def _residual_block(x, context, channels, dilation, dropout, use_physics, name):
    residual = x
    if x.shape[-1] != channels:
        residual = layers.Conv1D(channels, 1, padding="same", name=f"{name}_skip")(residual)

    y = layers.Conv1D(
        channels,
        3,
        padding="causal",
        dilation_rate=dilation,
        name=f"{name}_conv1",
    )(x)
    y = layers.LayerNormalization(name=f"{name}_norm1")(y)
    if use_physics:
        y = _film(y, context, channels, f"{name}_1")
    y = layers.Activation("swish", name=f"{name}_act1")(y)
    y = layers.Dropout(dropout, name=f"{name}_drop1")(y)

    y = layers.Conv1D(
        channels,
        3,
        padding="causal",
        dilation_rate=dilation,
        name=f"{name}_conv2",
    )(y)
    y = layers.LayerNormalization(name=f"{name}_norm2")(y)
    if use_physics:
        y = _film(y, context, channels, f"{name}_2")
    y = layers.Activation("swish", name=f"{name}_act2")(y)
    y = layers.Dropout(dropout, name=f"{name}_drop2")(y)
    return layers.Add(name=f"{name}_residual")([residual, y])


def build_model(
    n_in=156,
    horizon=16,
    n_context=8,
    channels=32,
    dilations=(1, 2, 4, 8, 16, 32),
    dropout=0.10,
    use_physics=True,
):
    """Build PC-TCN (use_physics=True) or the plain-TCN ablation."""
    history = keras.Input((n_in, 1), name="power_history")
    context = keras.Input((n_context,), name="physics_context")

    context_features = layers.Dense(32, activation="swish", name="context_mlp_1")(context)
    context_features = layers.Dense(32, activation="swish", name="context_mlp_2")(context_features)

    x = layers.Conv1D(channels, 1, padding="same", name="input_projection")(history)
    for dilation in dilations:
        x = _residual_block(
            x,
            context_features,
            channels,
            dilation,
            dropout,
            use_physics,
            name=f"tcn_d{dilation}",
        )

    x = LastTimeStep(name="last_causal_state")(x)
    if use_physics:
        x = layers.Concatenate(name="forecast_features")([x, context_features])
    else:
        # Connect the context input without allowing it to affect the ablation.
        zero_context = ZeroLike(name="ignore_physics_context")(context_features)
        x = layers.Concatenate(name="forecast_features")([x, zero_context])

    x = layers.Dense(64, activation="swish", name="forecast_dense")(x)
    logits = layers.Dense(horizon * 3, name="quantile_logits")(x)
    logits = layers.Reshape((horizon, 3), name="quantile_logit_matrix")(logits)
    output = OrderedQuantiles(name="ordered_quantiles")(logits)

    name = "PC_TCN" if use_physics else "TCN"
    return keras.Model([history, context], output, name=name)


def build_recurrent_baseline(kind, n_in=156, horizon=16, n_context=8):
    """RNN/GRU/CNN-RNN baselines with the same probabilistic output head."""
    kind = kind.lower()
    history = keras.Input((n_in, 1), name="power_history")
    context = keras.Input((n_context,), name="physics_context")
    if kind == "rnn":
        x = layers.SimpleRNN(64, activation="tanh", name="rnn")(history)
    elif kind == "gru":
        x = layers.GRU(64, name="gru")(history)
    elif kind == "cnn_rnn":
        x = layers.Conv1D(64, 3, activation="swish", name="conv")(history)
        x = layers.MaxPooling1D(2, name="pool")(x)
        x = layers.SimpleRNN(64, activation="tanh", name="rnn")(x)
    else:
        raise ValueError(f"Unknown baseline: {kind}")
    ignored_context = ZeroLike(name="ignore_physics_context")(context)
    x = layers.Concatenate(name="forecast_features")([x, ignored_context])
    x = layers.Dense(64, activation="swish", name="forecast_dense")(x)
    logits = layers.Dense(horizon * 3, name="quantile_logits")(x)
    logits = layers.Reshape((horizon, 3), name="quantile_logit_matrix")(logits)
    output = OrderedQuantiles(name="ordered_quantiles")(logits)
    return keras.Model([history, context], output, name=kind.upper())


def load_forecaster(path):
    """Load a saved model for inference without recompiling its custom loss."""
    return keras.models.load_model(path, compile=False)
