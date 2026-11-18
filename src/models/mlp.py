from tensorflow import keras
from tensorflow.keras import layers


def tab_branch(tab_input, adapt_data, units=(32, 16)):
    """Normalisation (adapted on TRAIN data) followed by Dense + ReLU layers."""
    norm = layers.Normalization(axis=-1, name="tab_norm")
    norm.adapt(adapt_data)
    x = norm(tab_input)
    for u in units:
        x = layers.Dense(u, activation="relu")(x)
    return x


def build_mlp(adapt_data):
    """Baseline: tabular body measurements only."""
    inp = keras.Input(shape=(4,), name="tab")
    x = tab_branch(inp, adapt_data)
    out = layers.Dense(1, name="yield")(x)
    return keras.Model(inp, out, name="mlp_only")