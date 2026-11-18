from tensorflow import keras
from tensorflow.keras import layers

from src.models.cnn import cnn_branch
from src.models.mlp import tab_branch


def build_fusion(adapt_data):
    """Multi-modal model: CNN(mask) + MLP(body measurements) -> concat -> regressor."""
    mask_in = keras.Input(shape=(128, 128, 1), name="mask")
    tab_in = keras.Input(shape=(4,), name="tab")
    c = cnn_branch(mask_in)
    t = tab_branch(tab_in, adapt_data, units=(16,))
    x = layers.Concatenate()([c, t])
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, name="yield")(x)
    return keras.Model([mask_in, tab_in], out, name="fusion")