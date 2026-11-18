from tensorflow import keras
from tensorflow.keras import layers


def cnn_branch(mask_input, filters=(16, 32, 64, 64)):
    """Conv2D -> BatchNorm -> ReLU -> MaxPool blocks, then Flatten + Dense.

    Flatten (not global pooling) keeps WHERE pieces are, which is what tells the
    network how long the layout is.
    """
    x = mask_input
    for f in filters:
        x = layers.Conv2D(f, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    return layers.Dense(64, activation="relu")(x)


def build_cnn():
    """Baseline: spatial mask only."""
    inp = keras.Input(shape=(128, 128, 1), name="mask")
    x = cnn_branch(inp)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, name="yield")(x)
    return keras.Model(inp, out, name="cnn_only")