"""Train the three neural models with the same split and seed.

Run from the repo root:  python -m src.train
"""
import argparse
import json

from tensorflow import keras

from src.data.layout import load_config
from src.features.preprocess import load_splits, save_preprocessor, ARTIFACT_DIR, ROOT
from src.models.cnn import build_cnn
from src.models.fusion import build_fusion
from src.models.mlp import build_mlp

REPORT_DIR = ROOT / "reports"


MODEL_INPUTS = {                    # which arrays each model takes
    "mlp_only": ["tab"],
    "cnn_only": ["mask"],
    "fusion": ["mask", "tab"],
}


def inputs_for(model, split):
    """Pick only the arrays a model expects (a plain array if it has one input)."""
    names = MODEL_INPUTS[model.name]
    if len(names) == 1:
        return split[names[0]]
    return {name: split[name] for name in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mlp_only", "cnn_only", "fusion"],
                    help="which models to train (default: all three)")
    ap.add_argument("--epochs", type=int, default=None, help="override max epochs")
    args = ap.parse_args()

    cfg = load_config()
    tcfg = cfg.get("train", {})
    seed = cfg["seed"]
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    splits, prep = load_splits()
    save_preprocessor(prep)
    train, val = splits["train"], splits["val"]

    builders = {
        "mlp_only": lambda: build_mlp(train["tab"]),
        "cnn_only": lambda: build_cnn(),
        "fusion": lambda: build_fusion(train["tab"]),
    }

    for name in args.models:
        build = builders[name]
        keras.utils.set_random_seed(seed)          # same seed for every model
        model = build()
        model.compile(optimizer=keras.optimizers.Adam(tcfg.get("lr", 1e-3)), loss="mse", metrics=["mae"])
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=tcfg.get("patience", 15),
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5),
        ]
        hist = model.fit(
            inputs_for(model, train), train["y"],
            validation_data=(inputs_for(model, val), val["y"]),
            epochs=args.epochs or tcfg.get("epochs", 80), batch_size=tcfg.get("batch_size", 32),
            callbacks=callbacks, verbose=0,
        )
        model.save(ARTIFACT_DIR / f"{name}.keras")
        with open(REPORT_DIR / f"history_{name}.json", "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in hist.history.items()}, f)
        best = min(hist.history["val_loss"])
        print(f"{name:9s} epochs run: {len(hist.history['loss']):3d} | best val_loss (standardised MSE): {best:.4f}")


if __name__ == "__main__":
    main()