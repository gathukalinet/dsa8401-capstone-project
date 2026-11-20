"""FastAPI service: predict fabric utilization (%) from body measurements + layout mask.

Run locally from the repo root:  uvicorn api.main:app --reload
Docs (auto-generated):           http://localhost:8000/docs
"""
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from tensorflow import keras

from src.features.preprocess import RAW_FEATS, load_preprocessor, make_tabular, to_pct

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "fusion.keras"
MASK_SIZE = 128

# ranges the model was trained on; outside them the prediction is an extrapolation
TRAIN_RANGE = {"waist_circ": (58.0, 110.0), "hip_circ": (80.0, 135.0), "total_length": (85.0, 112.0)}

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = keras.models.load_model(MODEL_PATH)
    state["prep"] = load_preprocessor()
    yield
    state.clear()


app = FastAPI(
    title="Trouser marker yield predictor",
    description="Multi-modal (CNN + MLP) surrogate that predicts fabric utilization for a cutting layout.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    # body measurements in cm; any of them may be omitted (they are imputed with training medians)
    waist_circ: float | None = Field(None, gt=0, description="waist circumference, cm")
    hip_circ: float | None = Field(None, gt=0, description="hip circumference, cm")
    total_length: float | None = Field(None, gt=0, description="trouser total length, cm")
    # layout mask: 128 rows x 128 columns of 0/1 (1 = pattern piece), rendered at the training scale
    mask: list[list[int]]

    @field_validator("mask")
    @classmethod
    def check_mask(cls, v):
        if len(v) != MASK_SIZE or any(len(r) != MASK_SIZE for r in v):
            raise ValueError(f"mask must be {MASK_SIZE} x {MASK_SIZE}")
        if any(px not in (0, 1) for r in v for px in r):
            raise ValueError("mask values must be 0 or 1")
        return v


class PredictResponse(BaseModel):
    utilization_pct: float
    waste_pct: float
    model: str
    imputed_fields: list[str]
    warnings: list[str]


@app.get("/health")
def health():
    return {"status": "ok" if "model" in state else "loading", "model": "fusion"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    prep = state["prep"]

    values = {c: getattr(req, c) for c in RAW_FEATS}
    imputed = [c for c, v in values.items() if v is None]
    warnings = [
        f"{c}={v} is outside the training range {TRAIN_RANGE[c]}; prediction is an extrapolation"
        for c, v in values.items()
        if v is not None and not (TRAIN_RANGE[c][0] <= v <= TRAIN_RANGE[c][1])
    ]

    tab = make_tabular(pd.DataFrame([values]), prep)                       # (1, 4), imputed + ratio
    mask = np.asarray(req.mask, dtype="float32").reshape(1, MASK_SIZE, MASK_SIZE, 1)
    z = state["model"].predict({"mask": mask, "tab": tab}, verbose=0)
    pct = float(to_pct(z, prep)[0])

    return PredictResponse(
        utilization_pct=round(pct, 2),
        waste_pct=round(100.0 - pct, 2),
        model="fusion",
        imputed_fields=imputed,
        warnings=warnings,
    )