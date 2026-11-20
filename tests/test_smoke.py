"""API smoke tests. Run from the repo root:  pytest -q"""
import copy
import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = json.loads((ROOT / "examples" / "payload.json").read_text())


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_plausible_number():
    with TestClient(app) as client:
        r = client.post("/predict", json=PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert 20 <= body["utilization_pct"] <= 90
        assert abs(body["utilization_pct"] + body["waste_pct"] - 100) < 0.05
        assert abs(body["utilization_pct"] - PAYLOAD["expected_utilization_pct"]) < 10


def test_missing_measurement_is_imputed():
    payload = copy.deepcopy(PAYLOAD)
    del payload["hip_circ"]
    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 200 and r.json()["imputed_fields"] == ["hip_circ"]


def test_bad_mask_shape_rejected():
    payload = copy.deepcopy(PAYLOAD)
    payload["mask"] = payload["mask"][:64]
    with TestClient(app) as client:
        assert client.post("/predict", json=payload).status_code == 422