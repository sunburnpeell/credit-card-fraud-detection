import pytest
from fastapi.testclient import TestClient

from fraud_detection.api import MODEL_PATH, app

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="train the model first")


def test_predict():
    c = TestClient(app)
    body = {"Time": 1000, "Amount": 25.5, "features": {f"V{i}": 0.0 for i in range(1, 29)}}
    r = c.post("/predict", json=body)
    assert r.status_code == 200
    j = r.json()
    assert 0 <= j["fraud_probability"] <= 1 and len(j["top_factors"]) == 3
