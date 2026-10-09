import pytest
from fastapi.testclient import TestClient

from heartrisk import config

pytestmark = pytest.mark.skipif(not config.MODEL_FILE.exists(),
                                reason="run `python -m heartrisk.train` first")


@pytest.fixture(scope="module")
def client():
    from heartrisk.api import app
    with TestClient(app) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["status"] == "ready"


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["confidence"] <= 1
    assert "x-request-id" in r.headers


def test_predict_rejects_out_of_range(client, sample_patient):
    sample_patient["age"] = 150
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_predict_rejects_missing_field(client, sample_patient):
    sample_patient.pop("chol")
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_batch_endpoint(client, sample_patient):
    r = client.post("/predict/batch", json={"patients": [sample_patient] * 3})
    assert r.status_code == 200 and len(r.json()) == 3


def test_metrics_exposed(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "api_requests_total" in text and "model_predictions_total" in text
