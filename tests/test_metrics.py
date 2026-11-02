import pytest
from genebridge_backend import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"

def test_metrics_endpoint(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert b"genebridge_predictions_total" in res.data
    assert b"genebridge_model_accuracy_ratio" in res.data
    assert b"genebridge_inference_duration_seconds" in res.data

def test_prediction_and_metric_increment(client):
    payload = {
        "condition": "spinal_cord_injury",
        "age": 32,
        "severity": "moderate",
        "gender": "male"
    }
    pred_res = client.post("/api/predict", json=payload)
    assert pred_res.status_code == 200
    data = pred_res.get_json()
    assert "results" in data
    assert len(data["results"]) > 0

    # Verify metrics updated
    metrics_res = client.get("/metrics")
    assert metrics_res.status_code == 200
    assert b'condition="spinal_cord_injury"' in metrics_res.data
    assert b'status="success"' in metrics_res.data

