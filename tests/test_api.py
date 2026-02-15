import pytest
from unittest.mock import patch


SAMPLE_TRANSACTION = {
    "Time": 0,
    "V1": -1.359807, "V2": -0.072781, "V3": 2.536346, "V4": 1.378155,
    "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
    "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
    "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
    "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
    "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
    "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
    "Amount": 149.62,
}


# ---------- HEALTH CHECK ----------

@pytest.mark.anyio
async def test_health_check(client):
    """GET / should return 200 with service info."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Fraud Detection API"
    assert "model_loaded" in data


# ---------- PREDICT — VALID INPUT ----------

@pytest.mark.anyio
async def test_predict_not_fraud(client, mock_model):
    """POST /predict with valid data should return fraudulent=false."""
    with patch("app.main.model", mock_model):
        response = await client.post("/predict", json=SAMPLE_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    assert data["fraudulent"] is False
    mock_model.predict.assert_called_once()


@pytest.mark.anyio
async def test_predict_fraud(client, mock_fraud_model):
    """POST /predict should return fraudulent=true when model predicts 1."""
    with patch("app.main.model", mock_fraud_model):
        response = await client.post("/predict", json=SAMPLE_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    assert data["fraudulent"] is True


# ---------- PREDICT — PARTIAL INPUT (DEFAULTS) ----------

@pytest.mark.anyio
async def test_predict_partial_input(client, mock_model):
    """POST /predict with only some fields should work (Pydantic defaults fill rest)."""
    partial_data = {"V1": -1.35, "Amount": 100.0}
    with patch("app.main.model", mock_model):
        response = await client.post("/predict", json=partial_data)
    assert response.status_code == 200
    assert "fraudulent" in response.json()


# ---------- PREDICT — NO MODEL LOADED ----------

@pytest.mark.anyio
async def test_predict_no_model(client):
    """POST /predict should return 503 when model is not loaded."""
    with patch("app.main.model", None):
        response = await client.post("/predict", json=SAMPLE_TRANSACTION)
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]


# ---------- PREDICT — INVALID INPUT ----------

@pytest.mark.anyio
async def test_predict_invalid_type(client, mock_model):
    """POST /predict with wrong types should return 422 validation error."""
    bad_data = {"V1": "not_a_number", "Amount": "abc"}
    with patch("app.main.model", mock_model):
        response = await client.post("/predict", json=bad_data)
    assert response.status_code == 422
