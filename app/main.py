import os
import time
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import pandas as pd


# ---------- STRUCTURED LOGGING ----------
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)


# ---------- MODEL LOADING ----------
model = None


def load_model():
    """Load model from local path first, then fall back to MLflow registry."""
    global model

    # Try local model directory first (for Docker/Render deployment)
    local_model_path = os.getenv("MODEL_PATH", "model")
    if os.path.exists(local_model_path) and os.path.exists(os.path.join(local_model_path, "model.pkl")):
        try:
            import pickle
            with open(os.path.join(local_model_path, "model.pkl"), "rb") as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from local path: {local_model_path}")
            return
        except Exception as e:
            logger.warning(f"Failed to load local model: {e}")

    # Fall back to MLflow registry
    try:
        import mlflow.pyfunc

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
        mlflow.set_tracking_uri(tracking_uri)

        model_name = os.getenv("MODEL_NAME", "FraudDetectionModel")
        model_stage = os.getenv("MODEL_STAGE", "")

        if model_stage:
            model_uri = f"models:/{model_name}/{model_stage}"
        else:
            model_uri = f"models:/{model_name}/latest"

        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Model loaded from MLflow: {model_uri}")
    except Exception as e:
        logger.warning(f"Could not load model: {e}. /predict will be unavailable.")
        model = None


# ---------- LIFESPAN ----------
@asynccontextmanager
async def lifespan(app):
    logger.info("Starting Fraud Detection API...")
    load_model()
    yield
    logger.info("Shutting down Fraud Detection API...")


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection using MLflow-served model",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- REQUEST LOGGING MIDDLEWARE ----------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({duration_ms}ms)"
    )
    return response


# ---------- REQUEST / RESPONSE SCHEMAS ----------
class TransactionInput(BaseModel):
    Time: float = Field(default=0, description="Seconds elapsed between this and the first transaction")
    V1: float = Field(default=0)
    V2: float = Field(default=0)
    V3: float = Field(default=0)
    V4: float = Field(default=0)
    V5: float = Field(default=0)
    V6: float = Field(default=0)
    V7: float = Field(default=0)
    V8: float = Field(default=0)
    V9: float = Field(default=0)
    V10: float = Field(default=0)
    V11: float = Field(default=0)
    V12: float = Field(default=0)
    V13: float = Field(default=0)
    V14: float = Field(default=0)
    V15: float = Field(default=0)
    V16: float = Field(default=0)
    V17: float = Field(default=0)
    V18: float = Field(default=0)
    V19: float = Field(default=0)
    V20: float = Field(default=0)
    V21: float = Field(default=0)
    V22: float = Field(default=0)
    V23: float = Field(default=0)
    V24: float = Field(default=0)
    V25: float = Field(default=0)
    V26: float = Field(default=0)
    V27: float = Field(default=0)
    V28: float = Field(default=0)
    Amount: float = Field(default=0, description="Transaction amount")


class PredictionResponse(BaseModel):
    fraudulent: bool
    confidence: str = "Model prediction (binary)"


# ---------- ENDPOINTS ----------
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "Fraud Detection API",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: TransactionInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train and register a model first using model_pipeline.py",
        )

    try:
        df = pd.DataFrame([data.model_dump()])
        expected_columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        df = df[expected_columns]

        prediction = model.predict(df)[0]
        return PredictionResponse(fraudulent=bool(prediction))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
