"""FastAPI model-serving app with request logging and Prometheus metrics.

Run locally:  uvicorn heartrisk.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)
from pydantic import BaseModel, ConfigDict, Field

from heartrisk import __version__, config
from heartrisk.predict import RiskModel


# ---------------------------------------------------------------- logging ---
class JsonFormatter(logging.Formatter):
    def format(self, record):
        base = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        base.update(getattr(record, "extra_fields", {}))
        return json.dumps(base)


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("heartrisk.api")
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    logger.handlers.clear()
    handlers = [logging.StreamHandler(sys.stdout)]
    if os.getenv("LOG_FILE"):
        handlers.append(logging.FileHandler(os.environ["LOG_FILE"]))
    for h in handlers:
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
    logger.propagate = False
    return logger


log = _setup_logging()

# ---------------------------------------------------------------- metrics ---
REQUESTS = Counter("api_requests_total", "HTTP requests", ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("model_predictions_total", "Predictions served", ["outcome"])
PROBABILITY = Histogram("model_prediction_probability", "Predicted disease probability",
                        buckets=[i / 10 for i in range(1, 11)])
MODEL_INFO = Gauge("model_info", "Loaded model (value is always 1)",
                   ["model_name", "model_version"])
INVALID = Counter("api_invalid_requests_total", "Requests rejected by validation")


# ---------------------------------------------------------------- schemas ---
class PatientFeatures(BaseModel):
    """The 13 Cleveland attributes, validated against clinically plausible ranges."""
    model_config = ConfigDict(json_schema_extra={"example": {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6}})

    age: int = Field(..., ge=18, le=100, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain type (4 = asymptomatic)")
    trestbps: float = Field(..., ge=70, le=220, description="Resting BP (mm Hg)")
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(..., ge=60, le=220, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=7, description="ST depression vs rest")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST segment")
    ca: int | None = Field(None, ge=0, le=3, description="Major vessels coloured (0-3)")
    thal: Literal[3, 6, 7] | None = Field(None, description="3 normal, 6 fixed, 7 reversible")


class PredictionOut(BaseModel):
    prediction: int
    label: str
    probability_disease: float
    confidence: float
    risk_band: str
    threshold: float
    model_version: str


class BatchIn(BaseModel):
    patients: list[PatientFeatures] = Field(..., min_length=1, max_length=100)


# -------------------------------------------------------------------- app ---
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH", str(config.MODEL_FILE))
    meta_path = os.getenv("MODEL_META_PATH", str(config.MODEL_META_FILE))
    STATE["model"] = RiskModel.load(model_path, meta_path)
    m = STATE["model"]
    MODEL_INFO.labels(m.meta.get("model_name", "unknown"), m.version).set(1)
    log.info("model loaded", extra={"extra_fields": {
        "model_name": m.meta.get("model_name"), "model_version": m.version,
        "threshold": m.threshold}})
    yield
    STATE.clear()


app = FastAPI(title="Heart Disease Risk API", version=__version__, lifespan=lifespan,
              description="Predicts heart disease risk from 13 UCI Cleveland attributes.")


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        endpoint = request.scope.get("route").path if request.scope.get("route") else "unmatched"
        if endpoint != "/metrics":
            REQUESTS.labels(request.method, endpoint, str(status)).inc()
            LATENCY.labels(endpoint).observe(elapsed)
            if status == 422:
                INVALID.inc()
            log.info("request", extra={"extra_fields": {
                "request_id": request_id, "method": request.method, "path": endpoint,
                "status": status, "latency_ms": round(elapsed * 1000, 2),
                "client": request.client.host if request.client else None}})


def _model() -> RiskModel:
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail="model not loaded")
    return STATE["model"]


def _serve(patients: list[PatientFeatures]) -> list[dict]:
    results = _model().predict([p.model_dump() for p in patients])
    for r in results:
        PREDICTIONS.labels("disease" if r["prediction"] else "no_disease").inc()
        PROBABILITY.observe(r["probability_disease"])
    log.info("prediction", extra={"extra_fields": {
        "n": len(results), "outcomes": [r["prediction"] for r in results],
        "probabilities": [r["probability_disease"] for r in results]}})
    return results


@app.get("/")
def root():
    return {"service": "heart-disease-risk-api", "version": __version__,
            "docs": "/docs", "endpoints": ["/predict", "/predict/batch", "/health",
                                           "/ready", "/model-info", "/metrics"]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    m = _model()
    return {"status": "ready", "model_version": m.version}


@app.get("/model-info")
def model_info():
    meta = dict(_model().meta)
    meta.pop("features", None)
    return meta


@app.post("/predict", response_model=PredictionOut)
def predict(patient: PatientFeatures):
    return _serve([patient])[0]


@app.post("/predict/batch", response_model=list[PredictionOut])
def predict_batch(batch: BatchIn):
    return _serve(batch.patients)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
