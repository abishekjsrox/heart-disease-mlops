# Model-serving image for the heart disease risk API
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/models/model.joblib \
    MODEL_META_PATH=/app/models/model_meta.json

WORKDIR /app
RUN useradd --create-home --uid 10001 appuser

# dependencies first so this layer is cached between code changes
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src ./src
COPY models/model.joblib models/model_meta.json ./models/
COPY samples ./samples

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# one worker per container; scale out with Kubernetes replicas instead
CMD ["uvicorn", "heartrisk.api:app", "--host", "0.0.0.0", "--port", "8000"]
