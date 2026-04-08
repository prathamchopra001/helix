# Inference service — CPU/ONNX runtime (default profile)
# For TRT GPU inference use Dockerfile.tensorrt
FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Install shared package first (changes less often → better layer caching)
COPY shared/ /opt/helix/shared/
RUN pip install --no-cache-dir /opt/helix/shared/

# Install inference service
COPY services/inference/ /opt/helix/inference/
RUN pip install --no-cache-dir /opt/helix/inference/

ENV PYTHONPATH="/opt/helix/shared/src:/opt/helix/inference/src"

EXPOSE 8000

CMD ["uvicorn", "inference.main:app", "--host", "0.0.0.0", "--port", "8000"]
