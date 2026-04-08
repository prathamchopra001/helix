FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY shared/ /opt/helix/shared/
RUN pip install --no-cache-dir /opt/helix/shared/

COPY services/monitoring/ /opt/helix/monitoring/
RUN pip install --no-cache-dir /opt/helix/monitoring/

ENV PYTHONPATH="/opt/helix/shared/src:/opt/helix/monitoring/src"

# Monitoring runs as a one-shot task from Airflow — no long-running server needed
CMD ["python", "-c", "from monitoring.main import run_monitoring; run_monitoring()"]
