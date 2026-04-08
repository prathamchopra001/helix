FROM apache/airflow:2.8.2-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Upgrade setuptools so setuptools.backends is available
RUN pip install --upgrade setuptools wheel

# Install shared package first — ingestion depends on it
COPY --chown=airflow:root shared/ /opt/helix/shared/
RUN pip install --no-cache-dir /opt/helix/shared/

# Install ingestion package
COPY --chown=airflow:root services/ingestion/ /opt/helix/ingestion/
RUN pip install --no-cache-dir /opt/helix/ingestion/

# Install ETL package
COPY --chown=airflow:root services/etl/ /opt/helix/etl/
RUN pip install --no-cache-dir /opt/helix/etl/

# Install training package
COPY --chown=airflow:root services/training/ /opt/helix/training/
RUN pip install --no-cache-dir /opt/helix/training/

# Install monitoring package
COPY --chown=airflow:root services/monitoring/ /opt/helix/monitoring/
RUN pip install --no-cache-dir /opt/helix/monitoring/
