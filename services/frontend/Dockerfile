# services/frontend/Dockerfile
# Build stage — installs Python dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

COPY services/frontend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Runtime stage — no build tools
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /install /usr/local
COPY services/frontend/src/app/ /app/

EXPOSE 8501

CMD ["streamlit", "run", "main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.baseUrlPath=frontend"]
