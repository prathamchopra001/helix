"""
Inference service configuration — all values come from environment variables.
Pydantic validates on startup; missing required vars raise an error immediately.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API auth — comma-separated list of valid API keys
    api_keys: str = "dev-key"

    # Postgres — used to fetch the last 60 bars for a ticker
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str
    postgres_user: str
    postgres_password: str

    # MinIO — model artifacts live here
    minio_endpoint: str = "minio:9000"
    minio_access_key: str
    minio_secret_key: str
    minio_bucket_models: str = "models"

    # MLflow — registry for finding the current Production model version
    mlflow_tracking_uri: str = "http://mlflow:5000"

    # Inference backend: "tensorrt" | "onnx" | "pytorch"
    # The loader tries TRT first regardless; this is the starting preference
    inference_backend: str = "tensorrt"

    # How often (seconds) to poll MLflow for a new Production model
    model_poll_interval: int = 60

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def valid_api_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    class Config:
        env_file = ".env"


settings = Settings()
