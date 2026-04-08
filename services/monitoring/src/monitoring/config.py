"""Monitoring service configuration."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class MonitoringConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    postgres_host: str
    postgres_port: int = 5432
    postgres_db: str
    postgres_user: str
    postgres_password: str

    drift_psi_threshold: float = 0.2
    drift_min_features: int = 3
    model_name: str = "helix_anomaly_detector"

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
