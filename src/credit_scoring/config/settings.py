"""Process environment settings, constructed by entry points at execution time."""

from pathlib import Path
from typing import Literal

from pydantic import Field, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Shared configuration. Developer dotenv files are never loaded implicitly."""

    model_config = SettingsConfigDict(env_file=None)

    app_env: str = Field(default="development", validation_alias="APP_ENV")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"] = Field(
        default="INFO", validation_alias="LOG_LEVEL"
    )
    dataset: Path | None = Field(default=None, validation_alias="CREDIT_SCORING_DATASET")
    max_iter: PositiveInt | None = Field(default=None, validation_alias="CREDIT_SCORING_MAX_ITER")
    retrain_max_iter: PositiveInt = Field(
        default=500, validation_alias="CREDIT_SCORING_RETRAIN_MAX_ITER"
    )
    max_missing_ratio: float = Field(
        default=0.10, ge=0, le=1, validation_alias="CREDIT_SCORING_MAX_MISSING_RATIO"
    )
    mlflow_tracking_uri: str | None = Field(
        default=None, validation_alias="CREDIT_SCORING_MLFLOW_TRACKING_URI"
    )
    mlflow_experiment: str = Field(
        default="credit-scoring", validation_alias="CREDIT_SCORING_MLFLOW_EXPERIMENT"
    )

    registry_enabled: bool = Field(
        default=False, validation_alias="CREDIT_SCORING_REGISTRY_ENABLED"
    )
    registered_model_name: str = Field(
        default="credit-scoring",
        min_length=1,
        validation_alias="CREDIT_SCORING_REGISTERED_MODEL_NAME",
    )
    promotion_policy_path: Path | None = Field(
        default=None, validation_alias="CREDIT_SCORING_PROMOTION_POLICY_PATH"
    )

    @field_validator("registered_model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Registered model name must not be blank")
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper()
