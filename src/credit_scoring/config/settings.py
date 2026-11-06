"""Process environment settings, constructed by entry points at execution time."""

from pathlib import Path
from typing import Literal

from pydantic import Field, PositiveInt, SecretStr, field_validator, model_validator
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


class ServingSettings(Settings):
    """Serving configuration; source requirements do not affect training commands."""

    model_config = SettingsConfigDict(env_file=None, hide_input_in_errors=True)

    serving_api_key: SecretStr | None = Field(
        default=None, validation_alias="CREDIT_SCORING_SERVING_API_KEY", repr=False
    )

    @field_validator("serving_api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().strip():
            raise ValueError("API key must not be blank")
        return value

    serving_model_source: Literal["registry", "local"] = Field(
        default="registry", validation_alias="CREDIT_SCORING_SERVING_MODEL_SOURCE"
    )
    serving_model_path: Path | None = Field(
        default=None, validation_alias="CREDIT_SCORING_SERVING_MODEL_PATH"
    )
    serving_host: str = Field(
        default="127.0.0.1", min_length=1, validation_alias="CREDIT_SCORING_SERVING_HOST"
    )
    serving_port: int = Field(
        default=8000, ge=1, le=65535, validation_alias="CREDIT_SCORING_SERVING_PORT"
    )
    risk_medium_threshold: float = Field(
        default=0.20,
        gt=0,
        lt=1,
        allow_inf_nan=False,
        validation_alias="CREDIT_SCORING_RISK_MEDIUM_THRESHOLD",
    )
    risk_high_threshold: float = Field(
        default=0.50,
        gt=0,
        lt=1,
        allow_inf_nan=False,
        validation_alias="CREDIT_SCORING_RISK_HIGH_THRESHOLD",
    )

    @model_validator(mode="after")
    def validate_serving(self) -> ServingSettings:
        if self.risk_medium_threshold >= self.risk_high_threshold:
            raise ValueError("Medium threshold must be below high threshold")
        if self.serving_model_source == "local":
            if self.serving_model_path is None or not self.serving_model_path.is_dir():
                raise ValueError("Local serving requires an existing model directory")
        elif not self.mlflow_tracking_uri or not self.mlflow_tracking_uri.strip():
            raise ValueError("Registry serving requires a nonempty tracking URI")
        return self
