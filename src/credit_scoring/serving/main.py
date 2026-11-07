"""Run the scoring API without reload or automatic model refresh."""

import argparse
from collections.abc import Sequence

import uvicorn
from pydantic import ValidationError

from credit_scoring.config.settings import ServingSettings
from credit_scoring.serving.app import create_app


def parse_args(argv: Sequence[str] | None = None) -> ServingSettings:
    parser = argparse.ArgumentParser(description="Serve a fixed credit-risk model snapshot")
    parser.add_argument(
        "--host", default=None, help="Bind host (overrides CREDIT_SCORING_SERVING_HOST)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port, 1-65535 (overrides CREDIT_SCORING_SERVING_PORT)",
    )
    args = parser.parse_args(argv)
    overrides = {
        name: value
        for name, value in {
            "CREDIT_SCORING_SERVING_HOST": args.host,
            "CREDIT_SCORING_SERVING_PORT": args.port,
        }.items()
        if value is not None
    }
    try:
        return ServingSettings(**overrides)
    except ValidationError as exc:
        aliases = {str(field.validation_alias) for field in ServingSettings.model_fields.values()}
        model_errors = {
            "Value error, Medium threshold must be below high threshold": [
                "CREDIT_SCORING_RISK_MEDIUM_THRESHOLD",
                "CREDIT_SCORING_RISK_HIGH_THRESHOLD",
            ],
            "Value error, Local serving requires an existing model directory": [
                "CREDIT_SCORING_SERVING_MODEL_PATH"
            ],
            "Value error, Registry serving requires a nonempty tracking URI": [
                "CREDIT_SCORING_MLFLOW_TRACKING_URI"
            ],
        }
        names: set[str] = set()
        for error in exc.errors(include_input=False):
            names.update(str(part) for part in error["loc"] if str(part) in aliases)
            if not error["loc"]:
                names.update(
                    model_errors.get(error["msg"], ["CREDIT_SCORING_SERVING_MODEL_SOURCE"])
                )
        parser.error("Invalid serving settings: " + ", ".join(sorted(names)))


def main() -> None:
    settings = parse_args()
    uvicorn.run(
        create_app(settings),
        host=settings.serving_host,
        port=settings.serving_port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
