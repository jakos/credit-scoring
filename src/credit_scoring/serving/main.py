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
    except ValidationError:
        parser.error(
            "Invalid serving settings; check model source, required source settings, thresholds, host and port"
        )


def main() -> None:
    settings = parse_args()
    uvicorn.run(create_app(settings), host=settings.serving_host, port=settings.serving_port)


if __name__ == "__main__":
    main()
