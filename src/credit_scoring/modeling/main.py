"""Compatibility imports; implementation lives in credit_scoring.training.main."""

from credit_scoring.training.main import (
    log_run_to_mlflow as log_run_to_mlflow,
)
from credit_scoring.training.main import (
    main as main,
)
from credit_scoring.training.main import (
    parse_args as parse_args,
)

if __name__ == "__main__":
    main()
