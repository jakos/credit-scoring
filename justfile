set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

check:
    uv run --locked ruff format --check .
    uv run --locked ruff check .
    uv run --locked mypy
    uv run --locked pytest -q
