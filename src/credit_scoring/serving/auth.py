"""Optional single-key authentication without credential diagnostics."""

import hmac

from starlette.types import Scope

PUBLIC_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})


def authorized(scope: Scope, key: bytes | None) -> bool:
    if key is None or scope["path"] in PUBLIC_PATHS:
        return True
    values = [value for name, value in scope["headers"] if name.lower() == b"x-api-key"]
    # ASGI header values are raw bytes; configured Unicode secrets use UTF-8.
    return len(values) == 1 and hmac.compare_digest(values[0], key)
