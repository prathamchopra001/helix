"""
API key authentication middleware.

Clients must send: X-API-Key: <key>
Valid keys are defined in the API_KEYS env var (comma-separated).

Returns 401 if header is missing, 403 if key is invalid.
"""
import json

from fastapi import Request
from inference.config import settings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from shared.logging import get_logger

log = get_logger(__name__)

API_KEY_HEADER = "X-API-Key"

# Paths that don't require auth
_PUBLIC_PATHS = {"/health", "/ready", "/metrics"}


def _json_response(status_code: int, detail: str) -> Response:
    return Response(
        content=json.dumps({"detail": detail}),
        status_code=status_code,
        media_type="application/json",
    )


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        api_key = request.headers.get(API_KEY_HEADER)

        if not api_key:
            log.warning(
                "auth_missing_key",
                path=request.url.path,
                correlation_id=request.headers.get("X-Correlation-ID", ""),
            )
            return _json_response(401, "X-API-Key header required")

        if api_key not in settings.valid_api_keys:
            log.warning(
                "auth_invalid_key",
                path=request.url.path,
                correlation_id=request.headers.get("X-Correlation-ID", ""),
            )
            return _json_response(403, "Invalid API key")

        return await call_next(request)
