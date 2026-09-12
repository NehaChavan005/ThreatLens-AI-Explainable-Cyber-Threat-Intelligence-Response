"""API key authentication for the unified IDS API.

Same contract as the legacy ML-IDS inference server: `X-API-Key` header on
HTTP routes, `api_key` query parameter for WebSocket connections. Auth can
be disabled entirely via `AUTH_ENABLED=false` for local development.
"""

import logging
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import settings

logger = logging.getLogger(__name__)

PUBLIC_PATHS = {
    "/",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/openapi.yaml",
    "/metrics",
}


def _safe_equals(provided: str, expected: str) -> bool:
    return secrets.compare_digest(provided.encode(), expected.encode())


def is_authorized(request: Request) -> bool:
    """Return True when the request carries a valid API key."""
    if not settings.AUTH_ENABLED or not settings.API_KEYS:
        return True
    provided = request.headers.get("X-API-Key", "")
    if not provided:
        return False
    return any(_safe_equals(provided, k) for k in settings.API_KEYS)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Enforce X-API-Key on every route that is not explicitly public."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if not settings.AUTH_ENABLED or not settings.API_KEYS:
            return await call_next(request)

        if path in PUBLIC_PATHS:
            return await call_next(request)
        if path.startswith("/dashboard") or path.startswith("/static"):
            return await call_next(request)

        if not is_authorized(request):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid X-API-Key header"},
            )

        return await call_next(request)


def verify_ws_api_key(api_key: str) -> bool:
    """Verify an API key supplied via a WebSocket query parameter."""
    if not settings.AUTH_ENABLED or not settings.API_KEYS:
        return True
    if not api_key:
        return False
    return any(_safe_equals(api_key, k) for k in settings.API_KEYS)