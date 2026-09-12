"""Unified ML-IDS + GenAI-IDS FastAPI entry point.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Routers:
    /predict         single unified prediction pipeline (FT-T primary)
    /explain         SHAP explainability
    /explain/advisory GenAI (IBM Granite) security advisory
    /api/alerts      alert & incident management
    /health          service health
    /metrics         Prometheus-compatible counters
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import alerts, dashboard, explainability, predictions
from .core.config import settings
from .core.security import APIKeyMiddleware
from .database import session as db_session
from .services.genai_service import genai_service
from .services.inference_service import inference_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()

    # --- Database (optional) -----------------------------------------
    db_ok = await db_session.init_db()
    if not db_ok:
        logger.warning("Running without persistence — alert storage disabled")

    # --- Model --------------------------------------------------------
    try:
        inference_service.load()
    except Exception as exc:
        logger.error("Model load failed at startup: %s", exc)

    yield

    await db_session.close_db()


app = FastAPI(
    title="Unified ML-IDS + GenAI-IDS API",
    version="1.0.0",
    description=(
        "Single prediction pipeline combining the ML-IDS FT-Transformer "
        "and GenAI-augmented XGBoost models with SHAP explainability and an "
        "IBM Granite advisory layer. Prediction is authoritative; the LLM "
        "layer is strictly advisory."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(APIKeyMiddleware)

# Routers
app.include_router(predictions.router)
app.include_router(explainability.router)
app.include_router(alerts.router)
app.include_router(dashboard.router)


# ---------------------------------------------------------------------------
# Static dashboard (optional; drop files into app/static/)
# ---------------------------------------------------------------------------
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/dashboard", StaticFiles(directory=static_dir, html=True), name="dashboard")


# ---------------------------------------------------------------------------
# Observability / health
# ---------------------------------------------------------------------------

# Prometheus-style counters (in-memory; replace with prometheus_client if added)
_METRICS = {"predictions_total": 0, "attacks_total": 0, "advisories_total": 0, "_started": time.time()}


@app.get("/")
async def root():
    return {
        "service": "Unified ML-IDS + GenAI-IDS",
        "version": app.version,
        "status": "running",
        "backend": inference_service.backend or "not-loaded",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Liveness/readiness report."""
    db_status = await db_session.health_check()

    overall = "healthy"
    if db_status.get("database") == "unavailable":
        overall = "degraded"
    elif db_status.get("database") == "unhealthy":
        overall = "unhealthy"
    if not inference_service.initialized:
        overall = "degraded"

    return {
        "status": overall,
        "model_initialized": inference_service.initialized,
        "model_backend": inference_service.backend or "none",
        "model_source": inference_service.model_source,
        "model_loaded_at": inference_service.model_loaded_at,
        "llm_enabled": genai_service.enabled,
        "database": db_status,
    }


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Simple Prometheus text metrics."""
    lines = []
    for name, value in _METRICS.items():
        if name.startswith("_"):
            continue
        lines.append(f"# HELP {name} counter")
        lines.append(f"{name} {value}")
    lines.append(f"uptime_seconds {time.time() - _METRICS['_started']:.0f}")
    resp_text = "\n".join(lines) + "\n"
    from fastapi.responses import PlainTextResponse  # noqa: PLC0415

    return PlainTextResponse(status_code=200, content=resp_text)


async def _record_metrics(request, call_next):
    """Middleware recording simple in-memory counters."""
    response = await call_next(request)
    path = request.url.path
    if path == "/predict":
        _METRICS["predictions_total"] += 1
    return response


app.middleware("http")(_record_metrics)