"""Shared fixtures for the ThreatLens test suite.

Env vars are set BEFORE app/main is imported so the eager `Settings`
instance picks them up. Tests that need a different setting value mutate the
singleton directly (it is read at call time by all services).
"""

import os
import tempfile

_TMPDIR = os.path.join(tempfile.gettempdir(), "threatlens_tests")
os.makedirs(_TMPDIR, exist_ok=True)
_DB_PATH = os.path.join(_TMPDIR, "test_mlids.db")
if os.path.exists(_DB_PATH):
    os.remove(_DB_PATH)

os.environ["MODEL_BACKEND"] = "xgboost"
os.environ["AUTH_ENABLED"] = "false"
os.environ["LLM_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_DB_PATH}"
os.environ["ALERT_CORRELATION_ENABLED"] = "true"
os.environ["ALERT_CORRELATION_MIN_ALERTS"] = "2"
os.environ["ALERT_CORRELATION_WINDOW_HOURS"] = "24"
os.environ["SHAP_BACKGROUND_SAMPLE"] = "50"
os.environ["LOG_NEGATIVE_PREDICTIONS"] = "false"
os.environ["ALERT_NOTIFICATION_ENABLED"] = "false"

import json  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.core.config import settings  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def settings_override():
    """Force-set fields on the frozen `Settings` singleton for a test, then
    restore them. Services read settings at call time, so this is safe."""
    saved = {}

    def _override(**kwargs):
        for key, value in kwargs.items():
            saved[key] = getattr(settings, key)
            object.__setattr__(settings, key, value)

    yield _override

    for key, value in saved.items():
        object.__setattr__(settings, key, value)


def load_sample(name: str) -> dict:
    path = os.path.join(os.path.dirname(__file__), "..", "app", "static", f"{name}.json")
    with open(os.path.realpath(path), encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def attack_features():
    feats = load_sample("sample_attack")
    feats["src_ip"] = "10.0.0.5"
    feats["dst_ip"] = "192.168.1.9"
    return feats


@pytest.fixture(scope="session")
def benign_features():
    feats = load_sample("sample_benign")
    feats["src_ip"] = "10.0.0.9"
    feats["dst_ip"] = "192.168.1.15"
    return feats


def make_payload(features, include_explanation=False, include_advisory=False):
    """PredictionRequest is a FLAT body (feature fields + src/dst + flags), so
    the test payload mirrors the dashboard exactly — no `features` wrapper."""
    payload = dict(features)
    if include_explanation:
        payload["include_explanation"] = True
    if include_advisory:
        payload["include_advisory"] = True
    return payload