"""Centralised configuration for the unified IDS API.

All runtime behaviour is controlled through environment variables with
production-safe defaults. No secrets live in code.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_list(name: str) -> List[str]:
    return [k.strip() for k in os.getenv(name, "").split(",") if k.strip()]


@dataclass(frozen=True)
class Settings:
    # --- Paths -----------------------------------------------------------
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    APP_DIR: Path = Path(__file__).resolve().parent.parent
    MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models")))
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", str(BASE_DIR / "logs")))

    # --- FT-Transformer (primary model) ----------------------------------
    FTTRANSFORMER_CKPT: str = os.getenv(
        "FTTRANSFORMER_CKPT",
        str(MODELS_DIR / "ft_transformer" / "unified_ft_transformer.pt"),
    )
    FTTRANSFORMER_METADATA: str = os.getenv(
        "FTTRANSFORMER_METADATA",
        str(MODELS_DIR / "ft_transformer" / "unified_metadata.json"),
    )
    FTTRANSFORMER_SCALER: str = os.getenv(
        "FTTRANSFORMER_SCALER",
        str(MODELS_DIR / "ft_transformer" / "unified_scaler.pkl"),
    )

    # --- XGBoost (GenAI-augmented fallback model) ------------------------
    XGBOOST_MODEL: str = os.getenv(
        "XGBOOST_MODEL",
        str(MODELS_DIR / "baseline" / "ids_xgboost_genai.pkl"),
    )
    XGBOOST_SCALER: str = os.getenv(
        "XGBOOST_SCALER",
        str(MODELS_DIR / "baseline" / "feature_scaler_genai.pkl"),
    )
    XGBOOST_LABEL_ENCODER: str = os.getenv(
        "XGBOOST_LABEL_ENCODER",
        str(MODELS_DIR / "baseline" / "label_encoder_genai.pkl"),
    )
    FEATURE_MAPPING: str = os.getenv(
        "FEATURE_MAPPING", str(APP_DIR / "feature_mapping.json")
    )
    FEATURE_MAPPING_GENAI: str = os.getenv(
        "FEATURE_MAPPING_GENAI", str(APP_DIR / "feature_mapping_genai.json")
    )

    # --- Model selection ---------------------------------------------------
    # "ft_transformer" | "xgboost" | "auto"
    MODEL_BACKEND: str = os.getenv("MODEL_BACKEND", "auto")

    # --- IBM Granite LLM advisory -----------------------------------------
    LLM_ENABLED: bool = _env_bool("LLM_ENABLED", "true")
    LLM_API_URL: str = os.getenv(
        "LLM_API_URL",
        "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29",
    )
    LLM_MODEL_ID: str = os.getenv("LLM_MODEL_ID", "ibm/granite-20b-code-instruct-v2")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_PROJECT_ID: str = os.getenv("LLM_PROJECT_ID", "")
    LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "20"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "1024"))
    LLM_ADVISORY_TOP_N: int = int(os.getenv("LLM_ADVISORY_TOP_N", "6"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    # --- API / security ---------------------------------------------------
    AUTH_ENABLED: bool = _env_bool("AUTH_ENABLED", "true")
    API_KEYS: List[str] = field(default_factory=lambda: _env_list("ML_IDS_API_KEYS"))

    # --- Persistence ------------------------------------------------------
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "sqlite+aiosqlite:///./mlids.db"
    )

    # --- Alerting ---------------------------------------------------------
    ALERT_DEDUP_WINDOW_SECONDS: int = int(
        os.getenv("ALERT_DEDUP_WINDOW_SECONDS", "300")
    )
    ALERT_CREATE_ON_ATTACK: bool = _env_bool("ALERT_CREATE_ON_ATTACK", "true")
    LOG_NEGATIVE_PREDICTIONS: bool = _env_bool("LOG_NEGATIVE_PREDICTIONS", "false")

    # --- Incident correlation ---------------------------------------------
    ALERT_CORRELATION_ENABLED: bool = _env_bool("ALERT_CORRELATION_ENABLED", "true")
    ALERT_CORRELATION_MIN_ALERTS: int = int(os.getenv("ALERT_CORRELATION_MIN_ALERTS", "2"))
    ALERT_CORRELATION_WINDOW_HOURS: int = int(os.getenv("ALERT_CORRELATION_WINDOW_HOURS", "24"))

    # --- Notifications (email / Slack / webhook) --------------------------
    ALERT_NOTIFICATION_ENABLED: bool = _env_bool("ALERT_NOTIFICATION_ENABLED", "false")
    NOTIFICATION_WEBHOOK_URL: str = os.getenv("NOTIFICATION_WEBHOOK_URL", "")
    SLACK_WEBHOOK_URL: str = os.getenv("SLACK_WEBHOOK_URL", "")
    SMTP_HOST: str = os.getenv("SMTP_HOST", "")
    SMTP_PORT: str = os.getenv("SMTP_PORT", "587")
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM: str = os.getenv("SMTP_FROM", "")
    SMTP_TO: List[str] = field(default_factory=lambda: _env_list("SMTP_TO"))
    SMTP_USE_SSL: bool = _env_bool("SMTP_USE_SSL", "false")
    SMTP_STARTTLS: bool = _env_bool("SMTP_STARTTLS", "true")
    SMTP_TIMEOUT: float = float(os.getenv("SMTP_TIMEOUT", "10"))

    # --- Class labels (override of the documented CIC-IDS2017 group map) --
    CLASS_LABELS_PATH: str = os.getenv(
        "CLASS_LABELS_PATH",
        str(MODELS_DIR / "ft_transformer" / "class_labels.json"),
    )

    # --- SHAP --------------------------------------------------------------
    SHAP_BACKGROUND_SAMPLE: int = int(os.getenv("SHAP_BACKGROUND_SAMPLE", "100"))

    @property
    def log_dir(self) -> Path:
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        return self.LOG_DIR

    def ensure_dirs(self) -> None:
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()