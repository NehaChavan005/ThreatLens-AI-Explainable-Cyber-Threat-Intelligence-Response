"""Unified inference service — ONE prediction pipeline.

Boot order:
  1. FT-Transformer  (primary, PyTorch checkpoint + StandardScaler)
  2. XGBoost         (fallback, GenAI-augmented model + scaler + label encoder)

Both backends accept the exact same canonical 76-feature dict produced by
`PredictionRequest.feature_dict()`. Missing features are padded with 0.0,
NaN/Inf are replaced, and the appropriate scaler is applied before inference.

The single public contract is `InferenceService.predict(feature_dict)`
returning a `PredictionResult` dataclass (class id, name, confidence,
per-class probabilities, is_attack, backend name).
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..core.config import Settings, settings

logger = logging.getLogger(__name__)


# =============================================================================
# FT-Transformer architecture (mirrors notebooks/ft_transformer_optuna_sweep.py)
# =============================================================================

try:  # torch is optional at import time; the service reports if unavailable
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer for tabular CIC-IDS2017 flows.

    Identical construction to training so a checkpoint can be loaded
    directly from `torch.load(...)["state_dict"]`.
    """

    def __init__(self, n_features, n_classes, d_token, n_blocks, n_heads,
                 ff_factor, dropout):
        super().__init__()
        self.tokenizer_w = nn.Parameter(torch.empty(n_features, d_token))
        self.tokenizer_b = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.kaiming_uniform_(self.tokenizer_w, a=5 ** 0.5)
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads,
            dim_feedforward=int(d_token * ff_factor),
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_blocks)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_classes)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.tokenizer_w + self.tokenizer_b
        cls = self.cls.expand(x.size(0), -1, -1)
        z = torch.cat([cls, tokens], dim=1)
        z = self.encoder(z)
        return self.head(self.norm(z[:, 0]))


# =============================================================================
# Result container
# =============================================================================


@dataclass
class PredictionResult:
    class_id: int
    class_name: str
    confidence: float
    probabilities: np.ndarray  # (n_classes,)
    class_labels: List[str]
    is_attack: bool
    backend: str

    def proba_map(self) -> Dict[str, float]:
        return {lbl: round(float(p), 6) for lbl, p in zip(self.class_labels, self.probabilities)}


# =============================================================================
# Default class label mapping (documented in ML-IDS UNIFIED_MODEL.md)
# =============================================================================

_DEFAULT_CLASS_LABELS = [
    "Benign", "Analysis", "Backdoor", "DoS", "Exploits",
    "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
]

_PERSISTENT_ATTACK_BACKENDS = {"xgboost", "ft_transformer"}


def _sanitize_label(label: str) -> str:
    """Repair label encoding corruption (U+FFFD) from the pickled encoder.

    The GenAI-augmented XGBoost label encoder was serialized with non-ASCII
    dashes that later decoded as the Unicode replacement character, producing
    names such as ``Web Attack \ufffd Brute Force``. Rewrite them to clean
    ASCII so class names, severity mapping and playbook lookup work.
    """
    if not label:
        return label
    cleaned = label.replace("\ufffd", " ").replace("\ufeff", "")
    return re.sub(r"\s+", " ", cleaned).strip()


# =============================================================================
# Service
# =============================================================================


class InferenceService:
    """Singleton that owns the single active model backend."""

    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        self.backend: Optional[str] = None        # "ft_transformer" | "xgboost"
        self.model = None
        self.scaler = None
        self.label_encoder = None                 # xgboost backend only
        self.feature_names: List[str] = []        # canonical order for backend
        self.feature_map: Dict[str, str] = {}     # api key -> backend feature name
        self.class_labels: List[str] = _DEFAULT_CLASS_LABELS
        self.initialized = False
        self.model_source = "none"
        self.model_loaded_at: Optional[str] = None
        self._column_dict: Dict[str, List[str]] = {}   # column dict for SHAP

        self._load_feature_maps()
        self._load_class_labels()

    # ------------------------------------------------------------------ util
    def _load_feature_maps(self) -> None:
        with open(self.cfg.FEATURE_MAPPING) as fh:
            pretty = json.load(fh)  # api key -> "Pretty Name" (FT-T order)
        self.feature_map_ft = pretty
        self.api_keys = list(pretty.keys())

        with open(self.cfg.FEATURE_MAPPING_GENAI) as fh:
            self.feature_map_genai = json.load(fh)  # GenAI col -> api key

    def _load_class_labels(self) -> None:
        path = Path(self.cfg.CLASS_LABELS_PATH)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                labels = data.get("labels") or data if isinstance(data, list) else None
                if labels:
                    self.class_labels = list(labels)
                    logger.info("Class labels loaded from %s", path)
            except Exception as exc:
                logger.warning("Failed to load class labels from %s: %s", path, exc)

    # ----------------------------------------------------------------- load
    def load(self) -> None:
        """Load the active backend (FT-T first, XGBoost fallback)."""
        if self.initialized:
            return

        if self.cfg.MODEL_BACKEND == "ft_transformer":
            ok = self._load_ft_transformer()
        elif self.cfg.MODEL_BACKEND == "xgboost":
            ok = self._load_xgboost()
        else:  # auto
            ok = self._load_ft_transformer()
            if not ok:
                ok = self._load_xgboost()

        if not ok:
            raise RuntimeError("No usable model backend was found")

        self.initialized = True

    def _load_ft_transformer(self) -> bool:
        if not _TORCH_AVAILABLE:
            logger.warning("torch unavailable — skipping FT-Transformer backend")
            return False

        ckpt_path = Path(self.cfg.FTTRANSFORMER_CKPT)
        scaler_path = Path(self.cfg.FTTRANSFORMER_SCALER)
        meta_path = Path(self.cfg.FTTRANSFORMER_METADATA)

        if not (ckpt_path.exists() and scaler_path.exists()):
            logger.warning(
                "FT-Transformer artifacts missing (ckpt=%s scaler=%s) — falling back",
                ckpt_path, scaler_path,
            )
            return False

        try:
            import joblib

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

            n_feat = int(ckpt.get("n_features", len(self.feature_map_ft)))
            n_cls = int(ckpt.get("n_classes", len(self.class_labels)))
            cfg = ckpt.get("config", {}) or {}

            model = FTTransformer(
                n_features=n_feat,
                n_classes=n_cls,
                d_token=int(cfg.get("d_token", 256)),
                n_blocks=int(cfg.get("n_blocks", 3)),
                n_heads=int(cfg.get("n_heads", 8)),
                ff_factor=float(cfg.get("ff_factor", 2.0)),
                dropout=float(cfg.get("dropout", 0.0985)),
            )
            model.load_state_dict(ckpt["state_dict"])
            model.to(device)
            model.eval()

            scaler = joblib.load(scaler_path)

            meta = {}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
            feature_names = ckpt.get("feature_names") or meta.get("feature_names")

            self.model = model
            self.scaler = scaler
            self.backend = "ft_transformer"
            self.feature_names = list(feature_names) if feature_names else list(self.feature_map_ft.values())
            self.feature_map = self.feature_map_ft
            self.model_source = f"ft_transformer:{ckpt_path.name}"
            self.model_loaded_at = ckpt.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._build_column_dict()
            logger.info("FT-Transformer loaded (%d features, %d classes)", n_feat, n_cls)
            return True

        except Exception as exc:
            logger.error("Failed to load FT-Transformer: %s", exc)
            self._reset_backend()
            return False

    def _load_xgboost(self) -> bool:
        model_path = Path(self.cfg.XGBOOST_MODEL)
        scaler_path = Path(self.cfg.XGBOOST_SCALER)
        le_path = Path(self.cfg.XGBOOST_LABEL_ENCODER)

        if not all(p.exists() for p in (model_path, scaler_path, le_path)):
            logger.warning("XGBoost artifacts missing — cannot load fallback")
            return False

        try:
            import joblib

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(le_path)
            self.backend = "xgboost"

            # Keep the exact column order the scaler was fit on.
            self.feature_names = list(self.scaler.feature_names_in_)
            self.feature_map = self.feature_map_genai
            self.class_labels = [_sanitize_label(str(c)) for c in self.label_encoder.classes_]
            self.model_source = f"xgboost:{model_path.name}"
            self.model_loaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._build_column_dict()
            logger.info("XGBoost loaded (%d features, %d classes)", len(self.feature_names), len(self.class_labels))
            return True

        except Exception as exc:
            logger.error("Failed to load XGBoost fallback: %s", exc)
            self._reset_backend()
            return False

    def _build_column_dict(self) -> None:
        """Map backend feature name -> list of API keys that populate it.

        Used by the SHAP service to translate raw feature values.
        """
        col = {name: [] for name in self.feature_names}
        inverse: Dict[str, str] = {}
        for api_key, backend_name in self.feature_map.items():
            inverse.setdefault(backend_name, api_key)
        for name in self.feature_names:
            col[name] = [inverse.get(name, name) or name]
        self._column_dict = col

    def _reset_backend(self) -> None:
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.backend = None

    # -------------------------------------------------------------- predict
    def _build_input_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Map the canonical feature dict onto backend-ordered values."""
        vector = []
        for backend_name in self.feature_names:
            api_key = self.feature_map.get(backend_name, backend_name)
            value = features.get(api_key, 0.0) or 0.0
            if isinstance(value, (int, float)) and not np.isfinite(float(value)):
                value = 0.0
            vector.append(float(value))
        return np.asarray(vector, dtype=np.float32).reshape(1, -1)

    @staticmethod
    def _sanitize(x: np.ndarray) -> np.ndarray:
        return np.nan_to_num(x, nan=0.0, posinf=1e9, neginf=-1e9)

    def predict(self, features: Dict[str, float]) -> PredictionResult:
        """Run a single purpose-built prediction for a canonical feature dict."""
        if not self.initialized:
            self.load()

        raw = self._build_input_vector(features)
        clean = self._sanitize(raw)
        scaled = self._apply_scaler(clean)

        if self.backend == "ft_transformer":
            return self._predict_ft(scaled)
        return self._predict_xgb(scaled)

    def _apply_scaler(self, clean: np.ndarray) -> np.ndarray:
        """Apply the backend scaler, preserving column names when supported."""
        if self.scaler is not None and hasattr(self.scaler, "feature_names_in_"):
            import pandas as pd  # noqa: PLC0415

            frame = pd.DataFrame(clean, columns=list(self.scaler.feature_names_in_))
            return self.scaler.transform(frame).astype(np.float32)
        return self.scaler.transform(clean).astype(np.float32)

    def _predict_ft(self, scaled: np.ndarray) -> PredictionResult:
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            t = torch.from_numpy(scaled).to(device)
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.model(t)
            else:
                logits = self.model(t)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]

        class_id = int(np.argmax(probs))
        return self._to_result(class_id, probs)

    def _predict_xgb(self, scaled: np.ndarray) -> PredictionResult:
        probs = self.model.predict_proba(scaled)[0]
        class_id = int(np.argmax(probs))
        return self._to_result(class_id, probs)

    def _to_result(self, class_id: int, probs: np.ndarray) -> PredictionResult:
        labels = self.class_labels
        if len(probs) != len(labels):
            # Guard against backend exposing more classes than configured.
            labels = [f"Class_{i}" for i in range(len(probs))]
        name = labels[class_id]
        return PredictionResult(
            class_id=class_id,
            class_name=name,
            confidence=float(probs[class_id]),
            probabilities=probs,
            class_labels=labels,
            is_attack=name.lower() != "benign",
            backend=self.backend or "unknown",
        )

    # ------------------------------------------------------------- explain
    def scaled_input(self, features: Dict[str, float]) -> np.ndarray:
        """Return the sanitized+scaled input vector (used by SHAP service)."""
        if not self.initialized:
            self.load()
        clean = self._sanitize(self._build_input_vector(features))
        return self._apply_scaler(clean)

    def raw_input(self, features: Dict[str, float]) -> np.ndarray:
        return self._build_input_vector(features)

    @property
    def column_dict(self) -> Dict[str, List[str]]:
        return self._column_dict


# Global singleton
inference_service = InferenceService()