"""SHAP explainability service.

Chooses the explainer by backend:
  - XGBoost:  `shap.TreeExplainer`      -> exact, fast, one-time construction
  - FT-Transformer: `shap.KernelExplainer` -> model-agnostic; a small scaled
    background sample is drawn from the processed dataset.

Both return the same structured `FeatureContribution` list so the upper
layers (GenAI advisory, explainability API) never see SHAP internals.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from ..core.config import Settings, settings
from ..schemas.responses import FeatureContribution

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    top_features: List[FeatureContribution] = field(default_factory=list)
    base_value: float = 0.0
    bias_position: int = 0
    explainer_type: str = "tree"
    status: str = "ok"


class ShapService:
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        self.shap = self._try_import_shap()
        self._explainer = None
        self._explainer_backend: Optional[str] = None
        self._background: Optional[np.ndarray] = None

    @staticmethod
    def _try_import_shap():
        try:
            import shap  # noqa: PLC0415

            return shap
        except Exception as exc:  # pragma: no cover
            logger.warning("shap not installed: %s", exc)
            return None

    # ------------------------------------------------------------------ setup
    def _ensure_explainer(self, inference) -> None:
        """Build the backend-appropriate explainer if not already built."""
        backend = inference.backend
        if self._explainer is not None and self._explainer_backend == backend:
            return
        if self.shap is None or inference.model is None:
            self._explainer = None
            self._explainer_backend = backend
            return

        if backend == "xgboost":
            self._explainer = self.shap.TreeExplainer(inference.model)
            self._explainer_backend = backend
            logger.info("SHAP TreeExplainer built for XGBoost")
        else:
            background = self._load_background(inference)
            predict_fn = self._ft_predict_fn(inference)
            self._explainer = self.shap.KernelExplainer(predict_fn, background, link="identity")
            self._explainer_backend = backend
            logger.info("SHAP KernelExplainer built for FT-Transformer (background=%d rows)", len(background))

    def _load_background(self, inference) -> np.ndarray:
        if self._background is not None:
            return self._background

        # Raw feature file. The active backend scaler must be applied here; a
        # pre-scaled file would be double-standardized for any backend.
        raw_path = Path(self.cfg.BASE_DIR) / "data" / "processed" / "clean_data_unscaled.csv"
        klass_path = Path(self.cfg.BASE_DIR) / "data" / "processed" / "clean_data.csv"
        candidate_paths = (
            [raw_path, klass_path, Path("/app/data/processed/clean_data_unscaled.csv")]
            if raw_path.exists()
            else [klass_path, Path("/app/data/processed/clean_data.csv")]
        )
        sample_size = int(getattr(self.cfg, "SHAP_BACKGROUND_SAMPLE", 100))

        candidates = [p for p in candidate_paths if p.exists()]
        if candidates:
            try:
                import pandas as pd  # noqa: PLC0415

                df = pd.read_csv(next(iter(candidates)), nrows=max(1000, sample_size * 10))
                df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                if len(df) > sample_size:
                    df = df.sample(sample_size, random_state=42)
                bg = self._map_to_scaled(df, inference)
                self._background = bg
                logger.info("SHAP background sample loaded from %s", str(next(iter(candidates))))
                return bg
            except Exception as exc:
                logger.warning("Failed to load SHAP background: %s", exc)

        # Degenerate fallback: default vector padded with zeros.
        self._background = np.zeros((sample_size, len(inference.feature_names)), dtype=np.float32)
        logger.warning("No background dataset found — using zero default background")
        return self._background

    def _map_to_scaled(self, df, inference) -> np.ndarray:
        """Translate dataframe columns into scaled backend input via the
        canonical feature mapping."""
        if df.empty:
            return np.zeros((1, len(inference.feature_names)), dtype=np.float32)

        rows = []
        for _, row in df.iterrows():
            feats = {}
            for backend_name in inference.feature_names:
                api_key = inference.feature_map.get(backend_name, backend_name)
                value = 0.0
                if api_key in row:
                    try:
                        value = float(row[api_key])
                    except (TypeError, ValueError):
                        value = 0.0
                feats[api_key] = value
            rows.append(feats)

        matrix = np.asarray(
            [
                inference._build_input_vector(r)
                for r in rows
            ],
            dtype=np.float32,
        ).reshape(len(rows), -1)
        matrix = inference._sanitize(matrix)
        return inference.scaler.transform(matrix).astype(np.float32)

    def _ft_predict_fn(self, inference) -> Callable[[np.ndarray], np.ndarray]:
        """Predict probability matrix for scaled numpy input (B, F) -> (B, C)."""
        import torch  # noqa: PLC0415

        device = next(inference.model.parameters()).device

        def predict(vectors: np.ndarray, batch_size: int = 256) -> np.ndarray:
            out = []
            for i in range(0, len(vectors), batch_size):
                chunk = np.asarray(vectors[i : i + batch_size], dtype=np.float32)
                chunk = inference._sanitize(chunk)
                with torch.inference_mode():
                    t = torch.from_numpy(chunk).to(device)
                    if device.type == "cuda":
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            logits = inference.model(t)
                    else:
                        logits = inference.model(t)
                    out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            return np.concatenate(out, axis=0)

        return predict

    # ---------------------------------------------------------------- compute
    def explain(
        self,
        inference,
        feature_dict: dict,
        top_k: int = 5,
        bias_position: Optional[int] = None,
    ) -> ExplanationResult:
        """Return SHAP top-k contributions for a single canonical flow."""
        if self.shap is None:
            return ExplanationResult(status="unavailable", explainer_type="none")

        self._ensure_explainer(inference)

        if self._explainer is None:
            return ExplanationResult(status="unavailable", explainer_type=inference.backend or "none")

        x = inference.scaled_input(feature_dict)

        if inference.backend == "xgboost":
            return self._explain_xgb(inference, x, feature_dict, top_k)
        return self._explain_ft(iteration_x=x, inference=inference, feature_dict=feature_dict, top_k=top_k, bias_position=bias_position)

    def _explain_xgb(self, inference, x, feature_dict, top_k) -> ExplanationResult:
        raw = inference.raw_input(feature_dict).flatten()
        try:
            sv = self._explainer.shap_values(x)
        except Exception as exc:
            logger.error("TreeExplainer.shap_values failed: %s", exc)
            return ExplanationResult(status="error", explainer_type="tree")

        # sv shape: (1, F, C) or (1, F) or list-of-arrays
        predicted = int(np.argmax(inference.model.predict_proba(x), axis=1)[0]) if hasattr(inference.model, "predict_proba") else 0
        contrib = self._select_class_slice(sv, predicted)
        return self._pack(inference, feature_dict, raw, contrib, top_k, "tree", predicted)

    def _explain_ft(self, iteration_x, inference, feature_dict, top_k, bias_position):
        raw = inference.raw_input(feature_dict).flatten()
        try:
            sv = self._explainer.shap_values(iteration_x)
        except Exception as exc:
            logger.error("KernelExplainer.shap_values failed: %s", exc)
            return ExplanationResult(status="error", explainer_type="kernel")

        if bias_position is None:
            import torch  # noqa: PLC0415

            device = next(inference.model.parameters()).device
            with torch.inference_mode():
                t = torch.from_numpy(iteration_x).to(device)
                if device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits = inference.model(t)
                else:
                    logits = inference.model(t)
                probs = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]
            bias_position = int(np.argmax(probs))

        contrib = self._select_class_slice(sv, bias_position)
        return self._pack(inference, feature_dict, raw, contrib, top_k, "kernel", bias_position)

    @staticmethod
    def _select_class_slice(sv, class_idx: int) -> np.ndarray:
        """Normalize arbitrary SHAP output to a 1D contribution vector."""
        try:
            if isinstance(sv, list):
                arr = np.asarray(sv[class_idx])
                if arr.ndim == 2:
                    return arr[0]
                return arr
            arr = np.asarray(sv)
            if arr.ndim == 3:
                return arr[0, :, class_idx]
            if arr.ndim == 2:
                return arr[0]
            return arr
        except Exception:
            return np.asarray(sv).flatten()

    def _pack(self, inference, feature_dict, raw, contrib, top_k, explainer_type, bias_position) -> ExplanationResult:
        contrib = np.asarray(contrib, dtype=np.float64).flatten()

        pairs = []
        for i, name in enumerate(inference.feature_names):
            api_key = inference.column_dict.get(name, [name])[0]
            raw_val = float(raw[i]) if i < len(raw) else 0.0
            shap_val = float(contrib[i]) if i < len(contrib) else 0.0
            direction = "increase" if shap_val >= 0 else "decrease"
            pairs.append(
                FeatureContribution(
                    feature=api_key,
                    shap_value=round(shap_val, 6),
                    abs_shap_value=round(abs(shap_val), 6),
                    raw_value=round(raw_val, 4),
                    direction=direction,
                )
            )

        pairs.sort(key=lambda c: c.abs_shap_value, reverse=True)
        base_value = float(np.asarray(self._explainer.expected_value).flatten()[0]) if hasattr(self._explainer, "expected_value") else 0.0

        return ExplanationResult(
            top_features=pairs[:top_k],
            base_value=round(base_value, 6),
            bias_position=bias_position,
            explainer_type=explainer_type,
            status="ok",
        )


# Global singleton
shap_service = ShapService()