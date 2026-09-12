"""Prediction API — a single, unified prediction pipeline.

The endpoint never branches on model internals: `inference_service.predict`
owns backend selection (FT-Transformer primary, XGBoost fallback). Optional
SHAP explanation, evidence, GenAI advisory and defensive-action
recommendations are layered on top and are purely advisory.
"""

import logging
import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..database.session import get_db
from ..schemas.requests import PredictionRequest
from ..schemas.responses import (
    AdvisoryResponse,
    BatchPredictionItem,
    BatchPredictionResponse,
    DefensiveAction,
    PredictionResponse,
    ShapExplanation,
)
from ..services.alert_service import alert_service
from ..services.defensive_action_service import defensive_action_service
from ..services.explainable_ai import explainable_ai
from ..services.genai_service import genai_service
from ..services.inference_service import inference_service
from ..services.notification_service import notification_service
from ..services.shap_service import shap_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.get("", response_model=PredictionResponse)
async def predict_get(
    features: PredictionRequest = Depends(),
    db: AsyncSession = Depends(get_db),
) -> PredictionResponse:
    """GET convenience alias for /predict (query-param features)."""
    return await _run_prediction(features, db)


@router.post("", response_model=PredictionResponse)
async def predict(
    payload: PredictionRequest,
    db: AsyncSession = Depends(get_db),
) -> PredictionResponse:
    """Run the unified prediction pipeline for a network flow.

    Body fields: the 76 canonical CIC-IDS2017 features via `feature_mapping.json`
    keys plus optional `src_ip`, `dst_ip`, `destination_port`,
    `include_explanation`, and `include_advisory`.
    """
    return await _run_prediction(payload, db)


async def _run_prediction(
    payload: PredictionRequest, db: AsyncSession
) -> PredictionResponse:
    if not inference_service.initialized:
        try:
            inference_service.load()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}")

    start = time.perf_counter()
    result = inference_service.predict(payload.feature_dict())
    latency_ms = (time.perf_counter() - start) * 1000.0

    severity = (
        alert_service.classify_severity(result.class_name, result.confidence).value
        if result.is_attack
        else None
    )

    explanation_model = None
    advisory: AdvisoryResponse | None = None

    if payload.include_explanation or payload.include_advisory:
        explanation = shap_service.explain(
            inference_service,
            payload.feature_dict(),
            top_k=6,
        )
        if explanation.status == "ok":
            explanation_model = ShapExplanation(
                **explainable_ai.enrich(explanation),
                prediction_class=result.class_name,
                is_attack=result.is_attack,
                confidence=round(float(result.confidence), 6),
                severity=severity,
            )
        else:
            explanation_model = ShapExplanation(status=explanation.status)

    defensive_actions: list[DefensiveAction] = defensive_action_service.recommend(
        class_name=result.class_name,
        confidence=result.confidence,
        severity=severity or "medium",
        src_ip=payload.src_ip,
        dst_ip=payload.dst_ip,
        is_attack=result.is_attack,
    )

    if payload.include_advisory:
        adv = genai_service.run_advisory(
            prediction={
                "class_id": result.class_id,
                "class_name": result.class_name,
                "confidence": result.confidence,
                "is_attack": result.is_attack,
                "probabilities": result.proba_map(),
            },
            top_shap_features=(
                [c.model_dump() for c in explanation_model.top_features]
                if explanation_model
                else []
            ),
            network_context=payload.context_dict(),
            severity=severity,
            evidence_summary=(
                explanation_model.evidence_summary if explanation_model else None
            ),
            defensive_actions=[a.model_dump() for a in defensive_actions],
        )
        advisory = AdvisoryResponse(
            advisory=adv.advisory,
            status=adv.status,
            provider=adv.provider,
            model_id=adv.model_id,
            generated=adv.generated,
            error=adv.error,
        )

    _log_prediction(result, payload, latency_ms)

    alert_id = None
    if result.is_attack and settings.ALERT_CREATE_ON_ATTACK and (payload.src_ip or payload.dst_ip):
        try:
            alert = await alert_service.create_alert(
                db,
                attack_type=result.class_name,
                src_ip=payload.src_ip or "unknown",
                dst_ip=payload.dst_ip,
                features=payload.feature_dict(),
                prediction_score=result.class_id,
                confidence=result.confidence,
                model_backend=result.backend,
                llm_insight=advisory.advisory if advisory else None,
            )
            if alert:
                alert_id = alert.id
                await notification_service.notify_alert_async(
                    {
                        "id": alert.id,
                        "attack_type": alert.attack_type,
                        "severity": severity_label(alert.severity),
                        "src_ip": alert.src_ip,
                        "dst_ip": alert.dst_ip,
                        "confidence": alert.confidence,
                        "timestamp": str(alert.timestamp),
                    }
                )
        except Exception as exc:
            logger.warning("Alert creation failed: %s", exc)
            alert_id = None

    return PredictionResponse(
        prediction=[result.class_id],
        prediction_class=result.class_name,
        is_attack=result.is_attack,
        confidence=round(result.confidence, 6),
        severity=severity,
        probabilities=[round(float(p), 6) for p in result.probabilities],
        class_probabilities=result.proba_map(),
        model_backend=result.backend,
        validation_warnings=payload._validation_warnings,
        latency_ms=round(latency_ms, 2),
        alert_id=alert_id,
        explanation=explanation_model,
        advisory=advisory,
        defensive_actions=defensive_actions,
    )


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> BatchPredictionResponse:
    """Batch CSV inference — each row goes through the SAME single pipeline.

    Accepts a CSV whose columns match the backend scaler columns (pretty names)
    or the canonical snake_case API keys. Non-numeric cells are treated as 0.
    """
    if not inference_service.initialized:
        try:
            inference_service.load()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}")

    import io  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    try:
        raw = file.file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    backend_cols = list(inference_service.feature_names)
    mapper = inference_service.feature_map  # pretty -> api_key
    items: list[BatchPredictionItem] = []
    attacks = 0

    for idx, row in df.iterrows():
        try:
            feats: dict = {}
            for col in backend_cols:
                api_key = mapper.get(col, col)
                val = row.get(api_key, row.get(col, 0.0))
                try:
                    feats[api_key] = float(val)
                except (TypeError, ValueError):
                    feats[api_key] = 0.0
            result = inference_service.predict(feats)
            severity = (
                alert_service.classify_severity(result.class_name, result.confidence).value
                if result.is_attack
                else None
            )
            if result.is_attack:
                attacks += 1
            items.append(
                BatchPredictionItem(
                    row=int(idx),
                    prediction_class=result.class_name,
                    is_attack=result.is_attack,
                    confidence=round(float(result.confidence), 6),
                    severity=severity,
                )
            )
        except Exception as exc:
            logger.warning("Batch row %s failed: %s", idx, exc)
            items.append(BatchPredictionItem(row=int(idx), error=str(exc)))

    return BatchPredictionResponse(total=len(items), attacks_detected=attacks, items=items)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_prediction(result, payload: PredictionRequest, latency_ms: float) -> None:
    """Append prediction results to the runtime log files (legacy parity)."""
    try:
        log_dir = settings.log_dir
        ts = datetime.utcnow().isoformat(timespec="seconds")
        line = (
            f"{ts} class={result.class_name} conf={result.confidence:.4f} "
            f"latency_ms={latency_ms:.1f}\n"
        )
        target = "positive_predictions.log" if result.is_attack else "negative_predictions.log"
        if result.is_attack or settings.LOG_NEGATIVE_PREDICTIONS:
            with open(log_dir / target, "a", encoding="utf-8") as fh:
                fh.write(line)
    except Exception as exc:
        logger.warning("Prediction logging skipped: %s", exc)


def severity_label(value) -> str:
    return str(value).upper() if value else "UNKNOWN"