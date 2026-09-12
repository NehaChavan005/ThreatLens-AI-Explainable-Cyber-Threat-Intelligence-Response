"""Explainability & GenAI advisory API.

- `POST /explain`          SHAP feature contributions for a flow, with an
                           analyst-readable evidence summary + interpretation
                           and the prediction context that produced them.
- `POST /explain/advisory` GenAI advisory; consumes (Prediction + SHAP
                           contributions + Network Context + defensive
                           recommendations) exactly as the GenAI service
                           contract dictates. AI wording is strictly advisory.
"""

import logging

from fastapi import APIRouter, HTTPException

from ..schemas.requests import AdvisoryRequest, ExplainRequest
from ..schemas.responses import AdvisoryResponse, ShapExplanation
from ..services.alert_service import alert_service
from ..services.defensive_action_service import defensive_action_service
from ..services.explainable_ai import explainable_ai
from ..services.genai_service import genai_service
from ..services.inference_service import inference_service
from ..services.shap_service import shap_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["explainability"])


def _ensure_model():
    if not inference_service.initialized:
        try:
            inference_service.load()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}")


@router.post("", response_model=ShapExplanation)
async def explain(req: ExplainRequest) -> ShapExplanation:
    """Return top-k SHAP contributions for a single flow plus evidence."""
    _ensure_model()

    result = inference_service.predict(req.features.feature_dict())
    severity = (
        alert_service.classify_severity(result.class_name, result.confidence).value
        if result.is_attack
        else None
    )

    explanation = shap_service.explain(
        inference_service,
        req.features.feature_dict(),
        top_k=req.top_k,
    )
    if explanation.status != "ok":
        raise HTTPException(status_code=503, detail=explanation.status)

    fields = explainable_ai.enrich(explanation)
    return ShapExplanation(
        **fields,
        prediction_class=result.class_name,
        is_attack=result.is_attack,
        confidence=round(float(result.confidence), 6),
        severity=severity,
    )


@router.post("/advisory", response_model=AdvisoryResponse)
async def advisory(req: AdvisoryRequest) -> AdvisoryResponse:
    """Generate a GenAI security advisory from prediction + SHAP + context +
    system-recommended defensive actions.

    The LLM is strictly advisory; failures degrade to a heuristic insight.
    The util level only ever sees the caller-supplied prediction/features.
    """
    pred = req.prediction.model_dump()
    sev = req.severity or (
        alert_service.classify_severity(pred.get("class_name", "unknown"), pred.get("confidence", 0.0)).value
        if pred.get("is_attack")
        else None
    )

    actions = defensive_action_service.recommend(
        class_name=pred.get("class_name", ""),
        confidence=pred.get("confidence", 0.0),
        severity=sev or "medium",
        src_ip=(req.network_context or {}).get("src_ip"),
        dst_ip=(req.network_context or {}).get("dst_ip"),
        is_attack=bool(pred.get("is_attack")),
    )

    adv = genai_service.run_advisory(
        prediction=pred,
        top_shap_features=req.top_shap_features,
        network_context=req.network_context,
        include_remediation=req.include_remediation,
        severity=sev,
        evidence_summary=req.evidence_summary or None,
        defensive_actions=[a.model_dump() for a in actions],
    )
    return AdvisoryResponse(
        advisory=adv.advisory,
        status=adv.status,
        provider=adv.provider,
        model_id=adv.model_id,
        generated=adv.generated,
        error=adv.error,
    )