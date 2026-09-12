"""Response models for prediction, explainability, advisory, and alerts."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class FeatureContribution(BaseModel):
    """One SHAP contribution entry for a single feature."""
    feature: str
    shap_value: float = Field(0.0)
    abs_shap_value: float = Field(0.0)
    raw_value: float = Field(0.0)
    direction: str = Field("increase")  # increase|decrease toward prediction


class DefensiveAction(BaseModel):
    action: str
    reason: str
    priority: str = Field("MEDIUM")
    urgency: str = Field("Normal")
    approval_required: bool = Field(True)
    category: str = Field("unknown")


class ShapExplanation(BaseModel):
    top_features: List[FeatureContribution] = Field(default_factory=list)
    base_value: float = 0.0
    bias_position: int = 0  # class-index column with greatest |SHAP|
    explainer_type: str = "tree"
    status: str = "ok"
    evidence_summary: List[str] = Field(default_factory=list)
    interpretation: str = ""
    explanation_source: str = ""  # e.g. "TreeExplainer" or "KernelExplainer"
    # Additive prediction context (fields were added without breaking the old shape)
    prediction_class: Optional[str] = None
    is_attack: bool = False
    confidence: Optional[float] = None
    severity: Optional[str] = None


class AdvisoryResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    advisory: str = ""
    status: str = "ok"
    provider: str = "ibm-granite"
    model_id: str = ""
    generated: bool = False
    error: Optional[str] = None


class PredictionResponse(BaseModel):
    prediction: List[int] = Field(default_factory=list)
    prediction_class: Optional[str] = None
    is_attack: bool = False
    confidence: float = 0.0
    severity: Optional[str] = None
    probabilities: List[float] = Field(default_factory=list)
    class_probabilities: Dict[str, float] = Field(default_factory=dict)
    model_backend: str = "ft_transformer"
    validation_warnings: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    alert_id: Optional[int] = None
    explanation: Optional[ShapExplanation] = None
    advisory: Optional[AdvisoryResponse] = None
    defensive_actions: List[DefensiveAction] = Field(default_factory=list)


class AlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    attack_type: str
    severity: str
    src_ip: str
    dst_ip: Optional[str] = None
    timestamp: datetime
    prediction_score: Optional[float] = None
    confidence: Optional[float] = None
    model_backend: Optional[str] = None
    llm_insight: Optional[str] = None
    acknowledged: bool = False
    incident_id: Optional[int] = None
    notes: Optional[str] = None


class AlertUpdate(BaseModel):
    acknowledged: Optional[bool] = None
    notes: Optional[str] = None
    severity: Optional[str] = None


class IncidentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: Optional[str] = None
    status: str
    severity: str
    assigned_to: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    notes: Optional[str] = None


class IncidentCreate(BaseModel):
    title: str
    description: Optional[str] = None
    status: str = "open"
    severity: str = "medium"
    assigned_to: Optional[str] = None


class IncidentUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    severity: Optional[str] = None
    assigned_to: Optional[str] = None
    notes: Optional[str] = None


class IncidentDetail(BaseModel):
    """Analyst-facing, correlated view of an incident."""
    incident: IncidentResponse
    alert_ids: List[int] = Field(default_factory=list)
    related_alerts: List[AlertResponse] = Field(default_factory=list)
    attack_types: List[str] = Field(default_factory=list)
    highest_severity: str = "medium"
    avg_confidence: Optional[float] = None
    total_alerts: int = 0
    source_ips: List[str] = Field(default_factory=list)
    destination_ips: List[str] = Field(default_factory=list)
    timeline: List[datetime] = Field(default_factory=list)
    explanation: Optional[str] = None
    defensive_actions: List[DefensiveAction] = Field(default_factory=list)


class DashboardStats(BaseModel):
    total_alerts: int = 0
    total_incidents: int = 0
    active_incidents: int = 0
    alerts_by_severity: Dict[str, int] = Field(default_factory=dict)
    time_period_hours: int = 24


class BatchPredictionItem(BaseModel):
    row: int
    prediction_class: Optional[str] = None
    is_attack: bool = False
    confidence: float = 0.0
    severity: Optional[str] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    total: int
    attacks_detected: int
    items: List[BatchPredictionItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_initialized: bool
    model_backend: str
    model_source: str
    model_loaded_at: Optional[str] = None
    llm_enabled: bool
    database: Dict[str, Any]