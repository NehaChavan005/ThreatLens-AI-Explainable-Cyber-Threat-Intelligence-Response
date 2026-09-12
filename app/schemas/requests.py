"""Request models.

A single canonical 76-feature `PredictionRequest` drives both model
backends (FT-Transformer and XGBoost). Optional network-context fields are
used for alerting and the LLM advisory layer.
"""

import math
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# Canonical 76-feature payload (keys match app/feature_mapping.json)
# =============================================================================


class PredictionRequest(BaseModel):
    """Network-flow features plus optional context for alerting."""

    flow_duration: float = Field(0.0)
    tot_fwd_pkts: float = Field(0.0)
    tot_bwd_pkts: float = Field(0.0)
    totlen_fwd_pkts: float = Field(0.0)
    totlen_bwd_pkts: float = Field(0.0)
    fwd_pkt_len_max: float = Field(0.0)
    fwd_pkt_len_min: float = Field(0.0)
    fwd_pkt_len_mean: float = Field(0.0)
    fwd_pkt_len_std: float = Field(0.0)
    bwd_pkt_len_max: float = Field(0.0)
    bwd_pkt_len_min: float = Field(0.0)
    bwd_pkt_len_mean: float = Field(0.0)
    bwd_pkt_len_std: float = Field(0.0)
    flow_byts_s: float = Field(0.0)
    flow_pkts_s: float = Field(0.0)
    flow_iat_mean: float = Field(0.0)
    flow_iat_std: float = Field(0.0)
    flow_iat_max: float = Field(0.0)
    flow_iat_min: float = Field(0.0)
    fwd_iat_tot: float = Field(0.0)
    fwd_iat_mean: float = Field(0.0)
    fwd_iat_std: float = Field(0.0)
    fwd_iat_max: float = Field(0.0)
    fwd_iat_min: float = Field(0.0)
    bwd_iat_tot: float = Field(0.0)
    bwd_iat_mean: float = Field(0.0)
    bwd_iat_std: float = Field(0.0)
    bwd_iat_max: float = Field(0.0)
    bwd_iat_min: float = Field(0.0)
    fwd_psh_flags: float = Field(0.0)
    bwd_psh_flags: float = Field(0.0)
    fwd_urg_flags: float = Field(0.0)
    bwd_urg_flags: float = Field(0.0)
    fwd_header_len: float = Field(0.0)
    bwd_header_len: float = Field(0.0)
    fwd_pkts_s: float = Field(0.0)
    bwd_pkts_s: float = Field(0.0)
    pkt_len_min: float = Field(0.0)
    pkt_len_max: float = Field(0.0)
    pkt_len_mean: float = Field(0.0)
    pkt_len_std: float = Field(0.0)
    pkt_len_var: float = Field(0.0)
    fin_flag_cnt: float = Field(0.0)
    syn_flag_cnt: float = Field(0.0)
    rst_flag_cnt: float = Field(0.0)
    psh_flag_cnt: float = Field(0.0)
    ack_flag_cnt: float = Field(0.0)
    urg_flag_cnt: float = Field(0.0)
    cwr_flag_count: float = Field(0.0)
    ece_flag_cnt: float = Field(0.0)
    down_up_ratio: float = Field(0.0)
    pkt_size_avg: float = Field(0.0)
    fwd_seg_size_avg: float = Field(0.0)
    bwd_seg_size_avg: float = Field(0.0)
    fwd_byts_b_avg: float = Field(0.0)
    fwd_pkts_b_avg: float = Field(0.0)
    fwd_blk_rate_avg: float = Field(0.0)
    bwd_byts_b_avg: float = Field(0.0)
    bwd_pkts_b_avg: float = Field(0.0)
    bwd_blk_rate_avg: float = Field(0.0)
    subflow_fwd_pkts: float = Field(0.0)
    subflow_fwd_byts: float = Field(0.0)
    subflow_bwd_pkts: float = Field(0.0)
    subflow_bwd_byts: float = Field(0.0)
    init_fwd_win_byts: float = Field(0.0)
    init_bwd_win_byts: float = Field(0.0)
    fwd_act_data_pkts: float = Field(0.0)
    fwd_seg_size_min: float = Field(0.0)
    active_mean: float = Field(0.0)
    active_std: float = Field(0.0)
    active_max: float = Field(0.0)
    active_min: float = Field(0.0)
    idle_mean: float = Field(0.0)
    idle_std: float = Field(0.0)
    idle_max: float = Field(0.0)
    idle_min: float = Field(0.0)

    # --- Optional fields ------------------------------------------------
    destination_port: float = Field(0.0, description="Used by the XGBoost backend")
    src_ip: Optional[str] = Field(None)
    dst_ip: Optional[str] = Field(None)

    # --- Service flags (non-feature behavior switches) ------------------
    include_explanation: bool = Field(
        False, description="Compute and return SHAP top features"
    )
    include_advisory: bool = Field(
        False,
        description="Generate a GenAI security advisory "
        "(prediction + SHAP + network context)",
    )

    _validation_warnings: List[str] = []

    @field_validator("*", mode="before")
    @classmethod
    def sanitize_numeric(cls, v, info):
        if info.field_name in ("src_ip", "dst_ip", "include_explanation", "include_advisory"):
            return v
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return 0.0
        return v

    @model_validator(mode="after")
    def clamp_out_of_range(self):
        warnings = []
        for name in self._non_negative_fields():
            val = getattr(self, name, None)
            if val is not None and val < 0:
                warnings.append(f"{name}: negative value {val} clamped to 0")
                setattr(self, name, 0.0)

        for name in self._flag_fields():
            val = getattr(self, name, None)
            if val is None:
                continue
            if val < 0:
                warnings.append(f"{name}: value {val} clamped to 0")
                setattr(self, name, 0.0)
            elif val > 1:
                warnings.append(f"{name}: value {val} clamped to 1")
                setattr(self, name, 1.0)

        if len(warnings) > 25:
            warnings = warnings[:25] + ["...more warnings truncated"]
        self._validation_warnings = warnings
        return self

    @staticmethod
    def _non_negative_fields():
        return {
            "flow_duration", "tot_fwd_pkts", "tot_bwd_pkts",
            "totlen_fwd_pkts", "totlen_bwd_pkts",
            "fwd_pkt_len_max", "fwd_pkt_len_min", "fwd_pkt_len_mean",
            "bwd_pkt_len_max", "bwd_pkt_len_min", "bwd_pkt_len_mean",
            "pkt_len_min", "pkt_len_max", "pkt_len_mean",
            "fwd_act_data_pkts", "subflow_fwd_pkts", "subflow_fwd_byts",
            "subflow_bwd_pkts", "subflow_bwd_byts",
        }

    @staticmethod
    def _flag_fields():
        return {
            "fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt", "psh_flag_cnt",
            "ack_flag_cnt", "urg_flag_cnt", "cwr_flag_count", "ece_flag_cnt",
            "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags",
        }

    def feature_dict(self) -> dict:
        """Return the canonical snake_case feature values as a plain dict."""
        return self.model_dump(
            exclude={"src_ip", "dst_ip", "include_explanation", "include_advisory"}
        )

    def context_dict(self) -> dict:
        """Return the contextual / optional fields used for alerting + LLM."""
        context = {"src_ip": self.src_ip, "dst_ip": self.dst_ip}
        if self.destination_port:
            context["destination_port"] = self.destination_port
        return context


# =============================================================================
# Explainability / LLM-advisory request models
# =============================================================================


class ExplainRequest(BaseModel):
    """Requests SHAP explanation for a single flow."""
    features: PredictionRequest
    top_k: int = Field(5, ge=1, le=76)


class AdvisoryRequest(BaseModel):
    """Requests a GenAI security advisory for an already-predicted flow."""
    prediction: "PredictionResultSummary"
    top_shap_features: List[dict] = Field(default_factory=list)
    network_context: dict = Field(default_factory=dict)
    include_remediation: bool = True
    severity: Optional[str] = None
    evidence_summary: Optional[List[str]] = None


class PredictionResultSummary(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    is_attack: bool
    probabilities: dict

    model_config = {"extra": "allow"}


# permit forward reference declared above to resolve
AdvisoryRequest.model_rebuild()