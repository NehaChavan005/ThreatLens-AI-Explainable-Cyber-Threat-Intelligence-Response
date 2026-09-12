"""Explainable-AI helpers: turn SHAP contributions into analyst language.

The ML model produces numbers; this module produces meaning. It converts the
top SHAP feature contributions into:

  1. per-feature human-readable interpretations (a plain-English reason for
     why the feature mattered and which direction it pushed the prediction),
  2. a concise "Threat Evidence" summary of the strongest signals,
  3. a single fused explanation paragraph suitable for the dashboard and for
     the GenAI advisory input.

No fictional numbers are introduced here: every statement is derived from an
actual feature contribution, its real observed value, and its direction.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import Settings, settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature lexicon: pretty-name -> plain-English meaning of the observed value.
# The observed value of a feature is what the contribution actually refers to,
# so interpretations are grounded in the real network metric.
# ---------------------------------------------------------------------------
FEATURE_LEXICON: Dict[str, str] = {
    "Destination Port": "the destination service port",
    "Flow Duration": "connection duration",
    "Total Fwd Packets": "forward traffic volume (packet count)",
    "Total Backward Packets": "reverse traffic volume (packet count)",
    "Total Length of Fwd Packets": "forward data volume (bytes)",
    "Total Length of Bwd Packets": "reverse data volume (bytes)",
    "Fwd Packet Length Max": "largest forward packet size",
    "Fwd Packet Length Min": "smallest forward packet size",
    "Fwd Packet Length Mean": "average forward packet size",
    "Fwd Packet Length Std": "forward packet-size variability",
    "Bwd Packet Length Max": "largest reverse packet size",
    "Bwd Packet Length Min": "smallest reverse packet size",
    "Bwd Packet Length Mean": "average reverse packet size",
    "Bwd Packet Length Std": "reverse packet-size variability",
    "Flow Bytes/s": "network throughput (bytes per second)",
    "Flow Packets/s": "packet rate (packets per second)",
    "Flow IAT Mean": "average inter-arrival time between packets",
    "Flow IAT Std": "inter-arrival time variability",
    "Flow IAT Max": "longest gap between packets in the flow",
    "Flow IAT Min": "shortest gap between packets in the flow",
    "Fwd IAT Total": "total forward inter-arrival spread",
    "Fwd IAT Mean": "average forward inter-arrival time",
    "Fwd IAT Std": "forward inter-arrival time variability",
    "Fwd IAT Max": "longest forward packet gap",
    "Fwd IAT Min": "shortest forward packet gap",
    "Bwd IAT Total": "total reverse inter-arrival spread",
    "Bwd IAT Mean": "average reverse inter-arrival time",
    "Bwd IAT Std": "reverse inter-arrival time variability",
    "Bwd IAT Max": "longest reverse packet gap",
    "Bwd IAT Min": "shortest reverse packet gap",
    "Fwd PSH Flags": "forward PUSH flag usage",
    "Bwd PSH Flags": "reverse PUSH flag usage",
    "Fwd URG Flags": "forward URGENT flag usage",
    "Bwd URG Flags": "reverse URGENT flag usage",
    "Fwd Header Length": "forward header overhead",
    "Bwd Header Length": "reverse header overhead",
    "Fwd Packets/s": "forward packet rate",
    "Bwd Packets/s": "reverse packet rate",
    "Min Packet Length": "smallest observed packet size",
    "Max Packet Length": "largest observed packet size",
    "Packet Length Mean": "average observed packet size",
    "Packet Length Std": "overall packet-size variability",
    "Packet Length Variance": "overall packet-size variance",
    "FIN Flag Count": "FIN (session-end) flag count",
    "SYN Flag Count": "SYN (connection-request) flag count",
    "RST Flag Count": "RST (connection-reset) flag count",
    "PSH Flag Count": "PSH (push) flag count",
    "ACK Flag Count": "ACK (acknowledgement) flag count",
    "URG Flag Count": "URG (urgent) flag count",
    "CWE Flag Count": "CWR (congestion-window-reduced) flag count",
    "ECE Flag Count": "ECE (explicit-congestion-notification) flag count",
    "Down/Up Ratio": "ratio of reverse to forward traffic",
    "Average Packet Size": "average packet size across the flow",
    "Avg Fwd Segment Size": "average forward payload segment size",
    "Avg Bwd Segment Size": "average reverse payload segment size",
    "Fwd Avg Bytes/Bulk": "bulk forward transfer size",
    "Fwd Avg Packets/Bulk": "bulk forward transfer packet count",
    "Fwd Avg Bulk Rate": "bulk forward transfer rate",
    "Bwd Avg Bytes/Bulk": "bulk reverse transfer size",
    "Bwd Avg Packets/Bulk": "bulk reverse transfer packet count",
    "Bwd Avg Bulk Rate": "bulk reverse transfer rate",
    "Subflow Fwd Packets": "forward sub-flow packet count",
    "Subflow Fwd Bytes": "forward sub-flow byte count",
    "Subflow Bwd Packets": "reverse sub-flow packet count",
    "Subflow Bwd Bytes": "reverse sub-flow byte count",
    "Init_Win_bytes_forward": "sender's initial TCP window (forward)",
    "Init_Win_bytes_backward": "sender's initial TCP window (reverse)",
    "act_data_pkt_fwd": "forward packets carrying payload data",
    "min_seg_size_forward": "minimum forward TCP segment size",
    "Active Mean": "average time a connection was active",
    "Active Std": "variability in active-connection time",
    "Active Max": "longest active-connection time",
    "Active Min": "shortest active-connection time",
    "Idle Mean": "average idle time between bursts",
    "Idle Std": "variability in idle time",
    "Idle Max": "longest idle period",
    "Idle Min": "shortest idle period",
}

# Evidence phrasing for signals that commonly drive attack classifications.
# A feature is only ever reported with one of these when its contribution is
# actually in the top set with the matching direction.
SIGNAL_RULES = [
    ("Port Scan", "Destination Port", "any", "activity concentrated on specific destination services"),
    ("SYN Flood/Fingerprinting", "SYN Flag Count", "increase", "elevated SYN (connection-request) count"),
    ("Connection Reset Behaviour", "RST Flag Count", "increase", "elevated RST (connection-reset) count"),
    ("Persistent Connections", "Active Mean", "increase", "abnormally long-lived active connections"),
    ("Sporadic Connections", "Idle Mean", "any", "abnormal idle-time pattern between activity bursts"),
    ("Sweeping/Irregular Flow Sizes", "Packet Length Std", "any", "high variability in packet sizes"),
    ("Connection Attempt Rate", "Flow Packets/s", "increase", "elevated packet rate"),
    ("Data Volume", "Flow Bytes/s", "increase", "elevated throughput"),
    ("Lateral Movement Pattern", "Bwd Packet Length Mean", "increase", "abnormally large reverse-direction packets"),
    ("Bulk Transfer", "Avg Bwd Segment Size", "increase", "abnormally large reverse payload segments"),
    ("Slow/Low-and-Slow Behaviour", "Flow IAT Mean", "increase", "abnormally slow/intermittent packet cadence"),
    ("Slow/Throttled Behaviour", "Flow IAT Std", "increase", "highly irregular inter-arrival timing"),
    ("Short-Lived Traffic Burst", "Flow Duration", "any", "abnormal connection duration"),
    ("Elevated Connection Count", "Total Fwd Packets", "increase", "elevated forward connection activity"),
    ("Elevated Connection Count", "Total Backward Packets", "increase", "elevated reverse connection activity"),
]


class ExplainableAI:
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        self._pretty_to_api: Dict[str, str] = {}
        self._api_to_pretty: Dict[str, str] = {}

    # ------------------------------------------------------------ mappings
    def _load_mappings(self) -> None:
        if self._api_to_pretty:
            return
        try:
            with open(self.cfg.FEATURE_MAPPING_GENAI, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for pretty, api in data.items():
                self._pretty_to_api[str(pretty)] = str(api)
                self._api_to_pretty[str(api)] = str(pretty)
        except Exception as exc:  # pragma: no cover - config dependent
            logger.warning("Could not load feature mapping for explanations: %s", exc)

    def pretty_name(self, api_key: str) -> str:
        self._load_mappings()
        return self._api_to_pretty.get(api_key, api_key)

    def meaning_of(self, pretty: str) -> str:
        return FEATURE_LEXICON.get(pretty, "the observed value of this feature")

    # ------------------------------------------------------------ evidence
    def evidence_summary(self, top_features: List[Dict[str, Any]]) -> List[str]:
        """Return a de-duplicated list of analyst-readable evidence bullets.

        A signal is emitted only when its feature actually appears in the top
        contributions with a compatible direction, or when the feature is
        listed with direction "any".
        """
        if not top_features:
            return []

        by_feature: Dict[str, Dict[str, Any]] = {}
        for fc in top_features:
            by_feature.setdefault(fc.get("feature", ""), fc)

        signals: List[str] = []
        seen: set = set()
        for label, feat, direction, text in SIGNAL_RULES:
            fc = by_feature.get(feat)
            if fc is None:
                continue
            if direction == "any" or fc.get("direction") == direction:
                if label not in seen:
                    seen.add(label)
                    signals.append(f"{label}: {text}")
        return signals

    def interpretation_lines(self, top_features: List[Dict[str, Any]]) -> List[str]:
        """One grounded sentence per top feature explaining its role."""
        lines: List[str] = []
        for fc in top_features[:5]:
            api = fc.get("feature", "")
            pretty = self.pretty_name(api)
            meaning = self.meaning_of(pretty)
            direction = fc.get("direction", "increase")
            raw = fc.get("raw_value", 0.0)
            verb = "contributed strongly to the threat classification" if direction == "increase" else "pulled the classification away from the threat"
            lines.append(
                f"{pretty} ({meaning}) {verb}; observed value {self._fmt(raw)} "
                f"with SHAP impact {fc.get('shap_value', 0.0):+.4f}."
            )
        return lines

    def interpretation(self, top_features: List[Dict[str, Any]]) -> str:
        lines = self.interpretation_lines(top_features)
        if not lines:
            return "No SHAP evidence was available for this prediction."
        return " ".join(lines)

    @staticmethod
    def _fmt(value: Any) -> str:
        try:
            fval = float(value)
        except (TypeError, ValueError):
            return str(value)
        if abs(fval) >= 1000:
            return f"{fval:,.1f}"
        return f"{fval:.4f}"

    # ------------------------------------------------------------ response
    @staticmethod
    def source_label(explainer_type: str) -> str:
        """Plain label for where the explanation came from."""
        mapping = {
            "tree": "TreeExplainer",
            "kernel": "KernelExplainer",
            "none": "fallback explanation",
        }
        return mapping.get(explainer_type or "", explainer_type or "unknown")

    def enrich(self, explanation) -> Dict[str, Any]:
        """Convert a shap_service ExplanationResult into response fields with
        grounded evidence + interpretation attached."""
        top = [
            fc.model_dump() if hasattr(fc, "model_dump") else dict(fc)
            for fc in getattr(explanation, "top_features", [])
        ]
        return {
            "top_features": getattr(explanation, "top_features", []),
            "base_value": getattr(explanation, "base_value", 0.0),
            "bias_position": getattr(explanation, "bias_position", 0),
            "explainer_type": getattr(explanation, "explainer_type", "tree"),
            "status": getattr(explanation, "status", "ok"),
            "evidence_summary": self.evidence_summary(top),
            "interpretation": self.interpretation(top),
            "explanation_source": self.source_label(getattr(explanation, "explainer_type", "")),
        }


# Global singleton
explainable_ai = ExplainableAI()