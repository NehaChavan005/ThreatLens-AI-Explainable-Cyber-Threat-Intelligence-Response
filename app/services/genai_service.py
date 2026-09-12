"""GenAI advisory layer — strictly advisory, never part of the prediction.

The LLM receives three inputs only:
  1. The ML prediction result   (class, confidence, probabilities)
  2. The SHAP explanation       (top-k contributing features)
  3. The network flow context   (src/dst IP, ports)

Output is a contextual security insight with recommended actions. If the
LLM is unreachable, the service degrades to a deterministic heuristic
advisory so the API keeps working.
"""

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.config import Settings, settings

logger = logging.getLogger(__name__)

SEVERITY_ATLAS: Dict[str, str] = {
    "high": "This is a high-severity event. Treat as an active security incident.",
    "critical": "This is a critical security event. Immediate containment is required.",
    "medium": "This is a medium-severity event. Validate before escalation.",
    "low": "This is a low-severity event. Log it and monitor for recurrence.",
}


@dataclass
class Advisory:
    advisory: str
    status: str = "ok"
    provider: str = "ibm-granite"
    model_id: str = ""
    generated: bool = False
    error: Optional[str] = None


class GenAIService:
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg

    # ----------------------------------------------------------------- enabled
    @property
    def enabled(self) -> bool:
        return (
            self.cfg.LLM_ENABLED
            and bool(self.cfg.LLM_API_KEY)
            and bool(self.cfg.LLM_API_URL)
        )

# ------------------------------------------------------------ prompt build
    def build_prompt(
        self,
        prediction: "PredictionResultSummary",
        top_shap_features: List[Dict[str, Any]],
        network_context: Dict[str, Any],
        include_remediation: bool = True,
        severity: Optional[str] = None,
        evidence_summary: Optional[List[str]] = None,
        defensive_actions: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Construct the single advisory prompt from the provided inputs.

        The LLM sees prediction, SHAP evidence, network context, severity
        and the system-computed defensive recommendations. It is explicitly
        instructed that the ML prediction is authoritative and that it must
        never invent indicators that are not listed below.
        """
        lines: List[str] = [
            "You are a senior SOC analyst providing a concise security advisory for an automated Intrusion Detection System.",
            "Use ONLY the evidence provided below. Be factual, specific, and actionable. No generic filler.",
            "The ML prediction below is authoritative — do NOT challenge or override it.",
            "Do NOT invent IP addresses, ports, or indicators that are not explicitly listed.",
            "",
            "=== ML PREDICTION ===",
            f"- Predicted class: {prediction['class_name']} (class id {prediction['class_id']})",
            f"- Confidence: {prediction['confidence']:.4f}",
            f"- Attack: {'yes' if prediction['is_attack'] else 'no'}",
            f"- Risk/severity: {severity or 'not assessed'}",
        ]

        if prediction.get("probabilities"):
            probs = sorted(prediction["probabilities"].items(), key=lambda kv: kv[1], reverse=True)[:3]
            probs_txt = ", ".join(f"{k}={v:.3f}" for k, v in probs)
            lines.append(f"- Top-3 class probabilities: {probs_txt}")

        lines.append("")
        lines.append("=== SHAP EXPLAINABILITY (top contributing features) ===")
        if top_shap_features:
            for fc in top_shap_features[: self.cfg.LLM_ADVISORY_TOP_N]:
                lines.append(
                    f"- {fc.get('feature', '?')} (raw={fc.get('raw_value', '?')}, "
                    f"shap={fc.get('shap_value', '?')}, pushes prediction "
                    f"{fc.get('direction', 'increase')})"
                )
        else:
            lines.append("- No SHAP features available.")

        lines.append("")
        lines.append("=== THREAT EVIDENCE (system-generated) ===")
        if evidence_summary:
            lines.extend(f"- {item}" for item in evidence_summary)
        else:
            lines.append("- No additional evidence beyond the SHAP features above.")

        lines.append("")
        lines.append("=== SYSTEM-RECOMMENDED DEFENSIVE ACTIONS (advisory only) ===")
        if defensive_actions:
            for i, a in enumerate(defensive_actions, start=1):
                lines.append(
                    f"- [{a.get('priority', 'MEDIUM')}] {a.get('action', '?')} "
                    f"({a.get('urgency', 'Normal')}; "
                    f"{'approval required' if a.get('approval_required') else 'review '}): "
                    f"{a.get('reason', '')}"
                )
        else:
            lines.append("- No recommended defensive actions for this event.")

        lines.append("")
        lines.append("=== NETWORK CONTEXT ===")
        nc = network_context or {}
        lines.append(f"- Source IP: {nc.get('src_ip') or 'unknown'}")
        lines.append(f"- Destination IP: {nc.get('dst_ip') or 'unknown'}")
        if nc.get("destination_port"):
            lines.append(f"- Destination port: {nc['destination_port']}")

        if include_remediation:
            lines += [
                "",
                "Provide:",
                "1. What happened? A 2-3 sentence executive summary of the event.",
                "2. Why was it detected? Which evidence supports the detection.",
                "3. Potential impact of the event.",
                "4. What the analyst should investigate next.",
                "5. Which recommended defensive actions to prioritize and why.",
            ]
        else:
            lines += ["", "Provide a brief factual summary only. No remediation."]

        lines.append("")
        lines.append("Respond with a single well-structured advisory.")
        return "\n".join(lines)

    # ------------------------------------------------------------- invocation
    def generate(self, prompt: str, max_retries: int = 1) -> Optional[str]:
        """Call IBM Granite (Watsonx.ai text generation) and return the text."""
        payload = {
            "parameters": {
                "max_new_tokens": self.cfg.LLM_MAX_TOKENS,
                "min_new_tokens": 15,
                "temperature": self.cfg.LLM_TEMPERATURE,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
            },
            "model_id": self.cfg.LLM_MODEL_ID,
            "input": prompt,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.cfg.LLM_API_KEY}",
        }
        url = self.cfg.LLM_API_URL
        if self.cfg.LLM_PROJECT_ID:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}project_id={self.cfg.LLM_PROJECT_ID}"

        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.cfg.LLM_TIMEOUT) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return self._extract_text(body)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning("LLM attempt %d failed: %s", attempt + 1, exc)

        if last_error:
            logger.error("LLM unavailable after %d attempts: %s", max_retries + 1, last_error)
        return None

    @staticmethod
    def _extract_text(body: Dict[str, Any]) -> Optional[str]:
        if "results" in body:
            texts = [r.get("generated_text", "") for r in body["results"] if r.get("generated_text")]
            if texts:
                return texts[0].strip()
        if "outputs" in body:
            for out in body["outputs"]:
                if isinstance(out, dict) and out.get("text"):
                    return out["text"].strip()
        if "generated_text" in body:
            return body["generated_text"].strip()
        return None

    # ------------------------------------------------------------- orchestrator
    def run_advisory(
        self,
        prediction: "PredictionResultSummary",
        top_shap_features: List[Dict[str, Any]],
        network_context: Dict[str, Any],
        include_remediation: bool = True,
        severity: Optional[str] = None,
        evidence_summary: Optional[List[str]] = None,
        defensive_actions: Optional[List[Dict[str, Any]]] = None,
    ) -> Advisory:
        """Generate an advisory, falling back to a heuristic when unavailable."""
        prompt = self.build_prompt(
            prediction,
            top_shap_features,
            network_context,
            include_remediation,
            severity=severity,
            evidence_summary=evidence_summary,
            defensive_actions=defensive_actions,
        )

        if not self.enabled:
            return Advisory(
                advisory=self._heuristic_advisory(
                    prediction,
                    top_shap_features,
                    network_context,
                    severity=severity,
                    evidence_summary=evidence_summary,
                    defensive_actions=defensive_actions,
                ),
                status="degraded",
                provider="heuristic",
                model_id="",
                generated=False,
                error="LLM not configured (LLM_ENABLED/LLM_API_KEY/LLM_API_URL)",
            )

        text = self.generate(prompt)
        if text:
            clean = re.sub(r"\s*\n{3,}", "\n\n", text).strip()
            return Advisory(
                advisory=clean,
                status="ok",
                provider="ibm-granite",
                model_id=self.cfg.LLM_MODEL_ID,
                generated=True,
            )

        return Advisory(
            advisory=self._heuristic_advisory(
                prediction,
                top_shap_features,
                network_context,
                severity=severity,
                evidence_summary=evidence_summary,
                defensive_actions=defensive_actions,
            ),
            status="degraded",
            provider="heuristic",
            model_id=self.cfg.LLM_MODEL_ID,
            generated=False,
            error="LLM call failed or returned empty output",
        )

    # ------------------------------------------------------------- heuristic
    @staticmethod
    def _heuristic_advisory(
        prediction,
        top_shap_features,
        network_context,
        severity: Optional[str] = None,
        evidence_summary: Optional[List[str]] = None,
        defensive_actions: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        name = prediction.get("class_name", "unknown")
        conf = prediction.get("confidence", 0.0)
        is_attack = prediction.get("is_attack", False)

        if not is_attack:
            return (
                f"No immediate threat: flow classified as {name} "
                f"(confidence {conf:.1%}). Continue routine monitoring."
            )

        sev = (severity or ("high" if conf >= 0.9 else ("medium" if conf >= 0.7 else "low"))).lower()
        sev_line = SEVERITY_ATLAS.get(sev, "Monitor this event.")

        top = ""
        if top_shap_features:
            names = ", ".join(fc.get("feature", "?") for fc in top_shap_features[:5])
            top = f"Key contributing features: {names}."

        evidence = ""
        if evidence_summary:
            evidence = " Evidence: " + "; ".join(evidence_summary[:4]) + "."

        actions = ""
        if defensive_actions:
            high = [
                f"[{a.get('priority')}] {a.get('action')}"
                for a in defensive_actions
                if a.get("priority") in ("HIGH", "CRITICAL")
            ]
            if high:
                actions = " Priority actions: " + " | ".join(high[:4]) + "."

        src = (network_context or {}).get("src_ip") or "unknown"
        return (
            f"{sev_line} Flow from {src} classified as {name} "
            f"(confidence {conf:.1%}). {top}{evidence}{actions} "
            "Investigate the source, validate against correlated events, and "
            "apply the recommended defensive actions with analyst approval."
        )


# Global singleton
genai_service = GenAIService()