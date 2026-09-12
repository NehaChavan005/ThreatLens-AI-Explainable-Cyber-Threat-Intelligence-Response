"""Defensive action recommendation engine.

ThreatLens AI is a detection + guidance system, NOT an autonomous response
system. This service maps a confirmed ML classification to a concrete set of
recommended defensive actions. Every action is:

  * derived from the actual attack class, severity, confidence and context,
  * advisory only, and
  * gated behind explicit analyst approval (`approval_required`).

No action is ever executed here — no blocking, isolation, reset, or file
changes happen automatically.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..schemas.responses import DefensiveAction

logger = logging.getLogger(__name__)

Priority = str  # LOW | MEDIUM | HIGH | CRITICAL

# ---------------------------------------------------------------------------
# Attack-class taxonomy actually supported by the served model (15 classes of
# the GenAI-augmented XGBoost fallback, CIC-IDS2017 naming).
# ---------------------------------------------------------------------------
_ATTACK_CATEGORIES: Dict[str, str] = {
    "portscan": "reconnaissance",
    "bot": "malware",
    "ddos": "dos",
    "dos": "dos",
    "dos goldeneye": "dos",
    "dos hulk": "dos",
    "dos slowhttptest": "dos",
    "dos slowloris": "dos",
    "ftp-patator": "brute_force",
    "ssh-patator": "brute_force",
    "web attack brute force": "brute_force",
    "web attack sql injection": "web_attack",
    "web attack xss": "web_attack",
    "infiltration": "infiltration",
    "heartbleed": "exposure",
    "benign": "benign",
}

# ---------------------------------------------------------------------------
# Playbooks: (action, reason_template, base_priority, urgency, needs_approval)
# Reason templates are rendered with {src_ip} and {dst_ip} when present.
# ---------------------------------------------------------------------------
ActionSpec = Tuple[str, str, Priority, str, bool]

_PLAYBOOKS: Dict[str, List[ActionSpec]] = {
    "reconnaissance": [
        (
            "Investigate source IP",
            "The source address generated repeated scanning/connection attempts characteristic of reconnaissance.",
            "MEDIUM",
            "Normal",
            False,
        ),
        (
            "Review scanning scope and targets",
            "Confirm which destination services/ports the source probed and whether they are exposed.",
            "MEDIUM",
            "Normal",
            False,
        ),
        (
            "Increase monitoring on targeted systems",
            "Targeted hosts were probed and may be subject to follow-up exploitation.",
            "HIGH",
            "High",
            False,
        ),
        (
            "Consider temporary source blocking after validation",
            "If the source is confirmed unauthorized, a temporary block reduces exposure while monitoring continues.",
            "HIGH",
            "High",
            True,
        ),
    ],
    "brute_force": [
        (
            "Investigate source addresses",
            "Repeated authentication-failure patterns indicate credential brute forcing.",
            "HIGH",
            "High",
            False,
        ),
        (
            "Review authentication and access logs",
            "Confirm whether any accounts were successfully authenticated from the attacking source.",
            "HIGH",
            "High",
            False,
        ),
        (
            "Enforce account protection and rate limiting",
            "Lockouts/rate limits slow ongoing credential attacks against user accounts.",
            "MEDIUM",
            "Normal",
            False,
        ),
        (
            "Consider temporary source blocking",
            "If authenticated attempts continue, block the source after analyst validation.",
            "HIGH",
            "High",
            True,
        ),
        (
            "Initiate credential reset if compromise is confirmed",
            "Any account identified as compromised must have its credentials rotated immediately.",
            "CRITICAL",
            "Immediate",
            True,
        ),
    ],
    "dos": [
        (
            "Investigate traffic source and volume",
            "High-volume or low-and-slow traffic patterns indicate resource-exhaustion activity.",
            "HIGH",
            "High",
            False,
        ),
        (
            "Apply rate limiting / connection throttling",
            "Limiting per-source rates mitigates ongoing flooding without dropping the service.",
            "HIGH",
            "High",
            True,
        ),
        (
            "Review firewall/WAF controls",
            "Confirm the edge defenses can absorb or filter the observed attack pattern.",
            "MEDIUM",
            "Normal",
            False,
        ),
        (
            "Isolate affected services if necessary",
            "If the service is degraded, temporarily isolate it to protect adjacent systems.",
            "CRITICAL",
            "Immediate",
            True,
        ),
        (
            "Escalate high-severity events",
            "Sustained availability impact warrants immediate escalation to the incident team.",
            "CRITICAL",
            "Immediate",
            False,
        ),
    ],
    "infiltration": [
        (
            "Isolate affected endpoint after analyst validation",
            "Suspected intrusion warrants isolating the host from the network to limit lateral movement.",
            "CRITICAL",
            "Immediate",
            True,
        ),
        (
            "Inspect endpoint and network logs",
            "Establish what the attacker accessed and the entry vector used.",
            "HIGH",
            "High",
            False,
        ),
        (
            "Identify persistence indicators",
            "Check for scheduled tasks, services, and startup modifications that indicate footholds.",
            "HIGH",
            "High",
            False,
        ),
        (
            "Investigate lateral movement",
            "Look for internal connections from the compromised host toward other assets.",
            "HIGH",
            "High",
            False,
        ),
        (
            "Escalate incident",
            "Confirmed infiltration is a serious security incident that requires escalation.",
            "CRITICAL",
            "Immediate",
            False,
        ),
    ],
    "web_attack": [
        (
            "Investigate destination and payload",
            "Web application attack traffic was observed against the destination service.",
            "HIGH",
            "High",
            False,
        ),
        (
            "Review unusual outbound traffic",
            "Determine whether data was exfiltrated through the web channel.",
            "HIGH",
            "High",
            False,
        ),
        (
            "Inspect affected endpoint and application logs",
            "Identify the vulnerable request path and whether exploitation succeeded.",
            "HIGH",
            "High",
            False,
        ),
        (
            "Restrict suspicious outbound communication",
            "If exfiltration is suspected, restrict unauthorized outbound connections after validation.",
            "CRITICAL",
            "Immediate",
            True,
        ),
        (
            "Preserve evidence",
            "Retain captured traffic/events for incident investigation.",
            "HIGH",
            "High",
            False,
        ),
        (
            "Escalate for high-confidence, high-severity events",
            "Confirmed exploitation with data impact requires immediate escalation.",
            "CRITICAL",
            "Immediate",
            False,
        ),
    ],
    "malware": [
        (
            "Isolate affected endpoint after validation",
            "Bot/malware activity indicates an infected host may be under attacker control.",
            "CRITICAL",
            "Immediate",
            True,
        ),
        (
            "Inspect process and network activity",
            "Identify the malicious process and command-and-control communication path.",
            "HIGH",
            "High",
            False,
        ),
        (
            "Quarantine confirmed malicious artifacts",
            "Samples should be quarantined for analysis once confirmed malicious.",
            "HIGH",
            "High",
            True,
        ),
        (
            "Investigate lateral movement",
            "Botnets commonly spread internally; check adjacent hosts for infection.",
            "HIGH",
            "High",
            False,
        ),
        (
            "Escalate incident",
            "Confirmed malware activity warrants incident escalation.",
            "CRITICAL",
            "Immediate",
            False,
        ),
    ],
    "exposure": [
        (
            "Investigate affected endpoint and services",
            "The Heartbleed-class identification indicates an SSL/TLS memory-exposure risk.",
            "HIGH",
            "High",
            False,
        ),
        (
            "Correlate with certificates and TLS library versions",
            "Confirm whether the impacted service runs a vulnerable TLS implementation.",
            "MEDIUM",
            "Normal",
            False,
        ),
        (
            "Limited sensitive data may have been disclosed",
            "Memory exfiltration can leak session keys, credentials, or data; treat flows as potentially exposed.",
            "HIGH",
            "High",
            False,
        ),
        (
            "Rotate secrets and session credentials if exposure is confirmed",
            "Rotate any secrets/keys that may have been disclosed through the vulnerable service.",
            "CRITICAL",
            "Immediate",
            True,
        ),
        (
            "Escalate for high-confidence or repeated exposure events",
            "Repeated exposure events indicate an active exploitation attempt.",
            "HIGH",
            "High",
            False,
        ),
    ],
}


class DefensiveActionService:
    """Translate a prediction into analyst-approved defensive guidance."""

    @staticmethod
    def _norm(value: str) -> str:
        """Fold a class name to its lookup key: lowercase, strip punctuation."""
        return re.sub(r"[^a-z0-9 ]", "", (value or "").strip().lower())

    def category_for(self, class_name: str) -> str:
        cached = getattr(self, "_norm_categories", None)
        if cached is None:
            cached = self._norm_categories = {
                self._norm(k): v for k, v in _ATTACK_CATEGORIES.items()
            }
        return cached.get(self._norm(class_name), "benign")

    def supported_attack_classes(self) -> List[str]:
        """Attack classes the current model can actually detect."""
        from ..services.inference_service import inference_service  # noqa: PLC0415

        if inference_service.initialized and inference_service.class_labels:
            return [c for c in inference_service.class_labels if c != "BENIGN"]
        return sorted({k.title() for k in _ATTACK_CATEGORIES if k != "benign"})

    def recommend(
        self,
        class_name: str,
        confidence: float,
        severity: str = "medium",
        src_ip: Optional[str] = None,
        dst_ip: Optional[str] = None,
        is_attack: bool = True,
    ) -> List[DefensiveAction]:
        """Return ordered, attack-specific defensive recommendations.

        Pure function: never executes or mutates anything.
        """
        if not is_attack:
            return []

        category = self.category_for(class_name)
        if category == "benign":
            return []

        specs = _PLAYBOOKS.get(category, [])
        sev = (severity or "medium").lower()
        high_severity = sev in ("high", "critical")

        actions: List[DefensiveAction] = []
        for action, reason_tpl, base, urgency, approval in specs:
            reason = reason_tpl.format(
                src_ip=src_ip or "the source address",
                dst_ip=dst_ip or "the destination",
            )
            priority = base
            # Escalate containment-oriented actions when severity demands it.
            if high_severity and urgency == "Immediate" and base != "CRITICAL":
                priority = "CRITICAL"
            if high_severity and urgency == "High" and base == "MEDIUM":
                priority = "HIGH"
            actions.append(
                DefensiveAction(
                    action=action,
                    reason=reason,
                    priority=priority,
                    urgency=urgency,
                    approval_required=approval,
                    category=category,
                )
            )
        return actions

    def as_prompt_block(self, actions: List[DefensiveAction]) -> str:
        """Render recommendations as structured prompt text for the LLM."""
        if not actions:
            return "- No defensive actions recommended for this event."
        lines = []
        for i, a in enumerate(
            sorted(actions, key=lambda x: -("CRITICAL HIGH MEDIUM LOW".split().index(x.priority))),
            start=1,
        ):
            approval = "requires analyst approval" if a.approval_required else "analyst review recommended"
            lines.append(
                f"{i}. [{a.priority}] {a.action} ({a.urgency}; {approval}) - {a.reason}"
            )
        return "\n".join(lines)


# Global singleton
defensive_action_service = DefensiveActionService()