"""Alert persistence and management for the unified IDS.

Alerts are created only when the ML pipeline flags an attack, and only when
a database session is available. Deduplication is windowed by src_ip +
attack_type to avoid alert storms on repeated malicious flows.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..database.models import Alert, Incident, IncidentStatus, SeverityLevel

logger = logging.getLogger(__name__)

_HIGH = {
    "ddos", "dos", "brute force", "sql injection", "web attack",
    "exploits", "reconnaissance",
}
_CRITICAL = {"infiltration", "heartbleed", "botnet", "worms", "shellcode", "backdoor"}


class AlertService:
    def classify_severity(self, attack_type: str, confidence: Optional[float] = None) -> SeverityLevel:
        """Map attack family + confidence to a severity level."""
        key = attack_type.lower()
        if any(c in key for c in _CRITICAL):
            return SeverityLevel.CRITICAL
        if any(c in key for c in _HIGH):
            return SeverityLevel.HIGH if (confidence or 0) >= 0.7 else SeverityLevel.MEDIUM
        if confidence:
            if confidence > 0.95:
                return SeverityLevel.CRITICAL
            if confidence > 0.85:
                return SeverityLevel.HIGH
            if confidence > 0.7:
                return SeverityLevel.MEDIUM
            return SeverityLevel.LOW
        return SeverityLevel.MEDIUM

    async def check_duplicate(self, db: AsyncSession, src_ip: str, attack_type: str) -> Optional[Alert]:
        cutoff = datetime.utcnow() - timedelta(seconds=settings.ALERT_DEDUP_WINDOW_SECONDS)
        res = await db.execute(
            sa.select(Alert)
            .where(
                Alert.src_ip == src_ip,
                Alert.attack_type == attack_type,
                Alert.timestamp >= cutoff,
            )
            .order_by(Alert.timestamp.desc())
            .limit(1)
        )
        return res.scalar_one_or_none()

    async def create_alert(
        self,
        db: AsyncSession,
        *,
        attack_type: str,
        src_ip: str,
        dst_ip: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None,
        prediction_score: Optional[float] = None,
        confidence: Optional[float] = None,
        model_backend: Optional[str] = None,
        llm_insight: Optional[str] = None,
    ) -> Optional[Alert]:
        """Create an alert (deduplicated). Returns the existing alert when a
        duplicate within the dedup window is found, or None when no DB."""
        if db is None:
            return None

        existing = await self.check_duplicate(db, src_ip, attack_type)
        if existing:
            logger.info(
                "Duplicate alert suppressed id=%s type=%s src=%s",
                existing.id, attack_type, src_ip,
            )
            # A repeated flow is still part of a campaign — keep the incident
            # correlation up to date even when no new row is stored.
            if settings.ALERT_CORRELATION_ENABLED:
                await self.correlate_alert(db, existing)
            return existing

        severity = self.classify_severity(attack_type, confidence or prediction_score)
        alert = Alert(
            attack_type=attack_type,
            severity=severity,
            src_ip=src_ip,
            dst_ip=dst_ip,
            features=features,
            prediction_score=prediction_score,
            confidence=confidence,
            model_backend=model_backend,
            llm_insight=llm_insight,
        )
        db.add(alert)
        await db.commit()
        await db.refresh(alert)
        logger.info("Alert created id=%s type=%s src=%s sev=%s", alert.id, attack_type, src_ip, severity.value)

        if settings.ALERT_CORRELATION_ENABLED:
            await self.correlate_alert(db, alert)

        return alert

    # ------------------------------------------------------------ correlation
    async def correlate_alert(self, db: AsyncSession, alert: Alert) -> Optional[Incident]:
        """Associate an alert with an existing open incident for the same source,
        or create one when several related alerts accumulate.

        Pure correlation — no blocking, isolation or other enforced actions.
        """
        try:
            src_ip = alert.src_ip

            open_res = await db.execute(
                sa.select(Incident).where(
                    Incident.status.in_([IncidentStatus.OPEN, IncidentStatus.INVESTIGATING])
                ).order_by(Incident.created_at.desc()).limit(20)
            )
            open_incidents = open_res.scalars().all()
            for inc in open_incidents:
                linked = await db.execute(
                    sa.select(Alert).where(Alert.incident_id == inc.id, Alert.src_ip == src_ip).limit(1)
                )
                if linked.scalar_one_or_none() is not None:
                    alert.incident_id = inc.id
                    await db.commit()
                    logger.info("Alert %s linked to existing incident %s", alert.id, inc.id)
                    return inc

            window = max(1, settings.ALERT_CORRELATION_WINDOW_HOURS)
            cutoff = datetime.utcnow() - timedelta(hours=window)
            related = await db.execute(
                sa.select(Alert)
                .where(Alert.src_ip == src_ip, Alert.timestamp >= cutoff)
                .order_by(Alert.timestamp.asc())
            )
            related_alerts = related.scalars().all()

            if len(related_alerts) >= settings.ALERT_CORRELATION_MIN_ALERTS:
                attack_types = sorted({a.attack_type for a in related_alerts})
                incident = Incident(
                    title=f"Campaign from {src_ip}: {attack_types[0] if attack_types else 'multiple attacks'}",
                    description=(
                        f"{len(related_alerts)} related alerts from {src_ip} within "
                        f"the last {window}h. Correlated into a single working incident."
                    ),
                    severity=max(related_alerts, key=lambda a: a.severity.value if hasattr(a.severity, "value") else str(a.severity)).severity,
                    status=IncidentStatus.OPEN,
                )
                db.add(incident)
                await db.commit()
                await db.refresh(incident)
                for a in related_alerts:
                    a.incident_id = incident.id
                await db.commit()
                logger.info(
                    "Correlated %d alerts into incident %s",
                    len(related_alerts), incident.id,
                )
                return incident
        except Exception as exc:
            logger.warning("Alert correlation skipped: %s", exc)
        return None

    async def acknowledge(self, db: AsyncSession, alert_id: int) -> Optional[Alert]:
        if db is None:
            return None
        res = await db.execute(sa.select(Alert).where(Alert.id == alert_id))
        alert = res.scalar_one_or_none()
        if not alert:
            return None
        alert.acknowledged = True
        await db.commit()
        await db.refresh(alert)
        return alert

    async def create_incident(self, db: AsyncSession, *, title: str, description: Optional[str] = None,
                              severity: str = "medium", assigned_to: Optional[str] = None,
                              alert_id: Optional[int] = None) -> Optional[Incident]:
        if db is None:
            return None
        incident = Incident(
            title=title,
            description=description,
            severity=self._parse_severity(severity),
            assigned_to=assigned_to,
        )
        db.add(incident)
        await db.commit()
        await db.refresh(incident)
        if alert_id:
            res = await db.execute(sa.select(Alert).where(Alert.id == alert_id))
            alert = res.scalar_one_or_none()
            if alert:
                alert.incident_id = incident.id
                await db.commit()
        return incident

    @staticmethod
    def _parse_severity(value: str) -> SeverityLevel:
        try:
            return SeverityLevel(value.lower())
        except ValueError:
            return SeverityLevel.MEDIUM


alert_service = AlertService()