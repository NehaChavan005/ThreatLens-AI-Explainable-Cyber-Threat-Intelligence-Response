"""Alert & incident management API.

Reuses the legacy ML-IDS semantics: list/filter alerts, acknowledge, update,
delete, and manage incidents. Every endpoint degrades gracefully (503) when
the database is unavailable.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.models import Alert, Incident, IncidentStatus, SeverityLevel
from ..database.session import get_db
from ..schemas.responses import (
    AlertResponse,
    AlertUpdate,
    IncidentCreate,
    IncidentDetail,
    IncidentResponse,
    IncidentUpdate,
)
from ..services.alert_service import alert_service
from ..services.defensive_action_service import defensive_action_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


def _require_db(db: AsyncSession) -> AsyncSession:
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return db


@router.get("", response_model=List[AlertResponse])
async def list_alerts(
    severity: Optional[str] = Query(None),
    src_ip: Optional[str] = Query(None),
    attack_type: Optional[str] = Query(None),
    acknowledged: Optional[bool] = Query(None),
    hours: int = Query(24, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List alerts with optional filtering and pagination."""
    db = _require_db(db)

    query = select(Alert)
    conditions = []

    if severity:
        try:
            conditions.append(Alert.severity == SeverityLevel(severity.lower()))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
    if src_ip:
        conditions.append(Alert.src_ip == src_ip)
    if attack_type:
        conditions.append(Alert.attack_type == attack_type)
    if acknowledged is not None:
        conditions.append(Alert.acknowledged == acknowledged)
    if hours:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        conditions.append(Alert.timestamp >= cutoff)

    if conditions:
        query = query.where(and_(*conditions))

    result = await db.execute(
        query.order_by(desc(Alert.timestamp)).limit(limit).offset(offset)
    )
    return result.scalars().all()


# =============================================================================
# Incidents — MUST be registered before /{alert_id} so "/incidents" is not
# captured by the int path-param route.
# =============================================================================


@router.post("/incidents", response_model=IncidentResponse)
async def create_incident(
    create: IncidentCreate,
    alert_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Create an incident, optionally linking an existing alert."""
    db = _require_db(db)
    try:
        SeverityLevel(create.severity.lower())
        IncidentStatus(create.status.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid severity or status")

    incident = await alert_service.create_incident(
        db,
        title=create.title,
        description=create.description,
        severity=create.severity,
        assigned_to=create.assigned_to,
        alert_id=alert_id,
    )
    return incident


@router.get("/incidents", response_model=List[IncidentResponse])
async def list_incidents(
    status: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    db = _require_db(db)
    query = select(Incident)
    conditions = []
    if status:
        conditions.append(Incident.status == IncidentStatus(status.lower()))
    if severity:
        conditions.append(Incident.severity == SeverityLevel(severity.lower()))
    if conditions:
        query = query.where(and_(*conditions))
    result = await db.execute(
        query.order_by(desc(Incident.created_at)).limit(limit).offset(offset)
    )
    return result.scalars().all()


@router.get("/incidents/{incident_id}", response_model=IncidentResponse)
async def get_incident(incident_id: int, db: AsyncSession = Depends(get_db)):
    db = _require_db(db)
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    incident = result.scalar_one_or_none()
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
    return incident


@router.get("/incidents/{incident_id}/detail", response_model=IncidentDetail)
async def get_incident_detail(incident_id: int, db: AsyncSession = Depends(get_db)):
    """Analyst-facing correlated view of an incident (alerts, evidence,
    recommendations, timeline)."""
    db = _require_db(db)
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    incident = result.scalar_one_or_none()
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    alerts_res = await db.execute(
        select(Alert).where(Alert.incident_id == incident.id).order_by(Alert.timestamp.asc())
    )
    related = list(alerts_res.scalars().all())

    severities = [a.severity.value if hasattr(a.severity, "value") else str(a.severity) for a in related]
    rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    highest = max(severities, key=lambda s: rank.get(s, 0)) if severities else "medium"

    confs = [a.confidence for a in related if a.confidence is not None]
    avg_conf = round(sum(confs) / len(confs), 4) if confs else None

    attack_types = sorted({a.attack_type for a in related})
    sources = sorted({a.src_ip for a in related})
    dests = sorted({a.dst_ip for a in related if a.dst_ip})

    explanation = None
    best = max(related, key=lambda a: a.confidence or 0, default=None)
    if best and best.llm_insight:
        explanation = best.llm_insight

    actions = []
    if best is not None:
        actions = defensive_action_service.recommend(
            class_name=best.attack_type,
            confidence=best.confidence or 0.0,
            severity=highest,
            src_ip=best.src_ip,
            dst_ip=best.dst_ip,
            is_attack=True,
        )

    return IncidentDetail(
        incident=IncidentResponse.model_validate(incident),
        alert_ids=[a.id for a in related],
        related_alerts=[AlertResponse.model_validate(a) for a in related],
        attack_types=attack_types,
        highest_severity=highest,
        avg_confidence=avg_conf,
        total_alerts=len(related),
        source_ips=sources,
        destination_ips=dests,
        timeline=[a.timestamp for a in related],
        explanation=explanation,
        defensive_actions=actions,
    )


@router.put("/incidents/{incident_id}", response_model=IncidentResponse)
async def update_incident(
    incident_id: int, update: IncidentUpdate, db: AsyncSession = Depends(get_db)
):
    db = _require_db(db)
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    incident = result.scalar_one_or_none()
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    updates = update.model_dump(exclude_none=True)
    for key, value in updates.items():
        setattr(incident, key, value)
    if update.status == "resolved" and not incident.resolved_at:
        incident.resolved_at = datetime.utcnow()

    await db.commit()
    await db.refresh(incident)
    return incident


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: int, db: AsyncSession = Depends(get_db)):
    db = _require_db(db)
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return alert


@router.put("/{alert_id}", response_model=AlertResponse)
async def update_alert(
    alert_id: int, update: AlertUpdate, db: AsyncSession = Depends(get_db)
):
    db = _require_db(db)
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

    if update.acknowledged is not None:
        alert.acknowledged = update.acknowledged
    if update.notes is not None:
        alert.notes = update.notes
    if update.severity is not None:
        try:
            alert.severity = SeverityLevel(update.severity.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid severity: {update.severity}")

    await db.commit()
    await db.refresh(alert)
    return alert


@router.post("/{alert_id}/acknowledge", response_model=AlertResponse)
async def acknowledge_alert(alert_id: int, db: AsyncSession = Depends(get_db)):
    db = _require_db(db)
    alert = await alert_service.acknowledge(db, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return alert


@router.delete("/{alert_id}")
async def delete_alert(alert_id: int, db: AsyncSession = Depends(get_db)):
    db = _require_db(db)
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    await db.delete(alert)
    await db.commit()
    return {"message": f"Alert {alert_id} deleted"}