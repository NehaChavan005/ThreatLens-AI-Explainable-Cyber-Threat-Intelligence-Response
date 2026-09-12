"""Dashboard data API — read-only aggregations for the analyst dashboard.

Analytics are computed server-side from persisted alerts/incidents. No write
operations live here. Degrades to 503 (or empty aggregates) when the database
is unavailable.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.models import Alert, Incident, IncidentStatus, SeverityLevel
from ..database.session import get_db
from ..schemas.responses import AlertResponse, DashboardStats

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


def _require_db(db: AsyncSession) -> AsyncSession:
    if db is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Database not available")
    return db


@router.get("/stats", response_model=DashboardStats)
async def stats(
    hours: int = Query(24, ge=1, le=24 * 30),
    db: AsyncSession = Depends(get_db),
):
    db = _require_db(db)
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    total_result = await db.execute(
        select(func.count(Alert.id)).where(Alert.timestamp >= cutoff)
    )
    total_alerts = total_result.scalar_one()

    incidents_result = await db.execute(select(func.count(Incident.id)))
    total_incidents = incidents_result.scalar_one()

    active_result = await db.execute(
        select(func.count(Incident.id)).where(
            Incident.status.in_([IncidentStatus.OPEN, IncidentStatus.INVESTIGATING])
        )
    )
    active_incidents = active_result.scalar_one()

    sev_result = await db.execute(
        select(Alert.severity, func.count(Alert.id))
        .where(Alert.timestamp >= cutoff)
        .group_by(Alert.severity)
    )
    by_severity = {
        (sev.value if hasattr(sev, "value") else str(sev)): count
        for sev, count in sev_result.all()
    }

    return DashboardStats(
        total_alerts=total_alerts,
        total_incidents=total_incidents,
        active_incidents=active_incidents,
        alerts_by_severity=by_severity,
        time_period_hours=hours,
    )


@router.get("/recent-alerts", response_model=List[AlertResponse])
async def recent_alerts(
    limit: int = Query(20, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    db = _require_db(db)
    result = await db.execute(
        select(Alert).order_by(Alert.timestamp.desc()).limit(limit)
    )
    return result.scalars().all()


@router.get("/attack-distribution")
async def attack_distribution(
    hours: int = Query(24, ge=1, le=24 * 30),
    db: AsyncSession = Depends(get_db),
):
    db = _require_db(db)
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    result = await db.execute(
        select(Alert.attack_type, func.count(Alert.id))
        .where(Alert.timestamp >= cutoff)
        .group_by(Alert.attack_type)
        .order_by(func.count(Alert.id).desc())
    )
    return {attack: count for attack, count in result.all()}


@router.get("/top-attackers")
async def top_attackers(
    limit: int = Query(10, ge=1, le=50),
    hours: int = Query(24, ge=1, le=24 * 30),
    db: AsyncSession = Depends(get_db),
):
    db = _require_db(db)
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    result = await db.execute(
        select(Alert.src_ip, Alert.attack_type, Alert.severity, Alert.id, Alert.timestamp)
        .where(Alert.timestamp >= cutoff)
        .order_by(Alert.src_ip, Alert.timestamp.desc())
    )
    groups: dict = {}
    for src_ip, attack, sev, alert_id, ts in result.all():
        s = sev.value if hasattr(sev, "value") else str(sev)
        entry = groups.setdefault(
            src_ip,
            {"src_ip": src_ip, "attack_count": 0, "max_severity": "low", "attack_types": set()},
        )
        entry["attack_count"] += 1
        entry["attack_types"].add(attack)
        if rank.get(s, 0) > rank.get(entry["max_severity"], 0):
            entry["max_severity"] = s
    rows = [
        {
            "src_ip": e["src_ip"],
            "attack_count": e["attack_count"],
            "max_severity": e["max_severity"],
            "attack_types": sorted(e["attack_types"]),
        }
        for e in groups.values()
    ]
    rows.sort(key=lambda r: r["attack_count"], reverse=True)
    return rows[:limit]


@router.get("/timeline")
async def timeline(
    hours: int = Query(24, ge=1, le=24 * 30),
    interval_minutes: int = Query(60, ge=5, le=60 * 24),
    db: AsyncSession = Depends(get_db),
):
    """Per-interval alert counts by severity over the last `hours`."""
    db = _require_db(db)
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    result = await db.execute(
        select(Alert.timestamp, Alert.severity)
        .where(Alert.timestamp >= cutoff)
        .order_by(Alert.timestamp)
    )
    buckets: dict = {}
    for ts, sev in result.all():
        s = sev.value if hasattr(sev, "value") else str(sev)
        b = int((ts - cutoff).total_seconds() // (interval_minutes * 60))
        bucket_key = (cutoff + timedelta(minutes=b * interval_minutes)).strftime("%Y-%m-%dT%H:%M")
        entry = buckets.setdefault(bucket_key, {"low": 0, "medium": 0, "high": 0, "critical": 0})
        entry[s] = entry.get(s, 0) + 1

    return [
        {"timestamp": key, **counts}
        for key, counts in sorted(buckets.items())
    ]