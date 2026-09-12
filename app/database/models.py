"""SQLAlchemy ORM models for the unified IDS.

Keeps the same schema and enums as the legacy ML-IDS inference server so
existing alerts/incidents remain compatible.
"""

import enum

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class SeverityLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(str, enum.Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    attack_type = Column(String(100), nullable=False, index=True)
    severity = Column(
        SQLEnum(SeverityLevel), nullable=False, default=SeverityLevel.MEDIUM, index=True
    )
    src_ip = Column(String(45), nullable=False, index=True)
    dst_ip = Column(String(45), nullable=True)
    timestamp = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    features = Column(JSON, nullable=True)
    prediction_score = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    model_backend = Column(String(20), nullable=True)
    llm_insight = Column(Text, nullable=True)
    acknowledged = Column(Boolean, default=False, nullable=False)
    incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=True, index=True)
    notes = Column(Text, nullable=True)

    incident = relationship("Incident", back_populates="alerts")

    def __repr__(self):
        return f"<Alert(id={self.id}, type={self.attack_type}, sev={self.severity}, src={self.src_ip})>"


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(
        SQLEnum(IncidentStatus), nullable=False, default=IncidentStatus.OPEN, index=True
    )
    severity = Column(
        SQLEnum(SeverityLevel), nullable=False, default=SeverityLevel.MEDIUM, index=True
    )
    assigned_to = Column(String(100), nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, nullable=True)

    alerts = relationship("Alert", back_populates="incident")

    def __repr__(self):
        return f"<Incident(id={self.id}, title={self.title}, status={self.status})>"