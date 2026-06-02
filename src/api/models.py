"""
Pydantic models for the Store Intelligence System API.
Covers all event types, API request/response schemas, and validation.
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ──────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────

class EventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"
    # Legacy aliases from sample data
    entry = "entry"
    exit = "exit"
    zone_entered = "zone_entered"
    zone_exited = "zone_exited"
    queue_completed = "queue_completed"
    queue_abandoned = "queue_abandoned"


class ZoneType(str, Enum):
    SHELF = "SHELF"
    DISPLAY = "DISPLAY"
    BILLING = "BILLING"
    PASSAGE = "PASSAGE"
    FITTING = "FITTING"


class Gender(str, Enum):
    M = "M"
    F = "F"
    UNKNOWN = "UNKNOWN"


class AgeBucket(str, Enum):
    UNDER_18 = "Under-18"
    AGE_18_24 = "18-24"
    AGE_25_34 = "25-34"
    AGE_35_44 = "35-44"
    AGE_45_54 = "45-54"
    AGE_55_PLUS = "55+"


class AnomalyType(str, Enum):
    QUEUE_SPIKE = "QUEUE_SPIKE"
    CONVERSION_DROP = "CONVERSION_DROP"
    DEAD_ZONE = "DEAD_ZONE"
    STALE_FEED = "STALE_FEED"
    OVERCROWDING = "OVERCROWDING"
    HIGH_ABANDONMENT = "HIGH_ABANDONMENT"


class AnomalySeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ──────────────────────────────────────────────
# Event Schemas (Ingest)
# ──────────────────────────────────────────────

class EventMetadata(BaseModel):
    """Flexible metadata bag for extra event context."""
    direction: Optional[str] = None          # "IN" | "OUT"
    bbox: Optional[List[float]] = None       # [x1,y1,x2,y2] in pixels
    track_score: Optional[float] = None      # tracker confidence
    frame_id: Optional[int] = None
    zone_hotspot_x: Optional[float] = None
    zone_hotspot_y: Optional[float] = None
    queue_position_at_join: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None


class StoreEvent(BaseModel):
    """Canonical event object consumed by /events/ingest."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str  # accepts both canonical and legacy types
    store_id: str
    camera_id: str
    visitor_id: Optional[str] = None       # hashed / anonymous token
    id_token: Optional[str] = None         # alias from legacy schema
    track_id: Optional[int] = None
    timestamp: Optional[datetime] = None
    event_timestamp: Optional[datetime] = None   # legacy alias
    event_time: Optional[datetime] = None        # legacy alias
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    zone_type: Optional[str] = None
    dwell_ms: Optional[int] = None
    is_staff: bool = False
    is_face_hidden: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    gender: Optional[str] = None
    gender_pred: Optional[str] = None           # legacy alias
    age: Optional[int] = None
    age_pred: Optional[int] = None              # legacy alias
    age_bucket: Optional[str] = None
    group_id: Optional[str] = None
    group_size: Optional[int] = None
    # Queue-specific
    wait_seconds: Optional[int] = None
    abandoned: Optional[bool] = None
    queue_join_ts: Optional[datetime] = None
    queue_served_ts: Optional[datetime] = None
    queue_exit_ts: Optional[datetime] = None
    queue_position_at_join: Optional[int] = None
    # Spatial
    zone_hotspot_x: Optional[float] = None
    zone_hotspot_y: Optional[float] = None
    metadata: Optional[EventMetadata] = None

    @model_validator(mode="after")
    def normalise_fields(self) -> "StoreEvent":
        """Unify legacy field names to canonical ones."""
        if not self.visitor_id:
            self.visitor_id = self.id_token or (
                f"track_{self.track_id}" if self.track_id else None
            )
        if not self.timestamp:
            self.timestamp = (
                self.event_timestamp or self.event_time or datetime.utcnow()
            )
        if not self.gender:
            self.gender = self.gender_pred
        if not self.age and self.age_pred:
            self.age = self.age_pred
        return self


class IngestRequest(BaseModel):
    """Batch ingest payload — up to 500 events."""
    events: List[StoreEvent] = Field(..., min_length=1, max_length=500)
    source: Optional[str] = Field(default="pipeline", description="Ingestion source identifier")

    @field_validator("events")
    @classmethod
    def check_unique_ids(cls, events: List[StoreEvent]) -> List[StoreEvent]:
        ids = [e.event_id for e in events if e.event_id]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate event_ids in batch — must be unique")
        return events


class IngestResponse(BaseModel):
    accepted: int
    rejected: int
    duplicate_skipped: int
    errors: List[Dict[str, str]] = []
    ingest_ts: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────
# Metrics API Response
# ──────────────────────────────────────────────

class ZoneDwellMetric(BaseModel):
    zone_id: str
    zone_name: str
    zone_type: str
    avg_dwell_ms: float
    visits: int
    is_revenue_zone: bool


class StoreMetrics(BaseModel):
    store_id: str
    as_of: datetime
    window_minutes: int = 60
    unique_visitors: int
    staff_count: int
    net_visitors: int                  # unique_visitors - staff_count
    current_occupancy: int             # currently inside
    conversion_rate: float             # purchasers / net_visitors
    avg_zone_dwell_ms: float
    queue_depth: int                   # people currently in billing queue
    queue_abandonment_rate: float
    avg_wait_seconds: float
    peak_hour: Optional[str] = None
    zone_breakdown: List[ZoneDwellMetric] = []


# ──────────────────────────────────────────────
# Funnel API Response
# ──────────────────────────────────────────────

class FunnelStage(BaseModel):
    stage: str
    label: str
    count: int
    drop_off_pct: float = 0.0
    conversion_pct: float = 100.0


class StoreFunnel(BaseModel):
    store_id: str
    period_start: datetime
    period_end: datetime
    stages: List[FunnelStage]
    overall_conversion_rate: float


# ──────────────────────────────────────────────
# Heatmap API Response
# ──────────────────────────────────────────────

class HeatmapCell(BaseModel):
    zone_id: str
    zone_name: str
    zone_type: str
    visit_score: float = Field(ge=0, le=100)      # normalised 0-100
    dwell_score: float = Field(ge=0, le=100)       # normalised 0-100
    combined_score: float = Field(ge=0, le=100)
    raw_visits: int
    raw_dwell_ms: float
    hotspot_x: Optional[float] = None
    hotspot_y: Optional[float] = None


class StoreHeatmap(BaseModel):
    store_id: str
    as_of: datetime
    window_hours: int = 1
    cells: List[HeatmapCell]


# ──────────────────────────────────────────────
# Anomaly API Response
# ──────────────────────────────────────────────

class Anomaly(BaseModel):
    anomaly_id: str = Field(default_factory=lambda: str(uuid4()))
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    store_id: str
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    description: str
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    baseline_value: Optional[float] = None
    recommended_action: Optional[str] = None
    is_active: bool = True


class AnomalyResponse(BaseModel):
    store_id: str
    as_of: datetime
    active_anomalies: List[Anomaly]
    total_active: int


# ──────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────

class CameraStatus(BaseModel):
    camera_id: str
    last_event_ts: Optional[datetime] = None
    lag_seconds: Optional[float] = None
    is_stale: bool = False
    status: str = "OK"   # OK | STALE | OFFLINE


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    db_connected: bool = True
    event_count: int = 0
    store_count: int = 0
    cameras: List[CameraStatus] = []
    warnings: List[str] = []
    uptime_seconds: float = 0.0
    as_of: datetime = Field(default_factory=datetime.utcnow)
