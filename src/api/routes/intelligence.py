"""
FastAPI routes for the Store Intelligence System.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import EventRecord, get_session
from ..models import IngestRequest, IngestResponse, StoreMetrics, StoreFunnel, StoreHeatmap, AnomalyResponse, HealthResponse, CameraStatus
from ...analytics.engine import compute_store_metrics, compute_store_funnel, compute_heatmap, detect_anomalies

router = APIRouter()


# ─── Event Ingest ────────────────────────────────────────────────────────────

@router.post("/events/ingest", response_model=IngestResponse, tags=["Events"])
async def ingest_events(
    payload: IngestRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Ingest a batch of store events (entry, exit, zone, queue).
    Idempotent — duplicates are silently skipped.
    """
    accepted, rejected, duplicates = 0, 0, 0
    errors = []

    for ev in payload.events:
        try:
            # Idempotency check
            existing = await session.execute(
                select(EventRecord).where(EventRecord.event_id == ev.event_id)
            )
            if existing.scalar_one_or_none():
                duplicates += 1
                continue

            record = EventRecord(
                event_id=ev.event_id,
                event_type=ev.event_type,
                store_id=ev.store_id,
                camera_id=ev.camera_id,
                visitor_id=ev.visitor_id,
                track_id=ev.track_id,
                timestamp=ev.timestamp or datetime.utcnow(),
                zone_id=ev.zone_id,
                zone_name=ev.zone_name,
                zone_type=ev.zone_type,
                dwell_ms=ev.dwell_ms,
                is_staff=ev.is_staff,
                is_face_hidden=ev.is_face_hidden,
                confidence=ev.confidence,
                gender=ev.gender,
                age=ev.age,
                age_bucket=ev.age_bucket,
                group_id=ev.group_id,
                group_size=ev.group_size,
                wait_seconds=ev.wait_seconds,
                abandoned=ev.abandoned,
                queue_position_at_join=ev.queue_position_at_join,
                queue_join_ts=ev.queue_join_ts,
                queue_served_ts=ev.queue_served_ts,
                queue_exit_ts=ev.queue_exit_ts,
                zone_hotspot_x=ev.zone_hotspot_x,
                zone_hotspot_y=ev.zone_hotspot_y,
            )
            session.add(record)
            accepted += 1
        except Exception as e:
            rejected += 1
            errors.append({"event_id": ev.event_id, "error": str(e)})

    await session.commit()
    return IngestResponse(
        accepted=accepted, rejected=rejected,
        duplicate_skipped=duplicates, errors=errors,
    )


# ─── Metrics ─────────────────────────────────────────────────────────────────

@router.get("/stores/{store_id}/metrics", response_model=StoreMetrics, tags=["Intelligence"])
async def get_store_metrics(
    store_id: str,
    window_minutes: int = Query(default=60, ge=1, le=1440),
    session: AsyncSession = Depends(get_session),
):
    """Real-time store metrics for the last N minutes."""
    return await compute_store_metrics(session, store_id, window_minutes)


@router.get("/stores/{store_id}/funnel", response_model=StoreFunnel, tags=["Intelligence"])
async def get_store_funnel(
    store_id: str,
    period_hours: int = Query(default=24, ge=1, le=168),
    session: AsyncSession = Depends(get_session),
):
    """Customer journey funnel: entry → zone engagement → billing → conversion."""
    return await compute_store_funnel(session, store_id, period_hours)


@router.get("/stores/{store_id}/heatmap", response_model=StoreHeatmap, tags=["Intelligence"])
async def get_store_heatmap(
    store_id: str,
    window_hours: int = Query(default=1, ge=1, le=24),
    session: AsyncSession = Depends(get_session),
):
    """Zone engagement heatmap ranked by visit + dwell scores."""
    return await compute_heatmap(session, store_id, window_hours)


@router.get("/stores/{store_id}/anomalies", response_model=AnomalyResponse, tags=["Intelligence"])
async def get_store_anomalies(
    store_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Active anomalies: queue spikes, dead zones, stale feeds, high abandonment."""
    return await detect_anomalies(session, store_id)


# ─── Visitor Timeline ────────────────────────────────────────────────────────

@router.get("/stores/{store_id}/visitor/{visitor_id}/journey", tags=["Visitors"])
async def get_visitor_journey(
    store_id: str,
    visitor_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Full event timeline for a single anonymised visitor token."""
    rows = await session.execute(
        select(EventRecord).where(
            EventRecord.store_id == store_id,
            EventRecord.visitor_id == visitor_id,
        ).order_by(EventRecord.timestamp)
    )
    events = rows.scalars().all()
    if not events:
        raise HTTPException(status_code=404, detail="Visitor not found")
    return {
        "visitor_id": visitor_id,
        "store_id": store_id,
        "event_count": len(events),
        "journey": [
            {
                "event_type": e.event_type,
                "timestamp": e.timestamp,
                "camera_id": e.camera_id,
                "zone_id": e.zone_id,
                "zone_name": e.zone_name,
                "dwell_ms": e.dwell_ms,
            }
            for e in events
        ],
    }


# ─── Health Check ────────────────────────────────────────────────────────────

@router.get("/health", tags=["System"])
async def health_check(request: Request, session: AsyncSession = Depends(get_session)):
    """Pipeline health: DB connectivity, event count, stale camera detection."""
    db_error = getattr(request.app.state, "db_error", None)
    if db_error:
        return JSONResponse(status_code=500, content={"status": "error", "db_error": db_error})
    
    from datetime import timedelta
    now = datetime.utcnow()

    event_count = (await session.execute(select(func.count()).select_from(EventRecord))).scalar() or 0
    store_count = (await session.execute(select(func.count(func.distinct(EventRecord.store_id))).select_from(EventRecord))).scalar() or 0

    cam_rows = await session.execute(
        select(EventRecord.store_id, EventRecord.camera_id, func.max(EventRecord.timestamp).label("last_ts"))
        .group_by(EventRecord.store_id, EventRecord.camera_id)
    )
    cameras = []
    warnings = []
    for row in cam_rows:
        lag = (now - row.last_ts).total_seconds() if row.last_ts else None
        is_stale = lag is not None and lag > 120
        status = "STALE" if is_stale else "OK"
        if is_stale:
            warnings.append(f"{row.store_id}/{row.camera_id} stale ({lag:.0f}s)")
        cameras.append(CameraStatus(camera_id=f"{row.store_id}/{row.camera_id}",
                                    last_event_ts=row.last_ts, lag_seconds=lag,
                                    is_stale=is_stale, status=status))

    return HealthResponse(
        db_connected=True, event_count=event_count,
        store_count=store_count, cameras=cameras, warnings=warnings,
        as_of=now,
    )
