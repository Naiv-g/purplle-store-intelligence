"""
Analytics engine — computes real-time metrics, funnels, heatmaps, and
anomaly detection from the event database.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..api.database import EventRecord
from ..api.models import (
    Anomaly, AnomalyResponse, AnomalySeverity, AnomalyType,
    FunnelStage, HeatmapCell, StoreFunnel, StoreHeatmap,
    StoreMetrics, ZoneDwellMetric,
)

# Thresholds
QUEUE_SPIKE_THRESHOLD = 8
QUEUE_CRITICAL_THRESHOLD = 15
ABANDONMENT_WARNING_RATE = 0.30
ABANDONMENT_CRITICAL_RATE = 0.50
DEAD_ZONE_VISIT_FLOOR = 2
STALE_FEED_SECONDS = 120
OCCUPANCY_WARNING = 30


async def compute_store_metrics(
    session: AsyncSession, store_id: str, window_minutes: int = 60,
) -> StoreMetrics:
    since = datetime.utcnow() - timedelta(minutes=window_minutes)

    entry_rows = await session.execute(
        select(EventRecord).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type.in_(["entry", "ENTRY"]),
            EventRecord.timestamp >= since,
        )
    )
    entries = entry_rows.scalars().all()
    unique_visitors = len({e.visitor_id for e in entries if not e.is_staff})
    staff_count = len({e.visitor_id for e in entries if e.is_staff})
    net_visitors = max(0, unique_visitors)

    all_entries_ct = (await session.execute(
        select(func.count()).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type.in_(["entry", "ENTRY"]),
            EventRecord.is_staff == False,
        )
    )).scalar() or 0
    all_exits_ct = (await session.execute(
        select(func.count()).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type.in_(["exit", "EXIT"]),
        )
    )).scalar() or 0
    occupancy = max(0, all_entries_ct - all_exits_ct)

    zone_rows = await session.execute(
        select(EventRecord).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type.in_(["zone_exited", "ZONE_EXIT"]),
            EventRecord.timestamp >= since,
            EventRecord.dwell_ms.isnot(None),
        )
    )
    zone_events = zone_rows.scalars().all()
    zone_map: Dict[str, Dict] = {}
    for ev in zone_events:
        zid = ev.zone_id or "unknown"
        if zid not in zone_map:
            zone_map[zid] = {"zone_name": ev.zone_name or zid, "zone_type": ev.zone_type or "UNKNOWN", "total_dwell": 0, "visits": 0}
        zone_map[zid]["total_dwell"] += ev.dwell_ms or 0
        zone_map[zid]["visits"] += 1

    zone_breakdown = [
        ZoneDwellMetric(zone_id=zid, zone_name=v["zone_name"], zone_type=v["zone_type"],
                        avg_dwell_ms=v["total_dwell"] / max(1, v["visits"]),
                        visits=v["visits"], is_revenue_zone=True)
        for zid, v in zone_map.items()
    ]
    avg_zone_dwell = sum(z.avg_dwell_ms for z in zone_breakdown) / max(1, len(zone_breakdown))

    queue_rows = await session.execute(
        select(EventRecord).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type.in_(["queue_completed", "queue_abandoned"]),
            EventRecord.timestamp >= since,
        )
    )
    queue_events = queue_rows.scalars().all()
    total_q = len(queue_events)
    abandoned_q = sum(1 for q in queue_events if q.abandoned)
    abandonment_rate = abandoned_q / max(1, total_q)
    avg_wait = sum((q.wait_seconds or 0) for q in queue_events) / max(1, total_q)

    purchasers = (await session.execute(
        select(func.count(func.distinct(EventRecord.visitor_id))).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "queue_completed",
            EventRecord.abandoned == False,
            EventRecord.timestamp >= since,
        )
    )).scalar() or 0
    conversion_rate = purchasers / max(1, net_visitors)

    return StoreMetrics(
        store_id=store_id, as_of=datetime.utcnow(), window_minutes=window_minutes,
        unique_visitors=unique_visitors, staff_count=staff_count, net_visitors=net_visitors,
        current_occupancy=occupancy, conversion_rate=round(conversion_rate, 4),
        avg_zone_dwell_ms=round(avg_zone_dwell, 1), queue_depth=0,
        queue_abandonment_rate=round(abandonment_rate, 4),
        avg_wait_seconds=round(avg_wait, 1), zone_breakdown=zone_breakdown,
    )


async def compute_store_funnel(
    session: AsyncSession, store_id: str, period_hours: int = 24,
) -> StoreFunnel:
    since = datetime.utcnow() - timedelta(hours=period_hours)

    async def count(event_types: list) -> int:
        r = await session.execute(
            select(func.count(func.distinct(EventRecord.visitor_id))).where(
                EventRecord.store_id == store_id,
                EventRecord.timestamp >= since,
                EventRecord.event_type.in_(event_types),
                EventRecord.is_staff == False,
            )
        )
        return r.scalar() or 0

    foot_traffic = await count(["entry", "ENTRY"])
    engaged = await count(["zone_entered", "ZONE_ENTER", "zone_exited", "ZONE_EXIT"])
    queue_joins = await count(["queue_completed", "queue_abandoned"])
    converted = await count(["queue_completed"])

    stages, prev = [], foot_traffic or 1
    for stage, label, cnt in [
        ("foot_traffic", "Entered Store", foot_traffic),
        ("engaged", "Engaged with Zone", engaged),
        ("queue_intent", "Joined Billing Queue", queue_joins),
        ("converted", "Completed Purchase", converted),
    ]:
        drop = round((1 - cnt / max(1, prev)) * 100, 1)
        conv = round(cnt / max(1, foot_traffic) * 100, 1)
        stages.append(FunnelStage(stage=stage, label=label, count=cnt, drop_off_pct=drop, conversion_pct=conv))
        prev = cnt or 1

    return StoreFunnel(
        store_id=store_id, period_start=since, period_end=datetime.utcnow(),
        stages=stages, overall_conversion_rate=round(converted / max(1, foot_traffic) * 100, 2),
    )


async def compute_heatmap(
    session: AsyncSession, store_id: str, window_hours: int = 1,
) -> StoreHeatmap:
    since = datetime.utcnow() - timedelta(hours=window_hours)
    rows = await session.execute(
        select(EventRecord).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type.in_(["zone_exited", "ZONE_EXIT", "zone_entered", "ZONE_ENTER"]),
            EventRecord.timestamp >= since,
            EventRecord.zone_id.isnot(None),
        )
    )
    events = rows.scalars().all()
    zone_stats: Dict[str, Dict] = {}
    for ev in events:
        zid = ev.zone_id
        if zid not in zone_stats:
            zone_stats[zid] = {"zone_name": ev.zone_name or zid, "zone_type": ev.zone_type or "UNKNOWN",
                               "visits": 0, "total_dwell": 0.0,
                               "hotspot_x": ev.zone_hotspot_x, "hotspot_y": ev.zone_hotspot_y}
        zone_stats[zid]["visits"] += 1
        zone_stats[zid]["total_dwell"] += ev.dwell_ms or 0

    if not zone_stats:
        return StoreHeatmap(store_id=store_id, as_of=datetime.utcnow(), window_hours=window_hours, cells=[])

    max_v = max(z["visits"] for z in zone_stats.values()) or 1
    max_d = max(z["total_dwell"] for z in zone_stats.values()) or 1
    cells = []
    for zid, z in zone_stats.items():
        vs = round(z["visits"] / max_v * 100, 1)
        ds = round(z["total_dwell"] / max_d * 100, 1)
        cells.append(HeatmapCell(zone_id=zid, zone_name=z["zone_name"], zone_type=z["zone_type"],
                                 visit_score=vs, dwell_score=ds, combined_score=round((vs + ds) / 2, 1),
                                 raw_visits=z["visits"], raw_dwell_ms=z["total_dwell"],
                                 hotspot_x=z["hotspot_x"], hotspot_y=z["hotspot_y"]))
    return StoreHeatmap(store_id=store_id, as_of=datetime.utcnow(), window_hours=window_hours,
                        cells=sorted(cells, key=lambda c: c.combined_score, reverse=True))


async def detect_anomalies(session: AsyncSession, store_id: str) -> AnomalyResponse:
    anomalies: List[Anomaly] = []
    now = datetime.utcnow()

    # Queue spike
    q_depth = (await session.execute(
        select(func.count()).where(
            EventRecord.store_id == store_id,
            EventRecord.zone_type == "BILLING",
            EventRecord.event_type.in_(["zone_entered", "ZONE_ENTER"]),
            EventRecord.timestamp >= now - timedelta(minutes=15),
        )
    )).scalar() or 0

    if q_depth >= QUEUE_CRITICAL_THRESHOLD:
        anomalies.append(Anomaly(anomaly_type=AnomalyType.QUEUE_SPIKE, severity=AnomalySeverity.CRITICAL,
            store_id=store_id, zone_name="Billing Counter",
            description=f"Queue depth {q_depth} exceeds CRITICAL threshold",
            current_value=float(q_depth), threshold_value=float(QUEUE_CRITICAL_THRESHOLD),
            recommended_action="Open additional billing counters immediately"))
    elif q_depth >= QUEUE_SPIKE_THRESHOLD:
        anomalies.append(Anomaly(anomaly_type=AnomalyType.QUEUE_SPIKE, severity=AnomalySeverity.HIGH,
            store_id=store_id, zone_name="Billing Counter",
            description=f"Queue depth {q_depth} exceeds warning threshold",
            current_value=float(q_depth), threshold_value=float(QUEUE_SPIKE_THRESHOLD),
            recommended_action="Open a secondary billing counter"))

    # Queue abandonment
    q_ev = (await session.execute(
        select(EventRecord).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type.in_(["queue_completed", "queue_abandoned"]),
            EventRecord.timestamp >= now - timedelta(hours=1),
        )
    )).scalars().all()
    if len(q_ev) >= 5:
        aband_rate = sum(1 for e in q_ev if e.abandoned) / len(q_ev)
        if aband_rate >= ABANDONMENT_CRITICAL_RATE:
            anomalies.append(Anomaly(anomaly_type=AnomalyType.HIGH_ABANDONMENT,
                severity=AnomalySeverity.CRITICAL, store_id=store_id,
                description=f"Queue abandonment {aband_rate:.0%} is critically high",
                current_value=aband_rate, threshold_value=ABANDONMENT_CRITICAL_RATE,
                recommended_action="Immediate intervention required"))
        elif aband_rate >= ABANDONMENT_WARNING_RATE:
            anomalies.append(Anomaly(anomaly_type=AnomalyType.HIGH_ABANDONMENT,
                severity=AnomalySeverity.MEDIUM, store_id=store_id,
                description=f"Queue abandonment {aband_rate:.0%} above warning",
                current_value=aband_rate, threshold_value=ABANDONMENT_WARNING_RATE,
                recommended_action="Monitor queue and prepare backup counter"))

    # Stale feed detection
    last_rows = await session.execute(
        select(EventRecord.camera_id, func.max(EventRecord.timestamp).label("last_ts"))
        .where(EventRecord.store_id == store_id)
        .group_by(EventRecord.camera_id)
    )
    for row in last_rows:
        if row.last_ts:
            lag = (now - row.last_ts).total_seconds()
            if lag > STALE_FEED_SECONDS:
                anomalies.append(Anomaly(anomaly_type=AnomalyType.STALE_FEED,
                    severity=AnomalySeverity.HIGH if lag > 300 else AnomalySeverity.MEDIUM,
                    store_id=store_id,
                    description=f"Camera {row.camera_id} silent for {lag:.0f}s",
                    current_value=lag, threshold_value=float(STALE_FEED_SECONDS),
                    recommended_action=f"Check camera {row.camera_id} connectivity"))

    return AnomalyResponse(store_id=store_id, as_of=now,
                           active_anomalies=anomalies, total_active=len(anomalies))
