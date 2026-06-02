"""
Data Simulator — seeds the database with realistic store events
based on the provided sample_events.jsonl and POS transactions.

Run: python scripts/seed_demo_data.py

This script:
1. Loads sample events from the provided JSONL files
2. Generates additional synthetic traffic for a full-day simulation
3. POSTs everything to the ingest API
"""
from __future__ import annotations

import asyncio
import json
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import httpx
from loguru import logger

BASE_URL = "https://purplle-store-intelligence.vercel.app/api/v1"
DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG_DIR = Path(__file__).parent.parent / "config"

STORES = ["ST1076", "ST1008"]

ZONE_CONFIGS = {
    "ST1076": [
        {"zone_id": "PURPLLE_MUM_1076_Z01", "zone_name": "Left Shelf", "zone_type": "SHELF"},
        {"zone_id": "PURPLLE_MUM_1076_Z02", "zone_name": "Center Display", "zone_type": "DISPLAY"},
        {"zone_id": "PURPLLE_MUM_1076_Z03", "zone_name": "Lipstick Aisle", "zone_type": "SHELF"},
        {"zone_id": "PURPLLE_MUM_1076_Z_BILLING_01", "zone_name": "Billing Counter", "zone_type": "BILLING"},
    ],
    "ST1008": [
        {"zone_id": "ST1008_Z01", "zone_name": "Skincare Section", "zone_type": "SHELF"},
        {"zone_id": "ST1008_Z02", "zone_name": "Makeup Aisle", "zone_type": "SHELF"},
        {"zone_id": "ST1008_Z_BILLING_01", "zone_name": "Billing Counter", "zone_type": "BILLING"},
    ],
}

CAMERAS = {
    "ST1076": ["CAM1", "CAM2", "CAM3", "CAM4", "CAM5"],
    "ST1008": ["CAM1", "CAM2", "CAM3", "CAM4"],
}

AGE_BUCKETS = ["18-24", "25-34", "35-44", "45-54"]
GENDERS = ["M", "F"]


def random_visitor_id(store_id: str, track_id: int) -> str:
    return f"TRK_{store_id}_{track_id:06d}"


def random_age():
    age = int(random.gauss(30, 10))
    age = max(16, min(65, age))
    if age < 18: bucket = "Under-18"
    elif age < 25: bucket = "18-24"
    elif age < 35: bucket = "25-34"
    elif age < 45: bucket = "35-44"
    elif age < 55: bucket = "45-54"
    else: bucket = "55+"
    return age, bucket


def generate_visitor_events(store_id: str, track_id: int, base_time: datetime) -> List[dict]:
    """Generate a complete visitor journey: entry → zones → optional billing → exit."""
    visitor_id = random_visitor_id(store_id, track_id)
    gender = random.choice(GENDERS)
    age, age_bucket = random_age()
    is_staff = random.random() < 0.05   # 5% staff
    camera_entry = CAMERAS[store_id][0]
    events = []

    def make_event(event_type: str, ts: datetime, **extra) -> dict:
        return {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "store_id": store_id,
            "camera_id": camera_entry,
            "visitor_id": visitor_id,
            "track_id": track_id,
            "timestamp": ts.isoformat(),
            "is_staff": is_staff,
            "gender": gender,
            "age": age,
            "age_bucket": age_bucket,
            "confidence": round(random.uniform(0.82, 0.99), 3),
            **extra,
        }

    # Entry
    t = base_time
    events.append(make_event("entry", t, camera_id=camera_entry))
    t += timedelta(seconds=random.randint(10, 60))

    if is_staff:
        t += timedelta(minutes=random.randint(60, 180))
        events.append(make_event("exit", t, camera_id=camera_entry))
        return events

    # Browse 1-3 zones
    zones = ZONE_CONFIGS[store_id]
    non_billing = [z for z in zones if z["zone_type"] != "BILLING"]
    visit_zones = random.sample(non_billing, k=random.randint(1, min(3, len(non_billing))))

    for zone in visit_zones:
        cam = random.choice(CAMERAS[store_id][1:])
        events.append(make_event("zone_entered", t, camera_id=cam,
                                 zone_id=zone["zone_id"], zone_name=zone["zone_name"],
                                 zone_type=zone["zone_type"],
                                 zone_hotspot_x=round(random.uniform(200, 600), 1),
                                 zone_hotspot_y=round(random.uniform(150, 400), 1)))
        dwell = random.randint(15, 120)
        t += timedelta(seconds=dwell)
        events.append(make_event("zone_exited", t, camera_id=cam,
                                 zone_id=zone["zone_id"], zone_name=zone["zone_name"],
                                 zone_type=zone["zone_type"],
                                 dwell_ms=dwell * 1000))
        t += timedelta(seconds=random.randint(5, 30))

    # 60% conversion rate: go to billing
    if random.random() < 0.60:
        billing_zone = next(z for z in zones if z["zone_type"] == "BILLING")
        cam = CAMERAS[store_id][-1]
        queue_join = t
        wait = random.randint(5, 90)
        abandoned = random.random() < 0.15  # 15% abandon
        queue_exit = t + timedelta(seconds=wait + random.randint(30, 120))

        q_event = make_event("queue_completed" if not abandoned else "queue_abandoned",
                             queue_exit, camera_id=cam,
                             zone_id=billing_zone["zone_id"],
                             zone_name=billing_zone["zone_name"],
                             zone_type=billing_zone["zone_type"],
                             wait_seconds=wait,
                             abandoned=abandoned,
                             queue_join_ts=queue_join.isoformat(),
                             queue_exit_ts=queue_exit.isoformat(),
                             queue_position_at_join=random.randint(1, 5))
        if not abandoned:
            q_event["queue_served_ts"] = (queue_join + timedelta(seconds=wait)).isoformat()
        events.append(q_event)
        t = queue_exit + timedelta(seconds=random.randint(10, 30))

    # Exit
    events.append(make_event("exit", t, camera_id=camera_entry))
    return events


def generate_full_day(store_id: str, date: datetime, num_visitors: int = 120) -> List[dict]:
    """Simulate a full store-day with realistic traffic patterns."""
    open_h, close_h = 10, 22
    all_events = []
    track_counter = random.randint(10000, 20000)

    # Traffic distribution: morning slow, lunch peak, evening peak
    for v in range(num_visitors):
        track_counter += 1
        # Time distribution
        r = random.random()
        if r < 0.15:        hour = random.uniform(open_h, 12)
        elif r < 0.35:      hour = random.uniform(12, 14)
        elif r < 0.55:      hour = random.uniform(14, 17)
        elif r < 0.80:      hour = random.uniform(17, 20)
        else:               hour = random.uniform(20, close_h - 0.5)
        h = int(hour)
        m = int((hour - h) * 60)
        base_time = date.replace(hour=h, minute=m, second=random.randint(0, 59))
        events = generate_visitor_events(store_id, track_counter, base_time)
        all_events.extend(events)

    return all_events


async def load_sample_events() -> List[dict]:
    """Load events from the provided sample JSONL file."""
    sample_path = DATA_DIR / "sample_events.jsonl"
    events = []
    if not sample_path.exists():
        # Try to find it in parent
        for path in [
            Path(__file__).parent.parent.parent / "sample_eventsbe42122.jsonl",
        ]:
            if path.exists():
                sample_path = path
                break

    if sample_path.exists():
        with open(sample_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                # Normalise to canonical schema
                ev = {
                    "event_id": raw.get("queue_event_id") or str(uuid.uuid4()),
                    "event_type": raw.get("event_type", "entry"),
                    "store_id": raw.get("store_id") or raw.get("store_code", "ST1076"),
                    "camera_id": raw.get("camera_id", "CAM1"),
                    "visitor_id": raw.get("id_token") or (f"TRK_{raw.get('track_id', 0):06d}"),
                    "track_id": raw.get("track_id"),
                    "timestamp": raw.get("event_timestamp") or raw.get("event_time") or raw.get("queue_exit_ts"),
                    "is_staff": raw.get("is_staff", False),
                    "gender": raw.get("gender_pred") or raw.get("gender"),
                    "age": raw.get("age_pred") or raw.get("age"),
                    "age_bucket": raw.get("age_bucket"),
                    "confidence": 0.95,
                    "zone_id": raw.get("zone_id"),
                    "zone_name": raw.get("zone_name"),
                    "zone_type": raw.get("zone_type"),
                    "wait_seconds": raw.get("wait_seconds"),
                    "abandoned": raw.get("abandoned"),
                    "queue_join_ts": raw.get("queue_join_ts"),
                    "queue_served_ts": raw.get("queue_served_ts"),
                    "queue_exit_ts": raw.get("queue_exit_ts"),
                    "queue_position_at_join": raw.get("queue_position_at_join"),
                    "zone_hotspot_x": raw.get("zone_hotspot_x"),
                    "zone_hotspot_y": raw.get("zone_hotspot_y"),
                }
                events.append(ev)
        logger.info(f"📂 Loaded {len(events)} events from sample JSONL")
    return events


async def ingest_batch(client: httpx.AsyncClient, events: List[dict], label: str) -> dict:
    """Send a batch to the ingest endpoint."""
    BATCH_SIZE = 100
    total_accepted = 0
    for i in range(0, len(events), BATCH_SIZE):
        batch = events[i:i+BATCH_SIZE]
        try:
            resp = await client.post(f"{BASE_URL}/events/ingest",
                                     json={"events": batch, "source": "seed"},
                                     timeout=30)
            resp.raise_for_status()
            data = resp.json()
            total_accepted += data.get("accepted", 0)
        except Exception as e:
            logger.error(f"Batch failed: {e}")
    logger.info(f"✅ [{label}] Ingested {total_accepted}/{len(events)} events")
    return {"accepted": total_accepted, "total": len(events)}


async def main():
    logger.info("🌱 Starting demo data seed…")

    # Wait for API to be ready
    async with httpx.AsyncClient() as client:
        for _ in range(20):
            try:
                r = await client.get(f"{BASE_URL}/health", timeout=3)
                if r.status_code == 200:
                    logger.info("✅ API is ready")
                    break
            except Exception:
                pass
            await asyncio.sleep(1)
        else:
            logger.error("❌ API not responding after 20s. Make sure the server is running.")
            sys.exit(1)

        # 1. Load and ingest sample events from provided data
        sample_events = await load_sample_events()

        # Normalise store IDs
        for ev in sample_events:
            if "store_1076" in str(ev.get("store_id", "")):
                ev["store_id"] = "ST1076"

        # 2. Generate synthetic full-day data
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday = today - timedelta(days=1)

        all_synthetic = []
        for store in STORES:
            logger.info(f"🏪 Generating events for {store}…")
            all_synthetic.extend(generate_full_day(store, yesterday, num_visitors=80))
            all_synthetic.extend(generate_full_day(store, today, num_visitors=50))

        logger.info(f"📊 Generated {len(all_synthetic)} synthetic events + {len(sample_events)} sample events")

        # Combine and ingest
        all_events = sample_events + all_synthetic

        for store in STORES:
            store_events = [e for e in all_events if e.get("store_id") == store or e.get("store_id") == "ST1076" and store == "ST1076"]
            if store_events:
                await ingest_batch(client, store_events, store)

        logger.info("🎉 Seed complete! Dashboard should now show live data.")
        logger.info(f"   → Open: http://localhost:8000/dashboard")
        logger.info(f"   → API docs: http://localhost:8000/docs")


if __name__ == "__main__":
    asyncio.run(main())
