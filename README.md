# 🏪 Purplle Store Intelligence System

> **Purplle Tech Challenge 2026 — Round 2 Submission**
>
> End-to-end AI-powered retail analytics from raw CCTV footage — combining computer vision, real-time event streaming, anomaly detection, and a live intelligence dashboard.

---

## 🏗️ Architecture Overview

```
CCTV Cameras
     │
     ▼
┌──────────────────────────────┐
│   Video Processing Pipeline  │  YOLOv8 detection + ByteTrack tracking
│   src/pipeline/              │  Zone classification, entry/exit logic
└───────────────┬──────────────┘
                │ Canonical Events (JSON)
                ▼
┌──────────────────────────────┐
│   FastAPI Ingest Endpoint    │  POST /api/v1/events/ingest
│   Idempotent · Batch · Async │  Up to 500 events/batch
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│   SQLite / PostgreSQL        │  WAL mode, indexed by store+timestamp
│   Event Store                │
└───────────────┬──────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
┌─────────────┐  ┌──────────────────┐
│  Analytics  │  │  Anomaly Engine  │
│  Engine     │  │  Rule-based +    │
│  Metrics,   │  │  Threshold alerts│
│  Funnel,    │  └──────────────────┘
│  Heatmap    │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────┐
│  REST API  +  Live Dashboard │
│  /api/v1/* endpoints         │
│  Auto-refreshing UI (15s)    │
└──────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1 — Local (Python)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the API server
uvicorn src.api.main:app --reload --port 8000

# 3. Seed with demo data (new terminal)
python scripts/seed_demo_data.py

# 4. Open dashboard
# http://localhost:8000/dashboard

# 5. View API docs
# http://localhost:8000/docs
```

### Option 2 — Docker

```bash
docker-compose up --build
# Dashboard: http://localhost:8000/dashboard
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/events/ingest` | Batch ingest store events (idempotent) |
| `GET`  | `/api/v1/stores/{id}/metrics` | Real-time KPIs (`?window_minutes=60`) |
| `GET`  | `/api/v1/stores/{id}/funnel` | Conversion funnel (`?period_hours=24`) |
| `GET`  | `/api/v1/stores/{id}/heatmap` | Zone engagement heatmap |
| `GET`  | `/api/v1/stores/{id}/anomalies` | Active anomalies & alerts |
| `GET`  | `/api/v1/stores/{id}/visitor/{vid}/journey` | Individual visitor timeline |
| `GET`  | `/api/v1/health` | System health + camera status |

### Event Ingest Example

```bash
curl -X POST http://localhost:8000/api/v1/events/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "events": [{
      "event_type": "entry",
      "store_id": "ST1076",
      "camera_id": "CAM1",
      "visitor_id": "ID_60001",
      "timestamp": "2026-06-01T10:15:00",
      "gender": "F",
      "age": 28,
      "age_bucket": "25-34",
      "is_staff": false,
      "confidence": 0.94
    }],
    "source": "pipeline"
  }'
```

---

## 🧠 Event Schema

```jsonc
// Entry / Exit
{ "event_type": "entry"|"exit", "store_id", "camera_id",
  "visitor_id", "timestamp", "gender", "age", "age_bucket",
  "is_staff", "is_face_hidden", "group_id", "group_size", "confidence" }

// Zone Enter / Exit
{ "event_type": "zone_entered"|"zone_exited",
  "zone_id", "zone_name", "zone_type",
  "dwell_ms",  // populated on zone_exited
  "zone_hotspot_x", "zone_hotspot_y" }

// Queue Events
{ "event_type": "queue_completed"|"queue_abandoned",
  "zone_id", "zone_type": "BILLING",
  "wait_seconds", "abandoned": true|false,
  "queue_join_ts", "queue_served_ts", "queue_exit_ts",
  "queue_position_at_join" }
```

---

## 📹 Vision Pipeline

**Model**: YOLOv8n (ultralytics) — real-time person detection at 30+ FPS on GPU, 8+ FPS CPU-only.

**Tracker**: ByteTrack (via `boxmot`) — maintains persistent IDs across frames with IoU + appearance matching.

**Zone Classification**: Polygon/bbox intersection per zone per frame. Dwell time accumulated per track.

**Staff Detection**: Flagged via uniform colour classifier (HSV range matching) or optional staff badge detection.

**Demo Mode**: When video files are unavailable, the pipeline generates realistic synthetic events via `scripts/seed_demo_data.py`, modelling real traffic patterns (morning slow, lunch + evening peaks).

---

## 🔍 Anomaly Detection

| Anomaly | Trigger | Severity |
|---------|---------|----------|
| `QUEUE_SPIKE` | >8 in queue | HIGH |
| `QUEUE_SPIKE` | >15 in queue | CRITICAL |
| `HIGH_ABANDONMENT` | >30% abandon rate | MEDIUM |
| `HIGH_ABANDONMENT` | >50% abandon rate | CRITICAL |
| `DEAD_ZONE` | <2 visits/hr in revenue zone | LOW |
| `STALE_FEED` | No camera events for >2 min | MEDIUM/HIGH |
| `OVERCROWDING` | Occupancy >30 | MEDIUM |

---

## 🏗️ Design Decisions

### Why SQLite + async instead of Kafka + Flink?
For a hackathon submission with unknown infra, SQLite with WAL mode handles thousands of events/sec with zero ops overhead. The architecture is designed to swap to PostgreSQL via a single env var. A real production deployment would add Kafka for stream ingestion and Flink/Spark for aggregation.

### Why rule-based anomaly detection?
Statistical baselines require weeks of historical data. Rule-based thresholds with configurable values provide immediate, interpretable alerts without cold-start problems. They can be layered with ML anomaly models (Isolation Forest, LSTM) once sufficient data accumulates.

### Why YOLOv8n (nano)?
Edge-deployable. Runs at 25+ FPS on a Raspberry Pi 4, making it viable for in-store deployment on cheap NVR hardware without cloud GPU costs. Accuracy trade-off is acceptable for person detection at typical CCTV resolutions.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📁 Project Structure

```
store-intelligence-system/
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI app + CORS + static serving
│   │   ├── models.py        # Pydantic schemas (all event types)
│   │   ├── database.py      # Async SQLAlchemy ORM
│   │   └── routes/
│   │       └── intelligence.py  # All API endpoints
│   ├── analytics/
│   │   └── engine.py        # Metrics, funnel, heatmap, anomaly
│   ├── pipeline/
│   │   └── video_processor.py  # YOLO + ByteTrack pipeline
│   └── dashboard/
│       └── index.html       # Live auto-refreshing dashboard
├── scripts/
│   └── seed_demo_data.py    # Demo data generator
├── config/
│   └── store_layout.json    # Zone/camera configuration
├── data/
│   ├── sample_events.jsonl  # Provided sample events
│   └── pos_transactions.csv # Provided POS data
├── tests/
│   └── test_api.py          # Pytest API tests
├── run_pipeline.py          # CLI runner
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

*Built with FastAPI · SQLAlchemy · YOLOv8 · Chart.js · Docker*
