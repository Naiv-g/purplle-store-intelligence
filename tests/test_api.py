"""
API endpoint tests using pytest + httpx AsyncClient.
Run: pytest tests/ -v
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from src.api.main import app
from src.api.database import init_db


@pytest_asyncio.fixture(scope="module")
async def client():
    await init_db()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert "event_count" in data


@pytest.mark.asyncio
async def test_ingest_single_event(client):
    event = {
        "event_type": "entry",
        "store_id": "ST_TEST",
        "camera_id": "CAM1",
        "visitor_id": "TEST_VISITOR_001",
        "timestamp": "2026-06-01T10:00:00",
        "is_staff": False,
        "gender": "F",
        "age": 28,
        "age_bucket": "25-34",
        "confidence": 0.95,
    }
    r = await client.post("/api/v1/events/ingest", json={"events": [event]})
    assert r.status_code == 200
    data = r.json()
    assert data["accepted"] == 1
    assert data["rejected"] == 0


@pytest.mark.asyncio
async def test_ingest_idempotency(client):
    """Same event_id should be counted as duplicate."""
    event = {
        "event_id": "DEDUP_TEST_001",
        "event_type": "entry",
        "store_id": "ST_TEST",
        "camera_id": "CAM1",
        "visitor_id": "TEST_VISITOR_DEDUP",
        "timestamp": "2026-06-01T10:01:00",
    }
    r1 = await client.post("/api/v1/events/ingest", json={"events": [event]})
    r2 = await client.post("/api/v1/events/ingest", json={"events": [event]})
    assert r1.json()["accepted"] == 1
    assert r2.json()["duplicate_skipped"] == 1


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    r = await client.get("/api/v1/stores/ST_TEST/metrics?window_minutes=60")
    assert r.status_code == 200
    data = r.json()
    assert "unique_visitors" in data
    assert "conversion_rate" in data
    assert "current_occupancy" in data


@pytest.mark.asyncio
async def test_funnel_endpoint(client):
    r = await client.get("/api/v1/stores/ST_TEST/funnel?period_hours=24")
    assert r.status_code == 200
    data = r.json()
    assert "stages" in data
    assert len(data["stages"]) == 4
    assert "overall_conversion_rate" in data


@pytest.mark.asyncio
async def test_heatmap_endpoint(client):
    r = await client.get("/api/v1/stores/ST_TEST/heatmap?window_hours=1")
    assert r.status_code == 200
    data = r.json()
    assert "cells" in data


@pytest.mark.asyncio
async def test_anomalies_endpoint(client):
    r = await client.get("/api/v1/stores/ST_TEST/anomalies")
    assert r.status_code == 200
    data = r.json()
    assert "active_anomalies" in data
    assert "total_active" in data


@pytest.mark.asyncio
async def test_visitor_journey_not_found(client):
    r = await client.get("/api/v1/stores/ST_TEST/visitor/NONEXISTENT/journey")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_ingest_batch(client):
    events = [
        {
            "event_type": "zone_entered",
            "store_id": "ST_TEST",
            "camera_id": "CAM2",
            "visitor_id": f"BATCH_V_{i}",
            "timestamp": f"2026-06-01T11:{i:02d}:00",
            "zone_id": "ZONE_TEST_01",
            "zone_name": "Test Zone",
            "zone_type": "SHELF",
        }
        for i in range(10)
    ]
    r = await client.post("/api/v1/events/ingest", json={"events": events})
    assert r.status_code == 200
    data = r.json()
    assert data["accepted"] == 10
