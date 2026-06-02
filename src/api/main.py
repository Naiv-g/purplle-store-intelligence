"""
FastAPI application entry-point for the Purplle Store Intelligence System.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from .database import init_db
from .routes.intelligence import router as intel_router

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Initialising database…")
    try:
        await init_db()
        logger.info("✅ Database ready")
    except Exception as e:
        logger.error(f"❌ Database init failed: {e}")
        app.state.db_error = str(e)
    yield
    logger.info("👋 Shutting down")


app = FastAPI(
    title="Purplle Store Intelligence System",
    description=(
        "End-to-end AI-powered store analytics from CCTV footage. "
        "Provides real-time footfall, zone engagement, queue intelligence, "
        "conversion funnel, heatmap, and anomaly detection APIs."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{ms:.1f}"
    response.headers["X-API-Version"] = "1.0.0"
    return response


# ── Mount static dashboard ───────────────────────────────────────────────────
import os, pathlib
static_dir = pathlib.Path(__file__).parent.parent / "dashboard"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/dashboard", include_in_schema=False)
    async def serve_dashboard():
        return FileResponse(str(static_dir / "index.html"))

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(str(static_dir / "index.html"))


# Include routes at both /api/v1 (for local) and /v1 (for Vercel path stripping)
app.include_router(intel_router, prefix="/api/v1")
app.include_router(intel_router, prefix="/v1")


@app.get("/api/v1/uptime", tags=["System"])
async def uptime():
    return {"uptime_seconds": round(time.time() - START_TIME, 1)}
