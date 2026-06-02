"""
Run the complete Store Intelligence System pipeline.
Usage:
  python run_pipeline.py --store ST1076 --video-dir "path/to/videos"
  python run_pipeline.py --demo   # runs with simulated data only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
from loguru import logger

API = "http://localhost:8000/api/v1"


async def run_demo_pipeline():
    """Run seeder and verify API is responding."""
    logger.info("🚀 Starting demo pipeline…")
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/seed_demo_data.py"],
        capture_output=False
    )
    return result.returncode == 0


async def run_video_pipeline(store_id: str, video_dir: str):
    """Process CCTV videos and push events to API."""
    from src.pipeline.video_processor import VideoProcessor, Zone, flush_events_to_api

    config_path = Path("config/store_layout.json")
    with open(config_path) as f:
        layout = json.load(f)

    store_cfg = next((s for s in layout["stores"] if s["store_id"] == store_id), None)
    if not store_cfg:
        logger.error(f"Store {store_id} not found in layout config")
        return

    zones = [
        Zone(
            zone_id=z["zone_id"], zone_name=z["zone_name"],
            zone_type=z["zone_type"], is_revenue_zone=z["is_revenue_zone"],
            bbox=tuple(z["bbox"])
        )
        for z in store_cfg["zones"]
    ]

    video_path = Path(video_dir)
    tasks = []

    for cam in store_cfg["cameras"]:
        # Try to match video files to cameras
        candidates = list(video_path.glob(f"*{cam['type']}*")) + list(video_path.glob("*.mp4"))
        if not candidates:
            logger.warning(f"No video found for {cam['camera_id']}, skipping")
            continue

        vp = VideoProcessor(
            store_id=store_id,
            camera_id=cam["camera_id"],
            video_path=str(candidates[0]),
            zones=zones,
            api_endpoint=f"{API}/events/ingest",
        )

        async def process_cam(processor=vp):
            logger.info(f"▶ Processing camera {processor.camera_id}")
            await asyncio.get_event_loop().run_in_executor(None, processor.run)
            events = processor.get_and_clear_buffer()
            async with httpx.AsyncClient() as client:
                if events:
                    await flush_events_to_api(events, f"{API}/events/ingest")
            logger.info(f"✅ Camera {processor.camera_id} done")

        tasks.append(process_cam())

    if tasks:
        await asyncio.gather(*tasks)
    else:
        logger.warning("No camera tasks created — check video directory")


def main():
    parser = argparse.ArgumentParser(description="Purplle Store Intelligence Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run with demo/simulated data")
    parser.add_argument("--store", default="ST1076", help="Store ID to process")
    parser.add_argument("--video-dir", help="Directory containing CCTV video files")
    args = parser.parse_args()

    if args.demo or not args.video_dir:
        asyncio.run(run_demo_pipeline())
    else:
        asyncio.run(run_video_pipeline(args.store, args.video_dir))


if __name__ == "__main__":
    main()
