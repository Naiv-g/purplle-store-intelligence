"""
Vercel entry point — imports and re-exports the FastAPI app.
Vercel looks for a WSGI/ASGI `app` object in api/index.py
"""
import sys
import os

# Add project root to path so src.* imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.main import app  # noqa: F401 — Vercel picks up `app`
