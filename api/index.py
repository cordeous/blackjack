"""
Vercel serverless entry point.
Re-exports the FastAPI app from the root api.py.
"""
import sys
from pathlib import Path

# Add project root to path so engine/agents packages resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import app  # noqa: F401 — Vercel picks up `app` from this module
