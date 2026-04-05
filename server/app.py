"""OpenEnv server entry point. Imports the FastAPI app from the main server module."""

# Re-export the app so OpenEnv can find it at server/app.py
import sys
import os

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app  # noqa: F401, E402