# server/app.py
import sys
import os

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Change this from "from server import app" to:
from main import app  # noqa: F401, E402