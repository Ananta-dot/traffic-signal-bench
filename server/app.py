"""OpenEnv server entry point. Imports the FastAPI app from the main server module."""

import sys
import os
import uvicorn

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: F401, E402

def main():
    """Run the Uvicorn server for multi-mode deployment."""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()