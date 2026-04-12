"""
Shim for OpenEnv Validator
Required by the Hackathon validation script to pass physical file checks.
"""
import sys
from pathlib import Path
import uvicorn

# Ensure root directory is on the path so backend modules can be resolved
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.api.app import app

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
