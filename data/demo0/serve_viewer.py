from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

import uvicorn


if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8765)
