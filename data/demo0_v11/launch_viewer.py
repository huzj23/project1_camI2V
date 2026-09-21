from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


HERE = Path(__file__).resolve().parent
FROZEN_RUN = "demo0_v11_d0real"


def healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/api/runs", timeout=2) as response:
            runs = json.load(response)
        return any(run.get("run_id") == FROZEN_RUN for run in runs)
    except (OSError, ValueError, urllib.error.URLError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the local Demo0 viewer and open it")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--no-browser", action="store_true", help="Health-check only; useful for automated tests")
    args = parser.parse_args()
    base_url = f"http://127.0.0.1:{args.port}"
    if not healthy(base_url):
        runtime = HERE / "runtime"
        runtime.mkdir(exist_ok=True)
        log_path = runtime / f"viewer_{args.port}.log"
        with log_path.open("ab") as log:
            kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
            subprocess.Popen(
                [sys.executable, str(HERE / "serve_viewer.py"), "--host", "127.0.0.1", "--port", str(args.port)],
                cwd=HERE,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                **kwargs,
            )
        for _ in range(40):
            if healthy(base_url):
                break
            time.sleep(0.25)
        else:
            raise RuntimeError(f"Viewer did not become healthy. Check {log_path}; port {args.port} may be occupied.")
    print(f"Demo0 viewer ready: {base_url}")
    if not args.no_browser:
        webbrowser.open(base_url)


if __name__ == "__main__":
    main()
