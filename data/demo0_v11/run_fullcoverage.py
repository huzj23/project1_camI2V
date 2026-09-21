from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from core import write_json
from pipeline import (
    candidates_window,
    extract_window,
    load_config,
    metrics_all,
    metrics_window,
    prepare,
    render_window,
    run_root,
    validate,
    vote_window,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "fullcoverage_config.json"
STAGES = ("prepared", "features_extracted", "candidates_ready", "vote_ready", "rendered", "complete_d0_real")


def stage(directory: Path) -> str:
    path = directory / "window_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))["status"] if path.is_file() else "missing"


def update_progress(root: Path, config: dict, windows: list[dict], active: str | None) -> None:
    statuses = {window["window_id"]: stage(root / window["window_id"]) for window in windows}
    completed = sum(value == "complete_d0_real" for value in statuses.values())
    write_json(root / "coverage_progress.json", {
        "schema_version": "demo0-v1.1-fullcoverage-progress",
        "run_id": config["run_id"],
        "total_new_windows": len(windows),
        "completed_new_windows": completed,
        "active_window": active,
        "window_status": statuses,
        "updated_at_unix": time.time(),
    })
    print(f"[fullcoverage] {completed}/{len(windows)} complete; active={active}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume-safe full coverage D0-Real inference")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config, config_dir = load_config(args.config)
    windows = config["windows"]
    root = run_root(config, config_dir)
    if (root / ".frozen.json").exists():
        raise RuntimeError("Expanded run is unexpectedly frozen")
    if not (root / "run_manifest.json").is_file():
        validate(config, config_dir, windows)
        prepare(config, config_dir, windows, None)
    else:
        missing = [window["window_id"] for window in windows if stage(root / window["window_id"]) == "missing"]
        if missing:
            raise RuntimeError(f"Run manifest exists but some prepared windows are missing: {missing}")

    update_progress(root, config, windows, None)
    for window in windows:
        directory = root / window["window_id"]
        current = stage(directory)
        if current not in STAGES:
            raise RuntimeError(f"Unknown window stage {current}: {directory}")
        update_progress(root, config, windows, window["window_id"])
        if STAGES.index(current) < STAGES.index("features_extracted"):
            extract_window(config, config_dir, window)
            current = stage(directory)
        if STAGES.index(current) < STAGES.index("candidates_ready"):
            candidates_window(config, config_dir, window)
            current = stage(directory)
        if STAGES.index(current) < STAGES.index("vote_ready"):
            vote_window(config, config_dir, window)
            current = stage(directory)
        if STAGES.index(current) < STAGES.index("rendered"):
            render_window(config, config_dir, window)
            current = stage(directory)
        if STAGES.index(current) < STAGES.index("complete_d0_real"):
            metrics_window(config, config_dir, window)
        update_progress(root, config, windows, None)
    metrics_all(config, config_dir, windows)
    update_progress(root, config, windows, None)
    print(f"Full coverage run complete: {len(windows)} new windows", flush=True)


if __name__ == "__main__":
    main()
