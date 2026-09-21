from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from core import inspect_video, write_json


HERE = Path(__file__).resolve().parent
DATA = HERE.parent
BASE = HERE / "config.json"
OUTPUT = HERE / "fullcoverage_config.json"
PLAN = HERE / "fullcoverage_windows.json"
RUN_ID = "demo0_v11_fullcoverage_d0real"

# Inclusive end points deliberately share one boundary frame.  No short
# window crosses either reversal in the asymmetric camera path (near 410 and
# 800); each interval is at most 121 encoded frames / ~2.02 s.
SEGMENTS: tuple[tuple[str, str, int, int, str, str], ...] = (
    ("simple", "startup", 0, 60, "startup_control", "Near-stationary startup; slight yaw variation, diagnostic only"),
    ("simple", "outbound", 60, 180, "startup_control", "Low-motion startup into translation"),
    ("simple", "outbound", 180, 300, "frozen", "Frozen multidepth conference window"),
    ("simple", "outbound", 300, 360, "translation", "Translation before foreground-tree crossing"),
    ("simple", "outbound", 360, 480, "frozen", "Frozen occlusion conference window"),
    ("simple", "outbound", 480, 600, "translation", "Translation after foreground-tree crossing"),
    ("simple", "outbound", 600, 719, "slowdown_control", "Translation slows near the end"),
    ("asymmetric", "startup", 0, 60, "startup_control", "Near-stationary startup; slight yaw variation, diagnostic only"),
    ("asymmetric", "outbound", 60, 120, "frozen", "Frozen asymmetric conference window"),
    ("asymmetric", "outbound", 120, 240, "translation", "Outbound translation and perspective scaling"),
    ("asymmetric", "outbound", 240, 360, "translation", "Outbound translation and perspective scaling"),
    ("asymmetric", "outbound", 360, 410, "turn_adjacent_control", "Short window ends at the first reversal"),
    ("asymmetric", "return", 410, 530, "translation", "Return path, no crossing of reversal"),
    ("asymmetric", "return", 530, 650, "translation", "Return path, no crossing of reversal"),
    ("asymmetric", "return", 650, 770, "translation", "Return path, no crossing of reversal"),
    ("asymmetric", "return", 770, 800, "turn_adjacent_control", "Short window ends at the second reversal"),
    ("asymmetric", "final_outbound", 800, 881, "translation", "Final short outbound segment after reversal"),
)

SOURCE: dict[str, dict[str, str]] = {
    "simple": {
        "video": "简单平动透视关系.mp4",
        "camera_csv": "简单平动透视关系.camera.csv",
        "metadata": "简单平动透视关系.camera.metadata.json",
        "display": "简单平动",
    },
    "asymmetric": {
        "video": "非对称建筑平动.mp4",
        "camera_csv": "非对称建筑平动.camera.csv",
        "metadata": "非对称建筑平动.camera.metadata.json",
        "display": "非对称建筑",
    },
}


def camera_summary(source: dict[str, str], start: int, end: int) -> dict[str, Any]:
    with (DATA / source["camera_csv"]).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    positions = np.asarray([[float(row[key]) for key in ("x", "y", "z")] for row in rows[start : end + 1]])
    yaw = np.asarray([float(row["yaw_deg"]) for row in rows[start : end + 1]])
    pitch = np.asarray([float(row["pitch_deg"]) for row in rows[start : end + 1]])
    return {
        "translation_norm": float(np.linalg.norm(positions[-1] - positions[0])),
        "yaw_range_deg": float(yaw.max() - yaw.min()),
        "pitch_range_deg": float(pitch.max() - pitch.min()),
    }


def main() -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    existing = {window["source_start_frame"]: window for window in base["windows"]}
    windows = []
    plan = []
    for name, path, start, end, category, note in SEGMENTS:
        source = SOURCE[name]
        video = inspect_video(DATA / source["video"])
        if not 0 <= start < end < video.frame_count or end - start > 120:
            raise ValueError(f"Invalid short window: {name} {start}..{end}")
        frozen = category == "frozen"
        window_id = (
            existing[start]["window_id"]
            if frozen and start in existing
            else f"{name}_{path}_{start:04d}_{end:04d}"
        )
        item = {
            "window_id": window_id,
            "display_name": f"{source['display']} · {path} · {start}–{end}" + (" · 已冻结" if frozen else ""),
            "category": category,
            "video": source["video"],
            "camera_csv": source["camera_csv"],
            "metadata": source["metadata"],
            "source_start_frame": start,
            "source_end_frame": end,
        }
        summary = camera_summary(source, start, end)
        plan.append({
            **item,
            "run_id": base["run_id"] if frozen else RUN_ID,
            "frozen": frozen,
            "frame_count": end - start + 1,
            "duration_seconds": (end - start + 1) / video.fps,
            "camera_summary_selection_only": summary,
            "note": note,
            "scientific_metric_role": "diagnostic_control" if "control" in category else "normal_D0_Real",
        })
        if not frozen:
            windows.append(item)

    for name, source in SOURCE.items():
        spans = [(item["source_start_frame"], item["source_end_frame"]) for item in plan if item["video"] == source["video"]]
        if spans[0][0] != 0 or spans[-1][1] != inspect_video(DATA / source["video"]).frame_count - 1:
            raise RuntimeError(f"Coverage endpoints incomplete: {name}")
        if any(left[1] != right[0] for left, right in zip(spans, spans[1:], strict=False)):
            raise RuntimeError(f"Coverage gaps: {name}")

    base["run_id"] = RUN_ID
    base["windows"] = windows
    base["coverage_manifest"] = PLAN.name
    write_json(OUTPUT, base)
    write_json(PLAN, {
        "schema_version": "demo0-v1.1-fullcoverage",
        "source_videos": {name: {**source, "frame_count": inspect_video(DATA / source["video"]).frame_count} for name, source in SOURCE.items()},
        "segment_count": len(plan),
        "frozen_segment_count": sum(int(item["frozen"]) for item in plan),
        "new_segment_count": len(windows),
        "overlap_policy": "Adjacent short windows share exactly their boundary frame",
        "windows": plan,
    })
    print(f"Wrote {len(windows)} new short windows; {len(plan)} total including frozen conference windows")


if __name__ == "__main__":
    main()
