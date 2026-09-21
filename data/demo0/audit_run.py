from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


REQUIRED_METHODS = {"B0", "B1", "B2", "B3", "B4", "B5"}
REQUIRED_ABLATIONS = {
    "B5_no_reject",
    "B5_no_spatial",
    "B5_hard_top1",
    "B5_posthoc_only",
    "B5_K2",
    "B5_K4",
    "B5_K8",
    "B5_L4",
    "B5_L8",
    "B5_L16",
}
REQUIRED_METRICS = {
    "PCK@1_token",
    "PCK@2_token",
    "Top8Recall",
    "EPE_token_mean",
    "EPE_token_median",
    "EPE_pixel_mean",
    "EPE_pixel_median",
    "RejectPrecision",
    "RejectRecall",
    "RejectF1",
    "FeatureWarpError",
    "RGBWarpError",
    "CycleConsistencyError_token",
    "Provisional-Demo0-Proxy-Score",
}


def video_properties(path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise AssertionError(f"Cannot open video: {path}")
    values = (
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        float(capture.get(cv2.CAP_PROP_FPS)),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    return values


def audit_clip(clip_dir: Path) -> dict[str, object]:
    manifest = json.loads((clip_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((clip_dir / "metrics.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise AssertionError(f"{clip_dir.name}: run is {manifest['status']}, not complete")
    if metrics["status"] != "provisional":
        raise AssertionError(f"{clip_dir.name}: missing provisional metric label")
    frame_count = int(manifest["processed_frame_count"])
    token_count = int(manifest["token_grid"]["count"])
    top_l = int(manifest["top_l"])
    slots = int(manifest["slot_count"])
    expected_shapes = {
        "topl_index": (frame_count, token_count, top_l),
        "topl_similarity": (frame_count, token_count, top_l),
        "topl_weight": (frame_count, token_count, top_l),
        "slot_assignment": (frame_count, token_count, top_l, slots + 1),
        "reject_prob": (frame_count, token_count),
        "motion_vectors": (frame_count, slots, 2),
        "token_confidence": (frame_count, token_count),
        "slot_id": (frame_count, token_count),
        "slot_prob": (frame_count, token_count, slots),
        "pred_ref_xy": (frame_count, token_count, 2),
    }
    shapes = {}
    for name, expected in expected_shapes.items():
        array = np.load(clip_dir / "arrays" / f"{name}.npy", mmap_mode="r")
        shapes[name] = list(array.shape)
        if array.shape != expected:
            raise AssertionError(f"{clip_dir.name}/{name}: {array.shape} != {expected}")
        if not np.isfinite(np.asarray(array[0])).all() or not np.isfinite(np.asarray(array[-1])).all():
            raise AssertionError(f"{clip_dir.name}/{name}: non-finite boundary frame")
    if not np.allclose(np.load(clip_dir / "arrays" / "motion_vectors.npy", mmap_mode="r")[0], 0):
        raise AssertionError(f"{clip_dir.name}: frame 0 is not zero displacement")

    slot_png = sorted((clip_dir / "overlays").glob("*.png"))
    flow_png = sorted((clip_dir / "overlays" / "flow").glob("*.png"))
    if len(slot_png) != frame_count or len(flow_png) != frame_count:
        raise AssertionError(
            f"{clip_dir.name}: overlay counts slot={len(slot_png)} flow={len(flow_png)} expected={frame_count}"
        )
    slot_video = video_properties(clip_dir / "overlay_preview.mp4")
    flow_video = video_properties(clip_dir / "overlay_flow_preview.mp4")
    if slot_video[0] != frame_count or flow_video[0] != frame_count:
        raise AssertionError(
            f"{clip_dir.name}: preview counts slot={slot_video[0]} flow={flow_video[0]} expected={frame_count}"
        )

    names = set(metrics["methods"])
    missing = (REQUIRED_METHODS | REQUIRED_ABLATIONS) - names
    if missing:
        raise AssertionError(f"{clip_dir.name}: missing methods/ablations {sorted(missing)}")
    for name, values in metrics["methods"].items():
        metric_missing = REQUIRED_METRICS - set(values)
        if metric_missing:
            raise AssertionError(f"{clip_dir.name}/{name}: missing metrics {sorted(metric_missing)}")
        if values.get("status") != "provisional":
            raise AssertionError(f"{clip_dir.name}/{name}: status is not provisional")
    if not (clip_dir / "legend.json").is_file() or not (clip_dir / "metrics.csv").is_file():
        raise AssertionError(f"{clip_dir.name}: legend or CSV missing")
    examples = json.loads((clip_dir / "query_examples.json").read_text(encoding="utf-8"))
    if len(examples) < 3:
        raise AssertionError(f"{clip_dir.name}: expected at least 3 static query heatmaps")
    for example in examples:
        if not (clip_dir / example["image"]).is_file():
            raise AssertionError(f"{clip_dir.name}: missing query heatmap {example['image']}")
    return {
        "clip_id": clip_dir.name,
        "frame_count": frame_count,
        "slot_overlay_png_count": len(slot_png),
        "flow_overlay_png_count": len(flow_png),
        "slot_preview": slot_video,
        "flow_preview": flow_video,
        "array_shapes": shapes,
        "methods": sorted(names),
        "metric_status": metrics["status"],
        "static_query_heatmap_count": len(examples),
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    reports = []
    demo0_root = run_dir.parents[1]
    if not (demo0_root / "backend" / "app.py").is_file():
        raise AssertionError("FastAPI backend missing")
    if not (demo0_root / "frontend" / "dist" / "index.html").is_file():
        raise AssertionError("Built React/Vite frontend missing")
    for clip in run_manifest["clips"]:
        clip_id = clip["clip"]["clip_id"]
        reports.append(audit_clip(run_dir / clip_id))
    if len(reports) != 2:
        raise AssertionError(f"Expected exactly two requested clips, found {len(reports)}")
    report = {
        "run_id": run_manifest["run_id"],
        "status": "passed",
        "interactive_viewer": {
            "backend": "backend/app.py",
            "frontend_build": "frontend/dist/index.html",
            "status": "present",
        },
        "clips": reports,
    }
    target = run_dir / "audit.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
