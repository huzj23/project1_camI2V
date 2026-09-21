from __future__ import annotations

import json
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
VENDOR = HERE.parents[1] / "demo0" / "vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


DEFAULT_RUN_ROOT = HERE.parent / "runs"
RUN_ROOT = Path(os.environ.get("DEMO0_V11_RUN_ROOT", DEFAULT_RUN_ROOT)).resolve()
ANNOTATION_LOCK = threading.Lock()

app = FastAPI(title="V²-DiT Demo0 v1.1 Viewer", version="1.1")


class Annotation(BaseModel):
    run_id: str
    window_id: str
    frame_id: int
    token_id: int
    mode: str
    hypothesis: str
    first_visible: bool | None = None
    last_visible: bool | None = None
    correct_candidate_ranks: list[int] = []
    should_reject: bool | None = None
    temporal_slot_consistent: bool | None = None
    category: str = ""
    note: str = ""


def safe_component(value: str) -> str:
    if not value or value in {".", ".."} or any(char in value for char in "/\\:"):
        raise HTTPException(400, "Invalid path component")
    return value


def run_dir(run_id: str) -> Path:
    path = RUN_ROOT / safe_component(run_id)
    if not (path / "run_manifest.json").is_file():
        raise HTTPException(404, f"Unknown run: {run_id}")
    return path


def window_dir(run_id: str, window_id: str) -> Path:
    path = run_dir(run_id) / safe_component(window_id)
    if not (path / "window_manifest.json").is_file():
        raise HTTPException(404, f"Unknown window: {window_id}")
    return path


@lru_cache(maxsize=128)
def read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=256)
def array(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def manifest(run_id: str, window_id: str) -> dict[str, Any]:
    return read_json(str(window_dir(run_id, window_id) / "window_manifest.json"))


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    result = []
    if not RUN_ROOT.is_dir():
        return result
    for path in sorted(RUN_ROOT.glob("*/run_manifest.json")):
        value = read_json(str(path))
        result.append(
            {
                "run_id": value["run_id"],
                "status": value.get("status"),
                "schema_version": value.get("schema_version"),
                "execution_mode": "D0-Real",
                "vace_checkpoint_used": bool(value.get("vace_checkpoint_used", False)),
                "legacy_label": value.get("legacy_baseline", {}).get("label"),
            }
        )
    return result


@app.get("/api/runs/{run_id}/windows")
def list_windows(run_id: str) -> list[dict[str, Any]]:
    root = run_dir(run_id)
    result = []
    for path in sorted(root.glob("*/window_manifest.json")):
        value = read_json(str(path))
        result.append(
            {
                "window_id": value["window_id"],
                "display_name": value["display_name"],
                "category": value["category"],
                "frame_count": value["frame_count"],
                "fps": value["fps"],
                "status": value.get("status"),
                "metric_status": value.get("metric_status"),
                "token_grid": value.get("token_grid"),
                "reference_modes": ["F", "FL"],
                "hypotheses": ["H0", "H1"],
                "representation": "dense_multiscale_rootsift_rgb_v1",
            }
        )
    return result


@app.get("/api/frame/{run_id}/{window_id}/{frame_id}")
def frame_image(run_id: str, window_id: str, frame_id: int) -> Response:
    directory = window_dir(run_id, window_id)
    info = manifest(run_id, window_id)
    total = int(info["frame_count"])
    if frame_id < 0 or frame_id >= total:
        raise HTTPException(400, f"frame_id must be in [0,{total - 1}]")
    capture = cv2.VideoCapture(str(directory / "window.mp4"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise HTTPException(500, "Could not read frame")
    encoded_ok, encoded = cv2.imencode(".jpg", frame, (cv2.IMWRITE_JPEG_QUALITY, 94))
    if not encoded_ok:
        raise HTTPException(500, "Could not encode frame")
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


@app.get("/api/query")
def query_token(
    run_id: str,
    window_id: str,
    frame_id: int = Query(ge=0),
    token_id: int = Query(ge=0),
    mode: str = Query(pattern="^(F|FL)$"),
    hypothesis: str = Query(pattern="^(H0|H1)$"),
    top_l: int = Query(default=8, ge=1, le=16),
) -> JSONResponse:
    directory = window_dir(run_id, window_id)
    info = manifest(run_id, window_id)
    total = int(info["frame_count"])
    grid = info["token_grid"]
    grid_h, grid_w, stride = int(grid["height"]), int(grid["width"]), int(grid["stride"])
    token_count = grid_h * grid_w
    if frame_id >= total or token_id >= token_count:
        raise HTTPException(400, "frame_id or token_id out of range")
    mode_dir = directory / "modes" / mode
    method_dir = mode_dir / hypothesis
    if not method_dir.is_dir():
        raise HTTPException(404, "Requested mode/hypothesis is not ready")

    raw_index = np.asarray(array(str(mode_dir / "raw_topl_index.npy"))[frame_id, token_id], dtype=np.int32)
    raw_score = np.asarray(array(str(mode_dir / "raw_topl_score.npy"))[frame_id, token_id], dtype=np.float32)
    endpoint = np.asarray(array(str(mode_dir / "reference_endpoint.npy"))[frame_id, token_id], dtype=np.uint8)
    local_id = np.asarray(array(str(mode_dir / "reference_local_id.npy"))[frame_id, token_id], dtype=np.int32)
    verified = np.asarray(array(str(method_dir / "verified_topl_weight.npy"))[frame_id, token_id], dtype=np.float32)
    candidate_slot = np.asarray(array(str(method_dir / "motion_slot_assignment.npy"))[frame_id, token_id], dtype=np.float32)
    sparse = np.asarray(array(str(method_dir / "sparse_reference_mask.npy"))[frame_id, token_id], dtype=bool)
    descriptors = array(str(directory / "features" / "descriptor.npy"))
    query_descriptor = np.asarray(descriptors[frame_id, token_id], dtype=np.float32)
    first_descriptor = np.asarray(descriptors[0], dtype=np.float32)
    last_descriptor = np.asarray(descriptors[-1], dtype=np.float32)
    first_dense = first_descriptor @ query_descriptor
    last_dense = last_descriptor @ query_descriptor
    fixed_min, fixed_max = 0.45, 1.0
    normalize = lambda value: np.clip((value - fixed_min) / (fixed_max - fixed_min), 0.0, 1.0)
    verified_heat_first = np.zeros(token_count, dtype=np.float32)
    verified_heat_last = np.zeros(token_count, dtype=np.float32)
    for rank in range(len(local_id)):
        if endpoint[rank] == 0:
            verified_heat_first[local_id[rank]] = max(verified_heat_first[local_id[rank]], verified[rank])
        else:
            verified_heat_last[local_id[rank]] = max(verified_heat_last[local_id[rank]], verified[rank])

    verified_order = np.argsort(-verified)
    verified_rank = np.empty_like(verified_order)
    verified_rank[verified_order] = np.arange(len(verified_order)) + 1
    candidates = []
    for rank in range(min(top_l, len(local_id))):
        reference_id = int(local_id[rank])
        candidates.append(
            {
                "raw_rank": rank + 1,
                "verified_rank": int(verified_rank[rank]),
                "combined_reference_index": int(raw_index[rank]),
                "endpoint": "first" if endpoint[rank] == 0 else "last",
                "reference_token_id": reference_id,
                "reference_xy": [reference_id % grid_w, reference_id // grid_w],
                "raw_score": float(raw_score[rank]),
                "verified_weight": float(verified[rank]),
                "sparse_allowed": bool(sparse[rank]),
                "best_motion_slot": int(candidate_slot[rank].argmax()),
                "motion_slot_probability": candidate_slot[rank].tolist(),
            }
        )

    token_slot = np.asarray(array(str(method_dir / "token_slot_probability.npy"))[frame_id, token_id], dtype=np.float32)
    slot_parameters = np.asarray(array(str(method_dir / "motion_slot_parameters.npy"))[frame_id], dtype=np.float32)
    response = {
        "run_id": run_id,
        "window_id": window_id,
        "frame_id": frame_id,
        "mode": mode,
        "hypothesis": hypothesis,
        "execution_mode": "D0-Real",
        "representation": "dense_multiscale_rootsift_rgb_v1",
        "vace_checkpoint_used": False,
        "token": {
            "id": token_id,
            "xy": [token_id % grid_w, token_id // grid_w],
            "nominal_pixel_box": [
                (token_id % grid_w) * stride,
                (token_id // grid_w) * stride,
                (token_id % grid_w + 1) * stride,
                (token_id // grid_w + 1) * stride,
            ],
            "descriptor_support_pixels": [24, 40],
            "texture_confidence": float(array(str(directory / "features" / "texture_confidence.npy"))[frame_id, token_id]),
        },
        "grid": {"height": grid_h, "width": grid_w, "stride": stride},
        "heatmap_scale": {"minimum": fixed_min, "maximum": fixed_max, "fixed_within_representation": True},
        "raw_heatmap": {
            "first": normalize(first_dense).reshape(grid_h, grid_w).tolist(),
            "last": normalize(last_dense).reshape(grid_h, grid_w).tolist(),
        },
        "verified_heatmap": {
            "first": verified_heat_first.reshape(grid_h, grid_w).tolist(),
            "last": verified_heat_last.reshape(grid_h, grid_w).tolist(),
        },
        "candidates": candidates,
        "selected_motion_slot": int(token_slot.argmax()),
        "token_slot_probability": token_slot.tolist(),
        "motion_slots": [
            {
                "slot_id": index,
                "scale_rate": float(value[0]),
                "vx": float(value[1]),
                "vy": float(value[2]),
                "token_probability": float(token_slot[index]),
            }
            for index, value in enumerate(slot_parameters)
        ],
        "reject_probability": float(array(str(method_dir / "reject_probability.npy"))[frame_id, token_id]),
        "confidence": float(array(str(method_dir / "token_confidence.npy"))[frame_id, token_id]),
        "predicted_reference_xy": np.asarray(
            array(str(method_dir / "predicted_reference_xy.npy"))[frame_id, token_id], dtype=np.float32
        ).tolist(),
        "selected_reference_endpoint": int(
            array(str(method_dir / "selected_reference_endpoint.npy"))[frame_id, token_id]
        ),
    }
    return JSONResponse(response)


@app.get("/api/metrics/{run_id}/{window_id}/{kind}")
def metrics(run_id: str, window_id: str, kind: str) -> dict[str, Any]:
    if kind not in {"candidate", "voting", "temporal"}:
        raise HTTPException(400, "kind must be candidate, voting or temporal")
    path = window_dir(run_id, window_id) / "metrics" / f"{kind}_metrics.json"
    if not path.is_file():
        raise HTTPException(404, "metrics not ready")
    return read_json(str(path))


@app.get("/api/overlay/{run_id}/{window_id}/{mode}/{hypothesis}/{kind}/{frame_id}")
def overlay(
    run_id: str,
    window_id: str,
    mode: str,
    hypothesis: str,
    kind: str,
    frame_id: int,
) -> FileResponse:
    if mode not in {"F", "FL"} or hypothesis not in {"H0", "H1"}:
        raise HTTPException(400, "Invalid mode or hypothesis")
    if kind not in {"motion_slots", "reference_source", "confidence", "motion_vectors"}:
        raise HTTPException(400, "Invalid overlay kind")
    path = window_dir(run_id, window_id) / "overlays" / mode / hypothesis / kind / f"{frame_id:06d}.png"
    if not path.is_file():
        raise HTTPException(404, "Overlay not ready")
    return FileResponse(path, media_type="image/png")


@app.get("/api/annotations/{run_id}")
def annotations(run_id: str) -> dict[str, Any]:
    path = run_dir(run_id) / "annotations" / "manual_queries.json"
    if not path.is_file():
        raise HTTPException(404, "Annotation file not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/annotations")
def save_annotation(value: Annotation) -> dict[str, Any]:
    root = run_dir(value.run_id)
    window_info = manifest(value.run_id, value.window_id)
    token_count = int(window_info["token_grid"]["count"])
    if value.frame_id < 0 or value.frame_id >= int(window_info["frame_count"]):
        raise HTTPException(400, "frame_id out of range")
    if value.token_id < 0 or value.token_id >= token_count:
        raise HTTPException(400, "token_id out of range")
    path = root / "annotations" / "manual_queries.json"
    with ANNOTATION_LOCK:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("frozen"):
            raise HTTPException(409, "Annotation set is frozen")
        record = value.model_dump()
        key = (value.window_id, value.frame_id, value.token_id, value.mode, value.hypothesis)
        replaced = False
        for index, existing in enumerate(document["queries"]):
            existing_key = (
                existing["window_id"],
                existing["frame_id"],
                existing["token_id"],
                existing["mode"],
                existing["hypothesis"],
            )
            if existing_key == key:
                document["queries"][index] = record
                replaced = True
                break
        if not replaced:
            document["queries"].append(record)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    return {"status": "saved", "replaced": replaced, "query_count": len(document["queries"])}


FRONTEND = HERE.parent / "frontend"
if FRONTEND.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
