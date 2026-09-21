from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles


HERE = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = HERE.parent / "runs"
RUN_ROOT = Path(os.environ.get("DEMO0_RUN_ROOT", DEFAULT_RUN_ROOT)).resolve()

app = FastAPI(title="V²-DiT Demo 0 Viewer", version="1.0")


def safe_component(value: str) -> str:
    if not value or value in {".", ".."} or any(char in value for char in "/\\:"):
        raise HTTPException(400, "Invalid path component")
    return value


def run_dir(run_id: str) -> Path:
    path = RUN_ROOT / safe_component(run_id)
    if not (path / "manifest.json").is_file():
        raise HTTPException(404, f"Unknown run: {run_id}")
    return path


def clip_dir(run_id: str, clip_id: str) -> Path:
    path = run_dir(run_id) / safe_component(clip_id)
    if not (path / "manifest.json").is_file():
        raise HTTPException(404, f"Unknown clip: {clip_id}")
    return path


@lru_cache(maxsize=32)
def read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=64)
def array(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def clip_manifest(run_id: str, clip_id: str) -> dict[str, Any]:
    path = clip_dir(run_id, clip_id) / "manifest.json"
    return read_json(str(path))


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    result = []
    if not RUN_ROOT.is_dir():
        return result
    for manifest_path in sorted(RUN_ROOT.glob("*/manifest.json")):
        manifest = read_json(str(manifest_path))
        result.append(
            {
                "run_id": manifest.get("run_id", manifest_path.parent.name),
                "schema_version": manifest.get("schema_version"),
                "device": manifest.get("device"),
                "clip_count": len(manifest.get("clips", [])),
            }
        )
    return result


@app.get("/api/runs/{run_id}/clips")
def list_clips(run_id: str) -> list[dict[str, Any]]:
    manifest = read_json(str(run_dir(run_id) / "manifest.json"))
    result = []
    for item in manifest.get("clips", []):
        clip = item["clip"]
        result.append(
            {
                "clip_id": clip["clip_id"],
                "display_name": clip["display_name"],
                "frame_count": item["processed_frame_count"],
                "fps": clip["fps"],
                "status": item["status"],
                "metric_status": item["metric_status"],
                "token_grid": item["token_grid"],
                "available_diffusion_steps": [0],
                "available_layers": ["resnet18.layer3"],
                "available_heads": ["single_projection"],
            }
        )
    return result


@app.get("/api/frame/{run_id}/{clip_id}/{frame_id}")
def frame_image(run_id: str, clip_id: str, frame_id: int) -> Response:
    manifest = clip_manifest(run_id, clip_id)
    total = int(manifest["processed_frame_count"])
    if frame_id < 0 or frame_id >= total:
        raise HTTPException(400, f"frame_id must be in [0, {total - 1}]")
    video = Path(manifest["clip"]["video_path"])
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise HTTPException(500, "Could not read video frame")
    grid = manifest["token_grid"]
    width = int(grid["width"]) * int(grid["stride"])
    height = int(grid["height"]) * int(grid["stride"])
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    encoded_ok, encoded = cv2.imencode(".jpg", frame, (cv2.IMWRITE_JPEG_QUALITY, 92))
    if not encoded_ok:
        raise HTTPException(500, "Could not encode frame")
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


@app.get("/api/query")
def query_token(
    run_id: str,
    clip_id: str,
    frame_id: int = Query(ge=0),
    token_id: int = Query(ge=0),
    top_k: int = Query(default=8, ge=1, le=16),
    diffusion_step: int = Query(default=0, ge=0),
    layer: str = Query(default="resnet18.layer3"),
    head: str = Query(default="single_projection"),
) -> JSONResponse:
    directory = clip_dir(run_id, clip_id)
    manifest = clip_manifest(run_id, clip_id)
    total = int(manifest["processed_frame_count"])
    grid_h = int(manifest["token_grid"]["height"])
    grid_w = int(manifest["token_grid"]["width"])
    stride = int(manifest["token_grid"]["stride"])
    token_count = grid_h * grid_w
    if frame_id >= total:
        raise HTTPException(400, f"frame_id must be in [0, {total - 1}]")
    if token_id >= token_count:
        raise HTTPException(400, f"token_id must be in [0, {token_count - 1}]")
    if diffusion_step != 0:
        raise HTTPException(400, "Demo 0 has exactly one recorded pseudo-step: diffusion_step=0")
    if layer != "resnet18.layer3" or head != "single_projection":
        raise HTTPException(400, "Only resnet18.layer3 / single_projection is recorded in this run")
    arrays = directory / "arrays"
    query_feature = np.asarray(array(str(arrays / "query_features.npy"))[frame_id, token_id], dtype=np.float32)
    reference_feature = np.asarray(array(str(arrays / "reference_features.npy")), dtype=np.float32)
    query_tensor = F.normalize(torch.from_numpy(query_feature), dim=-1)
    reference_tensor = F.normalize(torch.from_numpy(reference_feature), dim=-1)
    logits_tensor = reference_tensor @ query_tensor
    dense_weight_tensor = torch.softmax(logits_tensor / 0.07, dim=-1)
    logits = logits_tensor.numpy()
    dense_weight = dense_weight_tensor.numpy()
    normalized = (logits - logits.min()) / max(float(logits.max() - logits.min()), 1e-8)

    indices = np.asarray(array(str(arrays / "topl_index.npy"))[frame_id, token_id], dtype=np.int64)
    similarity = np.asarray(
        array(str(arrays / "topl_similarity.npy"))[frame_id, token_id], dtype=np.float32
    )
    verified = np.asarray(array(str(arrays / "topl_weight.npy"))[frame_id, token_id], dtype=np.float32)
    assignment = np.asarray(
        array(str(arrays / "slot_assignment.npy"))[frame_id, token_id], dtype=np.float32
    )
    motion = np.asarray(array(str(arrays / "motion_vectors.npy"))[frame_id], dtype=np.float32)
    slot_prob = np.asarray(array(str(arrays / "slot_prob.npy"))[frame_id, token_id], dtype=np.float32)
    reject_prob = float(array(str(arrays / "reject_prob.npy"))[frame_id, token_id])
    confidence = float(array(str(arrays / "token_confidence.npy"))[frame_id, token_id])
    chosen_slot = int(array(str(arrays / "slot_id.npy"))[frame_id, token_id])
    pred_ref_xy = np.asarray(array(str(arrays / "pred_ref_xy.npy"))[frame_id, token_id], dtype=np.float32)
    qx, qy = token_id % grid_w, token_id // grid_w
    candidates = []
    for rank, index in enumerate(indices[:top_k]):
        rx, ry = int(index % grid_w), int(index // grid_w)
        candidates.append(
            {
                "rank": rank + 1,
                "reference_token_id": int(index),
                "reference_xy": [rx, ry],
                "displacement_query_minus_reference": [qx - rx, qy - ry],
                "similarity": float(similarity[rank]),
                "raw_dense_attention_weight": float(dense_weight[index]),
                "verified_value_route_weight": float(verified[rank]),
                "slot_assignment": assignment[rank].tolist(),
                "best_motion_slot": int(np.argmax(assignment[rank, :-1])),
                "candidate_reject_probability": float(assignment[rank, -1]),
            }
        )
    response = {
        "run_id": run_id,
        "clip_id": clip_id,
        "frame_id": frame_id,
        "diffusion_step": diffusion_step,
        "diffusion_step_note": "Demo 0 frozen-image feature pass; not a denoising timestep.",
        "layer": layer,
        "attention_head": head,
        "token": {
            "id": token_id,
            "xy": [qx, qy],
            "nominal_pixel_box": [qx * stride, qy * stride, (qx + 1) * stride, (qy + 1) * stride],
            "receptive_field_note": manifest["token_grid"]["receptive_field_note"],
        },
        "grid": {"height": grid_h, "width": grid_w, "stride": stride},
        "dense_similarity_heatmap": normalized.reshape(grid_h, grid_w).tolist(),
        "dense_attention_heatmap": dense_weight.reshape(grid_h, grid_w).tolist(),
        "candidates": candidates,
        "motion_hypotheses": [
            {
                "slot_id": index,
                "dx": float(vector[0]),
                "dy": float(vector[1]),
                "token_soft_probability": float(slot_prob[index]),
            }
            for index, vector in enumerate(motion)
        ],
        "selected_motion_slot": chosen_slot,
        "reject_probability": reject_prob,
        "confidence": confidence,
        "predicted_reference_xy": pred_ref_xy.tolist(),
        "distinction": {
            "most_similar": "raw cosine/QK candidate before verification",
            "allowed_to_read": "verified_value_route_weight after motion/spatial/temporal/reject checks",
        },
        "ground_truth": None,
        "metric_status": "provisional",
    }
    return JSONResponse(response)


@app.get("/api/metrics/{run_id}/{clip_id}")
def metrics(run_id: str, clip_id: str) -> dict[str, Any]:
    return read_json(str(clip_dir(run_id, clip_id) / "metrics.json"))


@app.get("/api/overlay/{run_id}/{clip_id}/{scheme}/{frame_id}")
def overlay_image(run_id: str, clip_id: str, scheme: str, frame_id: int) -> FileResponse:
    if scheme not in {"slot", "flow"}:
        raise HTTPException(400, "scheme must be slot or flow")
    base = clip_dir(run_id, clip_id) / "overlays"
    path = (base if scheme == "slot" else base / "flow") / f"{frame_id:06d}.png"
    if not path.is_file():
        raise HTTPException(404, "overlay frame not found")
    return FileResponse(path, media_type="image/png")


FRONTEND_DIST = HERE.parent / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "status": "API ready",
            "frontend": "Run npm install && npm run build in frontend/, then restart uvicorn.",
        }
