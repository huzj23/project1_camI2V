from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RUN_ROOT = PROJECT_ROOT / "runs"
FRONTEND = PROJECT_ROOT / "frontend"
ANNOTATION_LOCK = threading.Lock()


class RequestError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def safe_component(value: str) -> str:
    if not value or value in {".", ".."} or any(char in value for char in "/\\:"):
        raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid path component")
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=256)
def load_array(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def run_dir(run_id: str) -> Path:
    path = RUN_ROOT / safe_component(run_id)
    if not (path / "run_manifest.json").is_file():
        raise RequestError(HTTPStatus.NOT_FOUND, f"Unknown run: {run_id}")
    return path


def window_dir(run_id: str, window_id: str) -> Path:
    path = run_dir(run_id) / safe_component(window_id)
    if not (path / "window_manifest.json").is_file():
        raise RequestError(HTTPStatus.NOT_FOUND, f"Unknown window: {window_id}")
    return path


def manifest(run_id: str, window_id: str) -> dict[str, Any]:
    return read_json(window_dir(run_id, window_id) / "window_manifest.json")


def list_runs() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not RUN_ROOT.is_dir():
        return result
    for path in sorted(RUN_ROOT.glob("*/run_manifest.json")):
        value = read_json(path)
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


def list_windows(run_id: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    values = [read_json(path) for path in run_dir(run_id).glob("*/window_manifest.json")]
    values.sort(key=lambda value: (
        value.get("source", {}).get("video_sha256", ""),
        int(value.get("source", {}).get("source_start_frame", 0)),
    ))
    for value in values:
        if value.get("status") != "complete_d0_real":
            continue
        directory = run_dir(run_id) / safe_component(value["window_id"])
        modes_root = directory / "modes"
        reference_modes = sorted(path.name for path in modes_root.iterdir() if path.is_dir()) if modes_root.is_dir() else []
        methods = sorted(
            {
                method_dir.name
                for mode in reference_modes
                for method_dir in (modes_root / mode).iterdir()
                if method_dir.is_dir() and (method_dir / "method_manifest.json").is_file()
            }
        )
        result.append(
            {
                "window_id": value["window_id"],
                "display_name": value["display_name"],
                "category": value["category"],
                "source_start_frame": value.get("source", {}).get("source_start_frame"),
                "source_end_frame": value.get("source", {}).get("source_end_frame"),
                "frame_count": value["frame_count"],
                "fps": value["fps"],
                "status": value.get("status"),
                "metric_status": value.get("metric_status"),
                "token_grid": value.get("token_grid"),
                "reference_modes": reference_modes,
                "methods": methods,
                "representation": "dense_multiscale_rootsift_rgb_v1",
            }
        )
    return result


def frame_bytes(run_id: str, window_id: str, frame_id: int) -> bytes:
    directory = window_dir(run_id, window_id)
    info = manifest(run_id, window_id)
    total = int(info["frame_count"])
    if frame_id < 0 or frame_id >= total:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"frame_id must be in [0,{total - 1}]")
    capture = cv2.VideoCapture(str(directory / "window.mp4"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, "Could not read frame")
    encoded_ok, encoded = cv2.imencode(".jpg", frame, (cv2.IMWRITE_JPEG_QUALITY, 94))
    if not encoded_ok:
        raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, "Could not encode frame")
    return encoded.tobytes()


def _required_query(query: dict[str, list[str]], name: str) -> str:
    if name not in query or not query[name]:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"Missing query parameter: {name}")
    return query[name][0]


def query_token(query: dict[str, list[str]]) -> dict[str, Any]:
    run_id = _required_query(query, "run_id")
    window_id = _required_query(query, "window_id")
    mode = _required_query(query, "mode")
    method = _required_query(query, "hypothesis")
    try:
        frame_id = int(_required_query(query, "frame_id"))
        token_id = int(_required_query(query, "token_id"))
        top_l = int(query.get("top_l", ["8"])[0])
    except ValueError as error:
        raise RequestError(HTTPStatus.BAD_REQUEST, "frame_id, token_id and top_l must be integers") from error
    if mode not in {"F", "FL"} or method not in {"P1", "P2", "P3", "P4", "P5"}:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid mode or ablation method")
    if not 1 <= top_l <= 16:
        raise RequestError(HTTPStatus.BAD_REQUEST, "top_l must be in [1,16]")

    directory = window_dir(run_id, window_id)
    info = manifest(run_id, window_id)
    total = int(info["frame_count"])
    grid = info["token_grid"]
    grid_h, grid_w, stride = int(grid["height"]), int(grid["width"]), int(grid["stride"])
    token_count = grid_h * grid_w
    if frame_id < 0 or frame_id >= total or token_id < 0 or token_id >= token_count:
        raise RequestError(HTTPStatus.BAD_REQUEST, "frame_id or token_id out of range")
    mode_dir = directory / "modes" / mode
    method_dir = mode_dir / method
    if not method_dir.is_dir():
        raise RequestError(HTTPStatus.NOT_FOUND, "Requested mode/hypothesis is not ready")

    def arr(path: Path) -> np.ndarray:
        return load_array(str(path))

    raw_index = np.asarray(arr(mode_dir / "raw_topl_index.npy")[frame_id, token_id], dtype=np.int32)
    raw_score = np.asarray(arr(mode_dir / "raw_topl_score.npy")[frame_id, token_id], dtype=np.float32)
    endpoint = np.asarray(arr(mode_dir / "reference_endpoint.npy")[frame_id, token_id], dtype=np.uint8)
    local_id = np.asarray(arr(mode_dir / "reference_local_id.npy")[frame_id, token_id], dtype=np.int32)
    verified = np.asarray(arr(method_dir / "verified_topl_weight.npy")[frame_id, token_id], dtype=np.float32)
    sparse_slot_storage = (method_dir / "motion_slot_top_id.npy").is_file()
    if sparse_slot_storage:
        candidate_slot_id = np.asarray(
            arr(method_dir / "motion_slot_top_id.npy")[frame_id, token_id], dtype=np.int32
        )
        candidate_slot_probability = np.asarray(
            arr(method_dir / "motion_slot_top_probability.npy")[frame_id, token_id], dtype=np.float32
        )
    else:
        candidate_slot = np.asarray(
            arr(method_dir / "motion_slot_assignment.npy")[frame_id, token_id], dtype=np.float32
        )
    sparse = np.asarray(arr(method_dir / "sparse_reference_mask.npy")[frame_id, token_id], dtype=bool)
    descriptors = arr(directory / "features" / "descriptor.npy")
    query_descriptor = np.asarray(descriptors[frame_id, token_id], dtype=np.float32)
    first_dense = np.asarray(descriptors[0], dtype=np.float32) @ query_descriptor
    last_dense = np.asarray(descriptors[-1], dtype=np.float32) @ query_descriptor

    fixed_min, fixed_max = 0.45, 1.0
    normalize = lambda value: np.clip((value - fixed_min) / (fixed_max - fixed_min), 0.0, 1.0)
    verified_heat_first = np.zeros(token_count, dtype=np.float32)
    verified_heat_last = np.zeros(token_count, dtype=np.float32)
    for rank in range(len(local_id)):
        heat = verified_heat_first if endpoint[rank] == 0 else verified_heat_last
        heat[local_id[rank]] = max(heat[local_id[rank]], verified[rank])

    verified_order = np.argsort(-verified)
    verified_rank = np.empty_like(verified_order)
    verified_rank[verified_order] = np.arange(len(verified_order)) + 1
    candidates = []
    for rank in range(len(local_id)):
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
                "best_motion_slot": (
                    int(candidate_slot_id[rank, 0])
                    if sparse_slot_storage and int(candidate_slot_id[rank, 0]) < 65535
                    else (-1 if sparse_slot_storage else int(candidate_slot[rank].argmax()))
                ),
                "motion_slot_probability": (
                    [
                        {"slot_id": int(slot_id), "probability": float(probability)}
                        for slot_id, probability in zip(candidate_slot_id[rank], candidate_slot_probability[rank])
                        if int(slot_id) < 65535 and float(probability) > 0.0
                    ]
                    if sparse_slot_storage
                    else candidate_slot[rank].tolist()
                ),
            }
        )

    if sparse_slot_storage:
        token_slot_id = np.asarray(arr(method_dir / "token_slot_top_id.npy")[frame_id, token_id], dtype=np.int32)
        token_slot_probability = np.asarray(
            arr(method_dir / "token_slot_top_probability.npy")[frame_id, token_id], dtype=np.float32
        )
        frame_slot_id = np.asarray(arr(method_dir / "token_slot_top_id.npy")[frame_id], dtype=np.int32)
        frame_slot_probability = np.asarray(
            arr(method_dir / "token_slot_top_probability.npy")[frame_id], dtype=np.float32
        )
    else:
        token_slot = np.asarray(arr(method_dir / "token_slot_probability.npy")[frame_id, token_id], dtype=np.float32)
        frame_slot = np.asarray(arr(method_dir / "token_slot_probability.npy")[frame_id], dtype=np.float32)
    frame_confidence = np.asarray(arr(method_dir / "token_confidence.npy")[frame_id], dtype=np.float32)
    frame_endpoint = np.asarray(arr(method_dir / "selected_reference_endpoint.npy")[frame_id], dtype=np.uint8)
    slot_parameters = np.asarray(arr(method_dir / "motion_slot_parameters.npy")[frame_id], dtype=np.float32)
    slot_capacity = int(slot_parameters.shape[0])
    method_manifest_path = method_dir / "method_manifest.json"
    method_manifest = read_json(method_manifest_path) if method_manifest_path.is_file() else {}
    slot_capacity_options = method_manifest.get("slot_capacity_options", [slot_capacity])
    if sparse_slot_storage:
        slot_active = np.asarray(arr(method_dir / "motion_slot_active.npy")[frame_id], dtype=bool)
        slot_mass = np.asarray(arr(method_dir / "motion_slot_mass.npy")[frame_id], dtype=np.float32)
        slot_age = np.asarray(arr(method_dir / "motion_slot_age.npy")[frame_id], dtype=np.int32)
        slot_missed = np.asarray(arr(method_dir / "motion_slot_missed.npy")[frame_id], dtype=np.int32)
        query_probability = {
            int(slot_id): float(probability)
            for slot_id, probability in zip(token_slot_id, token_slot_probability)
            if int(slot_id) < slot_capacity and float(probability) > 0.0
        }
        selected_motion_slot = int(token_slot_id[0]) if int(token_slot_id[0]) < slot_capacity else -1
        lifecycle_path = method_dir / "motion_slot_lifecycle.json"
        lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8")) if lifecycle_path.is_file() else []
        frame_lifecycle = [event for event in lifecycle if int(event.get("frame_id", -1)) == frame_id]
        motion_slots = [
            {
                "slot_id": index,
                "scale_rate": float(slot_parameters[index, 0]),
                "vx": float(slot_parameters[index, 1]),
                "vy": float(slot_parameters[index, 2]),
                "token_probability": query_probability.get(index, 0.0),
                "mass": float(slot_mass[index]),
                "age": int(slot_age[index]),
                "missed": int(slot_missed[index]),
                "active": True,
            }
            for index in np.flatnonzero(slot_active).tolist()
        ]
    else:
        selected_motion_slot = int(token_slot.argmax())
        frame_lifecycle = []
        motion_slots = [
            {
                "slot_id": index,
                "scale_rate": float(value[0]),
                "vx": float(value[1]),
                "vy": float(value[2]),
                "token_probability": float(token_slot[index]),
                "active": True,
            }
            for index, value in enumerate(slot_parameters)
        ]
    return {
        "run_id": run_id,
        "window_id": window_id,
        "frame_id": frame_id,
        "mode": mode,
        "hypothesis": method,
        "method": method,
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
            "texture_confidence": float(arr(directory / "features" / "texture_confidence.npy")[frame_id, token_id]),
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
        "requested_top_l": top_l,
        "selected_motion_slot": selected_motion_slot,
        "token_slot_probability": (None if sparse_slot_storage else token_slot.tolist()),
        "token_slot_sparse": (
            [
                {"slot_id": int(slot_id), "probability": float(probability)}
                for slot_id, probability in zip(token_slot_id, token_slot_probability)
                if int(slot_id) < slot_capacity and float(probability) > 0.0
            ]
            if sparse_slot_storage
            else None
        ),
        "slot_capacity": slot_capacity,
        "slot_capacity_options": slot_capacity_options,
        "active_slot_count": len(motion_slots),
        "slot_lifecycle": frame_lifecycle,
        "frame_motion": {
            "slot_probability": (None if sparse_slot_storage else frame_slot.reshape(grid_h, grid_w, -1).tolist()),
            "slot_ids": (
                frame_slot_id.reshape(grid_h, grid_w, -1).tolist() if sparse_slot_storage else None
            ),
            "slot_probabilities": (
                frame_slot_probability.reshape(grid_h, grid_w, -1).tolist()
                if sparse_slot_storage
                else None
            ),
            "confidence": frame_confidence.reshape(grid_h, grid_w).tolist(),
            "rejected": (frame_endpoint == 2).reshape(grid_h, grid_w).tolist(),
        },
        "motion_slots": motion_slots,
        "reject_probability": float(arr(method_dir / "reject_probability.npy")[frame_id, token_id]),
        "confidence": float(arr(method_dir / "token_confidence.npy")[frame_id, token_id]),
        "predicted_reference_xy": np.asarray(
            arr(method_dir / "predicted_reference_xy.npy")[frame_id, token_id], dtype=np.float32
        ).tolist(),
        "selected_reference_endpoint": int(
            arr(method_dir / "selected_reference_endpoint.npy")[frame_id, token_id]
        ),
    }


def save_annotation(value: dict[str, Any]) -> dict[str, Any]:
    required = {"run_id", "window_id", "frame_id", "token_id", "mode", "hypothesis"}
    missing = sorted(required.difference(value))
    if missing:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"Missing annotation fields: {', '.join(missing)}")
    root = run_dir(str(value["run_id"]))
    if (root / ".frozen.json").is_file():
        raise RequestError(HTTPStatus.CONFLICT, "This conference run is frozen; annotations are read-only")
    info = manifest(str(value["run_id"]), str(value["window_id"]))
    try:
        frame_id, token_id = int(value["frame_id"]), int(value["token_id"])
    except (TypeError, ValueError) as error:
        raise RequestError(HTTPStatus.BAD_REQUEST, "frame_id and token_id must be integers") from error
    if not 0 <= frame_id < int(info["frame_count"]):
        raise RequestError(HTTPStatus.BAD_REQUEST, "frame_id out of range")
    if not 0 <= token_id < int(info["token_grid"]["count"]):
        raise RequestError(HTTPStatus.BAD_REQUEST, "token_id out of range")
    if value["mode"] not in {"F", "FL"} or value["hypothesis"] not in {"P1", "P2", "P3", "P4", "P5"}:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid mode or ablation method")
    record = {
        "run_id": str(value["run_id"]),
        "window_id": str(value["window_id"]),
        "frame_id": frame_id,
        "token_id": token_id,
        "mode": value["mode"],
        "hypothesis": value["hypothesis"],
        "first_visible": value.get("first_visible"),
        "last_visible": value.get("last_visible"),
        "correct_candidate_ranks": [int(rank) for rank in value.get("correct_candidate_ranks", [])],
        "should_reject": value.get("should_reject"),
        "temporal_slot_consistent": value.get("temporal_slot_consistent"),
        "category": str(value.get("category", "")),
        "note": str(value.get("note", "")),
    }
    path = root / "annotations" / "manual_queries.json"
    with ANNOTATION_LOCK:
        document = read_json(path)
        if document.get("frozen"):
            raise RequestError(HTTPStatus.CONFLICT, "Annotation set is frozen")
        key = (record["window_id"], frame_id, token_id, record["mode"], record["hypothesis"])
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


class Demo0Handler(BaseHTTPRequestHandler):
    server_version = "Demo0Viewer/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def send_bytes(self, data: bytes, media_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, value: Any, status: int = HTTPStatus.OK) -> None:
        self.send_bytes(
            json.dumps(value, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def send_path(self, path: Path, media_type: str | None = None) -> None:
        if not path.is_file():
            raise RequestError(HTTPStatus.NOT_FOUND, "File not found")
        self.send_bytes(path.read_bytes(), media_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
            query = parse_qs(parsed.query)
            if parsed.path == "/api/runs":
                self.send_json(list_runs())
            elif len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "windows":
                self.send_json(list_windows(parts[2]))
            elif len(parts) == 5 and parts[:2] == ["api", "frame"]:
                self.send_bytes(frame_bytes(parts[2], parts[3], int(parts[4])), "image/jpeg")
            elif parsed.path == "/api/query":
                self.send_json(query_token(query))
            elif len(parts) == 5 and parts[:2] == ["api", "metrics"]:
                if parts[4] not in {"candidate", "voting", "temporal"}:
                    raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid metric kind")
                self.send_json(read_json(window_dir(parts[2], parts[3]) / "metrics" / f"{parts[4]}_metrics.json"))
            elif len(parts) == 8 and parts[:2] == ["api", "overlay"]:
                _, _, run_id, window_id, mode, method, kind, frame_id_text = parts
                if mode not in {"F", "FL"} or method not in {"P1", "P2", "P3", "P4", "P5"}:
                    raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid mode or ablation method")
                if kind not in {"motion_slots", "reference_source", "confidence", "motion_vectors"}:
                    raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid overlay kind")
                path = window_dir(run_id, window_id) / "overlays" / mode / method / kind / f"{int(frame_id_text):06d}.png"
                self.send_path(path, "image/png")
            elif len(parts) == 3 and parts[:2] == ["api", "annotations"]:
                self.send_json(read_json(run_dir(parts[2]) / "annotations" / "manual_queries.json"))
            elif parsed.path.startswith("/api/"):
                raise RequestError(HTTPStatus.NOT_FOUND, "Unknown API route")
            else:
                relative = "index.html" if parsed.path in {"", "/"} else unquote(parsed.path.lstrip("/"))
                target = (FRONTEND / relative).resolve()
                try:
                    target.relative_to(FRONTEND.resolve())
                except ValueError as error:
                    raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid static path") from error
                self.send_path(target)
        except RequestError as error:
            self.send_json({"error": error.message}, error.status)
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self.send_json({"error": f"Internal server error: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        try:
            if urlparse(self.path).path != "/api/annotations":
                raise RequestError(HTTPStatus.NOT_FOUND, "Unknown API route")
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid request body size")
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, dict):
                raise RequestError(HTTPStatus.BAD_REQUEST, "Annotation body must be an object")
            self.send_json(save_annotation(value))
        except RequestError as error:
            self.send_json({"error": error.message}, error.status)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self.send_json({"error": f"Internal server error: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dependency-free Demo0 v1.1 viewer server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Demo0Handler)
    print(f"Demo0 v1.1 viewer: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
