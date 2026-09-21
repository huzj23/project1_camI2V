from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np

from core import (
    EPS,
    MotionSlotState,
    _fit_dynamic_motion_slots,
    adjacent_flow_to_previous,
    adjacent_token_map,
    candidate_statistics,
    dense_token_descriptor,
    inspect_video,
    read_frame,
    sample_token_grid,
    sha256,
    softmax,
    token_grid,
    top_candidates,
    vote_and_verify,
    write_json,
)


def slot_palette_rgb(count: int) -> np.ndarray:
    """Deterministic colours for persistent ids, including capacities > 32."""
    if count <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    index = np.arange(count, dtype=np.float32)
    hue = np.mod(index * 137.507764, 360.0) / 2.0
    saturation = 170.0 + 65.0 * ((index.astype(np.int32) % 3) / 2.0)
    value = 220.0 + 35.0 * ((index.astype(np.int32) % 2))
    hsv = np.stack((hue, saturation, value), axis=-1).reshape(1, count, 3).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(count, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2-DiT Demo0 v1.1 D0-Real pipeline")
    parser.add_argument(
        "command",
        choices=("validate", "self-test", "prepare", "extract", "candidates", "vote", "render", "metrics", "run"),
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--window", action="append", default=[])
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if "extends" in value:
        parent_path = (path.parent / value["extends"]).resolve()
        parent, _ = load_config(parent_path)
        value = _merge_config(parent, value)
    return value, path.parent


def selected_windows(config: dict[str, Any], names: Iterable[str]) -> list[dict[str, Any]]:
    requested = set(names)
    windows = config["windows"]
    if not requested:
        return windows
    result = [item for item in windows if item["window_id"] in requested]
    missing = requested - {item["window_id"] for item in result}
    if missing:
        raise ValueError(f"Unknown window ids: {sorted(missing)}")
    return result


def resolved_paths(config: dict[str, Any], config_dir: Path, window: dict[str, Any]) -> dict[str, Path]:
    input_root = (config_dir / config["input_root"]).resolve()
    return {
        "video": input_root / window["video"],
        "camera_csv": input_root / window["camera_csv"],
        "metadata": input_root / window["metadata"],
    }


def run_root(config: dict[str, Any], config_dir: Path) -> Path:
    return (config_dir / config["output_root"] / config["run_id"]).resolve()


def validate(config: dict[str, Any], config_dir: Path, windows: list[dict[str, Any]]) -> dict[str, Any]:
    processing = config["processing"]
    width = int(processing["width"])
    height = int(processing["height"])
    stride = int(processing["token_stride"])
    if width % stride or height % stride:
        raise ValueError("Processing dimensions must be divisible by token_stride")
    if int(processing["top_l_primary"]) > int(processing["top_l_max"]):
        raise ValueError("top_l_primary cannot exceed top_l_max")
    voting = config["voting"]
    capacity = int(voting["slot_capacity"])
    allowed_capacities = [int(value) for value in voting.get("slot_capacity_options", [32, 128, 1024])]
    if capacity not in allowed_capacities:
        raise ValueError(f"slot_capacity must be one of {allowed_capacities}, got {capacity}")
    if not 1 <= int(voting.get("assignment_top_k", 3)) <= capacity:
        raise ValueError("assignment_top_k must be in [1, slot_capacity]")

    validated = []
    for window in windows:
        paths = resolved_paths(config, config_dir, window)
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        video = inspect_video(paths["video"])
        with paths["camera_csv"].open("r", encoding="utf-8-sig", newline="") as handle:
            camera_rows = list(csv.DictReader(handle))
        if len(camera_rows) != video.frame_count:
            raise ValueError(
                f"{window['window_id']}: video has {video.frame_count} frames but camera CSV has {len(camera_rows)}"
            )
        start = int(window["source_start_frame"])
        end = int(window["source_end_frame"])
        if start < 0 or end < start or end >= video.frame_count:
            raise ValueError(f"Invalid range for {window['window_id']}: {start}..{end}")
        first = camera_rows[start]
        last = camera_rows[end]
        delta = [float(last[key]) - float(first[key]) for key in ("x", "y", "z")]
        yaw_values = np.asarray([float(row["yaw_deg"]) for row in camera_rows[start : end + 1]])
        pitch_values = np.asarray([float(row["pitch_deg"]) for row in camera_rows[start : end + 1]])
        validated.append(
            {
                "window_id": window["window_id"],
                "source_range_inclusive": [start, end],
                "frame_count": end - start + 1,
                "duration_seconds": (end - start + 1) / video.fps,
                "source_video": str(video.path),
                "source_video_sha256": sha256(video.path),
                "camera_csv_sha256": sha256(paths["camera_csv"]),
                "camera_translation_xyz": delta,
                "camera_translation_norm": float(np.linalg.norm(delta)),
                "yaw_range_deg": [float(yaw_values.min()), float(yaw_values.max())],
                "pitch_range_deg": [float(pitch_values.min()), float(pitch_values.max())],
                "input_resolution": [video.width, video.height],
                "processing_resolution": [width, height],
                "fps": video.fps,
            }
        )
    report = {
        "schema_version": config["schema_version"],
        "status": "valid",
        "write_scope": str(config_dir.parent.resolve()),
        "grid": {"width": width // stride, "height": height // stride, "stride": stride},
        "windows": validated,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _contact_sheet(frames: list[np.ndarray], labels: list[str]) -> np.ndarray:
    thumbnails = []
    for frame, label in zip(frames, labels, strict=True):
        thumbnail = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumbnail, (0, 0), (320, 28), (10, 16, 26), -1)
        cv2.putText(thumbnail, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        thumbnails.append(thumbnail)
    return np.concatenate(thumbnails, axis=1)


def prepare_window(
    config: dict[str, Any],
    config_dir: Path,
    window: dict[str, Any],
    frame_limit: Optional[int],
) -> dict[str, Any]:
    paths = resolved_paths(config, config_dir, window)
    source = inspect_video(paths["video"])
    start = int(window["source_start_frame"])
    configured_end = int(window["source_end_frame"])
    end = min(configured_end, start + frame_limit - 1) if frame_limit else configured_end
    total = end - start + 1
    width = int(config["processing"]["width"])
    height = int(config["processing"]["height"])
    directory = run_root(config, config_dir) / window["window_id"]
    directory.mkdir(parents=True, exist_ok=True)
    video_out = directory / "window.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out), fourcc, source.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {video_out}")
    capture = cv2.VideoCapture(str(paths["video"]))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    selected_frames: list[np.ndarray] = []
    contact_ids = sorted(set(np.linspace(0, total - 1, 5).round().astype(int).tolist()))
    for local_id in range(total):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Unexpected end at source frame {start + local_id}")
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(resized)
        if local_id in contact_ids:
            selected_frames.append(resized.copy())
    capture.release()
    writer.release()
    if len(selected_frames) != len(contact_ids):
        raise RuntimeError("Could not collect contact-sheet frames")
    sheet = _contact_sheet(selected_frames, [f"local {i} / source {start + i}" for i in contact_ids])
    cv2.imwrite(str(directory / "contact_sheet.jpg"), sheet, (cv2.IMWRITE_JPEG_QUALITY, 94))

    with paths["camera_csv"].open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())
    output_fields = ["window_frame_index", "source_frame_index"] + fieldnames
    with (directory / "camera.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=output_fields)
        writer_csv.writeheader()
        for local_id, row in enumerate(rows[start : end + 1]):
            output = dict(row)
            output["window_frame_index"] = local_id
            output["source_frame_index"] = start + local_id
            writer_csv.writerow(output)

    manifest = {
        "schema_version": config["schema_version"],
        "window_id": window["window_id"],
        "display_name": window["display_name"],
        "category": window["category"],
        "status": "prepared",
        "source": {
            "video": str(paths["video"].resolve()),
            "camera_csv": str(paths["camera_csv"].resolve()),
            "metadata": str(paths["metadata"].resolve()),
            "source_start_frame": start,
            "source_end_frame": end,
            "configured_source_end_frame": configured_end,
            "video_sha256": sha256(paths["video"]),
            "camera_csv_sha256": sha256(paths["camera_csv"]),
        },
        "window_video": str(video_out.resolve()),
        "window_video_sha256": sha256(video_out),
        "frame_count": total,
        "fps": source.fps,
        "resolution": [width, height],
        "reference_frames": {"first": 0, "last": total - 1},
        "contact_sheet": "contact_sheet.jpg",
        "matching_geometry": "camera trajectory is recorded but not used for candidate scoring",
    }
    write_json(directory / "window_manifest.json", manifest)
    return manifest


def prepare(
    config: dict[str, Any],
    config_dir: Path,
    windows: list[dict[str, Any]],
    frame_limit: Optional[int],
) -> None:
    root = run_root(config, config_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifests = [prepare_window(config, config_dir, item, frame_limit) for item in windows]
    shutil.copyfile(config_dir / "config.json", root / "run_config.json")
    processing = config["processing"]
    stride = int(processing["token_stride"])
    grid_w = int(processing["width"]) // stride
    grid_h = int(processing["height"]) // stride
    write_json(
        root / "model_and_scheduler.json",
        {
            "execution_mode": "D0-Real",
            "representation": processing["descriptor"],
            "vace_checkpoint_used": False,
            "diffusion_scheduler": None,
            "diffusion_step": None,
            "warning": "This run uses real RGB frames and is not a VACE or denoising-step result.",
        },
    )
    write_json(
        root / "token_metadata.json",
        {
            "grid_height": grid_h,
            "grid_width": grid_w,
            "token_count": grid_h * grid_w,
            "nominal_pixel_box": [stride, stride],
            "descriptor_support_pixels": processing["descriptor_patch_sizes"],
            "representation": processing["descriptor"],
            "coordinate_convention": "xy token coordinates; pixel center=(xy+0.5)*stride",
            "receptive_field_note": "D0-Real descriptor support is explicitly bounded by descriptor_patch_sizes; it is not a VACE receptive field.",
        },
    )
    annotation = root / "annotations" / "manual_queries.json"
    if not annotation.exists():
        write_json(
            annotation,
            {
                "schema_version": "demo0-1.1-manual-annotations",
                "frozen": False,
                "queries": [],
                "instructions": "Record endpoint visibility, correct candidate ranks/regions, rejection and temporal slot judgement.",
            },
        )
    write_json(
        root / "run_manifest.json",
        {
            "schema_version": config["schema_version"],
            "run_id": config["run_id"],
            "status": "prepared",
            "legacy_baseline": {
                "path": str((config_dir.parent / "demo0" / "runs" / "demo0_v1_resnet18_l8_k4").resolve()),
                "label": "ResNet/DIS provisional legacy",
                "status": "failed_baseline",
            },
            "windows": manifests,
            "reference_modes": config["reference_modes"],
            "hypotheses": config["hypotheses"],
            "ablation_methods": config["ablation_methods"],
        },
    )
    print(f"Prepared {len(manifests)} windows under {root}")


def extract_window(config: dict[str, Any], config_dir: Path, window: dict[str, Any]) -> None:
    directory = run_root(config, config_dir) / window["window_id"]
    manifest = json.loads((directory / "window_manifest.json").read_text(encoding="utf-8"))
    total = int(manifest["frame_count"])
    width, height = map(int, manifest["resolution"])
    stride = int(config["processing"]["token_stride"])
    grid_h, grid_w = height // stride, width // stride
    token_count = grid_h * grid_w
    patch_sizes = [int(value) for value in config["processing"]["descriptor_patch_sizes"]]
    feature_dir = directory / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(directory / "window.mp4"))
    descriptor_map = None
    token_rgb = np.lib.format.open_memmap(
        feature_dir / "token_rgb.npy", mode="w+", dtype=np.float16, shape=(total, token_count, 3)
    )
    texture = np.lib.format.open_memmap(
        feature_dir / "texture_confidence.npy", mode="w+", dtype=np.float16, shape=(total, token_count)
    )
    for frame_id in range(total):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read prepared frame {frame_id}/{total}")
        descriptor, rgb, texture_value = dense_token_descriptor(
            frame, grid_h, grid_w, stride, patch_sizes
        )
        if descriptor_map is None:
            descriptor_map = np.lib.format.open_memmap(
                feature_dir / "descriptor.npy",
                mode="w+",
                dtype=np.float16,
                shape=(total, token_count, descriptor.shape[-1]),
            )
        descriptor_map[frame_id] = descriptor.astype(np.float16)
        token_rgb[frame_id] = rgb.astype(np.float16)
        texture[frame_id] = texture_value.astype(np.float16)
        if (frame_id + 1) % 20 == 0 or frame_id + 1 == total:
            descriptor_map.flush()
            token_rgb.flush()
            texture.flush()
            print(f"[{window['window_id']}] descriptors {frame_id + 1}/{total}", flush=True)
    capture.release()
    manifest["status"] = "features_extracted"
    manifest["token_grid"] = {"height": grid_h, "width": grid_w, "count": token_count, "stride": stride}
    manifest["descriptor_dim"] = int(descriptor_map.shape[-1])
    write_json(directory / "window_manifest.json", manifest)


def extract_all(config: dict[str, Any], config_dir: Path, windows: list[dict[str, Any]]) -> None:
    for window in windows:
        extract_window(config, config_dir, window)


def _force_anchor_self(
    index_value: np.ndarray,
    score_value: np.ndarray,
    query_descriptor: np.ndarray,
    reference: np.ndarray,
    reference_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Put the known endpoint self-token at raw rank 1, including ties."""
    result_index = index_value.copy()
    result_score = score_value.copy()
    for token_id in range(index_value.shape[0]):
        desired = reference_offset + token_id
        matches = np.flatnonzero(result_index[token_id] == desired)
        if len(matches):
            position = int(matches[0])
            result_index[token_id, 0], result_index[token_id, position] = (
                result_index[token_id, position],
                result_index[token_id, 0],
            )
            result_score[token_id, 0], result_score[token_id, position] = (
                result_score[token_id, position],
                result_score[token_id, 0],
            )
        else:
            result_index[token_id, 1:] = result_index[token_id, :-1]
            result_score[token_id, 1:] = result_score[token_id, :-1]
            result_index[token_id, 0] = desired
            result_score[token_id, 0] = float(query_descriptor[token_id] @ reference[desired])
    return result_index, result_score


def candidates_window(config: dict[str, Any], config_dir: Path, window: dict[str, Any]) -> None:
    directory = run_root(config, config_dir) / window["window_id"]
    manifest = json.loads((directory / "window_manifest.json").read_text(encoding="utf-8"))
    total = int(manifest["frame_count"])
    grid = manifest["token_grid"]
    token_count = int(grid["count"])
    top_l = int(config["processing"]["top_l_max"])
    descriptors = np.load(directory / "features" / "descriptor.npy", mmap_mode="r")

    for mode in config["reference_modes"]:
        output_dir = directory / "modes" / mode
        output_dir.mkdir(parents=True, exist_ok=True)
        if mode == "F":
            reference = np.asarray(descriptors[0], dtype=np.float32)
            endpoint_lookup = np.zeros(token_count, dtype=np.uint8)
            local_lookup = np.arange(token_count, dtype=np.uint16)
        elif mode == "FL":
            reference = np.concatenate(
                (np.asarray(descriptors[0], dtype=np.float32), np.asarray(descriptors[-1], dtype=np.float32)),
                axis=0,
            )
            endpoint_lookup = np.concatenate(
                (np.zeros(token_count, dtype=np.uint8), np.ones(token_count, dtype=np.uint8))
            )
            local_lookup = np.concatenate(
                (np.arange(token_count, dtype=np.uint16), np.arange(token_count, dtype=np.uint16))
            )
        else:
            raise ValueError(f"Unsupported reference mode: {mode}")

        raw_index = np.lib.format.open_memmap(
            output_dir / "raw_topl_index.npy", mode="w+", dtype=np.uint16, shape=(total, token_count, top_l)
        )
        raw_score = np.lib.format.open_memmap(
            output_dir / "raw_topl_score.npy", mode="w+", dtype=np.float16, shape=(total, token_count, top_l)
        )
        endpoint = np.lib.format.open_memmap(
            output_dir / "reference_endpoint.npy", mode="w+", dtype=np.uint8, shape=(total, token_count, top_l)
        )
        local_id = np.lib.format.open_memmap(
            output_dir / "reference_local_id.npy", mode="w+", dtype=np.uint16, shape=(total, token_count, top_l)
        )
        margin = np.lib.format.open_memmap(
            output_dir / "candidate_margin.npy", mode="w+", dtype=np.float16, shape=(total, token_count)
        )
        entropy = np.lib.format.open_memmap(
            output_dir / "candidate_entropy.npy", mode="w+", dtype=np.float16, shape=(total, token_count)
        )
        for frame_id in range(total):
            query_descriptor = np.asarray(descriptors[frame_id], dtype=np.float32)
            index_value, score_value = top_candidates(query_descriptor, reference, top_l)
            if frame_id == 0:
                index_value, score_value = _force_anchor_self(
                    index_value, score_value, query_descriptor, reference, 0
                )
            elif mode == "FL" and frame_id == total - 1:
                index_value, score_value = _force_anchor_self(
                    index_value, score_value, query_descriptor, reference, token_count
                )
            stats = candidate_statistics(score_value, float(config["processing"]["candidate_temperature"]))
            raw_index[frame_id] = index_value.astype(np.uint16)
            raw_score[frame_id] = score_value.astype(np.float16)
            endpoint[frame_id] = endpoint_lookup[index_value]
            local_id[frame_id] = local_lookup[index_value]
            margin[frame_id] = stats["margin"].astype(np.float16)
            entropy[frame_id] = stats["entropy"].astype(np.float16)
            if (frame_id + 1) % 20 == 0 or frame_id + 1 == total:
                for value in (raw_index, raw_score, endpoint, local_id, margin, entropy):
                    value.flush()
                print(f"[{window['window_id']}:{mode}] candidates {frame_id + 1}/{total}", flush=True)
    manifest["status"] = "candidates_ready"
    write_json(directory / "window_manifest.json", manifest)


def candidates_all(config: dict[str, Any], config_dir: Path, windows: list[dict[str, Any]]) -> None:
    for window in windows:
        candidates_window(config, config_dir, window)


def auxiliary_tracks(config: dict[str, Any], config_dir: Path, window: dict[str, Any]) -> None:
    directory = run_root(config, config_dir) / window["window_id"]
    manifest = json.loads((directory / "window_manifest.json").read_text(encoding="utf-8"))
    total = int(manifest["frame_count"])
    grid = manifest["token_grid"]
    grid_h, grid_w, stride = int(grid["height"]), int(grid["width"]), int(grid["stride"])
    token_count = int(grid["count"])
    aux = directory / "auxiliary_track_reference"
    aux.mkdir(parents=True, exist_ok=True)
    coordinates = token_grid(grid_h, grid_w)
    previous_xy = np.lib.format.open_memmap(
        aux / "previous_xy.npy", mode="w+", dtype=np.float16, shape=(total, token_count, 2)
    )
    next_xy = np.lib.format.open_memmap(
        aux / "next_xy.npy", mode="w+", dtype=np.float16, shape=(total, token_count, 2)
    )
    previous_reliability = np.lib.format.open_memmap(
        aux / "previous_reliability.npy", mode="w+", dtype=np.float16, shape=(total, token_count)
    )
    next_reliability = np.lib.format.open_memmap(
        aux / "next_reliability.npy", mode="w+", dtype=np.float16, shape=(total, token_count)
    )
    previous_xy[0] = coordinates
    next_xy[-1] = coordinates
    previous_reliability[0] = 1.0
    next_reliability[-1] = 1.0

    capture = cv2.VideoCapture(str(directory / "window.mp4"))
    ok, previous_frame = capture.read()
    if not ok:
        raise RuntimeError("Could not read first prepared frame")
    for frame_id in range(1, total):
        ok, current_frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read prepared frame {frame_id}")
        flow_current_previous, flow_previous_current = adjacent_flow_to_previous(previous_frame, current_frame)
        map_previous, reliability_previous = adjacent_token_map(
            flow_current_previous, flow_previous_current, grid_h, grid_w, stride
        )
        map_next, reliability_next = adjacent_token_map(
            flow_previous_current, flow_current_previous, grid_h, grid_w, stride
        )
        previous_xy[frame_id] = map_previous.astype(np.float16)
        previous_reliability[frame_id] = reliability_previous.astype(np.float16)
        next_xy[frame_id - 1] = map_next.astype(np.float16)
        next_reliability[frame_id - 1] = reliability_next.astype(np.float16)
        previous_frame = current_frame
        if frame_id % 20 == 0 or frame_id == total - 1:
            print(f"[{window['window_id']}] adjacent tracks {frame_id}/{total - 1}", flush=True)
    capture.release()
    for value in (previous_xy, next_xy, previous_reliability, next_reliability):
        value.flush()

    track_first = np.lib.format.open_memmap(
        aux / "track_to_first_xy.npy", mode="w+", dtype=np.float16, shape=(total, token_count, 2)
    )
    track_last = np.lib.format.open_memmap(
        aux / "track_to_last_xy.npy", mode="w+", dtype=np.float16, shape=(total, token_count, 2)
    )
    reliable_first = np.lib.format.open_memmap(
        aux / "track_to_first_reliability.npy", mode="w+", dtype=np.float16, shape=(total, token_count)
    )
    reliable_last = np.lib.format.open_memmap(
        aux / "track_to_last_reliability.npy", mode="w+", dtype=np.float16, shape=(total, token_count)
    )
    for frame_id in range(total):
        points = coordinates.copy()
        reliability = np.ones(token_count, dtype=np.float32)
        for step in range(frame_id, 0, -1):
            step_reliability = sample_token_grid(
                np.asarray(previous_reliability[step], dtype=np.float32)[:, None], points, grid_h, grid_w
            )[:, 0]
            points = sample_token_grid(
                np.asarray(previous_xy[step], dtype=np.float32), points, grid_h, grid_w
            )
            reliability *= np.clip(step_reliability, 0.0, 1.0)
        track_first[frame_id] = points.astype(np.float16)
        reliable_first[frame_id] = reliability.astype(np.float16)

        points = coordinates.copy()
        reliability = np.ones(token_count, dtype=np.float32)
        for step in range(frame_id, total - 1):
            step_reliability = sample_token_grid(
                np.asarray(next_reliability[step], dtype=np.float32)[:, None], points, grid_h, grid_w
            )[:, 0]
            points = sample_token_grid(
                np.asarray(next_xy[step], dtype=np.float32), points, grid_h, grid_w
            )
            reliability *= np.clip(step_reliability, 0.0, 1.0)
        track_last[frame_id] = points.astype(np.float16)
        reliable_last[frame_id] = reliability.astype(np.float16)
        if (frame_id + 1) % 20 == 0 or frame_id + 1 == total:
            print(f"[{window['window_id']}] composed tracks {frame_id + 1}/{total}", flush=True)
    for value in (track_first, track_last, reliable_first, reliable_last):
        value.flush()
    write_json(
        aux / "metadata.json",
        {
            "role": "auxiliary_track_reference",
            "algorithm": "OpenCV DIS MEDIUM only between adjacent frames, composed over the short window",
            "used_as_candidate_input": False,
            "used_as_training_label": False,
            "used_for": ["temporal slot propagation", "provisional candidate audit"],
            "warning": "Composed tracks can drift and are not geometry ground truth.",
            "frame_count": total,
            "token_count": token_count,
            "token_grid": {"height": grid_h, "width": grid_w, "stride": stride},
        },
    )


def _auxiliary_tracks_ready(aux: Path, total: int, token_count: int) -> bool:
    expected = {
        "previous_xy.npy": (total, token_count, 2),
        "next_xy.npy": (total, token_count, 2),
        "previous_reliability.npy": (total, token_count),
        "next_reliability.npy": (total, token_count),
        "track_to_first_xy.npy": (total, token_count, 2),
        "track_to_last_xy.npy": (total, token_count, 2),
        "track_to_first_reliability.npy": (total, token_count),
        "track_to_last_reliability.npy": (total, token_count),
    }
    try:
        return all(
            path.is_file() and np.load(path, mmap_mode="r").shape == shape
            for name, shape in expected.items()
            for path in (aux / name,)
        )
    except (OSError, ValueError):
        return False


def _allocate_vote_arrays(
    output: Path,
    total: int,
    token_count: int,
    top_l: int,
    slot_capacity: int,
    assignment_top_k: int,
) -> dict[str, np.ndarray]:
    output.mkdir(parents=True, exist_ok=True)
    specs = {
        "verified_topl_weight": (np.float16, (total, token_count, top_l)),
        # Slot membership is sparse in slot space. This keeps capacity=1024
        # practical instead of allocating [frame,token,candidate,capacity].
        "motion_slot_top_id": (np.uint16, (total, token_count, top_l, assignment_top_k)),
        "motion_slot_top_probability": (np.float16, (total, token_count, top_l, assignment_top_k)),
        "token_slot_top_id": (np.uint16, (total, token_count, assignment_top_k)),
        "token_slot_top_probability": (np.float16, (total, token_count, assignment_top_k)),
        "motion_slot_parameters": (np.float32, (total, slot_capacity, 3)),
        "motion_slot_active": (np.uint8, (total, slot_capacity)),
        "motion_slot_mass": (np.float32, (total, slot_capacity)),
        "motion_slot_age": (np.uint16, (total, slot_capacity)),
        "motion_slot_missed": (np.uint16, (total, slot_capacity)),
        "candidate_outlier_probability": (np.float16, (total, token_count, top_l)),
        "reject_probability": (np.float16, (total, token_count)),
        "token_confidence": (np.float16, (total, token_count)),
        "sparse_reference_mask": (np.uint8, (total, token_count, top_l)),
        "predicted_reference_xy": (np.float16, (total, token_count, 2)),
        "selected_reference_endpoint": (np.uint8, (total, token_count)),
    }
    return {
        name: np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        for name, (dtype, shape) in specs.items()
    }


def vote_window(config: dict[str, Any], config_dir: Path, window: dict[str, Any]) -> None:
    directory = run_root(config, config_dir) / window["window_id"]
    manifest = json.loads((directory / "window_manifest.json").read_text(encoding="utf-8"))
    total = int(manifest["frame_count"])
    grid = manifest["token_grid"]
    grid_h, grid_w = int(grid["height"]), int(grid["width"])
    token_count = int(grid["count"])
    top_l = int(config["processing"]["top_l_max"])
    texture = np.load(directory / "features" / "texture_confidence.npy", mmap_mode="r")
    token_rgb = np.load(directory / "features" / "token_rgb.npy", mmap_mode="r")

    auxiliary_dir = directory / "auxiliary_track_reference"
    if not _auxiliary_tracks_ready(auxiliary_dir, total, token_count):
        print(f"[{window['window_id']}] auxiliary cache missing or stale; rebuilding", flush=True)
        auxiliary_tracks(config, config_dir, window)
    previous_xy = np.load(auxiliary_dir / "previous_xy.npy", mmap_mode="r")
    previous_reliability = np.load(auxiliary_dir / "previous_reliability.npy", mmap_mode="r")

    for mode in config["reference_modes"]:
        mode_dir = directory / "modes" / mode
        raw_index = np.load(mode_dir / "raw_topl_index.npy", mmap_mode="r")
        raw_score = np.load(mode_dir / "raw_topl_score.npy", mmap_mode="r")
        endpoint = np.load(mode_dir / "reference_endpoint.npy", mmap_mode="r")
        local_id = np.load(mode_dir / "reference_local_id.npy", mmap_mode="r")
        for method, method_spec in config["ablation_methods"].items():
            method_vote_config = dict(config["voting"])
            if not method_spec["use_spatial"]:
                method_vote_config["spatial_logit_weight"] = 0.0
            if not method_spec["use_temporal"]:
                method_vote_config["temporal_logit_weight"] = 0.0
                method_vote_config["parameter_smoothing"] = 0.0
            slot_capacity = int(config["voting"]["slot_capacity"])
            assignment_top_k = int(config["voting"].get("assignment_top_k", 3))
            arrays = _allocate_vote_arrays(
                mode_dir / method, total, token_count, top_l, slot_capacity, assignment_top_k
            )
            write_json(
                mode_dir / method / "method_manifest.json",
                {
                    "method": method,
                    **method_spec,
                    "candidate_source": "P0 dense_multiscale_rootsift_rgb_v1 Top-16",
                    "primary_candidate_count": int(config["processing"]["top_l_primary"]),
                    "slot_storage": "sparse_top_k",
                    "slot_capacity": slot_capacity,
                    "slot_capacity_options": config["voting"].get("slot_capacity_options", [32, 128, 1024]),
                    "effective_vote_config": method_vote_config,
                },
            )
            previous_slot_state = None
            previous_token_slot = None
            lifecycle_log: list[dict[str, Any]] = []
            for frame_id in range(total):
                temporal_prior = None
                temporal_reliability = None
                temporal_state_ready = mode == "FL" or frame_id / max(total - 1, 1) >= 0.12
                if (
                    method_spec["use_temporal"]
                    and temporal_state_ready
                    and frame_id > 0
                    and previous_token_slot is not None
                ):
                    temporal_prior = sample_token_grid(
                        previous_token_slot,
                        np.asarray(previous_xy[frame_id], dtype=np.float32),
                        grid_h,
                        grid_w,
                    )
                    temporal_reliability = np.asarray(previous_reliability[frame_id], dtype=np.float32)
                result = vote_and_verify(
                    frame_id=frame_id,
                    frame_count=total,
                    raw_index=np.asarray(raw_index[frame_id], dtype=np.int32),
                    raw_score=np.asarray(raw_score[frame_id], dtype=np.float32),
                    reference_endpoint=np.asarray(endpoint[frame_id], dtype=np.uint8),
                    reference_local_id=np.asarray(local_id[frame_id], dtype=np.int32),
                    texture=np.asarray(texture[frame_id], dtype=np.float32),
                    token_rgb=np.asarray(token_rgb[frame_id], dtype=np.float32),
                    grid_h=grid_h,
                    grid_w=grid_w,
                    candidate_temperature=float(config["processing"]["candidate_temperature"]),
                    vote_config=method_vote_config,
                    hypothesis=method_spec["hypothesis"],
                    previous_slot_state=previous_slot_state,
                    previous_token_slot=previous_token_slot,
                    temporal_prior=temporal_prior,
                    temporal_reliability=temporal_reliability,
                    enable_rejection=bool(method_spec["use_rejection"]),
                    enable_sparse=bool(method_spec["use_sparse_value_mask"]),
                )
                token_order = np.argsort(-result.token_slot, axis=-1)[..., :assignment_top_k]
                token_probability = np.take_along_axis(result.token_slot, token_order, axis=-1)
                invalid_slot_id = np.uint16(65535)
                token_slot_id = np.where(token_probability > 0.0, token_order, invalid_slot_id).astype(np.uint16)
                candidate_slot_id = np.where(
                    result.candidate_slot_id >= 0, result.candidate_slot_id, invalid_slot_id
                ).astype(np.uint16)
                values = {
                    "verified_topl_weight": result.candidate_weight,
                    "motion_slot_top_id": candidate_slot_id,
                    "motion_slot_top_probability": result.candidate_slot_probability,
                    "token_slot_top_id": token_slot_id,
                    "token_slot_top_probability": token_probability,
                    "motion_slot_parameters": result.slot_parameters,
                    "motion_slot_active": result.slot_active,
                    "motion_slot_mass": result.slot_mass,
                    "motion_slot_age": result.slot_age,
                    "motion_slot_missed": result.slot_missed,
                    "candidate_outlier_probability": result.candidate_outlier_probability,
                    "reject_probability": result.reject_probability,
                    "token_confidence": result.confidence,
                    "sparse_reference_mask": result.sparse_mask,
                    "predicted_reference_xy": result.predicted_reference_xy,
                    "selected_reference_endpoint": result.selected_endpoint,
                }
                for name, value in values.items():
                    arrays[name][frame_id] = value.astype(arrays[name].dtype)
                for event in result.lifecycle_events:
                    lifecycle_log.append({"frame_id": frame_id, **event})
                previous_slot_state = result.slot_state
                if method_spec["use_temporal"] and temporal_state_ready:
                    previous_token_slot = result.token_slot
                else:
                    previous_token_slot = None
                if (frame_id + 1) % 20 == 0 or frame_id + 1 == total:
                    for value in arrays.values():
                        value.flush()
                    print(
                        f"[{window['window_id']}:{mode}:{method}] vote {frame_id + 1}/{total}",
                        flush=True,
                    )
            write_json(mode_dir / method / "motion_slot_lifecycle.json", lifecycle_log)
    manifest["status"] = "vote_ready"
    write_json(directory / "window_manifest.json", manifest)


def vote_all(config: dict[str, Any], config_dir: Path, windows: list[dict[str, Any]]) -> None:
    for window in windows:
        vote_window(config, config_dir, window)


def _draw_hatch(image: np.ndarray, mask: np.ndarray, stride: int) -> None:
    grid_h, grid_w = mask.shape
    for y in range(grid_h):
        for x in range(grid_w):
            if not mask[y, x]:
                continue
            x0, y0 = x * stride, y * stride
            cv2.line(image, (x0 + 2, y0 + stride - 3), (x0 + stride - 3, y0 + 2), (170, 170, 170), 1)


def _overlay_layer(frame: np.ndarray, color_grid: np.ndarray, alpha_grid: np.ndarray, stride: int) -> np.ndarray:
    color = cv2.resize(color_grid, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    alpha = cv2.resize(alpha_grid, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)[..., None]
    return np.clip(frame * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def render_window(config: dict[str, Any], config_dir: Path, window: dict[str, Any]) -> None:
    directory = run_root(config, config_dir) / window["window_id"]
    manifest = json.loads((directory / "window_manifest.json").read_text(encoding="utf-8"))
    total = int(manifest["frame_count"])
    width, height = map(int, manifest["resolution"])
    grid = manifest["token_grid"]
    grid_h, grid_w, stride = int(grid["height"]), int(grid["width"]), int(grid["stride"])
    slot_capacity = int(config["voting"]["slot_capacity"])
    palette_bgr = slot_palette_rgb(slot_capacity)[:, ::-1]
    alpha_base = float(config["visualization"]["overlay_alpha"])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for mode in config["reference_modes"]:
        for method, method_spec in config["ablation_methods"].items():
            source = directory / "modes" / mode / method
            sparse_slot_storage = (source / "token_slot_top_id.npy").is_file()
            if sparse_slot_storage:
                token_slot_id = np.load(source / "token_slot_top_id.npy", mmap_mode="r")
                token_slot_probability = np.load(source / "token_slot_top_probability.npy", mmap_mode="r")
                slot_active = np.load(source / "motion_slot_active.npy", mmap_mode="r")
            else:
                token_slot = np.load(source / "token_slot_probability.npy", mmap_mode="r")
            confidence = np.load(source / "token_confidence.npy", mmap_mode="r")
            endpoint = np.load(source / "selected_reference_endpoint.npy", mmap_mode="r")
            predicted_xy = np.load(source / "predicted_reference_xy.npy", mmap_mode="r")
            output_root = directory / "overlays" / mode / method
            kinds = ("motion_slots", "reference_source", "confidence", "motion_vectors")
            writers = {}
            for kind in kinds:
                target_dir = output_root / kind
                target_dir.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(output_root / f"overlay_{kind}.mp4"), fourcc, float(manifest["fps"]), (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not initialize overlay writer for {target_dir}")
                writers[kind] = writer

            capture = cv2.VideoCapture(str(directory / "window.mp4"))
            coordinates = token_grid(grid_h, grid_w)
            for frame_id in range(total):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not read frame {frame_id} for overlay")
                if sparse_slot_storage:
                    sparse_ids = np.asarray(token_slot_id[frame_id], dtype=np.int32)
                    sparse_prob = np.asarray(token_slot_probability[frame_id], dtype=np.float32)
                    invalid = sparse_ids >= slot_capacity
                    safe_ids = np.where(invalid, 0, sparse_ids)
                    ids = np.where(invalid[:, 0], -1, safe_ids[:, 0]).reshape(grid_h, grid_w)
                else:
                    slot_prob = np.asarray(token_slot[frame_id], dtype=np.float32)
                    ids = slot_prob.argmax(-1).reshape(grid_h, grid_w)
                conf = np.asarray(confidence[frame_id], dtype=np.float32).reshape(grid_h, grid_w)
                rejected = (
                    conf < float(config["voting"]["reject_confidence_threshold"])
                    if method_spec["use_rejection"]
                    else np.zeros_like(conf, dtype=bool)
                )
                alpha = alpha_base * (0.35 + 0.65 * conf)

                # Mix palette colours by the full slot posterior.  A hard
                # argmax made tiny probability changes look like crisp,
                # arbitrary region switches even when the slot was uncertain.
                if sparse_slot_storage:
                    display_prob = np.where(invalid, 0.0, sparse_prob)
                    display_prob = np.maximum(display_prob, 0.0) ** float(
                        config["visualization"]["slot_display_power"]
                    )
                    display_prob /= np.maximum(display_prob.sum(-1, keepdims=True), EPS)
                    gathered_color = palette_bgr[safe_ids].astype(np.float32)
                    slot_color = (display_prob[..., None] * gathered_color).sum(1).reshape(
                        grid_h, grid_w, 3
                    ).astype(np.uint8)
                    slot_certainty = sparse_prob[:, 0].reshape(grid_h, grid_w)
                    active_count = int(np.asarray(slot_active[frame_id], dtype=bool).sum())
                else:
                    slot_palette = palette_bgr[: slot_prob.shape[-1]].astype(np.float32)
                    display_prob = np.maximum(slot_prob, EPS) ** float(
                        config["visualization"]["slot_display_power"]
                    )
                    display_prob /= np.maximum(display_prob.sum(-1, keepdims=True), EPS)
                    slot_color = (display_prob @ slot_palette).reshape(grid_h, grid_w, 3).astype(np.uint8)
                    slot_certainty = slot_prob.max(-1).reshape(grid_h, grid_w)
                    active_count = slot_prob.shape[-1]
                motion_alpha = alpha * (0.55 + 0.45 * slot_certainty)
                motion_image = _overlay_layer(frame, slot_color, motion_alpha, stride)
                _draw_hatch(motion_image, rejected, stride)

                endpoint_grid = np.asarray(endpoint[frame_id], dtype=np.uint8).reshape(grid_h, grid_w)
                endpoint_color = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
                endpoint_color[endpoint_grid == 0] = (255, 150, 45)
                endpoint_color[endpoint_grid == 1] = (40, 145, 255)
                endpoint_color[endpoint_grid == 2] = (130, 130, 130)
                endpoint_image = _overlay_layer(frame, endpoint_color, alpha, stride)
                _draw_hatch(endpoint_image, rejected, stride)

                confidence_color = cv2.applyColorMap((np.clip(conf, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                confidence_image = _overlay_layer(frame, confidence_color, np.full_like(conf, alpha_base), stride)
                _draw_hatch(confidence_image, rejected, stride)

                vector_image = frame.copy()
                pred = np.asarray(predicted_xy[frame_id], dtype=np.float32)
                confidence_flat = conf.reshape(-1)
                for token_id in range(0, coordinates.shape[0], 8):
                    q = coordinates[token_id]
                    p = pred[token_id]
                    start = (int((q[0] + 0.5) * stride), int((q[1] + 0.5) * stride))
                    end = (int((p[0] + 0.5) * stride), int((p[1] + 0.5) * stride))
                    color = (255, 150, 45) if endpoint_grid.reshape(-1)[token_id] == 0 else (40, 145, 255)
                    # Rejected tokens have no trustworthy endpoint.  Drawing
                    # their arrows created long, misleading lines across the
                    # sky; rejection is already shown by the hatch overlay.
                    if rejected.reshape(-1)[token_id] or confidence_flat[token_id] < 0.35:
                        continue
                    end = (
                        int(np.clip(end[0], 0, width - 1)),
                        int(np.clip(end[1], 30, height - 1)),
                    )
                    cv2.arrowedLine(vector_image, start, end, color, 1, cv2.LINE_AA, tipLength=0.18)

                images = {
                    "motion_slots": motion_image,
                    "reference_source": endpoint_image,
                    "confidence": confidence_image,
                    "motion_vectors": vector_image,
                }
                header = (
                    f"{window['window_id']} | {mode} | {method} | frame {frame_id}/{total - 1} "
                    f"| active slots {active_count}/{slot_capacity}"
                )
                for kind, image in images.items():
                    cv2.rectangle(image, (0, 0), (width, 30), (8, 14, 24), -1)
                    kind_header = header + (" | conf>=0.35, 1/8 tokens" if kind == "motion_vectors" else "")
                    cv2.putText(image, kind_header, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.imwrite(str(output_root / kind / f"{frame_id:06d}.png"), image)
                    writers[kind].write(image)
                if (frame_id + 1) % 30 == 0 or frame_id + 1 == total:
                    print(f"[{window['window_id']}:{mode}:{method}] render {frame_id + 1}/{total}", flush=True)
            capture.release()
            for writer in writers.values():
                writer.release()
    manifest["status"] = "rendered"
    write_json(directory / "window_manifest.json", manifest)


def render_all(config: dict[str, Any], config_dir: Path, windows: list[dict[str, Any]]) -> None:
    for window in windows:
        render_window(config, config_dir, window)


def _candidate_correctness(
    endpoint: np.ndarray,
    local_id: np.ndarray,
    target_first: np.ndarray,
    target_last: np.ndarray,
    reliable_first: np.ndarray,
    reliable_last: np.ndarray,
    grid_h: int,
    grid_w: int,
    radius: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = token_grid(grid_h, grid_w)
    candidate_xy = coordinates[local_id]
    distance_first = np.linalg.norm(candidate_xy - target_first[:, None, :], axis=-1)
    distance_last = np.linalg.norm(candidate_xy - target_last[:, None, :], axis=-1)
    correct = ((endpoint == 0) & (distance_first <= radius)) | ((endpoint == 1) & (distance_last <= radius))
    valid = ((endpoint == 0) & (reliable_first[:, None] >= 0.20)) | (
        (endpoint == 1) & (reliable_last[:, None] >= 0.20)
    )
    correct &= valid
    query_valid = valid.any(-1)
    inside_first = (
        (target_first[:, 0] >= 0)
        & (target_first[:, 0] <= grid_w - 1)
        & (target_first[:, 1] >= 0)
        & (target_first[:, 1] <= grid_h - 1)
    )
    inside_last = (
        (target_last[:, 0] >= 0)
        & (target_last[:, 0] <= grid_w - 1)
        & (target_last[:, 1] >= 0)
        & (target_last[:, 1] <= grid_h - 1)
    )
    query_valid &= ((reliable_first >= 0.20) & inside_first) | ((reliable_last >= 0.20) & inside_last)
    return correct, query_valid


def _rank_metrics(correct: np.ndarray, valid: np.ndarray, weight: Optional[np.ndarray] = None) -> dict[str, Any]:
    if weight is None:
        order = np.broadcast_to(np.arange(correct.shape[-1]), correct.shape)
    else:
        order = np.argsort(-weight, axis=-1)
        correct = np.take_along_axis(correct, order, axis=-1)
    first = np.where(correct, np.arange(correct.shape[-1])[None, :], correct.shape[-1]).min(-1)
    count = int(valid.sum())
    if count == 0:
        return {"valid_queries": 0, "Hit@1": None, "Hit@8": None, "Hit@16": None, "MRR": None}
    return {
        "valid_queries": count,
        "Hit@1": float((first[valid] < 1).mean()),
        "Hit@8": float((first[valid] < min(8, correct.shape[-1])).mean()),
        "Hit@16": float((first[valid] < min(16, correct.shape[-1])).mean()),
        "MRR": float(np.where(first[valid] < correct.shape[-1], 1.0 / (first[valid] + 1), 0.0).mean()),
    }


def metrics_window(config: dict[str, Any], config_dir: Path, window: dict[str, Any]) -> None:
    directory = run_root(config, config_dir) / window["window_id"]
    manifest = json.loads((directory / "window_manifest.json").read_text(encoding="utf-8"))
    total = int(manifest["frame_count"])
    grid_h = int(manifest["token_grid"]["height"])
    grid_w = int(manifest["token_grid"]["width"])
    aux = directory / "auxiliary_track_reference"
    target_first = np.load(aux / "track_to_first_xy.npy", mmap_mode="r")
    target_last = np.load(aux / "track_to_last_xy.npy", mmap_mode="r")
    reliable_first = np.load(aux / "track_to_first_reliability.npy", mmap_mode="r")
    reliable_last = np.load(aux / "track_to_last_reliability.npy", mmap_mode="r")
    texture = np.load(directory / "features" / "texture_confidence.npy", mmap_mode="r")
    previous_xy = np.load(aux / "previous_xy.npy", mmap_mode="r")
    previous_reliability = np.load(aux / "previous_reliability.npy", mmap_mode="r")

    candidate_report: dict[str, Any] = {"status": "auxiliary_provisional", "modes": {}}
    voting_report: dict[str, Any] = {"status": "auxiliary_provisional", "methods": {}}
    temporal_report: dict[str, Any] = {"status": "auxiliary_provisional", "methods": {}}
    for mode in config["reference_modes"]:
        mode_dir = directory / "modes" / mode
        endpoint = np.load(mode_dir / "reference_endpoint.npy", mmap_mode="r")
        local_id = np.load(mode_dir / "reference_local_id.npy", mmap_mode="r")
        raw_score = np.load(mode_dir / "raw_topl_score.npy", mmap_mode="r")
        all_correct = []
        all_valid = []
        raw_weights = []
        visibility_groups: dict[str, list[np.ndarray]] = {
            "first_visible": [],
            "first_not_visible": [],
            "first_only": [],
            "last_only": [],
            "both_endpoints": [],
            "neither_endpoint": [],
        }
        for frame_id in range(total):
            correct, valid = _candidate_correctness(
                np.asarray(endpoint[frame_id], dtype=np.uint8),
                np.asarray(local_id[frame_id], dtype=np.int32),
                np.asarray(target_first[frame_id], dtype=np.float32),
                np.asarray(target_last[frame_id], dtype=np.float32),
                np.asarray(reliable_first[frame_id], dtype=np.float32),
                np.asarray(reliable_last[frame_id], dtype=np.float32) if mode == "FL" else np.zeros_like(reliable_last[frame_id]),
                grid_h,
                grid_w,
            )
            valid &= np.asarray(texture[frame_id], dtype=np.float32) >= float(config["voting"]["minimum_query_texture"])
            all_correct.append(correct)
            all_valid.append(valid)
            raw_weights.append(
                softmax(
                    np.asarray(raw_score[frame_id], dtype=np.float32)
                    / float(config["processing"]["candidate_temperature"]),
                    axis=-1,
                )
            )
            first_xy = np.asarray(target_first[frame_id], dtype=np.float32)
            last_xy = np.asarray(target_last[frame_id], dtype=np.float32)
            first_visible = (
                (np.asarray(reliable_first[frame_id], dtype=np.float32) >= 0.20)
                & (first_xy[:, 0] >= 0)
                & (first_xy[:, 0] <= grid_w - 1)
                & (first_xy[:, 1] >= 0)
                & (first_xy[:, 1] <= grid_h - 1)
            )
            last_visible = (
                (np.asarray(reliable_last[frame_id], dtype=np.float32) >= 0.20)
                & (last_xy[:, 0] >= 0)
                & (last_xy[:, 0] <= grid_w - 1)
                & (last_xy[:, 1] >= 0)
                & (last_xy[:, 1] <= grid_h - 1)
            )
            textured = np.asarray(texture[frame_id], dtype=np.float32) >= float(
                config["voting"]["minimum_query_texture"]
            )
            visibility_groups["first_visible"].append(first_visible & textured)
            visibility_groups["first_not_visible"].append(~first_visible & textured)
            visibility_groups["first_only"].append(first_visible & ~last_visible & textured)
            visibility_groups["last_only"].append(~first_visible & last_visible & textured)
            visibility_groups["both_endpoints"].append(first_visible & last_visible & textured)
            visibility_groups["neither_endpoint"].append(~first_visible & ~last_visible & textured)
        correct = np.concatenate(all_correct)
        valid = np.concatenate(all_valid)
        raw_weight = np.concatenate(raw_weights)
        raw_score_flat = np.asarray(raw_score, dtype=np.float32).reshape(-1, raw_score.shape[-1])
        normalized_entropy = -(
            raw_weight * np.log(np.maximum(raw_weight, EPS))
        ).sum(-1) / math.log(raw_weight.shape[-1])
        group_metrics = {
            name: _rank_metrics(correct, np.concatenate(masks))
            for name, masks in visibility_groups.items()
            if (mode == "FL" or name in {"first_visible", "first_not_visible"})
        }
        p0_metrics = _rank_metrics(correct, valid, raw_weight)
        p0_correct_weight = (raw_weight * correct.astype(np.float32)).sum(-1)
        candidate_report["modes"][mode] = {
            "P0": p0_metrics,
            "raw_candidate_order": _rank_metrics(correct, valid),
            "visibility_groups_provisional": group_metrics,
            "valid_fraction": float(valid.mean()),
            "mean_top_score": float(np.asarray(raw_score[..., 0], dtype=np.float32).mean()),
            "mean_top1_top2_margin": float((raw_score_flat[:, 0] - raw_score_flat[:, 1]).mean()),
            "mean_normalized_entropy": float(normalized_entropy.mean()),
            "mean_correct_candidate_weight": float(p0_correct_weight[valid].mean()) if valid.any() else None,
            "mean_texture": float(np.asarray(texture, dtype=np.float32).mean()),
            "label_source": "adjacent-flow composition; provisional, not ground truth",
        }

        for method, method_spec in config["ablation_methods"].items():
            method_key = f"{mode}_{method}"
            result_dir = mode_dir / method
            verified = np.asarray(np.load(result_dir / "verified_topl_weight.npy", mmap_mode="r"), dtype=np.float32)
            confidence = np.asarray(np.load(result_dir / "token_confidence.npy", mmap_mode="r"), dtype=np.float32)
            sparse = np.asarray(np.load(result_dir / "sparse_reference_mask.npy", mmap_mode="r"), dtype=bool)
            sparse_slot_path = result_dir / "token_slot_top_id.npy"
            if sparse_slot_path.is_file():
                slot_id = np.asarray(np.load(sparse_slot_path, mmap_mode="r")[..., 0], dtype=np.int32)
                slot = None
            else:
                slot = np.asarray(np.load(result_dir / "token_slot_probability.npy", mmap_mode="r"), dtype=np.float32)
                slot_id = None
            voting_report["methods"][method_key] = {
                "method_spec": method_spec,
                "verified": _rank_metrics(correct, valid, verified.reshape(-1, verified.shape[-1])),
                "accepted_fraction": (
                    float((confidence >= float(config["voting"]["reject_confidence_threshold"])).mean())
                    if method_spec["use_rejection"]
                    else 1.0
                ),
                "mean_confidence": float(confidence.mean()),
                "mean_value_candidates": float(sparse.sum(-1).mean()),
                "sparse_fraction_of_reference_bank": float(
                    sparse.sum(-1).mean() / ((1 if mode == "F" else 2) * grid_h * grid_w)
                ),
                "mean_correct_candidate_weight": (
                    float(((verified.reshape(-1, verified.shape[-1]) * correct).sum(-1)[valid]).mean())
                    if valid.any()
                    else None
                ),
                "delta_correct_weight_vs_P0": (
                    float(
                        (
                            (verified.reshape(-1, verified.shape[-1]) * correct).sum(-1)[valid]
                            - p0_correct_weight[valid]
                        ).mean()
                    )
                    if valid.any()
                    else None
                ),
                "sparse_correct_candidate_coverage": (
                    float(((sparse.reshape(-1, sparse.shape[-1]) & correct).any(-1)[valid]).mean())
                    if valid.any()
                    else None
                ),
                "reject_precision_recall": "pending frozen manual annotations",
            }
            flicker_values = []
            for frame_id in range(1, total):
                if slot_id is not None:
                    mapping = np.asarray(previous_xy[frame_id], dtype=np.float32)
                    map_x = mapping[:, 0].reshape(grid_h, grid_w)
                    map_y = mapping[:, 1].reshape(grid_h, grid_w)
                    warped_id = cv2.remap(
                        slot_id[frame_id - 1].reshape(grid_h, grid_w).astype(np.float32),
                        map_x,
                        map_y,
                        interpolation=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=65535,
                    ).reshape(-1).astype(np.int32)
                    current_id = slot_id[frame_id]
                else:
                    warped = sample_token_grid(
                        slot[frame_id - 1],
                        np.asarray(previous_xy[frame_id], dtype=np.float32),
                        grid_h,
                        grid_w,
                    )
                    warped_id = warped.argmax(-1)
                    current_id = slot[frame_id].argmax(-1)
                reliable = np.asarray(previous_reliability[frame_id], dtype=np.float32) >= 0.5
                if reliable.any():
                    flicker_values.append((current_id[reliable] != warped_id[reliable]).astype(np.float32))
            temporal_report["methods"][method_key] = {
                "method_spec": method_spec,
                "slot_flicker_rate": float(np.concatenate(flicker_values).mean()) if flicker_values else None,
                "adjacent_track_role": "disclosed temporal continuity signal and provisional audit only",
                "object_surface_persistence": "pending frozen manual annotations",
                "cross_boundary_merge_rate": "pending frozen manual annotations",
            }

    metrics_dir = directory / "metrics"
    write_json(metrics_dir / "candidate_metrics.json", candidate_report)
    write_json(metrics_dir / "voting_metrics.json", voting_report)
    write_json(metrics_dir / "temporal_metrics.json", temporal_report)
    manifest["status"] = "complete_d0_real"
    manifest["metric_status"] = "auxiliary_provisional_pending_manual_annotations"
    write_json(directory / "window_manifest.json", manifest)


def metrics_all(config: dict[str, Any], config_dir: Path, windows: list[dict[str, Any]]) -> None:
    for window in windows:
        metrics_window(config, config_dir, window)
    root = run_root(config, config_dir)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "complete_d0_real_pending_manual_audit"
    manifest["completed_at_unix"] = time.time()
    manifest["vace_checkpoint_used"] = False
    manifest["manual_annotations_frozen"] = False
    write_json(root / "run_manifest.json", manifest)


def self_test(config: dict[str, Any]) -> None:
    rng = np.random.default_rng(int(config["seed"]))
    grid_h, grid_w = 9, 12
    token_count = grid_h * grid_w
    top_l = int(config["processing"]["top_l_max"])
    query_xy = token_grid(grid_h, grid_w)
    true_ref = query_xy.copy()
    true_ref[:, 0] -= 2.0
    true_ref[:, 1] += 1.0
    true_ref[:, 0] = np.clip(true_ref[:, 0], 0, grid_w - 1)
    true_ref[:, 1] = np.clip(true_ref[:, 1], 0, grid_h - 1)
    local = np.empty((token_count, top_l), dtype=np.int32)
    local[:, 0] = np.rint(true_ref[:, 1]).astype(int) * grid_w + np.rint(true_ref[:, 0]).astype(int)
    for rank in range(1, top_l):
        local[:, rank] = rng.integers(0, token_count, size=token_count)
    raw_score = rng.normal(0.45, 0.025, size=(token_count, top_l)).astype(np.float32)
    raw_score[:, 0] = 0.62
    endpoint = np.zeros_like(local, dtype=np.uint8)
    texture = np.full(token_count, 0.9, dtype=np.float32)
    rgb = rng.random((token_count, 3), dtype=np.float32)
    result = vote_and_verify(
        frame_id=5,
        frame_count=11,
        raw_index=local,
        raw_score=raw_score,
        reference_endpoint=endpoint,
        reference_local_id=local,
        texture=texture,
        token_rgb=rgb,
        grid_h=grid_h,
        grid_w=grid_w,
        candidate_temperature=float(config["processing"]["candidate_temperature"]),
        vote_config=config["voting"],
        hypothesis="H1",
    )
    assert result.candidate_weight.shape == (token_count, top_l)
    assignment_top_k = int(config["voting"].get("assignment_top_k", 3))
    assert result.candidate_slot_id.shape == (token_count, top_l, assignment_top_k)
    assert 0 < int(result.slot_active.sum()) <= int(config["voting"]["slot_capacity"])
    assert any(event["type"] == "birth" for event in result.lifecycle_events), result.lifecycle_events
    assert np.isfinite(result.candidate_weight).all()
    assert np.allclose(result.candidate_weight.sum(-1), 1.0, atol=2e-4)
    assert result.sparse_mask.sum() > 0

    # Lifecycle regression checks use direct, controlled vote fields so every
    # required transition from the v1.1 plan is executable and recorded.
    lifecycle_config = dict(config["voting"])
    lifecycle_config.update(
        {
            "slot_capacity": 32,
            "minimum_slot_mass": 2.0,
            "birth_minimum_mass": 3.0,
            "termination_patience": 1,
            "split_minimum_improvement": 0.20,
            "split_parameter_separation": 0.50,
            "merge_parameter_threshold": 0.60,
            "parameter_smoothing": 0.0,
        }
    )
    sample_count = 80
    positions = np.zeros((sample_count, 1, 2), dtype=np.float32)
    weights = np.ones((sample_count, 1), dtype=np.float32)
    split_velocity = np.zeros((sample_count, 1, 2), dtype=np.float32)
    split_velocity[: sample_count // 2, 0, 0] = -2.0
    split_velocity[sample_count // 2 :, 0, 0] = 2.0
    seeded = MotionSlotState(
        parameters=np.zeros((32, 3), dtype=np.float32),
        alive=np.asarray([True] + [False] * 31),
        active=np.asarray([True] + [False] * 31),
        age=np.asarray([3] + [0] * 31, dtype=np.int32),
        missed=np.zeros(32, dtype=np.int32),
        mass=np.zeros(32, dtype=np.float32),
        generation=np.asarray([1] + [0] * 31, dtype=np.int32),
    )
    split_state, _, _, _, _, _, split_events = _fit_dynamic_motion_slots(
        split_velocity, positions, weights, lifecycle_config, "H0", seeded
    )
    assert any(event["type"] == "split" for event in split_events), split_events

    merged_seed = MotionSlotState(
        parameters=np.zeros((32, 3), dtype=np.float32),
        alive=np.asarray([True, True] + [False] * 30),
        active=np.asarray([True, True] + [False] * 30),
        age=np.asarray([3, 3] + [0] * 30, dtype=np.int32),
        missed=np.zeros(32, dtype=np.int32),
        mass=np.zeros(32, dtype=np.float32),
        generation=np.asarray([1, 1] + [0] * 30, dtype=np.int32),
    )
    merge_velocity = np.zeros((sample_count, 1, 2), dtype=np.float32)
    merged_state, _, _, _, _, _, merge_events = _fit_dynamic_motion_slots(
        merge_velocity, positions, weights, lifecycle_config, "H0", merged_seed
    )
    assert any(event["type"] == "merge" for event in merge_events), merge_events

    no_votes = np.zeros_like(weights)
    missing_state, _, _, _, _, _, missing_events = _fit_dynamic_motion_slots(
        merge_velocity, positions, no_votes, lifecycle_config, "H0", merged_state
    )
    terminated_state, _, _, _, _, _, termination_events = _fit_dynamic_motion_slots(
        merge_velocity, positions, no_votes, lifecycle_config, "H0", missing_state
    )
    assert any(event["type"] == "disappearance" for event in missing_events), missing_events
    assert any(event["type"] == "termination" for event in termination_events), termination_events
    print(
        json.dumps(
            {
                "status": "passed",
                "candidate_shape": list(result.candidate_weight.shape),
                "candidate_slot_sparse_shape": list(result.candidate_slot_id.shape),
                "active_slots": int(result.slot_active.sum()),
                "lifecycle_checks": ["birth", "split", "merge", "disappearance", "termination"],
                "mean_confidence": float(result.confidence.mean()),
                "mean_sparse_candidates": float(result.sparse_mask.sum(-1).mean()),
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    config, config_dir = load_config(args.config)
    frozen_marker = run_root(config, config_dir) / ".frozen.json"
    if frozen_marker.is_file() and args.command in {
        "prepare", "extract", "candidates", "vote", "render", "metrics", "run"
    }:
        raise RuntimeError(
            f"Run {config['run_id']} is frozen at {frozen_marker}; use a new run_id for any writes"
        )
    if args.method:
        requested_methods = set(args.method)
        unknown_methods = requested_methods.difference(config["ablation_methods"])
        if unknown_methods:
            raise ValueError(f"Unknown ablation methods: {sorted(unknown_methods)}")
        config = dict(config)
        config["ablation_methods"] = {
            method: spec
            for method, spec in config["ablation_methods"].items()
            if method in requested_methods
        }
    windows = selected_windows(config, args.window)
    if args.command == "validate":
        validate(config, config_dir, windows)
    elif args.command == "self-test":
        self_test(config)
    elif args.command == "prepare":
        validate(config, config_dir, windows)
        prepare(config, config_dir, windows, args.max_frames)
    elif args.command == "extract":
        extract_all(config, config_dir, windows)
    elif args.command == "candidates":
        candidates_all(config, config_dir, windows)
    elif args.command == "vote":
        vote_all(config, config_dir, windows)
    elif args.command == "render":
        render_all(config, config_dir, windows)
    elif args.command == "metrics":
        metrics_all(config, config_dir, windows)
    elif args.command == "run":
        validate(config, config_dir, windows)
        prepare(config, config_dir, windows, args.max_frames)
        extract_all(config, config_dir, windows)
        candidates_all(config, config_dir, windows)
        vote_all(config, config_dir, windows)
        render_all(config, config_dir, windows)
        metrics_all(config, config_dir, windows)


if __name__ == "__main__":
    main()
