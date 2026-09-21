from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core import token_grid, write_json


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "demo0_v11_d0real"
MODE = "FL"
METHOD = "P5"


@dataclass(frozen=True)
class Query:
    window_id: str
    frame_id: int
    x: int
    y: int
    category: str
    note: str


QUERIES = (
    Query("simple_multidepth_0180_0300", 60, 10, 14, "unique_texture", "window detail"),
    Query("simple_multidepth_0180_0300", 60, 10, 8, "repeated_texture", "wood roof repetition"),
    Query("simple_multidepth_0180_0300", 60, 19, 10, "building_corner", "near building edge"),
    Query("simple_multidepth_0180_0300", 60, 24, 8, "multi_depth", "tall background structure"),
    Query("simple_multidepth_0180_0300", 60, 42, 18, "multi_depth", "far house"),
    Query("simple_multidepth_0180_0300", 60, 0, 10, "tail_only", "left boundary newly visible by proxy"),
    Query("simple_multidepth_0180_0300", 60, 18, 8, "neither_visible", "endpoint visibility stress case"),
    Query("simple_multidepth_0180_0300", 60, 40, 5, "sky_low_texture", "uniform sky"),
    Query("simple_occlusion_0360_0480", 60, 21, 14, "unique_texture", "window detail behind foreground"),
    Query("simple_occlusion_0360_0480", 60, 20, 8, "repeated_texture", "wood roof repetition"),
    Query("simple_occlusion_0360_0480", 60, 30, 11, "building_corner", "house/tower boundary"),
    Query("simple_occlusion_0360_0480", 60, 42, 18, "multi_depth", "far house"),
    Query("simple_occlusion_0360_0480", 60, 13, 11, "occlusion", "foreground trunk crossing house"),
    Query("simple_occlusion_0360_0480", 60, 12, 10, "tail_only", "trunk region, tail-visible proxy"),
    Query("simple_occlusion_0360_0480", 60, 44, 22, "neither_visible", "right foreground stress case"),
    Query("simple_occlusion_0360_0480", 60, 40, 5, "sky_low_texture", "uniform sky"),
    Query("asymmetric_outbound_0060_0120", 30, 35, 14, "unique_texture", "torch/door detail"),
    Query("asymmetric_outbound_0060_0120", 30, 24, 10, "repeated_texture", "cobblestone repetition"),
    Query("asymmetric_outbound_0060_0120", 30, 42, 7, "building_corner", "roof corner"),
    Query("asymmetric_outbound_0060_0120", 30, 34, 19, "multi_depth", "front foundation"),
    Query("asymmetric_outbound_0060_0120", 30, 40, 11, "tail_only", "right wall, tail-visible proxy"),
    Query("asymmetric_outbound_0060_0120", 30, 21, 16, "neither_visible", "left silhouette boundary"),
    Query("asymmetric_outbound_0060_0120", 30, 28, 14, "unique_texture", "furnace detail"),
    Query("asymmetric_outbound_0060_0120", 30, 10, 5, "sky_low_texture", "uniform sky"),
)


# Human decisions made from the three-pane sheets at the native token scale.
# Ranks below are P5 verified ranks (V1 is the largest verified weight).  Several
# ranks may be correct when neighbouring 16 px tokens overlap the same physical
# detail.  The auxiliary flow suggestions were explicitly ignored when they
# disagreed with the visible scene content.
MANUAL_OVERRIDES: dict[int, dict[str, Any]] = {
    1: {"first_visible": True, "last_visible": True, "correct_verified": [1, 2, 3, 4, 5, 6, 8], "should_reject": False, "note": "Window corner remains visible at both endpoints; flow visibility failed."},
    2: {"first_visible": True, "last_visible": True, "correct_verified": [3, 4], "should_reject": False, "note": "Repeated roof texture; two token boxes agree with the physical roof patch."},
    3: {"first_visible": True, "last_visible": True, "correct_verified": [1, 3], "should_reject": False, "note": "Near-building eave edge remains visible; flow visibility failed."},
    4: {"first_visible": True, "last_visible": True, "correct_verified": [1, 2, 3, 4, 5, 6, 7], "should_reject": False, "note": "Distinct coloured tower is visible at both endpoints."},
    5: {"first_visible": True, "last_visible": True, "correct_verified": [1, 2, 3, 4, 5, 6, 7, 8], "should_reject": False, "note": "All displayed candidates overlap the same distant roof detail at token resolution."},
    6: {"first_visible": False, "last_visible": True, "correct_verified": [1, 2, 8], "should_reject": False, "note": "Left-boundary roof token is genuinely tail-only in this window."},
    7: {"category": "proxy_visibility_failure", "first_visible": True, "last_visible": True, "correct_verified": [1, 2, 4, 5, 6], "should_reject": False, "note": "The roof/white-column boundary is visible at both endpoints; this is not a neither-visible case."},
    8: {"first_visible": True, "last_visible": True, "correct_verified": [], "should_reject": True, "note": "Uniform sky is visible, but no physical token correspondence is visually identifiable."},
    9: {"first_visible": True, "last_visible": True, "correct_verified": [1, 2], "should_reject": False, "note": "Window corner remains visible behind the moving foreground trunk."},
    10: {"first_visible": True, "last_visible": True, "correct_verified": [1], "should_reject": False, "note": "Only the leading roof candidate is a conservative physical match."},
    11: {"first_visible": True, "last_visible": True, "correct_verified": [1, 2, 4, 5], "should_reject": False, "note": "House/tower boundary is visible and represented by four neighbouring token boxes."},
    12: {"first_visible": True, "last_visible": True, "correct_verified": [1, 2], "should_reject": False, "note": "Distant roof remains visible; composed-flow visibility was a false negative."},
    13: {"first_visible": True, "last_visible": True, "correct_verified": [], "should_reject": True, "note": "Foreground trunk token has no trunk match in Top-8; P5 incorrectly accepts roof-edge candidates."},
    14: {"first_visible": True, "last_visible": True, "correct_verified": [], "should_reject": True, "note": "Foreground trunk token again misses the physical match; P5 is a false accept."},
    15: {"category": "proxy_visibility_failure", "first_visible": True, "last_visible": True, "correct_verified": [2, 5, 7], "should_reject": False, "note": "Foreground grass is visible at both endpoints; this is not a neither-visible case."},
    16: {"first_visible": True, "last_visible": True, "correct_verified": [], "should_reject": True, "note": "Uniform sky is visible, but its physical correspondence is unidentifiable."},
    17: {"first_visible": True, "last_visible": True, "correct_verified": [1, 2, 3, 4, 5, 6, 8], "should_reject": False, "note": "Selected token is the right wooden wall, not the torch; neighbouring boxes match the same wall strip."},
    18: {"first_visible": True, "last_visible": True, "correct_verified": [1, 2, 3, 5, 7], "should_reject": False, "note": "Cobblestone silhouette is visible; repeated texture admits several neighbouring matches."},
    19: {"first_visible": True, "last_visible": True, "correct_verified": [1, 2, 3, 4, 5, 6, 7, 8], "should_reject": False, "note": "Stepped roof corner is plainly visible at both endpoints; flow visibility failed."},
    20: {"first_visible": True, "last_visible": True, "correct_verified": [2, 3, 6], "should_reject": False, "note": "Three candidates overlap the front-foundation detail at token resolution."},
    21: {"category": "proxy_visibility_failure", "first_visible": True, "last_visible": True, "correct_verified": [1, 5, 6], "should_reject": False, "note": "Right wall/eave token is visible at both endpoints; the tail-only proxy label was wrong."},
    22: {"category": "proxy_visibility_failure", "first_visible": True, "last_visible": True, "correct_verified": [1, 2, 3, 4, 5, 8], "should_reject": False, "note": "Left cobblestone silhouette is visible at both endpoints; this is not a neither-visible case."},
    23: {"first_visible": True, "last_visible": True, "correct_verified": [1, 2, 3, 4, 5, 6, 7, 8], "should_reject": False, "note": "Furnace stack is distinctive and represented by all displayed neighbouring candidates."},
    24: {"first_visible": True, "last_visible": True, "correct_verified": [], "should_reject": True, "note": "Uniform sky is visible, but no individual sky token can be verified across viewpoints."},
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_frame(path: Path, frame_id: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_id}: {path}")
    return frame


def visible(target: np.ndarray, reliability: float, grid_h: int, grid_w: int) -> bool:
    return bool(
        reliability >= 0.20
        and 0 <= target[0] <= grid_w - 1
        and 0 <= target[1] <= grid_h - 1
    )


def draw_token_box(image: np.ndarray, xy: tuple[int, int], stride: int, color: tuple[int, int, int], label: str) -> None:
    x, y = xy
    x0, y0 = x * stride, y * stride
    cv2.rectangle(image, (x0 + 1, y0 + 1), (x0 + stride - 2, y0 + stride - 2), color, 2)
    cv2.rectangle(image, (x0, max(0, y0 - 15)), (x0 + max(34, 7 * len(label)), y0), (7, 12, 20), -1)
    cv2.putText(image, label, (x0 + 2, max(11, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)


def crop(image: np.ndarray, xy: tuple[int, int], stride: int, size: int = 64) -> np.ndarray:
    cx, cy = int((xy[0] + 0.5) * stride), int((xy[1] + 0.5) * stride)
    padded = cv2.copyMakeBorder(image, size, size, size, size, cv2.BORDER_REFLECT_101)
    patch = padded[cy + size - size // 2 : cy + size + size // 2, cx + size - size // 2 : cx + size + size // 2]
    return cv2.resize(patch, (96, 96), interpolation=cv2.INTER_NEAREST)


def evaluate(query: Query, sheet_dir: Path, index: int) -> dict[str, Any]:
    directory = RUN / query.window_id
    info = read_json(directory / "window_manifest.json")
    grid_h = int(info["token_grid"]["height"])
    grid_w = int(info["token_grid"]["width"])
    stride = int(info["token_grid"]["stride"])
    token_id = query.y * grid_w + query.x
    total = int(info["frame_count"])
    mode_dir = directory / "modes" / MODE
    method_dir = mode_dir / METHOD

    raw_score = np.load(mode_dir / "raw_topl_score.npy", mmap_mode="r")
    endpoint = np.load(mode_dir / "reference_endpoint.npy", mmap_mode="r")
    local_id = np.load(mode_dir / "reference_local_id.npy", mmap_mode="r")
    verified = np.load(method_dir / "verified_topl_weight.npy", mmap_mode="r")
    sparse = np.load(method_dir / "sparse_reference_mask.npy", mmap_mode="r")
    confidence = np.load(method_dir / "token_confidence.npy", mmap_mode="r")
    reject_probability = np.load(method_dir / "reject_probability.npy", mmap_mode="r")
    token_slot = np.load(method_dir / "token_slot_probability.npy", mmap_mode="r")
    aux = directory / "auxiliary_track_reference"
    target_first = np.asarray(np.load(aux / "track_to_first_xy.npy", mmap_mode="r")[query.frame_id, token_id], dtype=np.float32)
    target_last = np.asarray(np.load(aux / "track_to_last_xy.npy", mmap_mode="r")[query.frame_id, token_id], dtype=np.float32)
    reliability_first = float(np.load(aux / "track_to_first_reliability.npy", mmap_mode="r")[query.frame_id, token_id])
    reliability_last = float(np.load(aux / "track_to_last_reliability.npy", mmap_mode="r")[query.frame_id, token_id])
    first_visible = visible(target_first, reliability_first, grid_h, grid_w)
    last_visible = visible(target_last, reliability_last, grid_h, grid_w)

    coords = token_grid(grid_h, grid_w)
    candidate_local = np.asarray(local_id[query.frame_id, token_id], dtype=np.int32)
    candidate_endpoint = np.asarray(endpoint[query.frame_id, token_id], dtype=np.uint8)
    candidate_xy = coords[candidate_local]
    correct = (
        ((candidate_endpoint == 0) & first_visible & (np.linalg.norm(candidate_xy - target_first[None], axis=-1) <= 1.25))
        | ((candidate_endpoint == 1) & last_visible & (np.linalg.norm(candidate_xy - target_last[None], axis=-1) <= 1.25))
    )
    verified_value = np.asarray(verified[query.frame_id, token_id], dtype=np.float32)
    verified_order = np.argsort(-verified_value)
    verified_rank = np.empty_like(verified_order)
    verified_rank[verified_order] = np.arange(len(verified_order)) + 1
    display = verified_order[:8]

    frames = [read_frame(directory / "window.mp4", frame_id) for frame_id in (0, query.frame_id, total - 1)]
    annotated = [frame.copy() for frame in frames]
    draw_token_box(annotated[1], (query.x, query.y), stride, (80, 255, 170), "QUERY")
    for rank in display:
        target_image = annotated[0] if candidate_endpoint[rank] == 0 else annotated[2]
        candidate = (int(candidate_xy[rank, 0]), int(candidate_xy[rank, 1]))
        color = (255, 170, 55) if candidate_endpoint[rank] == 0 else (55, 155, 255)
        draw_token_box(target_image, candidate, stride, color, f"V{int(verified_rank[rank])}/R{rank + 1}")
    for image, target, is_visible in ((annotated[0], target_first, first_visible), (annotated[2], target_last, last_visible)):
        if is_visible:
            center = (int((target[0] + 0.5) * stride), int((target[1] + 0.5) * stride))
            cv2.drawMarker(image, center, (255, 0, 255), cv2.MARKER_CROSS, 18, 2)

    header_h = 58
    panel = np.full((header_h + frames[0].shape[0] + 135, frames[0].shape[1] * 3, 3), 18, dtype=np.uint8)
    header = (
        f"{index:02d} | {query.window_id} | frame {query.frame_id} | token {token_id} ({query.x},{query.y}) | "
        f"{query.category} | Fvis={first_visible} Lvis={last_visible} | conf={float(confidence[query.frame_id, token_id]):.3f}"
    )
    cv2.putText(panel, header, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(panel, query.note, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 205, 225), 1, cv2.LINE_AA)
    for column, image in enumerate(annotated):
        panel[header_h : header_h + image.shape[0], column * image.shape[1] : (column + 1) * image.shape[1]] = image

    crop_y = header_h + frames[0].shape[0] + 4
    query_crop = crop(frames[1], (query.x, query.y), stride)
    panel[crop_y : crop_y + 96, 6 : 102] = query_crop
    cv2.putText(panel, "QUERY", (9, crop_y + 114), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 255, 170), 1, cv2.LINE_AA)
    for display_rank, rank in enumerate(display):
        source = frames[0] if candidate_endpoint[rank] == 0 else frames[2]
        xy = (int(candidate_xy[rank, 0]), int(candidate_xy[rank, 1]))
        tile = crop(source, xy, stride)
        x0 = 112 + display_rank * 104
        panel[crop_y : crop_y + 96, x0 : x0 + 96] = tile
        label = f"V{display_rank + 1}/R{rank + 1}{'F' if candidate_endpoint[rank] == 0 else 'L'}"
        if correct[rank]:
            cv2.rectangle(panel, (x0, crop_y), (x0 + 95, crop_y + 95), (0, 255, 255), 3)
            label += " OK"
        cv2.putText(panel, label, (x0, crop_y + 114), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (235, 235, 235), 1, cv2.LINE_AA)

    sheet_name = f"{index:02d}_{query.window_id}_{query.category}_f{query.frame_id:03d}_t{token_id:04d}.png"
    cv2.imwrite(str(sheet_dir / sheet_name), panel)

    local_slots = np.asarray(token_slot[max(0, query.frame_id - 2) : min(total, query.frame_id + 3), token_id], dtype=np.float32)
    selected_slot = int(np.asarray(token_slot[query.frame_id, token_id], dtype=np.float32).argmax())
    temporal_consistent = bool((local_slots.argmax(-1) == selected_slot).mean() >= 0.60)
    correct_raw = (np.flatnonzero(correct) + 1).astype(int).tolist()
    correct_verified = sorted(verified_rank[np.flatnonzero(correct)].astype(int).tolist())
    should_reject = query.category in {"sky_low_texture", "neither_visible"} or not correct.any()
    return {
        "run_id": "demo0_v11_d0real",
        "window_id": query.window_id,
        "frame_id": query.frame_id,
        "token_id": token_id,
        "token_xy": [query.x, query.y],
        "mode": MODE,
        "hypothesis": METHOD,
        "category": query.category,
        "first_visible": first_visible,
        "last_visible": last_visible,
        "correct_candidate_ranks": correct_raw,
        "correct_candidate_raw_ranks": correct_raw,
        "correct_candidate_verified_ranks": correct_verified,
        "should_reject": should_reject,
        "temporal_slot_consistent": temporal_consistent,
        "confidence": float(confidence[query.frame_id, token_id]),
        "reject_probability": float(reject_probability[query.frame_id, token_id]),
        "sparse_allowed_raw_ranks": (np.flatnonzero(np.asarray(sparse[query.frame_id, token_id], dtype=bool)) + 1).astype(int).tolist(),
        "raw_scores": np.asarray(raw_score[query.frame_id, token_id], dtype=np.float32).tolist(),
        "auxiliary_track": {
            "first_xy": target_first.tolist(),
            "last_xy": target_last.tolist(),
            "first_reliability": reliability_first,
            "last_reliability": reliability_last,
            "warning": "Adjacent-flow composition is only an aid, not ground truth.",
        },
        "review_sheet": sheet_name,
        "note": query.note,
        "review_status": "draft_pending_visual_confirmation",
    }


def render() -> None:
    annotation_dir = RUN / "annotations"
    sheet_dir = annotation_dir / "review_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    records = [evaluate(query, sheet_dir, index) for index, query in enumerate(QUERIES, 1)]
    write_json(
        annotation_dir / "draft_manual_queries.json",
        {
            "schema_version": "demo0-1.1-manual-annotations-draft",
            "frozen": False,
            "review_method": "three-pane visual review aided by disclosed adjacent-flow composition",
            "queries": records,
        },
    )
    print(f"Rendered {len(records)} review sheets to {sheet_dir}")


def apply_review() -> None:
    annotation_dir = RUN / "annotations"
    draft = read_json(annotation_dir / "draft_manual_queries.json")
    records = draft["queries"]
    if len(records) != len(MANUAL_OVERRIDES):
        raise RuntimeError("Manual decision count does not match the rendered query set")
    for index, record in enumerate(records, 1):
        override = MANUAL_OVERRIDES[index]
        verified_path = RUN / record["window_id"] / "modes" / MODE / METHOD / "verified_topl_weight.npy"
        verified = np.load(verified_path, mmap_mode="r")
        weights = np.asarray(verified[record["frame_id"], record["token_id"]], dtype=np.float32)
        order = np.argsort(-weights)
        correct_verified = [int(rank) for rank in override["correct_verified"]]
        correct_raw = sorted(int(order[rank - 1]) + 1 for rank in correct_verified)
        record.update({key: value for key, value in override.items() if key != "correct_verified"})
        record["correct_candidate_ranks"] = correct_raw
        record["correct_candidate_raw_ranks"] = correct_raw
        record["correct_candidate_verified_ranks"] = correct_verified
        record["review_status"] = "visually_confirmed"
        record["review_basis"] = "three-pane endpoint inspection at 16 px token resolution"
    draft["manual_review_note"] = (
        "No genuinely neither-endpoint-visible token was confirmed in these short windows. "
        "All three proxy-selected examples were reclassified as auxiliary visibility failures."
    )
    draft["coverage_gaps"] = ["confirmed_neither_endpoint_visible"]
    write_json(annotation_dir / "draft_manual_queries.json", draft)
    print(f"Applied visual decisions to {len(records)} draft queries")


def freeze() -> None:
    annotation_dir = RUN / "annotations"
    draft = read_json(annotation_dir / "draft_manual_queries.json")
    records = draft["queries"]
    if not records or any(record.get("review_status") != "visually_confirmed" for record in records):
        raise RuntimeError("Every draft query must be visually_confirmed before freezing")
    write_json(
        annotation_dir / "manual_queries.json",
        {
            "schema_version": "demo0-1.1-manual-annotations",
            "frozen": True,
            "review_method": draft["review_method"],
            "reviewer": "assistant visual audit",
            "manual_review_note": draft.get("manual_review_note", ""),
            "coverage_gaps": draft.get("coverage_gaps", []),
            "queries": records,
        },
    )
    print(f"Frozen {len(records)} manual queries")


def summarize_subset(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [record for record in records if not record["should_reject"]]
    rejected = [record for record in records if record["should_reject"]]
    reciprocal_raw: list[float] = []
    reciprocal_verified: list[float] = []
    correct_weights: list[float] = []
    hit1_raw = hit8_raw = hit16_raw = 0
    hit1_verified = hit8_verified = 0
    reject_correct = 0
    false_accepts: list[int] = []
    false_rejects: list[int] = []
    for record in records:
        raw = sorted(int(rank) for rank in record["correct_candidate_raw_ranks"])
        verified = sorted(int(rank) for rank in record["correct_candidate_verified_ranks"])
        predicted_reject = len(record["sparse_allowed_raw_ranks"]) == 0
        reject_correct += int(predicted_reject == bool(record["should_reject"]))
        if record["should_reject"] and not predicted_reject:
            false_accepts.append(int(record["token_id"]))
        if not record["should_reject"] and predicted_reject:
            false_rejects.append(int(record["token_id"]))
        if record["should_reject"]:
            continue
        best_raw = raw[0] if raw else None
        best_verified = verified[0] if verified else None
        reciprocal_raw.append(0.0 if best_raw is None else 1.0 / best_raw)
        reciprocal_verified.append(0.0 if best_verified is None else 1.0 / best_verified)
        hit1_raw += int(best_raw is not None and best_raw <= 1)
        hit8_raw += int(best_raw is not None and best_raw <= 8)
        hit16_raw += int(best_raw is not None and best_raw <= 16)
        hit1_verified += int(best_verified is not None and best_verified <= 1)
        hit8_verified += int(best_verified is not None and best_verified <= 8)
        window = RUN / record["window_id"] / "modes" / MODE / METHOD
        weights = np.load(window / "verified_topl_weight.npy", mmap_mode="r")
        vector = np.asarray(weights[record["frame_id"], record["token_id"]], dtype=np.float32)
        correct_weights.append(float(sum(vector[rank - 1] for rank in raw)))
    denom = max(1, len(eligible))
    return {
        "query_count": len(records),
        "matchable_query_count": len(eligible),
        "should_reject_count": len(rejected),
        "P0_raw": {
            "MRR": float(np.mean(reciprocal_raw)) if reciprocal_raw else 0.0,
            "Hit@1": hit1_raw / denom,
            "Hit@8": hit8_raw / denom,
            "Hit@16": hit16_raw / denom,
        },
        "P5_verified": {
            "MRR": float(np.mean(reciprocal_verified)) if reciprocal_verified else 0.0,
            "Hit@1": hit1_verified / denom,
            "Hit@8": hit8_verified / denom,
            "mean_correct_candidate_weight": float(np.mean(correct_weights)) if correct_weights else 0.0,
            "reject_accuracy": reject_correct / max(1, len(records)),
            "false_accept_token_ids": false_accepts,
            "false_reject_token_ids": false_rejects,
        },
    }


def metrics() -> None:
    annotation_dir = RUN / "annotations"
    frozen = read_json(annotation_dir / "manual_queries.json")
    if not frozen.get("frozen"):
        raise RuntimeError("Manual query set must be frozen before metrics")
    records = frozen["queries"]
    by_window = {
        window_id: summarize_subset([record for record in records if record["window_id"] == window_id])
        for window_id in sorted({record["window_id"] for record in records})
    }
    report = {
        "schema_version": "demo0-1.1-manual-metrics",
        "scope": "D0-Real, FL reference bank, P0 versus P5 on 24 visually audited queries",
        "warning": "Small diagnostic set; repeated textures are judged at 16 px token resolution and are not statistical ground truth.",
        "coverage_gaps": frozen.get("coverage_gaps", []),
        "overall": summarize_subset(records),
        "by_window": by_window,
    }
    write_json(annotation_dir / "manual_metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render or freeze Demo0 v1.1 manual audit queries")
    parser.add_argument("command", choices=("render", "apply-review", "freeze", "metrics"))
    args = parser.parse_args()
    if args.command == "render":
        render()
    elif args.command == "apply-review":
        apply_review()
    elif args.command == "freeze":
        freeze()
    else:
        metrics()


if __name__ == "__main__":
    main()
