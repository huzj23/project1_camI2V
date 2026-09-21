from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


EPS = 1e-8


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass
class VoteResult:
    candidate_weight: np.ndarray
    candidate_slot_id: np.ndarray
    candidate_slot_probability: np.ndarray
    token_slot: np.ndarray
    slot_parameters: np.ndarray
    slot_active: np.ndarray
    slot_mass: np.ndarray
    slot_age: np.ndarray
    slot_missed: np.ndarray
    candidate_outlier_probability: np.ndarray
    lifecycle_events: list[dict[str, Any]]
    slot_state: "MotionSlotState"
    reject_probability: np.ndarray
    confidence: np.ndarray
    sparse_mask: np.ndarray
    predicted_reference_xy: np.ndarray
    selected_endpoint: np.ndarray


@dataclass
class MotionSlotState:
    """Persistent dynamic-slot state.

    ``alive`` includes temporarily missing slots kept during the occlusion
    grace period. ``active`` means that the slot has support in this frame.
    Slot ids are stable array indices and are therefore safe to store in the
    sparse on-disk representation.
    """

    parameters: np.ndarray
    alive: np.ndarray
    active: np.ndarray
    age: np.ndarray
    missed: np.ndarray
    mass: np.ndarray
    generation: np.ndarray


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_video(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    info = VideoInfo(
        path=path.resolve(),
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    return info


def read_frame(path: Path, frame_id: int, width: int, height: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read {path} frame {frame_id}")
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def token_grid(grid_h: int, grid_w: int) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.arange(grid_h, dtype=np.float32),
        np.arange(grid_w, dtype=np.float32),
        indexing="ij",
    )
    return np.stack((xx, yy), axis=-1).reshape(-1, 2)


def token_centers(grid_h: int, grid_w: int, stride: int) -> np.ndarray:
    return (token_grid(grid_h, grid_w) + 0.5) * float(stride)


def _sample_image(array: np.ndarray, xy: np.ndarray) -> np.ndarray:
    map_x = xy[:, 0].astype(np.float32).reshape(-1, 1)
    map_y = xy[:, 1].astype(np.float32).reshape(-1, 1)
    sampled = cv2.remap(
        array,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return sampled.reshape((xy.shape[0],) + array.shape[2:])


def _fixed_sift(
    gray: np.ndarray,
    centers: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    sift = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=0.01,
        edgeThreshold=12,
        sigma=1.4,
    )
    keypoints = [
        cv2.KeyPoint(float(x), float(y), float(patch_size), 0.0)
        for x, y in centers
    ]
    returned, descriptor = sift.compute(gray, keypoints)
    output = np.zeros((centers.shape[0], 128), dtype=np.float32)
    if descriptor is None or returned is None:
        return output
    coordinate_to_id = {
        (round(float(x), 3), round(float(y), 3)): index
        for index, (x, y) in enumerate(centers)
    }
    for keypoint, value in zip(returned, descriptor, strict=False):
        key = (round(float(keypoint.pt[0]), 3), round(float(keypoint.pt[1]), 3))
        index = coordinate_to_id.get(key)
        if index is not None:
            value = value.astype(np.float32)
            value /= max(float(value.sum()), EPS)
            output[index] = np.sqrt(np.maximum(value, 0.0))
    output /= np.maximum(np.linalg.norm(output, axis=-1, keepdims=True), EPS)
    return output


def dense_token_descriptor(
    frame_bgr: np.ndarray,
    grid_h: int,
    grid_w: int,
    stride: int,
    patch_sizes: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return descriptor, token RGB and texture confidence.

    The descriptor has a stated, bounded local support. It is intentionally an
    interpretable D0-Real representation, not a VACE-token substitute.
    """

    centers = token_centers(grid_h, grid_w, stride)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0

    components: list[np.ndarray] = []
    scale_weight = 1.0 / math.sqrt(max(len(patch_sizes), 1))
    for patch_size in patch_sizes:
        components.append(_fixed_sift(gray, centers, patch_size) * scale_weight)

    kernel = max(3, int(round(stride * 1.5)) | 1)
    rgb_mean_image = cv2.boxFilter(rgb, -1, (kernel, kernel), normalize=True)
    rgb_sq_image = cv2.boxFilter(rgb * rgb, -1, (kernel, kernel), normalize=True)
    lab_mean_image = cv2.boxFilter(lab, -1, (kernel, kernel), normalize=True)
    rgb_mean = _sample_image(rgb_mean_image, centers)
    rgb_var = np.maximum(_sample_image(rgb_sq_image, centers) - rgb_mean * rgb_mean, 0.0)
    rgb_std = np.sqrt(rgb_var)
    lab_mean = _sample_image(lab_mean_image, centers)
    color = np.concatenate((rgb_mean, rgb_std, lab_mean), axis=-1)
    color /= np.maximum(np.linalg.norm(color, axis=-1, keepdims=True), EPS)
    components.append(color * 0.70)

    descriptor = np.concatenate(components, axis=-1).astype(np.float32)
    descriptor /= np.maximum(np.linalg.norm(descriptor, axis=-1, keepdims=True), EPS)

    gray_float = gray.astype(np.float32) / 255.0
    mean = cv2.boxFilter(gray_float, -1, (kernel, kernel), normalize=True)
    mean_sq = cv2.boxFilter(gray_float * gray_float, -1, (kernel, kernel), normalize=True)
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    gx = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.boxFilter(np.sqrt(gx * gx + gy * gy), -1, (kernel, kernel), normalize=True)
    texture_raw = _sample_image(local_std, centers) + 0.35 * _sample_image(gradient, centers)
    texture = np.clip(texture_raw.reshape(-1) / 0.18, 0.0, 1.0).astype(np.float32)
    return descriptor, rgb_mean.astype(np.float32), texture


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponential = np.exp(np.clip(shifted, -80.0, 0.0))
    return exponential / np.maximum(exponential.sum(axis=axis, keepdims=True), EPS)


def top_candidates(
    query: np.ndarray,
    reference: np.ndarray,
    top_l: int,
    block_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    count = query.shape[0]
    indices = np.empty((count, top_l), dtype=np.int32)
    scores = np.empty((count, top_l), dtype=np.float32)
    reference_t = reference.astype(np.float32).T
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        similarity = query[start:stop].astype(np.float32) @ reference_t
        unsorted = np.argpartition(similarity, -top_l, axis=-1)[:, -top_l:]
        candidate_score = np.take_along_axis(similarity, unsorted, axis=-1)
        order = np.argsort(-candidate_score, axis=-1)
        indices[start:stop] = np.take_along_axis(unsorted, order, axis=-1)
        scores[start:stop] = np.take_along_axis(candidate_score, order, axis=-1)
    return indices, scores


def candidate_statistics(scores: np.ndarray, temperature: float) -> dict[str, np.ndarray]:
    weights = softmax(scores / max(temperature, EPS), axis=-1)
    entropy = -(weights * np.log(np.maximum(weights, EPS))).sum(-1)
    entropy /= math.log(max(scores.shape[-1], 2))
    margin = scores[:, 0] - scores[:, 1]
    return {
        "weight": weights.astype(np.float32),
        "entropy": entropy.astype(np.float32),
        "margin": margin.astype(np.float32),
    }


def adjacent_flow_to_previous(
    previous_bgr: np.ndarray,
    current_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    previous_gray = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    estimator.setUseSpatialPropagation(True)
    flow_current_previous = estimator.calc(current_gray, previous_gray, None)
    reverse_estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    reverse_estimator.setUseSpatialPropagation(True)
    flow_previous_current = reverse_estimator.calc(previous_gray, current_gray, None)
    return flow_current_previous, flow_previous_current


def adjacent_token_map(
    flow_current_previous: np.ndarray,
    flow_previous_current: np.ndarray,
    grid_h: int,
    grid_w: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    centers = token_centers(grid_h, grid_w, stride)
    flow = _sample_image(flow_current_previous, centers)
    previous_pixels = centers + flow
    previous_grid_xy = previous_pixels / float(stride) - 0.5
    reverse = _sample_image(flow_previous_current, previous_pixels)
    fb_error = np.linalg.norm(flow + reverse, axis=-1)
    inside = (
        (previous_pixels[:, 0] >= 0)
        & (previous_pixels[:, 0] <= flow_current_previous.shape[1] - 1)
        & (previous_pixels[:, 1] >= 0)
        & (previous_pixels[:, 1] <= flow_current_previous.shape[0] - 1)
    )
    reliability = np.exp(-fb_error / 2.0) * inside.astype(np.float32)
    return previous_grid_xy.astype(np.float32), reliability.astype(np.float32)


def sample_token_grid(
    values: np.ndarray,
    sample_xy: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    map_x = sample_xy[:, 0].astype(np.float32).reshape(grid_h, grid_w)
    map_y = sample_xy[:, 1].astype(np.float32).reshape(grid_h, grid_w)
    value_image = values.reshape(grid_h, grid_w, -1).astype(np.float32)
    return cv2.remap(
        value_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).reshape(values.shape).astype(np.float32)


def warp_previous_token_values(
    previous_values: np.ndarray,
    flow_current_previous: np.ndarray,
    flow_previous_current: np.ndarray,
    grid_h: int,
    grid_w: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    previous_grid_xy, reliability = adjacent_token_map(
        flow_current_previous,
        flow_previous_current,
        grid_h,
        grid_w,
        stride,
    )
    warped = sample_token_grid(previous_values, previous_grid_xy, grid_h, grid_w)
    return warped.astype(np.float32), reliability.astype(np.float32)


def _predict_velocity(parameters: np.ndarray, position: np.ndarray) -> np.ndarray:
    scale_rate = parameters[:, 0]
    translation = parameters[:, 1:]
    return position[:, None, :] * scale_rate[None, :, None] + translation[None, :, :]


def _parameter_cost(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    probes = np.asarray(((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (0.0, 0.0)), dtype=np.float32)
    previous_field = _predict_velocity(previous, probes).transpose(1, 0, 2)
    current_field = _predict_velocity(current, probes).transpose(1, 0, 2)
    return np.linalg.norm(previous_field[:, None] - current_field[None], axis=-1).mean(-1)


def empty_motion_slot_state(capacity: int) -> MotionSlotState:
    return MotionSlotState(
        parameters=np.zeros((capacity, 3), dtype=np.float32),
        alive=np.zeros(capacity, dtype=bool),
        active=np.zeros(capacity, dtype=bool),
        age=np.zeros(capacity, dtype=np.int32),
        missed=np.zeros(capacity, dtype=np.int32),
        mass=np.zeros(capacity, dtype=np.float32),
        generation=np.zeros(capacity, dtype=np.int32),
    )


def _parameter_limit(flat_v: np.ndarray, flat_w: np.ndarray) -> float:
    valid = np.isfinite(flat_v).all(-1) & (flat_w > EPS)
    if not valid.any():
        return 2.0
    magnitude = np.linalg.norm(flat_v[valid], axis=-1)
    order = np.argsort(magnitude)
    cumulative = np.cumsum(flat_w[valid][order])
    cutoff = 0.97 * float(cumulative[-1])
    index = min(int(np.searchsorted(cumulative, cutoff)), len(order) - 1)
    return max(2.0, 1.5 * float(magnitude[order[index]]) + 1.0)


def _fit_one_parameter(
    flat_v: np.ndarray,
    flat_p: np.ndarray,
    weight: np.ndarray,
    hypothesis: str,
    limit: float,
    fallback: Optional[np.ndarray] = None,
) -> np.ndarray:
    total = float(weight.sum())
    value = np.zeros(3, dtype=np.float32) if fallback is None else np.asarray(fallback, dtype=np.float32).copy()
    if total <= 1e-5:
        return value
    if hypothesis == "H0":
        value[0] = 0.0
        value[1:] = (weight[:, None] * flat_v).sum(0) / total
    else:
        px, py = flat_p[:, 0], flat_p[:, 1]
        vx, vy = flat_v[:, 0], flat_v[:, 1]
        matrix = np.asarray(
            (
                ((weight * (px * px + py * py)).sum(), (weight * px).sum(), (weight * py).sum()),
                ((weight * px).sum(), total, 0.0),
                ((weight * py).sum(), 0.0, total),
            ),
            dtype=np.float64,
        )
        rhs = np.asarray(
            (
                (weight * (px * vx + py * vy)).sum(),
                (weight * vx).sum(),
                (weight * vy).sum(),
            ),
            dtype=np.float64,
        )
        ridge = np.diag((3e-2 * total, 1e-4 * total, 1e-4 * total))
        try:
            value = np.linalg.solve(matrix + ridge, rhs).astype(np.float32)
        except np.linalg.LinAlgError:
            pass
    value[0] = float(np.clip(value[0], -limit, limit))
    value[1:] = np.clip(value[1:], -limit, limit)
    return value


def _dynamic_em(
    flat_v: np.ndarray,
    flat_p: np.ndarray,
    flat_w: np.ndarray,
    parameters: np.ndarray,
    iterations: int,
    sigma: float,
    hypothesis: str,
    limit: float,
    outlier_prior: float,
    outlier_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit current hypotheses while preserving an explicit no-slot channel."""
    if len(parameters) == 0:
        shape = (len(flat_v), 0)
        return (
            parameters.astype(np.float32),
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
            np.ones(len(flat_v), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    parameters = np.asarray(parameters, dtype=np.float32).copy()
    slot_prior = np.full(len(parameters), (1.0 - outlier_prior) / len(parameters), dtype=np.float32)
    outlier_likelihood = float(outlier_prior) * math.exp(
        -(outlier_threshold * outlier_threshold) / (2.0 * sigma * sigma)
    )
    for _ in range(max(1, iterations)):
        predicted = _predict_velocity(parameters, flat_p)
        residual_sq = ((flat_v[:, None, :] - predicted) ** 2).sum(-1)
        fit = np.exp(-residual_sq / (2.0 * sigma * sigma))
        likelihood = fit * slot_prior[None, :]
        denominator = likelihood.sum(-1) + outlier_likelihood
        assignment = likelihood / np.maximum(denominator[:, None], EPS)
        outlier = outlier_likelihood / np.maximum(denominator, EPS)
        weighted = flat_w[:, None] * assignment
        mass = weighted.sum(0)
        for index in range(len(parameters)):
            parameters[index] = _fit_one_parameter(
                flat_v, flat_p, weighted[:, index], hypothesis, limit, parameters[index]
            )
        slot_prior = (mass + 1e-3) / max(float(mass.sum()) + 1e-3 * len(parameters), EPS)
        slot_prior *= 1.0 - outlier_prior
    predicted = _predict_velocity(parameters, flat_p)
    residual_sq = ((flat_v[:, None, :] - predicted) ** 2).sum(-1)
    fit = np.exp(-residual_sq / (2.0 * sigma * sigma))
    likelihood = fit * slot_prior[None, :]
    denominator = likelihood.sum(-1) + outlier_likelihood
    assignment = likelihood / np.maximum(denominator[:, None], EPS)
    outlier = outlier_likelihood / np.maximum(denominator, EPS)
    mass = (flat_w[:, None] * assignment).sum(0)
    return parameters, assignment.astype(np.float32), fit.astype(np.float32), outlier.astype(np.float32), mass.astype(np.float32)


def _weighted_residual_seed(
    flat_v: np.ndarray,
    flat_p: np.ndarray,
    unexplained_weight: np.ndarray,
    hypothesis: str,
    limit: float,
) -> np.ndarray:
    if float(unexplained_weight.sum()) <= EPS:
        return np.zeros(3, dtype=np.float32)
    order = np.argsort(unexplained_weight)
    keep = order[-min(256, len(order)) :]
    seed_weight = unexplained_weight[keep]
    return _fit_one_parameter(flat_v[keep], flat_p[keep], seed_weight, hypothesis, limit)


def _allocate_slot_id(state: MotionSlotState, reserved: set[int]) -> int:
    free = np.flatnonzero(~state.alive)
    for value in free:
        slot_id = int(value)
        if slot_id not in reserved:
            if state.generation[slot_id] > 0:
                state.generation[slot_id] += 1
            else:
                state.generation[slot_id] = 1
            return slot_id
    return -1


def _fit_dynamic_motion_slots(
    velocity: np.ndarray,
    position: np.ndarray,
    vote_weight: np.ndarray,
    vote_config: dict[str, Any],
    hypothesis: str,
    previous_state: Optional[MotionSlotState],
) -> tuple[
    MotionSlotState,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    """Generate, verify, split, merge and retire motion hypotheses.

    The slot capacity is only an upper bound. The number of fitted components
    follows the vote evidence in each frame; votes may remain unassigned.
    """
    capacity = int(vote_config["slot_capacity"])
    state = empty_motion_slot_state(capacity) if previous_state is None else MotionSlotState(
        parameters=np.asarray(previous_state.parameters, dtype=np.float32).copy(),
        alive=np.asarray(previous_state.alive, dtype=bool).copy(),
        active=np.zeros(capacity, dtype=bool),
        age=np.asarray(previous_state.age, dtype=np.int32).copy(),
        missed=np.asarray(previous_state.missed, dtype=np.int32).copy(),
        mass=np.zeros(capacity, dtype=np.float32),
        generation=np.asarray(previous_state.generation, dtype=np.int32).copy(),
    )
    flat_v = velocity.reshape(-1, 2).astype(np.float32)
    flat_p = position.reshape(-1, 2).astype(np.float32)
    flat_w = vote_weight.reshape(-1).astype(np.float32)
    valid = np.isfinite(flat_v).all(-1) & np.isfinite(flat_p).all(-1) & (flat_w > EPS)
    flat_w = np.where(valid, flat_w, 0.0).astype(np.float32)
    sigma = max(float(vote_config["residual_temperature"]), 0.05)
    limit = _parameter_limit(flat_v, flat_w)
    minimum_mass = float(vote_config.get("minimum_slot_mass", 3.0))
    birth_mass = float(vote_config.get("birth_minimum_mass", minimum_mass * 1.5))
    max_births = int(vote_config.get("max_births_per_frame", 12))
    max_splits = int(vote_config.get("max_splits_per_frame", 4))
    patience = int(vote_config.get("termination_patience", 3))
    iterations = int(vote_config["em_iterations"])
    outlier_prior = float(vote_config.get("outlier_prior", 0.12))
    outlier_threshold = float(vote_config.get("outlier_residual_threshold", 1.5))

    seed_ids = np.flatnonzero(state.alive).astype(np.int32).tolist()
    parameters = state.parameters[seed_ids].copy() if seed_ids else np.zeros((0, 3), dtype=np.float32)
    metadata: list[dict[str, Any]] = [
        {"origins": {int(slot_id)}, "split_from": None} for slot_id in seed_ids
    ]
    if len(parameters) == 0 and float(flat_w.sum()) > EPS:
        parameters = _weighted_residual_seed(flat_v, flat_p, flat_w, hypothesis, limit)[None, :]
        metadata = [{"origins": set(), "split_from": None}]

    parameters, assignment, fit, outlier, mass = _dynamic_em(
        flat_v, flat_p, flat_w, parameters, iterations, sigma, hypothesis, limit,
        outlier_prior, outlier_threshold,
    )

    # Retire unsupported local components before proposing new ones. This is
    # what permits disappearance instead of forcing every old slot to persist.
    if len(parameters):
        keep = mass >= minimum_mass
        if not keep.any() and float(mass.sum()) > EPS:
            keep[int(mass.argmax())] = True
        parameters = parameters[keep]
        metadata = [item for item, use in zip(metadata, keep) if use]
        parameters, assignment, fit, outlier, mass = _dynamic_em(
            flat_v, flat_p, flat_w, parameters, iterations, sigma, hypothesis, limit,
            outlier_prior, outlier_threshold,
        )

    # A supported component with two clearly separated residual modes splits.
    split_events_local: list[tuple[Optional[int], int]] = []
    for _ in range(max_splits):
        if len(parameters) >= capacity or len(parameters) == 0:
            break
        best: Optional[tuple[float, int, np.ndarray, np.ndarray]] = None
        predicted = _predict_velocity(parameters, flat_p)
        for index in range(len(parameters)):
            base_weight = flat_w * assignment[:, index]
            if float(base_weight.sum()) < 2.0 * minimum_mass:
                continue
            residual = flat_v - predicted[:, index]
            centre0 = np.average(residual, axis=0, weights=np.maximum(base_weight, EPS))
            first = int(np.argmax(base_weight * np.linalg.norm(residual - centre0, axis=-1)))
            centre1 = residual[first]
            second = int(np.argmax(base_weight * np.linalg.norm(residual - centre1, axis=-1)))
            centres = np.stack((centre1, residual[second])).astype(np.float32)
            labels = np.zeros(len(flat_v), dtype=np.int32)
            for _kmeans in range(6):
                distance = ((residual[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
                labels = distance.argmin(-1)
                for side in (0, 1):
                    selected = base_weight * (labels == side)
                    if float(selected.sum()) > EPS:
                        centres[side] = (selected[:, None] * residual).sum(0) / selected.sum()
            side_weight = [base_weight * (labels == side) for side in (0, 1)]
            if min(float(value.sum()) for value in side_weight) < minimum_mass:
                continue
            child = [
                _fit_one_parameter(flat_v, flat_p, value, hypothesis, limit, parameters[index])
                for value in side_weight
            ]
            old_error = float((base_weight * ((flat_v - predicted[:, index]) ** 2).sum(-1)).sum())
            child_pred = _predict_velocity(np.stack(child), flat_p)
            new_error = float((base_weight * ((flat_v[:, None] - child_pred) ** 2).sum(-1).min(-1)).sum())
            improvement = (old_error - new_error) / max(old_error, EPS)
            separation = float(_parameter_cost(np.stack(child[:1]), np.stack(child[1:]))[0, 0])
            if improvement >= float(vote_config.get("split_minimum_improvement", 0.30)) and separation >= float(
                vote_config.get("split_parameter_separation", 0.75)
            ):
                score = improvement * float(base_weight.sum())
                if best is None or score > best[0]:
                    best = (score, index, child[0], child[1])
        if best is None:
            break
        _, index, first_child, second_child = best
        parent_origin = min(metadata[index]["origins"]) if metadata[index]["origins"] else None
        parameters[index] = first_child
        parameters = np.concatenate((parameters, second_child[None, :]), axis=0)
        metadata.append({"origins": set(), "split_from": parent_origin})
        split_events_local.append((parent_origin, len(parameters) - 1))
        parameters, assignment, fit, outlier, mass = _dynamic_em(
            flat_v, flat_p, flat_w, parameters, iterations, sigma, hypothesis, limit,
            outlier_prior, outlier_threshold,
        )

    # Unexplained vote mass spawns new hypotheses. The loop stops from data
    # support, not from a preselected desired number of slots.
    for _ in range(max_births):
        if len(parameters) >= capacity:
            break
        unexplained_weight = flat_w * outlier
        if float(unexplained_weight.sum()) < birth_mass:
            break
        proposal = _weighted_residual_seed(flat_v, flat_p, unexplained_weight, hypothesis, limit)
        if len(parameters) and float(_parameter_cost(parameters, proposal[None, :]).min()) < float(
            vote_config.get("birth_parameter_separation", 0.90)
        ):
            break
        parameters = np.concatenate((parameters, proposal[None, :]), axis=0)
        metadata.append({"origins": set(), "split_from": None})
        parameters, assignment, fit, outlier, mass = _dynamic_em(
            flat_v, flat_p, flat_w, parameters, iterations, sigma, hypothesis, limit,
            outlier_prior, outlier_threshold,
        )

    # Similar hypotheses represent one motion state and are merged. Origin sets
    # are retained so the lifecycle log can name every consumed persistent id.
    merge_origins: list[set[int]] = []
    while len(parameters) > 1:
        cost = _parameter_cost(parameters, parameters)
        np.fill_diagonal(cost, np.inf)
        left, right = np.unravel_index(int(np.argmin(cost)), cost.shape)
        if float(cost[left, right]) >= float(vote_config.get("merge_parameter_threshold", 0.55)):
            break
        if right < left:
            left, right = right, left
        combined_weight = flat_w * (assignment[:, left] + assignment[:, right])
        parameters[left] = _fit_one_parameter(
            flat_v, flat_p, combined_weight, hypothesis, limit, parameters[left]
        )
        origins = set(metadata[left]["origins"]) | set(metadata[right]["origins"])
        if len(origins) > 1:
            merge_origins.append(origins)
        metadata[left]["origins"] = origins
        if metadata[left]["split_from"] is None:
            metadata[left]["split_from"] = metadata[right]["split_from"]
        parameters = np.delete(parameters, right, axis=0)
        del metadata[right]
        parameters, assignment, fit, outlier, mass = _dynamic_em(
            flat_v, flat_p, flat_w, parameters, iterations, sigma, hypothesis, limit,
            outlier_prior, outlier_threshold,
        )

    # Birth/split proposals can lose their support after the joint refit. Do
    # not publish such numerical remnants as visible slots.
    if len(parameters):
        keep = mass >= minimum_mass
        if not keep.any() and float(mass.sum()) > EPS:
            keep[int(mass.argmax())] = True
        parameters = parameters[keep]
        metadata = [item for item, use in zip(metadata, keep) if use]
        parameters, assignment, fit, outlier, mass = _dynamic_em(
            flat_v, flat_p, flat_w, parameters, iterations, sigma, hypothesis, limit,
            outlier_prior, outlier_threshold,
        )

    events: list[dict[str, Any]] = []
    reserved: set[int] = set()
    local_ids: list[int] = []
    previous_missed = state.missed.copy()
    for local_index, item in enumerate(metadata):
        available_origins = [value for value in item["origins"] if value not in reserved]
        if available_origins:
            origin_parameters = state.parameters[available_origins]
            closest = int(np.argmin(_parameter_cost(origin_parameters, parameters[local_index : local_index + 1])[:, 0]))
            slot_id = int(available_origins[closest])
        else:
            # A proposal may be the rediscovery of a component that lost its
            # local EM seed. Match it back to an unused live/dormant id before
            # declaring a new birth. Split children deliberately get a fresh
            # id so parent/child lineage remains unambiguous.
            slot_id = -1
            if item["split_from"] is None:
                matchable = [int(value) for value in np.flatnonzero(state.alive) if int(value) not in reserved]
                if matchable:
                    costs = _parameter_cost(
                        state.parameters[matchable], parameters[local_index : local_index + 1]
                    )[:, 0]
                    best_match = int(np.argmin(costs))
                    if float(costs[best_match]) <= float(vote_config.get("identity_match_threshold", 1.25)):
                        slot_id = matchable[best_match]
            if slot_id < 0:
                slot_id = _allocate_slot_id(state, reserved)
        if slot_id < 0:
            # Capacity is a hard safety ceiling; excess proposals remain in the
            # outlier channel and are never silently folded into another slot.
            local_ids.append(-1)
            continue
        reserved.add(slot_id)
        local_ids.append(slot_id)
        was_alive = bool(state.alive[slot_id])
        was_missing = int(previous_missed[slot_id])
        if not was_alive:
            event_type = "split" if item["split_from"] is not None else "birth"
            event: dict[str, Any] = {"type": event_type, "slot_id": slot_id}
            if item["split_from"] is not None:
                event["parent_slot_id"] = int(item["split_from"])
            events.append(event)
        elif was_missing > 0:
            events.append({"type": "reappearance", "slot_id": slot_id, "missing_frames": was_missing})
        old_parameter = state.parameters[slot_id].copy()
        smoothing = float(vote_config.get("parameter_smoothing", 0.0)) if was_alive and was_missing == 0 else 0.0
        state.parameters[slot_id] = smoothing * old_parameter + (1.0 - smoothing) * parameters[local_index]
        state.alive[slot_id] = True
        state.active[slot_id] = True
        state.age[slot_id] = state.age[slot_id] + 1 if was_alive else 1
        state.missed[slot_id] = 0
        state.mass[slot_id] = mass[local_index]

    # Merge consumes all origin ids except the selected survivor.
    for origins in merge_origins:
        survivors = sorted(origins & reserved)
        if not survivors:
            continue
        target = survivors[0]
        sources = sorted(value for value in origins if value != target and state.alive[value])
        if sources:
            for source in sources:
                state.alive[source] = False
                state.active[source] = False
                state.missed[source] = patience + 1
            events.append({"type": "merge", "source_slot_ids": sources, "target_slot_id": target})

    for slot_id in np.flatnonzero(state.alive & ~state.active):
        state.missed[slot_id] += 1
        if state.missed[slot_id] == 1:
            events.append({"type": "disappearance", "slot_id": int(slot_id)})
        if state.missed[slot_id] > patience:
            state.alive[slot_id] = False
            events.append({"type": "termination", "slot_id": int(slot_id)})

    usable = np.asarray([index for index, slot_id in enumerate(local_ids) if slot_id >= 0], dtype=np.int32)
    slot_ids = np.asarray([local_ids[index] for index in usable], dtype=np.int32)
    assignment = assignment[:, usable] if len(usable) else np.zeros((len(flat_v), 0), dtype=np.float32)
    fit = fit[:, usable] if len(usable) else np.zeros((len(flat_v), 0), dtype=np.float32)
    mass = mass[usable] if len(usable) else np.zeros(0, dtype=np.float32)
    prior = mass / max(float(mass.sum()), EPS)
    return (
        state,
        slot_ids,
        assignment.reshape(velocity.shape[:-1] + (len(slot_ids),)).astype(np.float32),
        fit.reshape(velocity.shape[:-1] + (len(slot_ids),)).astype(np.float32),
        prior.astype(np.float32),
        outlier.reshape(velocity.shape[:-1]).astype(np.float32),
        events,
    )


def edge_aware_neighbor_average(
    values: np.ndarray,
    token_rgb: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    value_image = values.reshape(grid_h, grid_w, -1)
    rgb_image = token_rgb.reshape(grid_h, grid_w, 3)
    total = value_image.copy()
    denominator = np.ones((grid_h, grid_w, 1), dtype=np.float32)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        shifted_value = np.roll(value_image, shift=(dy, dx), axis=(0, 1))
        shifted_rgb = np.roll(rgb_image, shift=(dy, dx), axis=(0, 1))
        valid = np.ones((grid_h, grid_w), dtype=bool)
        if dy < 0:
            valid[dy:, :] = False
        elif dy > 0:
            valid[:dy, :] = False
        if dx < 0:
            valid[:, dx:] = False
        elif dx > 0:
            valid[:, :dx] = False
        color_distance = ((rgb_image - shifted_rgb) ** 2).sum(-1)
        weight = np.exp(-color_distance / 0.035) * valid.astype(np.float32)
        total += shifted_value * weight[..., None]
        denominator += weight[..., None]
    result = total / np.maximum(denominator, EPS)
    result /= np.maximum(result.sum(-1, keepdims=True), EPS)
    return result.reshape(values.shape).astype(np.float32)


def _candidate_geometry(
    frame_id: int,
    frame_count: int,
    candidate_ref_xy: np.ndarray,
    candidate_endpoint: np.ndarray,
    query_xy: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = frame_id / max(frame_count - 1, 1)
    minimum_dt = 1.0 / max(frame_count - 1, 1)
    first_dt = max(alpha, minimum_dt)
    last_dt = max(1.0 - alpha, minimum_dt)
    q = np.broadcast_to(query_xy[:, None, :], candidate_ref_xy.shape)
    velocity_first = (q - candidate_ref_xy) / first_dt
    velocity_last = (candidate_ref_xy - q) / last_dt
    endpoint = candidate_endpoint[..., None]
    velocity = np.where(endpoint == 0, velocity_first, velocity_last)
    midpoint = 0.5 * (q + candidate_ref_xy)
    position = midpoint.copy()
    position[..., 0] = (position[..., 0] - (grid_w - 1) / 2.0) / max((grid_w - 1) / 2.0, 1.0)
    position[..., 1] = (position[..., 1] - (grid_h - 1) / 2.0) / max((grid_h - 1) / 2.0, 1.0)
    # Motion inferred from a reference only one or two frames away is highly
    # ill-conditioned: a one-token correspondence error becomes an enormous
    # normalized velocity.  Use the temporal lever arm as an explicit
    # reliability gate, and let appearance carry endpoint-near queries.
    baseline = np.where(candidate_endpoint == 0, alpha, 1.0 - alpha)
    geometry_reliability = np.clip(baseline / 0.15, 0.0, 1.0) ** 2
    return (
        velocity.astype(np.float32),
        position.astype(np.float32),
        geometry_reliability.astype(np.float32),
    )


def vote_and_verify(
    frame_id: int,
    frame_count: int,
    raw_index: np.ndarray,
    raw_score: np.ndarray,
    reference_endpoint: np.ndarray,
    reference_local_id: np.ndarray,
    texture: np.ndarray,
    token_rgb: np.ndarray,
    grid_h: int,
    grid_w: int,
    candidate_temperature: float,
    vote_config: dict[str, Any],
    hypothesis: str,
    previous_slot_state: Optional[MotionSlotState] = None,
    previous_token_slot: Optional[np.ndarray] = None,
    temporal_prior: Optional[np.ndarray] = None,
    temporal_reliability: Optional[np.ndarray] = None,
    enable_rejection: bool = True,
    enable_sparse: bool = True,
) -> VoteResult:
    del raw_index
    slot_capacity = int(vote_config["slot_capacity"])
    query_xy = token_grid(grid_h, grid_w)
    ref_xy_all = token_grid(grid_h, grid_w)
    candidate_ref_xy = ref_xy_all[reference_local_id]
    stats = candidate_statistics(raw_score, candidate_temperature)
    raw_weight = stats["weight"]
    clarity = np.clip(1.0 - stats["entropy"], 0.05, 1.0)
    voter = raw_weight * texture[:, None] * clarity[:, None]

    velocity, position, geometry_reliability = _candidate_geometry(
        frame_id,
        frame_count,
        candidate_ref_xy,
        reference_endpoint,
        query_xy,
        grid_h,
        grid_w,
    )
    voter *= geometry_reliability
    slot_state, active_slot_ids, assignment, fit, slot_prior, candidate_outlier, lifecycle_events = _fit_dynamic_motion_slots(
        velocity,
        position,
        voter,
        vote_config,
        hypothesis,
        previous_slot_state,
    )

    active_count = len(active_slot_ids)
    if active_count:
        initial_token_slot = (raw_weight[..., None] * assignment).sum(1)
        initial_token_slot /= np.maximum(initial_token_slot.sum(-1, keepdims=True), EPS)
        spatial = edge_aware_neighbor_average(initial_token_slot, token_rgb, grid_h, grid_w)
        candidate_fit = (assignment * fit).sum(-1)
        candidate_spatial = (assignment * spatial[:, None, :]).sum(-1)
        candidate_prior = (assignment * slot_prior[None, None, :]).sum(-1)
    else:
        initial_token_slot = np.zeros((raw_weight.shape[0], 0), dtype=np.float32)
        spatial = initial_token_slot.copy()
        candidate_fit = np.zeros_like(raw_weight, dtype=np.float32)
        candidate_spatial = np.zeros_like(raw_weight, dtype=np.float32)
        candidate_prior = np.zeros_like(raw_weight, dtype=np.float32)

    if active_count:
        if temporal_prior is None or previous_token_slot is None:
            temporal_active = np.full_like(initial_token_slot, 1.0 / active_count)
            temporal_reliability = np.zeros(initial_token_slot.shape[0], dtype=np.float32)
        else:
            temporal_active = np.asarray(temporal_prior[:, active_slot_ids], dtype=np.float32)
        temporal_reliability = np.asarray(temporal_reliability, dtype=np.float32)
        uniform = np.full_like(temporal_active, 1.0 / active_count)
        reliable_temporal = (
            temporal_active * temporal_reliability[:, None]
            + uniform * (1.0 - temporal_reliability[:, None])
        )
        reliable_temporal /= np.maximum(reliable_temporal.sum(-1, keepdims=True), EPS)
        candidate_temporal = (assignment * reliable_temporal[:, None, :]).sum(-1)
    else:
        candidate_temporal = np.zeros_like(raw_weight, dtype=np.float32)

    logits = raw_score / max(candidate_temperature, EPS)
    logits += geometry_reliability * float(vote_config["motion_logit_weight"]) * np.log(
        np.maximum(candidate_fit, 1e-4)
    )
    logits += geometry_reliability * float(vote_config["spatial_logit_weight"]) * np.log(
        np.maximum(candidate_spatial, 1e-4)
    )
    logits += geometry_reliability * float(vote_config["temporal_logit_weight"]) * np.log(
        np.maximum(candidate_temporal, 1e-4)
    )
    logits += geometry_reliability * float(vote_config["slot_prior_weight"]) * np.log(
        np.maximum(candidate_prior, 1e-4)
    )
    verified_weight = softmax(logits, axis=-1).astype(np.float32)

    token_slot_active = (verified_weight[..., None] * assignment).sum(1)
    if active_count:
        token_slot_active /= np.maximum(token_slot_active.sum(-1, keepdims=True), EPS)
    if active_count and float(vote_config["spatial_logit_weight"]) > 0.0:
        # Candidate support and the displayed motion set should share the same
        # edge-aware neighbourhood prior.  Two conservative mean-field steps
        # suppress isolated one-token colour islands without hard-merging
        # across strong appearance boundaries.
        spatial_seed = token_slot_active.copy()
        for _ in range(2):
            neighbor_slot = edge_aware_neighbor_average(token_slot_active, token_rgb, grid_h, grid_w)
            token_slot_active = 0.45 * spatial_seed + 0.55 * neighbor_slot
            token_slot_active /= np.maximum(token_slot_active.sum(-1, keepdims=True), EPS)
    token_slot = np.zeros((raw_weight.shape[0], slot_capacity), dtype=np.float32)
    if active_count:
        token_slot[:, active_slot_ids] = token_slot_active
    weighted_fit = (verified_weight * candidate_fit).sum(-1)
    weighted_spatial = (verified_weight * candidate_spatial).sum(-1)
    peak = verified_weight.max(-1)
    score_strength = np.clip((raw_score[:, 0] - 0.42) / 0.30, 0.0, 1.0)
    confidence = (
        0.24 * peak
        + 0.20 * clarity
        + 0.20 * weighted_fit
        + 0.16 * weighted_spatial
        + 0.20 * score_strength
    )
    confidence *= np.clip(1.0 - (raw_weight * candidate_outlier).sum(-1), 0.0, 1.0)
    confidence *= 0.25 + 0.75 * texture
    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)
    reject_probability = (1.0 - confidence).astype(np.float32)
    if not enable_rejection:
        reject_probability.fill(0.0)

    primary_l = min(8, verified_weight.shape[-1])
    if enable_sparse:
        sparse_mask = np.zeros_like(verified_weight, dtype=bool)
        sparse_mask[:, :primary_l] = verified_weight[:, :primary_l] >= float(
            vote_config["minimum_sparse_weight"]
        )
        # The P5 sparse route is deliberately restricted to the primary Top-8
        # set.  Top-9..16 remain available only for capacity diagnostics.
        best = verified_weight[:, :primary_l].argmax(-1)
        sparse_mask[np.arange(sparse_mask.shape[0]), best] = True
    else:
        # P1-P4 are ranking ablations and do not apply the P5 Value mask.
        sparse_mask = np.ones_like(verified_weight, dtype=bool)
        best = verified_weight.argmax(-1)
    rejected = (
        confidence < float(vote_config["reject_confidence_threshold"])
        if enable_rejection
        else np.zeros(confidence.shape, dtype=bool)
    )
    sparse_mask[rejected] = False

    accepted_weight = verified_weight * sparse_mask.astype(np.float32)
    accepted_weight /= np.maximum(accepted_weight.sum(-1, keepdims=True), EPS)
    predicted_reference_xy = (accepted_weight[..., None] * candidate_ref_xy).sum(1)
    selected_endpoint = np.take_along_axis(reference_endpoint, best[:, None], axis=-1)[:, 0]
    selected_endpoint[rejected] = 2

    if frame_id == 0:
        self_candidate = (reference_endpoint == 0) & (reference_local_id == np.arange(query_xy.shape[0])[:, None])
        has_self = self_candidate.any(-1)
        if has_self.any():
            self_rank = self_candidate.argmax(-1)
            verified_weight[has_self] = 0.0
            verified_weight[np.where(has_self)[0], self_rank[has_self]] = 1.0
            if enable_sparse:
                sparse_mask[has_self] = False
                sparse_mask[np.where(has_self)[0], self_rank[has_self]] = True
            predicted_reference_xy[has_self] = query_xy[has_self]
            selected_endpoint[has_self] = 0
            confidence[has_self] = 1.0
            reject_probability[has_self] = 0.0
    if frame_id == frame_count - 1 and (reference_endpoint == 1).any():
        self_candidate = (reference_endpoint == 1) & (reference_local_id == np.arange(query_xy.shape[0])[:, None])
        has_self = self_candidate.any(-1)
        if has_self.any():
            self_rank = self_candidate.argmax(-1)
            verified_weight[has_self] = 0.0
            verified_weight[np.where(has_self)[0], self_rank[has_self]] = 1.0
            if enable_sparse:
                sparse_mask[has_self] = False
                sparse_mask[np.where(has_self)[0], self_rank[has_self]] = True
            predicted_reference_xy[has_self] = query_xy[has_self]
            selected_endpoint[has_self] = 1
            confidence[has_self] = 1.0
            reject_probability[has_self] = 0.0

    assignment_top_k = int(vote_config.get("assignment_top_k", 3))
    candidate_slot_id = np.full(raw_weight.shape + (assignment_top_k,), -1, dtype=np.int32)
    candidate_slot_probability = np.zeros(raw_weight.shape + (assignment_top_k,), dtype=np.float32)
    if active_count:
        stored_count = min(assignment_top_k, active_count)
        local_order = np.argsort(-assignment, axis=-1)[..., :stored_count]
        candidate_slot_id[..., :stored_count] = active_slot_ids[local_order].astype(np.int32)
        candidate_slot_probability[..., :stored_count] = np.take_along_axis(
            assignment, local_order, axis=-1
        ).astype(np.float32)

    return VoteResult(
        candidate_weight=verified_weight,
        candidate_slot_id=candidate_slot_id,
        candidate_slot_probability=candidate_slot_probability,
        token_slot=token_slot,
        slot_parameters=slot_state.parameters,
        slot_active=slot_state.active,
        slot_mass=slot_state.mass,
        slot_age=slot_state.age,
        slot_missed=slot_state.missed,
        candidate_outlier_probability=candidate_outlier,
        lifecycle_events=lifecycle_events,
        slot_state=slot_state,
        reject_probability=reject_probability,
        confidence=confidence,
        sparse_mask=sparse_mask,
        predicted_reference_xy=predicted_reference_xy.astype(np.float32),
        selected_endpoint=selected_endpoint.astype(np.uint8),
    )
