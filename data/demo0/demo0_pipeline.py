from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from v2_core import (
    CandidateContext,
    CandidateScorer,
    FrozenResNet18,
    MethodPrediction,
    append_reject_slot,
    build_candidate_context,
    predict_methods,
    stable_slot_permutation,
    token_grid,
)


METHOD_NAMES = ("B0", "B1", "B2", "B3", "B4", "B5")
ABLATION_NAMES = (
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
)
PALETTE_RGB = np.asarray(
    (
        (230, 57, 70),
        (46, 196, 182),
        (69, 123, 255),
        (255, 183, 3),
        (131, 56, 236),
        (0, 168, 107),
        (247, 127, 0),
        (255, 99, 164),
    ),
    dtype=np.uint8,
)


@dataclass(frozen=True)
class ClipInfo:
    clip_id: str
    display_name: str
    video_path: Path
    camera_csv_path: Path
    metadata_path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    camera_rows: int


@dataclass
class WeakProxy:
    ref_xy_token: np.ndarray
    reject: np.ndarray
    fb_error_px: np.ndarray
    photometric_error: np.ndarray


class MetricAccumulator:
    def __init__(self) -> None:
        self.epe_token: list[np.ndarray] = []
        self.pck1: list[np.ndarray] = []
        self.pck2: list[np.ndarray] = []
        self.top8: list[np.ndarray] = []
        self.feature_error: list[np.ndarray] = []
        self.rgb_error: list[np.ndarray] = []
        self.cycle_error: list[np.ndarray] = []
        self.reject_true: list[np.ndarray] = []
        self.reject_pred: list[np.ndarray] = []
        self.coverage: list[np.ndarray] = []
        self.confidence: list[np.ndarray] = []

    @staticmethod
    def _cat(values: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(values) if values else np.empty(0, dtype=np.float32)

    def finalize(self, stride_px: int, tau: float) -> dict[str, Any]:
        epe = self._cat(self.epe_token)
        pck1 = self._cat(self.pck1)
        pck2 = self._cat(self.pck2)
        top8 = self._cat(self.top8)
        feature = self._cat(self.feature_error)
        rgb = self._cat(self.rgb_error)
        cycle = self._cat(self.cycle_error)
        true = self._cat(self.reject_true).astype(bool)
        pred = self._cat(self.reject_pred).astype(bool)
        coverage = self._cat(self.coverage)
        confidence = self._cat(self.confidence)
        tp = int(np.logical_and(true, pred).sum())
        fp = int(np.logical_and(~true, pred).sum())
        fn = int(np.logical_and(true, ~pred).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        reject_f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        pck1_mean = float(pck1.mean()) if pck1.size else float("nan")
        top8_mean = float(top8.mean()) if top8.size else float("nan")
        epe_mean = float(epe.mean()) if epe.size else float("nan")
        feature_mean = float(feature.mean()) if feature.size else float("nan")
        proxy_score = (
            0.30 * pck1_mean
            + 0.15 * top8_mean
            + 0.20 * math.exp(-epe_mean / 2.0)
            + 0.20 * reject_f1
            + 0.15 * math.exp(-feature_mean / max(tau, 1e-8))
        )
        return {
            "status": "provisional",
            "sample_count_valid": int(epe.size),
            "sample_count_all": int(true.size),
            "PCK@1_token": pck1_mean,
            "PCK@2_token": float(pck2.mean()) if pck2.size else float("nan"),
            "Top8Recall": top8_mean,
            "EPE_token_mean": epe_mean,
            "EPE_token_median": float(np.median(epe)) if epe.size else float("nan"),
            "EPE_pixel_mean": epe_mean * stride_px,
            "EPE_pixel_median": float(np.median(epe)) * stride_px if epe.size else float("nan"),
            "RejectPrecision": precision,
            "RejectRecall": recall,
            "RejectF1": reject_f1,
            "FeatureWarpError": feature_mean,
            "RGBWarpError": float(rgb.mean()) if rgb.size else float("nan"),
            "CycleConsistencyError_token": float(cycle.mean()) if cycle.size else float("nan"),
            "accepted_fraction": float(coverage.mean()) if coverage.size else float("nan"),
            "mean_confidence": float(confidence.mean()) if confidence.size else float("nan"),
            "Provisional-Demo0-Proxy-Score": proxy_score,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V²-DiT Demo 0 pipeline")
    parser.add_argument("command", choices=("validate", "self-test", "run"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    return config, path.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def inspect_clip(config_dir: Path, config: dict[str, Any], item: dict[str, Any]) -> ClipInfo:
    input_root = (config_dir / config["input_root"]).resolve()
    video = input_root / item["video"]
    camera_csv = input_root / item["camera_csv"]
    metadata_path = input_root / item["metadata"]
    for path in (video, camera_csv, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    with camera_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        camera_rows = sum(1 for _ in csv.DictReader(handle))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    alignment = metadata.get("alignment", {})
    expected = int(alignment.get("video_frame_count", frame_count))
    if frame_count != expected or camera_rows != frame_count:
        raise ValueError(
            f"Alignment mismatch for {item['display_name']}: "
            f"video={frame_count}, metadata={expected}, camera={camera_rows}"
        )
    if (width, height) != (
        int(metadata["flashback_export"]["width"]),
        int(metadata["flashback_export"]["height"]),
    ):
        raise ValueError(f"Resolution mismatch for {item['display_name']}")
    return ClipInfo(
        clip_id=item["clip_id"],
        display_name=item["display_name"],
        video_path=video,
        camera_csv_path=camera_csv,
        metadata_path=metadata_path,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        camera_rows=camera_rows,
    )


def validate_inputs(config: dict[str, Any], config_dir: Path) -> list[ClipInfo]:
    clips = [inspect_clip(config_dir, config, item) for item in config["clips"]]
    print(json.dumps([asdict_path_safe(clip) for clip in clips], ensure_ascii=False, indent=2))
    return clips


def asdict_path_safe(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: asdict_path_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: asdict_path_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [asdict_path_safe(item) for item in value]
    return value


def self_test(config: dict[str, Any], device: torch.device) -> None:
    torch.manual_seed(config["seed"])
    grid_h, grid_w, channels = 6, 8, 16
    reference = F.normalize(torch.randn(grid_h * grid_w, channels, device=device), dim=-1)
    query = reference.reshape(grid_h, grid_w, channels).roll(shifts=(1, 2), dims=(0, 1)).reshape(-1, channels)
    query = F.normalize(query + 0.02 * torch.randn_like(query), dim=-1)
    matcher = config["matching"]
    context = build_candidate_context(
        query,
        reference,
        grid_h,
        grid_w,
        matcher["top_l"],
        matcher["slot_count"],
        matcher["slot_iterations"],
        matcher["appearance_temperature"],
        matcher["slot_temperature"],
    )
    b4 = CandidateScorer(hidden_dim=config["training"]["hidden_dim"]).to(device)
    b5 = CandidateScorer(hidden_dim=config["training"]["hidden_dim"]).to(device)
    predictions = predict_methods(
        context,
        query,
        reference,
        b4,
        b5,
        matcher["appearance_temperature"],
    )
    loss = (
        predictions["B4"].routed_feature.square().mean()
        + predictions["B5"].routed_feature.square().mean()
        + predictions["B5"].reject_prob.mean()
    )
    loss.backward()
    assert context.topl_index.shape == (grid_h * grid_w, matcher["top_l"])
    assert context.slot_assignment.shape == (
        grid_h * grid_w,
        matcher["top_l"],
        matcher["slot_count"],
    )
    assert append_reject_slot(context.slot_assignment, predictions["B5"].reject_prob).shape[-1] == (
        matcher["slot_count"] + 1
    )
    assert any(parameter.grad is not None for parameter in b4.parameters())
    assert any(parameter.grad is not None for parameter in b5.parameters())
    if b4.parameter_count != b5.parameter_count:
        raise AssertionError("B4 and B5 parameter counts differ")
    print(
        json.dumps(
            {
                "status": "passed",
                "device": str(device),
                "forward_shapes": {
                    "topl_index": list(context.topl_index.shape),
                    "slot_assignment": list(context.slot_assignment.shape),
                    "motion_vectors": list(context.motion_vectors.shape),
                    "reject_prob": list(predictions["B5"].reject_prob.shape),
                },
                "backward": "passed",
                "B4_parameter_count": b4.parameter_count,
                "B5_parameter_count": b5.parameter_count,
            },
            indent=2,
        )
    )


def frame_to_tensor(frame_bgr: np.ndarray, width: int, height: int, device: torch.device) -> Tensor:
    resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).to(device=device, dtype=torch.float32)
    return tensor.permute(2, 0, 1) / 255.0


def extract_features(
    clip: ClipInfo,
    clip_dir: Path,
    encoder: FrozenResNet18,
    config: dict[str, Any],
    device: torch.device,
    batch_size: int,
    frame_limit: Optional[int],
) -> tuple[int, int, int, int]:
    encoder_cfg = config["encoder"]
    total = min(clip.frame_count, frame_limit) if frame_limit else clip.frame_count
    resize_w = int(encoder_cfg["resize_width"])
    resize_h = int(encoder_cfg["resize_height"])
    grid_h = resize_h // int(encoder_cfg["stride"])
    grid_w = resize_w // int(encoder_cfg["stride"])
    token_count = grid_h * grid_w
    feature_dim = int(encoder_cfg["projection_dim"])
    arrays = clip_dir / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    feature_map = np.lib.format.open_memmap(
        arrays / "query_features.npy",
        mode="w+",
        dtype=np.float16,
        shape=(total, token_count, feature_dim),
    )
    token_rgb = np.lib.format.open_memmap(
        arrays / "token_rgb.npy",
        mode="w+",
        dtype=np.float16,
        shape=(total, token_count, 3),
    )
    capture = cv2.VideoCapture(str(clip.video_path))
    batch: list[Tensor] = []
    batch_rgb: list[np.ndarray] = []
    batch_start = 0
    processed = 0
    while processed < total:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Unexpected end of {clip.video_path} at {processed}/{total}")
        tensor = frame_to_tensor(frame, resize_w, resize_h, device)
        batch.append(tensor)
        small_rgb = cv2.cvtColor(
            cv2.resize(frame, (grid_w, grid_h), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        batch_rgb.append(small_rgb.astype(np.float32) / 255.0)
        processed += 1
        if len(batch) == batch_size or processed == total:
            with torch.inference_mode():
                encoded = encoder(torch.stack(batch)).reshape(len(batch), token_count, feature_dim)
            end = batch_start + len(batch)
            feature_map[batch_start:end] = encoded.detach().cpu().numpy().astype(np.float16)
            token_rgb[batch_start:end] = np.stack(batch_rgb).reshape(len(batch), token_count, 3).astype(
                np.float16
            )
            feature_map.flush()
            token_rgb.flush()
            batch_start = end
            batch.clear()
            batch_rgb.clear()
            if processed % max(batch_size * 10, 1) == 0 or processed == total:
                print(f"[{clip.clip_id}] encoded {processed}/{total}", flush=True)
    capture.release()
    reference = np.asarray(feature_map[0], dtype=np.float16)
    np.save(arrays / "reference_features.npy", reference)
    return total, grid_h, grid_w, feature_dim


def choose_frames(total: int, stride: int, start: int, stop: int, limit: int) -> list[int]:
    frames = list(range(max(start, 1), max(stop, start + 1), stride))
    if stop - 1 >= start:
        frames.append(stop - 1)
    frames = sorted(set(frame for frame in frames if 0 < frame < total))
    if len(frames) > limit:
        indices = np.linspace(0, len(frames) - 1, limit).round().astype(int)
        frames = [frames[index] for index in sorted(set(indices.tolist()))]
    return frames


def read_resized_frame(video_path: Path, frame_id: int, width: int, height: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read {video_path} frame {frame_id}")
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def bilinear_sample(array: np.ndarray, xy: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    map_x = xy[:, 0].astype(np.float32).reshape(-1, 1)
    map_y = xy[:, 1].astype(np.float32).reshape(-1, 1)
    sampled = cv2.remap(
        array,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return sampled.reshape((xy.shape[0],) + array.shape[2:])


def weak_proxy_for_frame(
    reference_bgr: np.ndarray,
    current_bgr: np.ndarray,
    grid_h: int,
    grid_w: int,
    stride: int,
    fb_threshold: float,
) -> WeakProxy:
    ref_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
    dis_forward = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis_backward = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow_cur_ref = dis_forward.calc(cur_gray, ref_gray, None)
    flow_ref_cur = dis_backward.calc(ref_gray, cur_gray, None)
    grid = token_grid(grid_h, grid_w, torch.device("cpu")).numpy()
    centers = (grid + 0.5) * float(stride)
    flow_at_query = bilinear_sample(flow_cur_ref, centers)
    ref_pixels = centers + flow_at_query
    backward = bilinear_sample(flow_ref_cur, ref_pixels)
    fb_error = np.linalg.norm(flow_at_query + backward, axis=-1)
    inside = (
        (ref_pixels[:, 0] >= 0)
        & (ref_pixels[:, 0] <= reference_bgr.shape[1] - 1)
        & (ref_pixels[:, 1] >= 0)
        & (ref_pixels[:, 1] <= reference_bgr.shape[0] - 1)
    )
    ref_rgb = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    cur_rgb = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    sampled_ref = bilinear_sample(ref_rgb, ref_pixels)
    sampled_cur = bilinear_sample(cur_rgb, centers)
    photo = np.abs(sampled_ref - sampled_cur).mean(-1)
    reject = (~inside) | (fb_error > fb_threshold) | (photo > 0.30)
    return WeakProxy(
        ref_xy_token=(ref_pixels / float(stride) - 0.5).astype(np.float32),
        reject=reject.astype(bool),
        fb_error_px=fb_error.astype(np.float32),
        photometric_error=photo.astype(np.float32),
    )


def context_for_frame(
    feature_map: np.ndarray,
    reference: Tensor,
    frame_id: int,
    grid_h: int,
    grid_w: int,
    config: dict[str, Any],
    device: torch.device,
    previous_slots: Optional[Tensor] = None,
    top_l: Optional[int] = None,
    slot_count: Optional[int] = None,
) -> tuple[Tensor, CandidateContext]:
    matcher = config["matching"]
    query = torch.from_numpy(np.asarray(feature_map[frame_id], dtype=np.float32)).to(device)
    context = build_candidate_context(
        query,
        reference,
        grid_h,
        grid_w,
        int(top_l or matcher["top_l"]),
        int(slot_count or matcher["slot_count"]),
        int(matcher["slot_iterations"]),
        float(matcher["appearance_temperature"]),
        float(matcher["slot_temperature"]),
        previous_slots=previous_slots,
    )
    return query, context


def gather_training_data(
    clip: ClipInfo,
    clip_dir: Path,
    frame_ids: Iterable[int],
    total: int,
    grid_h: int,
    grid_w: int,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Tensor]:
    del total
    arrays = clip_dir / "arrays"
    feature_map = np.load(arrays / "query_features.npy", mmap_mode="r")
    reference = torch.from_numpy(np.asarray(feature_map[0], dtype=np.float32)).to(device)
    resize_w = int(config["encoder"]["resize_width"])
    resize_h = int(config["encoder"]["resize_height"])
    stride = int(config["encoder"]["stride"])
    fb_threshold = float(config["evaluation"]["forward_backward_threshold_pixels"])
    reference_bgr = read_resized_frame(clip.video_path, 0, resize_w, resize_h)
    collected: dict[str, list[Tensor]] = {
        "b4": [],
        "b5": [],
        "base": [],
        "target": [],
        "reject": [],
        "valid_candidate": [],
        "b0_feature_error": [],
    }
    weak_dir = clip_dir / "weak_proxy" / "calibration"
    weak_dir.mkdir(parents=True, exist_ok=True)
    for frame_id in frame_ids:
        query, context = context_for_frame(
            feature_map, reference, frame_id, grid_h, grid_w, config, device
        )
        current_bgr = read_resized_frame(clip.video_path, frame_id, resize_w, resize_h)
        proxy = weak_proxy_for_frame(
            reference_bgr, current_bgr, grid_h, grid_w, stride, fb_threshold
        )
        np.savez_compressed(
            weak_dir / f"{frame_id:06d}.npz",
            ref_xy_token=proxy.ref_xy_token,
            reject=proxy.reject,
            fb_error_px=proxy.fb_error_px,
            photometric_error=proxy.photometric_error,
        )
        target_xy = torch.from_numpy(proxy.ref_xy_token).to(device)
        distance = torch.linalg.vector_norm(context.candidate_ref_xy - target_xy[:, None, :], dim=-1)
        target = distance.argmin(-1)
        valid_candidate = distance.amin(-1) <= 2.0
        reject = torch.from_numpy(proxy.reject).to(device) | ~valid_candidate
        collected["b4"].append(context.b4_features.detach().cpu())
        collected["b5"].append(context.b5_features.detach().cpu())
        collected["base"].append(context.topl_similarity.detach().cpu())
        collected["target"].append(target.detach().cpu())
        collected["reject"].append(reject.detach().cpu())
        collected["valid_candidate"].append(valid_candidate.detach().cpu())
        b0_routed = reference[context.topl_index[:, 0]]
        b0_error = (1.0 - F.cosine_similarity(b0_routed, query, dim=-1)).clamp_min(0.0)
        collected["b0_feature_error"].append(b0_error[~reject].detach().cpu())
        print(f"[{clip.clip_id}] weak calibration frame {frame_id}", flush=True)
    return {key: torch.cat(value, dim=0) for key, value in collected.items()}


def train_scorer_pair(
    data: dict[str, Tensor],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[CandidateScorer, CandidateScorer, dict[str, Any]]:
    training = config["training"]
    matcher = config["matching"]
    b4 = CandidateScorer(hidden_dim=int(training["hidden_dim"])).to(device)
    b5 = CandidateScorer(hidden_dim=int(training["hidden_dim"])).to(device)
    if b4.parameter_count != b5.parameter_count:
        raise AssertionError("Equal-parameter control violated")
    optimizer = torch.optim.AdamW(
        list(b4.parameters()) + list(b5.parameters()),
        lr=float(training["learning_rate"]),
        weight_decay=1e-4,
    )
    sample_count = data["target"].shape[0]
    batch_size = int(training["batch_tokens"])
    history = []
    for epoch in range(int(training["epochs"])):
        order = torch.randperm(sample_count)
        epoch_losses = []
        for start in range(0, sample_count, batch_size):
            ids = order[start : start + batch_size]
            base = data["base"][ids].to(device)
            target = data["target"][ids].to(device)
            reject = data["reject"][ids].float().to(device)
            valid = data["valid_candidate"][ids].to(device) & ~reject.bool()
            total_loss = torch.zeros((), device=device)
            for scorer, key in ((b4, "b4"), (b5, "b5")):
                candidate_logits, reject_logits = scorer(data[key][ids].to(device))
                scores = base / float(matcher["appearance_temperature"]) + candidate_logits
                if valid.any():
                    classification = F.cross_entropy(scores[valid], target[valid])
                else:
                    classification = scores.sum() * 0.0
                weight = torch.softmax(scores.detach(), dim=-1)
                reject_token_logits = (weight * reject_logits).sum(-1)
                positive = reject.sum().clamp_min(1.0)
                negative = (1.0 - reject).sum().clamp_min(1.0)
                positive_weight = (negative / positive).clamp(1.0, 25.0)
                rejection = F.binary_cross_entropy_with_logits(
                    reject_token_logits,
                    reject,
                    pos_weight=positive_weight,
                )
                total_loss = total_loss + classification + 0.5 * rejection
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(b4.parameters()) + list(b5.parameters()), 5.0)
            optimizer.step()
            epoch_losses.append(float(total_loss.detach().cpu()))
        mean_loss = float(np.mean(epoch_losses))
        history.append(mean_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"training epoch {epoch + 1}/{training['epochs']} loss={mean_loss:.6f}", flush=True)
    return b4.eval(), b5.eval(), {
        "epochs": int(training["epochs"]),
        "sample_count": sample_count,
        "final_loss": history[-1],
        "loss_history": history,
        "B4_parameter_count": b4.parameter_count,
        "B5_parameter_count": b5.parameter_count,
    }


def allocate_prediction_arrays(
    arrays: Path,
    total: int,
    token_count: int,
    top_l: int,
    slot_count: int,
) -> dict[str, np.memmap]:
    specifications = {
        "topl_index": (np.int32, (total, token_count, top_l)),
        "topl_similarity": (np.float16, (total, token_count, top_l)),
        "topl_weight": (np.float16, (total, token_count, top_l)),
        "slot_assignment": (np.float16, (total, token_count, top_l, slot_count + 1)),
        "reject_prob": (np.float16, (total, token_count)),
        "motion_vectors": (np.float32, (total, slot_count, 2)),
        "token_confidence": (np.float16, (total, token_count)),
        "slot_id": (np.int16, (total, token_count)),
        "slot_prob": (np.float16, (total, token_count, slot_count)),
        "pred_ref_xy": (np.float16, (total, token_count, 2)),
    }
    return {
        name: np.lib.format.open_memmap(arrays / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        for name, (dtype, shape) in specifications.items()
    }


def run_full_predictions(
    clip: ClipInfo,
    clip_dir: Path,
    total: int,
    grid_h: int,
    grid_w: int,
    config: dict[str, Any],
    device: torch.device,
    b4: Optional[CandidateScorer],
    b5: Optional[CandidateScorer],
) -> None:
    matcher = config["matching"]
    arrays_dir = clip_dir / "arrays"
    features = np.load(arrays_dir / "query_features.npy", mmap_mode="r")
    reference = torch.from_numpy(np.asarray(features[0], dtype=np.float32)).to(device)
    outputs = allocate_prediction_arrays(
        arrays_dir,
        total,
        grid_h * grid_w,
        int(matcher["top_l"]),
        int(matcher["slot_count"]),
    )
    previous_slots: Optional[Tensor] = None
    coords = token_grid(grid_h, grid_w, device)
    for frame_id in range(total):
        query, context = context_for_frame(
            features,
            reference,
            frame_id,
            grid_h,
            grid_w,
            config,
            device,
            previous_slots,
        )
        order = stable_slot_permutation(context.motion_vectors, previous_slots)
        order_tensor = torch.from_numpy(order).to(device)
        context.motion_vectors = context.motion_vectors[order_tensor]
        context.slot_assignment = context.slot_assignment[..., order_tensor]
        previous_slots = context.motion_vectors.detach()
        methods = predict_methods(
            context,
            query,
            reference,
            b4,
            b5,
            float(matcher["appearance_temperature"]),
        )
        prediction = methods["B5"]
        slot_prob = (prediction.candidate_weight[..., None] * context.slot_assignment).sum(1)
        slot_prob = slot_prob * (1.0 - prediction.reject_prob[:, None])
        slot_id = slot_prob.argmax(-1)
        slot_id[prediction.reject_prob >= float(matcher["reject_threshold"])] = -1
        slot_with_reject = append_reject_slot(context.slot_assignment, prediction.reject_prob)

        if frame_id == 0:
            prediction.ref_xy = coords
            prediction.reject_prob.zero_()
            prediction.confidence.fill_(1.0)
            context.motion_vectors.zero_()
            slot_id.zero_()
            slot_prob.zero_()
            slot_prob[:, 0] = 1.0
            slot_with_reject[..., -1].zero_()

        values = {
            "topl_index": context.topl_index,
            "topl_similarity": context.topl_similarity,
            "topl_weight": prediction.candidate_weight,
            "slot_assignment": slot_with_reject,
            "reject_prob": prediction.reject_prob,
            "motion_vectors": context.motion_vectors,
            "token_confidence": prediction.confidence,
            "slot_id": slot_id,
            "slot_prob": slot_prob,
            "pred_ref_xy": prediction.ref_xy,
        }
        for name, tensor in values.items():
            outputs[name][frame_id] = tensor.detach().cpu().numpy().astype(outputs[name].dtype)
        if (frame_id + 1) % 50 == 0 or frame_id + 1 == total:
            for output in outputs.values():
                output.flush()
            print(f"[{clip.clip_id}] matched {frame_id + 1}/{total}", flush=True)


def scorer_prediction(
    context: CandidateContext,
    query: Tensor,
    reference: Tensor,
    b5: Optional[CandidateScorer],
    appearance_temperature: float,
) -> MethodPrediction:
    return predict_methods(
        context, query, reference, None, b5, appearance_temperature
    )["B5"]


def build_ablations(
    query: Tensor,
    reference: Tensor,
    base_context: CandidateContext,
    b5: Optional[CandidateScorer],
    config: dict[str, Any],
    feature_map: np.ndarray,
    frame_id: int,
    grid_h: int,
    grid_w: int,
    device: torch.device,
) -> dict[str, tuple[CandidateContext, MethodPrediction]]:
    temperature = float(config["matching"]["appearance_temperature"])
    base = scorer_prediction(base_context, query, reference, b5, temperature)
    result: dict[str, tuple[CandidateContext, MethodPrediction]] = {}
    no_reject = replace(base, reject_prob=torch.zeros_like(base.reject_prob))
    result["B5_no_reject"] = (base_context, no_reject)

    no_spatial_context = replace(base_context, b5_features=base_context.b5_features.clone())
    no_spatial_context.b5_features[..., 2] = 0.0
    result["B5_no_spatial"] = (
        no_spatial_context,
        scorer_prediction(no_spatial_context, query, reference, b5, temperature),
    )

    hard_index = base.candidate_weight.argmax(-1)
    hard_weight = F.one_hot(hard_index, num_classes=base.candidate_weight.shape[-1]).float()
    hard_ref_xy = (hard_weight[..., None] * base_context.candidate_ref_xy).sum(1)
    hard_feature = F.normalize(
        (hard_weight[..., None] * reference[base_context.topl_index]).sum(1), dim=-1
    )
    result["B5_hard_top1"] = (
        base_context,
        MethodPrediction(hard_ref_xy, hard_weight, base.reject_prob, base.confidence, hard_feature),
    )

    b1 = predict_methods(base_context, query, reference, None, b5, temperature)["B1"]
    result["B5_posthoc_only"] = (
        base_context,
        MethodPrediction(base.ref_xy, base.candidate_weight, base.reject_prob, base.confidence, b1.routed_feature),
    )

    for count in (2, 4, 8):
        _, context = context_for_frame(
            feature_map,
            reference,
            frame_id,
            grid_h,
            grid_w,
            config,
            device,
            top_l=int(config["matching"]["top_l"]),
            slot_count=count,
        )
        result[f"B5_K{count}"] = (
            context,
            scorer_prediction(context, query, reference, b5, temperature),
        )
    for count in (4, 8, 16):
        _, context = context_for_frame(
            feature_map,
            reference,
            frame_id,
            grid_h,
            grid_w,
            config,
            device,
            top_l=count,
            slot_count=int(config["matching"]["slot_count"]),
        )
        result[f"B5_L{count}"] = (
            context,
            scorer_prediction(context, query, reference, b5, temperature),
        )
    return result


def accumulate_metrics(
    accumulator: MetricAccumulator,
    prediction: MethodPrediction,
    context: CandidateContext,
    query: Tensor,
    proxy: WeakProxy,
    reference_rgb: np.ndarray,
    current_rgb: np.ndarray,
    reject_threshold: float,
) -> None:
    pred_xy = prediction.ref_xy.detach().cpu().numpy()
    true_xy = proxy.ref_xy_token
    valid = ~proxy.reject
    epe = np.linalg.norm(pred_xy - true_xy, axis=-1)
    candidate_xy = context.candidate_ref_xy.detach().cpu().numpy()
    candidate_distance = np.linalg.norm(candidate_xy - true_xy[:, None, :], axis=-1)
    top8 = candidate_distance.min(axis=-1) <= 1.0
    reject_pred = prediction.reject_prob.detach().cpu().numpy() >= reject_threshold
    feature_error = np.clip(1.0 - F.cosine_similarity(
        prediction.routed_feature, query, dim=-1
    ).detach().cpu().numpy(), 0.0, None)
    sampled_rgb = bilinear_sample(reference_rgb.reshape(context.grid_h, context.grid_w, 3), pred_xy)
    rgb_error = np.abs(sampled_rgb - current_rgb).mean(-1)

    rounded = np.rint(pred_xy).astype(np.int64)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, context.grid_w - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, context.grid_h - 1)
    ref_index = rounded[:, 1] * context.grid_w + rounded[:, 0]
    reverse = context.similarity.argmax(dim=0).detach().cpu().numpy()
    query_xy = token_grid(context.grid_h, context.grid_w, torch.device("cpu")).numpy()
    reverse_xy = query_xy[reverse[ref_index]]
    cycle_error = np.linalg.norm(reverse_xy - query_xy, axis=-1)

    accumulator.epe_token.append(epe[valid].astype(np.float32))
    accumulator.pck1.append((epe[valid] <= 1.0).astype(np.float32))
    accumulator.pck2.append((epe[valid] <= 2.0).astype(np.float32))
    accumulator.top8.append(top8[valid].astype(np.float32))
    accumulator.feature_error.append(feature_error[valid].astype(np.float32))
    accumulator.rgb_error.append(rgb_error[valid].astype(np.float32))
    accumulator.cycle_error.append(cycle_error[valid].astype(np.float32))
    accumulator.reject_true.append(proxy.reject.astype(bool))
    accumulator.reject_pred.append(reject_pred.astype(bool))
    accumulator.coverage.append((~reject_pred).astype(np.float32))
    accumulator.confidence.append(prediction.confidence.detach().cpu().numpy().astype(np.float32))


def evaluate_clip(
    clip: ClipInfo,
    clip_dir: Path,
    test_frames: list[int],
    grid_h: int,
    grid_w: int,
    config: dict[str, Any],
    device: torch.device,
    b4: Optional[CandidateScorer],
    b5: Optional[CandidateScorer],
    calibration_tau: Optional[float],
) -> dict[str, Any]:
    features = np.load(clip_dir / "arrays" / "query_features.npy", mmap_mode="r")
    token_rgb = np.load(clip_dir / "arrays" / "token_rgb.npy", mmap_mode="r")
    reference = torch.from_numpy(np.asarray(features[0], dtype=np.float32)).to(device)
    resize_w = int(config["encoder"]["resize_width"])
    resize_h = int(config["encoder"]["resize_height"])
    stride = int(config["encoder"]["stride"])
    threshold = float(config["evaluation"]["forward_backward_threshold_pixels"])
    reject_threshold = float(config["matching"]["reject_threshold"])
    reference_bgr = read_resized_frame(clip.video_path, 0, resize_w, resize_h)
    reference_rgb = np.asarray(token_rgb[0], dtype=np.float32)
    accumulators = {name: MetricAccumulator() for name in METHOD_NAMES + ABLATION_NAMES}
    weak_dir = clip_dir / "weak_proxy" / "test"
    weak_dir.mkdir(parents=True, exist_ok=True)
    for frame_id in test_frames:
        query, context = context_for_frame(
            features, reference, frame_id, grid_h, grid_w, config, device
        )
        current_bgr = read_resized_frame(clip.video_path, frame_id, resize_w, resize_h)
        proxy = weak_proxy_for_frame(
            reference_bgr, current_bgr, grid_h, grid_w, stride, threshold
        )
        np.savez_compressed(
            weak_dir / f"{frame_id:06d}.npz",
            ref_xy_token=proxy.ref_xy_token,
            reject=proxy.reject,
            fb_error_px=proxy.fb_error_px,
            photometric_error=proxy.photometric_error,
        )
        current_rgb = np.asarray(token_rgb[frame_id], dtype=np.float32)
        methods = predict_methods(
            context,
            query,
            reference,
            b4,
            b5,
            float(config["matching"]["appearance_temperature"]),
        )
        for name, prediction in methods.items():
            accumulate_metrics(
                accumulators[name],
                prediction,
                context,
                query,
                proxy,
                reference_rgb,
                current_rgb,
                reject_threshold,
            )
        ablations = build_ablations(
            query,
            reference,
            context,
            b5,
            config,
            features,
            frame_id,
            grid_h,
            grid_w,
            device,
        )
        for name, (variant_context, prediction) in ablations.items():
            accumulate_metrics(
                accumulators[name],
                prediction,
                variant_context,
                query,
                proxy,
                reference_rgb,
                current_rgb,
                reject_threshold,
            )
        print(f"[{clip.clip_id}] evaluated frame {frame_id}", flush=True)

    provisional_tau = (
        float(calibration_tau)
        if calibration_tau is not None
        else float(np.median(MetricAccumulator._cat(accumulators["B0"].feature_error)))
    )
    provisional_tau = max(provisional_tau, 1e-6)
    return {
        "status": "provisional",
        "reason": "No depth/visibility/entity mask; DIS bidirectional flow is a weak proxy, not geometry ground truth.",
        "test_frames": test_frames,
        "tau_calibration_note": (
            "Frozen B0 FeatureWarpError median on the disjoint calibration frames."
            if calibration_tau is not None
            else "B0 test median used only because training/calibration was skipped."
        ),
        "tau": provisional_tau,
        "methods": {
            name: accumulator.finalize(stride, provisional_tau)
            for name, accumulator in accumulators.items()
        },
        "not_applicable": {
            "diffusion_noise_ablation": "Demo 0 uses frozen image features and has no diffusion step.",
            "insertion_layer_ablation": "Reserved for Demo 1 when V² is inserted into a denoiser.",
        },
    }


def flow_color(vector_xy: np.ndarray, max_magnitude: float) -> np.ndarray:
    angle = np.arctan2(vector_xy[..., 1], vector_xy[..., 0])
    magnitude = np.linalg.norm(vector_xy, axis=-1)
    hsv = np.zeros(vector_xy.shape[:-1] + (3,), dtype=np.uint8)
    hsv[..., 0] = np.mod((angle + math.pi) * 90.0 / math.pi, 180).astype(np.uint8)
    hsv[..., 1] = 230
    hsv[..., 2] = np.clip(70 + 185 * magnitude / max(max_magnitude, 1e-6), 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def render_overlays(
    clip: ClipInfo,
    clip_dir: Path,
    total: int,
    grid_h: int,
    grid_w: int,
    config: dict[str, Any],
) -> None:
    arrays = clip_dir / "arrays"
    slot_id = np.load(arrays / "slot_id.npy", mmap_mode="r")
    reject = np.load(arrays / "reject_prob.npy", mmap_mode="r")
    confidence = np.load(arrays / "token_confidence.npy", mmap_mode="r")
    motion = np.load(arrays / "motion_vectors.npy", mmap_mode="r")
    resize_w = int(config["encoder"]["resize_width"])
    resize_h = int(config["encoder"]["resize_height"])
    alpha_base = float(config["visualization"]["overlay_alpha"])
    slot_dir = clip_dir / "overlays"
    flow_dir = clip_dir / "overlays" / "flow"
    slot_dir.mkdir(parents=True, exist_ok=True)
    flow_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    slot_writer = cv2.VideoWriter(
        str(clip_dir / "overlay_preview.mp4"), fourcc, clip.fps, (resize_w, resize_h)
    )
    flow_writer = cv2.VideoWriter(
        str(clip_dir / "overlay_flow_preview.mp4"), fourcc, clip.fps, (resize_w, resize_h)
    )
    if not slot_writer.isOpened() or not flow_writer.isOpened():
        raise RuntimeError("OpenCV could not initialize MP4 overlay writers")
    capture = cv2.VideoCapture(str(clip.video_path))
    max_magnitude = float(np.linalg.norm(np.asarray(motion), axis=-1).max())
    palette_bgr = PALETTE_RGB[:, ::-1]
    for frame_id in range(total):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Overlay read stopped at {frame_id}/{total}")
        frame = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
        ids = np.asarray(slot_id[frame_id], dtype=np.int32).reshape(grid_h, grid_w)
        reject_grid = np.asarray(reject[frame_id], dtype=np.float32).reshape(grid_h, grid_w)
        conf_grid = np.asarray(confidence[frame_id], dtype=np.float32).reshape(grid_h, grid_w)
        slot_color = np.full((grid_h, grid_w, 3), 55, dtype=np.uint8)
        for index in range(int(config["matching"]["slot_count"])):
            slot_color[ids == index] = palette_bgr[index % len(palette_bgr)]
        selected_vector = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
        for index in range(int(config["matching"]["slot_count"])):
            selected_vector[ids == index] = motion[frame_id, index]
        flow_grid = flow_color(selected_vector, max_magnitude)
        rejected = (ids < 0) | (reject_grid >= float(config["matching"]["reject_threshold"]))
        slot_color[rejected] = 35
        flow_grid[rejected] = 35
        alpha = alpha_base * np.clip(conf_grid * (1.0 - reject_grid), 0.12, 1.0)
        alpha[rejected] = alpha_base * 0.75
        alpha = cv2.resize(alpha, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)[..., None]
        slot_layer = cv2.resize(slot_color, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
        flow_layer = cv2.resize(flow_grid, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
        slot_overlay = np.clip(frame * (1.0 - alpha) + slot_layer * alpha, 0, 255).astype(np.uint8)
        flow_overlay = np.clip(frame * (1.0 - alpha) + flow_layer * alpha, 0, 255).astype(np.uint8)
        label = "reference / zero displacement" if frame_id == 0 else f"frame {frame_id}"
        for image in (slot_overlay, flow_overlay):
            cv2.putText(image, label, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            for index, vector in enumerate(motion[frame_id]):
                color = tuple(int(value) for value in palette_bgr[index % len(palette_bgr)])
                mean_conf = float(conf_grid[ids == index].mean()) if np.any(ids == index) else 0.0
                text = f"s{index}: ({vector[0]:+.1f},{vector[1]:+.1f}) c={mean_conf:.2f}"
                cv2.rectangle(image, (10, 36 + index * 22), (24, 50 + index * 22), color, -1)
                cv2.putText(
                    image,
                    text,
                    (31, 49 + index * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        cv2.imwrite(str(slot_dir / f"{frame_id:06d}.png"), slot_overlay)
        cv2.imwrite(str(flow_dir / f"{frame_id:06d}.png"), flow_overlay)
        slot_writer.write(slot_overlay)
        flow_writer.write(flow_overlay)
        if (frame_id + 1) % 100 == 0 or frame_id + 1 == total:
            print(f"[{clip.clip_id}] rendered overlays {frame_id + 1}/{total}", flush=True)
    capture.release()
    slot_writer.release()
    flow_writer.release()
    write_json(
        clip_dir / "legend.json",
        {
            "schema_version": config["schema_version"],
            "slot_palette_rgb": {str(i): PALETTE_RGB[i].tolist() for i in range(8)},
            "rejected_rgb": [35, 35, 35],
            "slot_overlay": "Discrete, Hungarian-stabilized slot colors.",
            "flow_overlay": "Hue encodes direction; brightness encodes motion magnitude.",
            "motion_vector_convention": "query_xy - reference_xy in token units",
            "first_frame": "reference / zero displacement",
        },
    )


def write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    rows = []
    for name, values in metrics["methods"].items():
        row = {"method": name}
        row.update(values)
        rows.append(row)
    keys = ["method"] + sorted({key for row in rows for key in row if key != "method"})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_clip(
    clip: ClipInfo,
    run_dir: Path,
    config: dict[str, Any],
    device: torch.device,
    encoder: FrozenResNet18,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    clip_dir = run_dir / clip.clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    total, grid_h, grid_w, feature_dim = extract_features(
        clip,
        clip_dir,
        encoder,
        config,
        device,
        args.batch_size,
        args.max_frames,
    )
    split = max(2, int(total * float(config["training"]["calibration_fraction"])))
    limit = int(config["evaluation"]["max_eval_frames_per_clip"])
    calibration_frames = choose_frames(
        total,
        int(config["training"]["calibration_stride"]),
        1,
        split,
        limit,
    )
    test_frames = choose_frames(
        total,
        int(config["evaluation"]["test_stride"]),
        split,
        total,
        limit,
    )
    b4: Optional[CandidateScorer]
    b5: Optional[CandidateScorer]
    training_report: dict[str, Any]
    calibration_tau: Optional[float]
    if args.skip_training or not calibration_frames:
        b4 = None
        b5 = None
        training_report = {"status": "skipped", "reason": "--skip-training or no calibration frames"}
        calibration_tau = None
    else:
        training_data = gather_training_data(
            clip,
            clip_dir,
            calibration_frames,
            total,
            grid_h,
            grid_w,
            config,
            device,
        )
        b4, b5, training_report = train_scorer_pair(training_data, config, device)
        calibration_errors = training_data["b0_feature_error"].numpy()
        calibration_tau = float(np.median(calibration_errors)) if calibration_errors.size else 1e-6
        calibration_tau = max(calibration_tau, 1e-6)
        training_report["frozen_calibration_tau"] = calibration_tau
        checkpoint = {
            "schema_version": config["schema_version"],
            "B4": b4.state_dict(),
            "B5": b5.state_dict(),
            "training_report": training_report,
        }
        torch.save(checkpoint, clip_dir / "scorers.pt")
    run_full_predictions(
        clip, clip_dir, total, grid_h, grid_w, config, device, b4, b5
    )
    metrics = evaluate_clip(
        clip, clip_dir, test_frames, grid_h, grid_w, config, device, b4, b5, calibration_tau
    )
    write_json(clip_dir / "metrics.json", metrics)
    write_metrics_csv(clip_dir / "metrics.csv", metrics)
    render_overlays(clip, clip_dir, total, grid_h, grid_w, config)
    manifest = {
        "schema_version": config["schema_version"],
        "status": "complete" if total == clip.frame_count else "partial_smoke_test",
        "clip": asdict_path_safe(clip),
        "processed_frame_count": total,
        "token_grid": {
            "height": grid_h,
            "width": grid_w,
            "count": grid_h * grid_w,
            "stride": int(config["encoder"]["stride"]),
            "nominal_token_box_pixels": [
                int(config["encoder"]["stride"]),
                int(config["encoder"]["stride"]),
            ],
            "receptive_field_note": "The real ResNet receptive field is larger than the nominal stride-16 box.",
        },
        "feature_dim": feature_dim,
        "top_l": int(config["matching"]["top_l"]),
        "slot_count": int(config["matching"]["slot_count"]),
        "calibration_frames": calibration_frames,
        "test_frames": test_frames,
        "training": training_report,
        "methods": list(METHOD_NAMES),
        "ablations": list(ABLATION_NAMES),
        "metric_status": "provisional",
        "input_sha256": {
            "video": sha256(clip.video_path),
            "camera_csv": sha256(clip.camera_csv_path),
            "metadata": sha256(clip.metadata_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json(clip_dir / "manifest.json", manifest)
    return manifest


def run_all(config: dict[str, Any], config_dir: Path, args: argparse.Namespace) -> None:
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    clips = [inspect_clip(config_dir, config, item) for item in config["clips"]]
    output_root = (config_dir / config["output_root"]).resolve()
    run_dir = output_root / config["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (config_dir / config["encoder"]["cache_dir"]).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache_dir))
    encoder = FrozenResNet18(int(config["encoder"]["projection_dim"]), seed).to(device).eval()
    manifests = []
    for clip in clips:
        manifests.append(run_clip(clip, run_dir, config, device, encoder, args))
    run_manifest = {
        "schema_version": config["schema_version"],
        "run_id": config["run_id"],
        "device": str(device),
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "clips": manifests,
        "scope_note": "Only the two user-selected Minecraft recordings were processed.",
        "vace_used": false_literal(),
    }
    write_json(run_dir / "manifest.json", run_manifest)
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2))


def false_literal() -> bool:
    return False


def main() -> None:
    args = parse_args()
    config, config_dir = load_config(args.config)
    device = torch.device(args.device)
    if args.command == "validate":
        validate_inputs(config, config_dir)
    elif args.command == "self-test":
        self_test(config, device)
    else:
        run_all(config, config_dir, args)


if __name__ == "__main__":
    main()
