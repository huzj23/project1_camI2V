from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_frame(path: Path, frame_id: int, width: int, height: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read {path} frame {frame_id}")
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def choose_token(clip_dir: Path, frame_id: int, grid_h: int, grid_w: int) -> int:
    confidence = np.asarray(np.load(clip_dir / "arrays" / "token_confidence.npy", mmap_mode="r")[frame_id])
    reject = np.asarray(np.load(clip_dir / "arrays" / "reject_prob.npy", mmap_mode="r")[frame_id])
    rgb = np.asarray(np.load(clip_dir / "arrays" / "token_rgb.npy", mmap_mode="r")[frame_id])
    yy, xx = np.mgrid[0:grid_h, 0:grid_w]
    center_prior = np.exp(
        -0.5 * (((xx - (grid_w - 1) / 2) / (grid_w * 0.30)) ** 2 + ((yy - grid_h * 0.60) / (grid_h * 0.32)) ** 2)
    ).reshape(-1)
    texture_proxy = np.std(rgb, axis=-1) + 0.15
    score = confidence * (1.0 - reject) * center_prior * texture_proxy
    score[(yy.reshape(-1) < 3) | (xx.reshape(-1) < 2) | (xx.reshape(-1) >= grid_w - 2)] = -1
    return int(np.argmax(score))


def draw_example(
    clip_dir: Path,
    video_path: Path,
    frame_id: int,
    token_id: int,
    display_name: str,
    grid_h: int,
    grid_w: int,
    stride: int,
) -> dict[str, object]:
    width, height = grid_w * stride, grid_h * stride
    current = read_frame(video_path, frame_id, width, height)
    reference = read_frame(video_path, 0, width, height)
    features = np.load(clip_dir / "arrays" / "query_features.npy", mmap_mode="r")
    q = np.asarray(features[frame_id, token_id], dtype=np.float32)
    r = np.asarray(features[0], dtype=np.float32)
    q /= max(float(np.linalg.norm(q)), 1e-8)
    r /= np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-8)
    similarity = r @ q
    lower, upper = np.percentile(similarity, (5.0, 99.0))
    normalized = np.clip((similarity - lower) / max(float(upper - lower), 1e-8), 0.0, 1.0)
    heat = cv2.applyColorMap((normalized.reshape(grid_h, grid_w) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_NEAREST)
    reference_heat = cv2.addWeighted(reference, 0.48, heat, 0.52, 0.0)

    qx, qy = token_id % grid_w, token_id // grid_w
    cv2.rectangle(
        current,
        (qx * stride + 1, qy * stride + 1),
        ((qx + 1) * stride - 2, (qy + 1) * stride - 2),
        (80, 255, 120),
        3,
    )
    indices = np.asarray(np.load(clip_dir / "arrays" / "topl_index.npy", mmap_mode="r")[frame_id, token_id])
    weights = np.asarray(np.load(clip_dir / "arrays" / "topl_weight.npy", mmap_mode="r")[frame_id, token_id])
    pred = np.asarray(np.load(clip_dir / "arrays" / "pred_ref_xy.npy", mmap_mode="r")[frame_id, token_id])
    for rank, index in enumerate(indices):
        x, y = int(index % grid_w), int(index // grid_w)
        color = (255, 255, 255) if rank == 0 else (25, 25, 25)
        cv2.rectangle(
            reference_heat,
            (x * stride + 1, y * stride + 1),
            ((x + 1) * stride - 2, (y + 1) * stride - 2),
            color,
            2,
        )
        cv2.putText(
            reference_heat,
            str(rank + 1),
            (x * stride + 3, y * stride + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            color,
            1,
            cv2.LINE_AA,
        )
    center = (int((pred[0] + 0.5) * stride), int((pred[1] + 0.5) * stride))
    cv2.circle(reference_heat, center, 7, (70, 255, 155), 3, cv2.LINE_AA)

    body = np.concatenate((current, reference_heat), axis=1)
    header = np.full((60, body.shape[1], 3), 15, dtype=np.uint8)
    cv2.putText(
        header,
        f"{clip_dir.name} | frame={frame_id} token={token_id} xy=({qx},{qy})",
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 245, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "LEFT: nominal query token   RIGHT: raw similarity heatmap + Top-L boxes + verified read point",
        (12, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (120, 225, 195),
        1,
        cv2.LINE_AA,
    )
    result = np.concatenate((header, body), axis=0)
    output_dir = clip_dir / "query_heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"frame_{frame_id:06d}_token_{token_id:04d}.png"
    if not cv2.imwrite(str(target), result):
        raise RuntimeError(f"Cannot write {target}")
    reject = float(np.load(clip_dir / "arrays" / "reject_prob.npy", mmap_mode="r")[frame_id, token_id])
    confidence = float(np.load(clip_dir / "arrays" / "token_confidence.npy", mmap_mode="r")[frame_id, token_id])
    return {
        "display_name": display_name,
        "frame_id": frame_id,
        "token_id": token_id,
        "query_xy": [qx, qy],
        "predicted_reference_xy": pred.tolist(),
        "topl_index": indices.tolist(),
        "verified_weight": weights.tolist(),
        "reject_probability": reject,
        "confidence": confidence,
        "image": str(target.relative_to(clip_dir)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    all_examples = {}
    for item in run_manifest["clips"]:
        clip = item["clip"]
        clip_dir = run_dir / clip["clip_id"]
        grid = item["token_grid"]
        test_frames = item["test_frames"]
        selected_frames = [test_frames[0], test_frames[len(test_frames) // 2], test_frames[-1]]
        examples = []
        for frame_id in selected_frames:
            token_id = choose_token(clip_dir, frame_id, grid["height"], grid["width"])
            examples.append(
                draw_example(
                    clip_dir,
                    Path(clip["video_path"]),
                    frame_id,
                    token_id,
                    clip["display_name"],
                    grid["height"],
                    grid["width"],
                    grid["stride"],
                )
            )
        write_json(clip_dir / "query_examples.json", examples)
        all_examples[clip["clip_id"]] = examples
    write_json(run_dir / "query_examples.json", all_examples)
    print(json.dumps(all_examples, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
