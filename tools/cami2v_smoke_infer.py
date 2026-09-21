import argparse
import gc
import json
import os
import time
import traceback

import numpy as np
import torch
import torchvision
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything


NEGATIVE_PROMPT = (
    "Fast movement, jittery motion, abrupt transitions, distorted body, "
    "blurry, artifacts."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke-test CamI2V and DynamiCrafter inference with official configs."
    )
    parser.add_argument(
        "--repo",
        default="/data/raw/huzijian/project1_camI2V/code/CamI2V",
        help="Path to the cloned CamI2V repository.",
    )
    parser.add_argument(
        "--out",
        default="/data/raw/huzijian/project1_camI2V/output",
        help="Directory for generated smoke-test videos.",
    )
    parser.add_argument("--steps", type=int, default=2, help="DDIM steps for smoke inference.")
    parser.add_argument(
        "--which",
        choices=["cami2v", "baseline", "both"],
        default="both",
        help="Run CamI2V, baseline DynamiCrafter, or both.",
    )
    return parser.parse_args()


def lazy_import_repo():
    from utils.utils import instantiate_from_config
    from CameraControl.data.single_image_for_inference import SingleImageForInference

    return instantiate_from_config, SingleImageForInference


def parse_ckpt(path):
    print("LOAD_TORCH", path, os.path.getsize(path), flush=True)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print("WEIGHTS_ONLY_FAILED_FALLBACK", repr(exc), flush=True)
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict) and "module" in obj:
        state_dict = obj["module"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    else:
        state_dict = obj

    state_dict = {
        key.replace("framestride_embed", "fps_embedding"): value
        for key, value in state_dict.items()
    }
    print("STATE_DICT_KEYS", len(state_dict), flush=True)
    return state_dict


def load_model(config_file, ckpt_path):
    instantiate_from_config, _ = lazy_import_repo()
    print("CONFIG", config_file, flush=True)
    config = OmegaConf.load(config_file)
    model = instantiate_from_config(config.model)

    state_dict = parse_ckpt(ckpt_path)
    try:
        result = model.load_state_dict(state_dict, strict=True)
        print("STRICT_LOAD_OK", result, flush=True)
    except Exception as exc:
        print("STRICT_LOAD_FAILED", repr(exc), flush=True)
        result = model.load_state_dict(state_dict, strict=False)
        print(
            "NONSTRICT_LOAD",
            len(result.missing_keys),
            "missing",
            len(result.unexpected_keys),
            "unexpected",
            flush=True,
        )
        print("MISSING_SAMPLE", result.missing_keys[:20], flush=True)
        print("UNEXPECTED_SAMPLE", result.unexpected_keys[:20], flush=True)

    model.uncond_type = "negative_prompt"
    if hasattr(model, "rand_cond_frame"):
        model.rand_cond_frame = False
    model = model.to(torch.float16).eval().cuda()
    return model


def camera_pose_lerp(c2w, target_frames):
    weights = torch.linspace(0, c2w.size(0) - 1, target_frames, dtype=c2w.dtype)
    left = weights.floor().long()
    right = weights.ceil().long()
    frac = weights.unsqueeze(-1).unsqueeze(-1).frac()
    return torch.lerp(c2w[left], c2w[right], frac)


def build_batch(width=256, height=256, camera_pose_type="zoom in"):
    _, SingleImageForInference = lazy_import_repo()
    caption = (
        "Tranquil mountain lake surrounded by lush greenery and towering peaks "
        "under a clear blue sky. dramatic camera movement, high quality, "
        "cinematic shot, photorealistic, detailed, smooth motion"
    )

    image_path = "demo/pexels/pexels-eberhardgross-534164.jpg"
    image = np.array(Image.open(image_path).convert("RGB"))
    with open("demo/camera_poses.json", "r", encoding="utf-8") as handle:
        pose_path = json.load(handle)[camera_pose_type]

    camera_data = torch.from_numpy(np.loadtxt(pose_path, comments="https"))
    w2cs_3x4 = camera_data[:, 7:].reshape(-1, 3, 4).to(torch.float32)
    bottom = torch.tensor([[[0, 0, 0, 1]]], dtype=torch.float32).repeat(
        w2cs_3x4.shape[0], 1, 1
    )
    w2cs_4x4 = torch.cat([w2cs_3x4, bottom], dim=1)
    c2ws_4x4 = w2cs_4x4.inverse()
    c2ws_lerp_4x4 = camera_pose_lerp(c2ws_4x4, 16)[:16]
    w2cs_lerp_4x4 = c2ws_lerp_4x4.inverse()

    processor = SingleImageForInference(
        video_length=16,
        resolution=(height, width),
        spatial_transform_type="resize_center_crop",
        device="cuda",
    )
    batch = processor.get_batch_input(
        image,
        caption,
        w2cs_lerp_4x4[:16, :3],
        frame_stride=2,
    )
    batch["cond_frame_index"] = torch.tensor([0], device="cuda", dtype=torch.long)
    return batch


def save_video(video, path, fps=10):
    n, c, t, h, w = video.shape
    video = torch.nn.functional.interpolate(
        rearrange(video, "n c t h w -> (n t) c h w"),
        (h // 2 * 2, w // 2 * 2),
        mode="bilinear",
    )
    video = rearrange(video, "(n t) c h w -> t n c h w", n=n, t=t)
    frames = [
        torchvision.utils.make_grid(framesheet, nrow=1, padding=0)
        for framesheet in video
    ]
    grid = torch.stack(frames, dim=0)
    grid = ((grid + 1.0) / 2.0 * 255).clamp(0, 255).to(torch.uint8)
    grid = grid.permute(0, 2, 3, 1)
    torchvision.io.write_video(
        path,
        grid.cpu(),
        fps=fps,
        video_codec="h264",
        options={"crf": "10"},
    )
    print("SAVED", path, os.path.getsize(path), flush=True)


def run_one(name, config_file, ckpt_path, steps, camera=False):
    print("RUN_START", name, "steps", steps, "camera", camera, flush=True)
    seed_everything(123)
    torch.cuda.empty_cache()
    batch = build_batch()
    model = load_model(config_file, ckpt_path)

    kwargs = {
        "ddim_steps": steps,
        "ddim_eta": 1.0,
        "unconditional_guidance_scale": 5.5,
        "timestep_spacing": "uniform_trailing",
        "guidance_rescale": 0.7,
        "negative_prompt": NEGATIVE_PROMPT,
    }
    if camera:
        kwargs.update(
            {
                "cond_frame_index": batch["cond_frame_index"],
                "camera_cfg": 1.0,
                "camera_cfg_scheduler": "constant",
                "enable_camera_condition": True,
                "trace_scale_factor": 1.0,
            }
        )

    start = time.time()
    with torch.no_grad(), torch.autocast("cuda"):
        output = model.log_images(batch, **kwargs)

    path = os.path.join(args.out, f"{name}_{steps}steps_smoke.mp4")
    save_video(output["samples"].detach().clamp(-1, 1).cpu(), path)
    print("RUN_DONE", name, "elapsed_sec", round(time.time() - start, 2), flush=True)

    del model, output, batch
    gc.collect()
    torch.cuda.empty_cache()
    return path


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.chdir(args.repo)
    print("CWD", os.getcwd(), flush=True)
    print(
        "TORCH",
        torch.__version__,
        "CUDA",
        torch.version.cuda,
        "AVAILABLE",
        torch.cuda.is_available(),
        "COUNT",
        torch.cuda.device_count(),
        flush=True,
    )

    try:
        generated = []
        if args.which in {"cami2v", "both"}:
            generated.append(
                run_one(
                    "cami2v_256",
                    "configs/inference/003_cami2v_256x256.yaml",
                    "ckpts/256_cami2v.pt",
                    args.steps,
                    camera=True,
                )
            )
        if args.which in {"baseline", "both"}:
            generated.append(
                run_one(
                    "dynamicrafter_256",
                    "configs/inference/000_dynamicrafter_256x256.yaml",
                    "pretrained_models/DynamiCrafter/model.ckpt",
                    args.steps,
                    camera=False,
                )
            )
        print("SMOKE_SUCCESS", generated, flush=True)
    except Exception:
        print("SMOKE_FAILED", flush=True)
        traceback.print_exc()
        raise
