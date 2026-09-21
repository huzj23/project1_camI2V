# -*- coding: utf-8 -*-
"""P0 探针 —— VACE 挂载点验证（阶段 A：不需要 GPU、不需要跑推理）

本文件是服务器上 /data/raw/huzijian/project1_camI2V/code/Wan2.1/codex_p0_probe.py
的本地镜像，用于仓库留档。

上游基线：Wan-Video/Wan2.1 @ 9737cba9c1c3c4d04b33fcad41c111989865d315
本文件为本项目新增（AGENTS.md R5），不修改上游任何文件。

阶段 A 只做三件事：
  A1  读 safetensors 头部，列出全部权重键与形状（不加载张量、不占显存）
  A2  在 meta 设备上按 config.json 实例化 VaceWanModel，逐键比对键集与形状
  A3  计算 token 网格数学：由 (F,H,W) 推出 L、首/尾帧 token 索引区间、
      以及各 vace block 与主干 block 的挂载路径

阶段 B（需要 GPU）：真实前向 + 逐层 c 干净度 + 零初始化恒等性。本文件不含。

运行方式（服务器）：
    source /data/raw/miniconda3/etc/profile.d/conda.sh
    conda activate /home/chenliang/.conda/envs/huzj_camI2V_VACE
    cd /data/raw/huzijian/project1_camI2V/code/Wan2.1
    python -u /data/raw/huzijian/project1_camI2V/code/Wan2.1/codex_p0_probe.py
"""
import json
import os
import sys

PROJ = "/data/raw/huzijian/project1_camI2V"
MDL = os.path.join(PROJ, "model", "Wan2.1-VACE-1.3B")
REPO = os.path.join(PROJ, "code", "Wan2.1")
sys.path.insert(0, REPO)

CKPT = os.path.join(MDL, "diffusion_pytorch_model.safetensors")
CFG = os.path.join(MDL, "config.json")


def a1_read_ckpt_keys():
    from safetensors import safe_open
    keys = {}
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        for k in f.keys():
            keys[k] = tuple(f.get_slice(k).get_shape())
    print("=== A1. checkpoint 权重结构（不加载张量） ===")
    print("文件: %s" % CKPT)
    print("键数: %d" % len(keys))
    groups = {}
    for k, shp in keys.items():
        parts = k.split(".")
        prefix = ".".join(parts[:2]) if parts[0] == "blocks" else parts[0]
        groups.setdefault(prefix, []).append((k, shp))
    for g in sorted(groups):
        print("  %-28s %4d 个" % (g, len(groups[g])))
    return keys


def a2_compare_with_model(ckpt_keys):
    import torch
    from wan.modules.vace_model import VaceWanModel

    with open(CFG) as fh:
        cfg = json.load(fh)
    accepted = ("vace_layers", "vace_in_dim", "model_type", "patch_size", "text_len",
                "in_dim", "dim", "ffn_dim", "freq_dim", "text_dim", "out_dim",
                "num_heads", "num_layers", "window_size", "qk_norm",
                "cross_attn_norm", "eps")
    kwargs = {k: cfg[k] for k in accepted if k in cfg}

    print("\n=== A2. 按 config.json 实例化（meta 设备，不占内存/显存） ===")
    print("构造参数: %s" % json.dumps(kwargs, ensure_ascii=False))
    with torch.device("meta"):
        model = VaceWanModel(**kwargs)

    msd = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    mk, ck = set(msd), set(ckpt_keys)

    missing = sorted(mk - ck)
    unexpected = sorted(ck - mk)
    shape_mismatch = sorted(k for k in (mk & ck) if msd[k] != ckpt_keys[k])

    print("模型键数: %d   checkpoint 键数: %d" % (len(mk), len(ck)))
    print("模型有/权重无 (missing)      : %d" % len(missing))
    for k in missing[:20]:
        print("    - %s  %s" % (k, msd[k]))
    print("权重有/模型无 (unexpected)   : %d" % len(unexpected))
    for k in unexpected[:20]:
        print("    + %s  %s" % (k, ckpt_keys[k]))
    print("形状不一致 (shape mismatch)  : %d" % len(shape_mismatch))
    for k in shape_mismatch[:20]:
        print("    ! %s  模型%s vs 权重%s" % (k, msd[k], ckpt_keys[k]))

    strict_ok = not (missing or unexpected or shape_mismatch)
    print("\n>>> STRICT_MATCH = %s" % strict_ok)

    print("\n--- 主干 blocks（前 2 个的完整模块路径） ---")
    names = [n for n, _ in model.named_modules()]
    for n in names:
        if n.startswith("blocks.0.") or n.startswith("blocks.1."):
            print("    %s" % n)
    print("--- vace_blocks（前 2 个的完整模块路径） ---")
    for n in names:
        if n.startswith("vace_blocks.0.") or n.startswith("vace_blocks.1."):
            print("    %s" % n)
    print("--- vace 相关顶层模块 ---")
    for n, m in model.named_modules():
        if n.count(".") == 0 and n in ("vace_blocks", "vace_patch_embedding", "patch_embedding",
                                       "blocks", "text_embedding", "time_embedding",
                                       "time_projection", "head"):
            print("    %-24s %s" % (n, type(m).__name__))
    return model, kwargs, strict_ok


def a3_token_math(vace_layers, patch=(1, 2, 2), vae_spatial=8, vae_temporal=4):
    print("\n=== A3. token 网格数学 ===")
    print("VAE: 空间 1/%d，时间 1/%d（F_z = 1 + (F-1)/%d）" % (vae_spatial, vae_temporal, vae_temporal))
    print("DiT patch: %s" % (patch,))
    print("展平顺序: x.flatten(2).transpose(1,2)  即 (F_z, H_p, W_p) 行优先，W 最快")
    print("")
    print("%-22s %-22s %-8s %-10s %s" % ("RGB (F,H,W)", "latent (Fz,Hz,Wz)", "grid Hp,Wp", "L",
                                         "首帧索引 / 尾帧索引"))
    cases = [(81, 480, 832), (81, 720, 1280), (16, 256, 256), (49, 480, 832), (21, 256, 256)]
    for F, H, W in cases:
        Fz = 1 + (F - 1) // vae_temporal
        Hz, Wz = H // vae_spatial, W // vae_spatial
        Hp, Wp = Hz // patch[1], Wz // patch[2]
        L = Fz * Hp * Wp
        per = Hp * Wp
        first = (0, per - 1)
        last = ((Fz - 1) * per, Fz * per - 1)
        print("%-22s %-22s %-8s %-10s [%d, %d] / [%d, %d]"
              % ("(%d,%d,%d)" % (F, H, W), "(%d,%d,%d)" % (Fz, Hz, Wz),
                 "%dx%d" % (Hp, Wp), L, first[0], first[1], last[0], last[1]))
    print("")
    print("--- 各 vace block 与主干 block 的对应 ---")
    mapping = {i: n for n, i in enumerate(vace_layers)}
    for i in vace_layers:
        print("    blocks[%2d].block_id = %2d   <->   vace_blocks[%2d]" % (i, mapping[i], mapping[i]))
    print("    主干 blocks 总数 = %d ；参与注入的 = %d ；不参与的 = %d"
          % (max(vace_layers) + 2, len(vace_layers), 30 - len(vace_layers)))


def main():
    print("上游基线 commit: 9737cba9c1c3c4d04b33fcad41c111989865d315")
    print("本脚本为新增文件，未修改上游任何内容")
    print("")
    ckpt_keys = a1_read_ckpt_keys()
    model, kwargs, strict_ok = a2_compare_with_model(ckpt_keys)
    a3_token_math(kwargs["vace_layers"])
    print("\n=== 阶段 A 结论 ===")
    print("STRICT_MATCH = %s" % strict_ok)
    print("（阶段 B 需要 GPU：真实前向 + c 逐层干净度 + 零初始化恒等性）")


if __name__ == "__main__":
    main()
