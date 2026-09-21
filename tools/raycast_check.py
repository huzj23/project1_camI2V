#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""raycast_check.py — A-3 + 投射器自检

做三件事：
  1. 读含天空版 CSV，打印相机位姿字段（确认坐标系与 FOV）
  2. 只加载相机附近区块的体素，用 Amanatides-Woo DDA 投射一张深度图
  3. 与 Flashback 导出的深度 mp4 并排存图，人工核对

不写任何上游文件；只读存档与本地素材。
"""
import csv
import math
import os
import struct
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mca_probe import read_chunk  # noqa: E402

MC = r"D:\Mc\.minecraft\versions\26.1.2-V2DiT-Data"
REGION = os.path.join(MC, r"saves\新的世界\dimensions\minecraft\overworld\region")
CSV = os.path.join(MC, r"v2dit_capture\trajectory\20260921-154140-flashback_export.csv")
DEPTH_MP4 = r"C:\Users\12447\Desktop\plan\ShortTerm_WeeksCounted\project4_CAMI2V\data\2026-09-21T15_25_05.mp4"
OUT = r"C:\Users\12447\Desktop\plan\ShortTerm_WeeksCounted\project4_CAMI2V\tmp\raycast"
os.makedirs(OUT, exist_ok=True)

EMPTY = {
    "minecraft:air", "minecraft:cave_air", "minecraft:void_air",
    "minecraft:water", "minecraft:short_grass", "minecraft:tall_grass",
    "minecraft:grass", "minecraft:fern", "minecraft:large_fern",
    "minecraft:dandelion", "minecraft:poppy", "minecraft:torch",
    "minecraft:wall_torch", "minecraft:lantern", "minecraft:vine",
    "minecraft:snow", "minecraft:light", "minecraft:barrier",
    "minecraft:oak_leaves", "minecraft:spruce_leaves", "minecraft:birch_leaves",
    "minecraft:glass", "minecraft:glass_pane", "minecraft:flower_pot",
}
EMPTY_PREFIX_SKIP = ("minecraft:potted_",)


def is_solid(name):
    if name in EMPTY:
        return False
    for p in EMPTY_PREFIX_SKIP:
        if name.startswith(p):
            return False
    return True


def read_pose(path, idx):
    with open(path, newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        for row in r:
            if int(row["frame_index"]) == idx:
                return row
    raise KeyError(idx)


def unpack_section(bs, nbits_min=4):
    """block_states.data 是 packed long 数组，每 long 放 floor(64/nbits) 个索引。"""
    pal = bs.get("palette", [])
    data = bs.get("data")
    if data is None:
        return pal, None, 0
    nbits = max(nbits_min, (len(pal) - 1).bit_length())
    per_long = 64 // nbits
    mask = (1 << nbits) - 1
    idx = np.zeros(4096, dtype=np.int32)
    p = 0
    for lv in data:
        v = lv & 0xFFFFFFFFFFFFFFFF
        for k in range(per_long):
            if p >= 4096:
                break
            idx[p] = (v >> (k * nbits)) & mask
            p += 1
        if p >= 4096:
            break
    return pal, idx, nbits


def load_voxels(cam_x, cam_z, radius_chunks=4):
    """加载相机附近区块的体素占用集合 {(x,y,z)}。"""
    solid = set()
    names = {}
    bx0, bz0 = int(math.floor(cam_x)) >> 9 << 9, int(math.floor(cam_z)) >> 9 << 9
    files = [f for f in os.listdir(REGION) if f.endswith(".mca")]
    loaded = 0
    for f in files:
        m = f.split(".")
        rx, rz = int(m[1]), int(m[2])
        # 该 region 覆盖 512x512 方块
        if abs(rx * 512 + 256 - cam_x) > radius_chunks * 16 + 512:
            continue
        if abs(rz * 512 + 256 - cam_z) > radius_chunks * 16 + 512:
            continue
        path = os.path.join(REGION, f)
        for dx in range(32):
            for dz in range(32):
                cx, cz = rx * 32 + dx, rz * 32 + dz
                chunk_x, chunk_z = cx * 16, cz * 16
                if abs(chunk_x + 8 - cam_x) > radius_chunks * 16:
                    continue
                if abs(chunk_z + 8 - cam_z) > radius_chunks * 16:
                    continue
                try:
                    nbt = read_chunk(path, cx, cz)
                except Exception:
                    continue
                if nbt is None:
                    continue
                loaded += 1
                for s in nbt.get("sections", []):
                    y0 = s.get("Y", 0) * 16
                    bs = s.get("block_states")
                    if not bs:
                        continue
                    pal, idx, nbits = unpack_section(bs)
                    if idx is None:
                        # 单一调色板
                        nm = pal[0].get("Name") if pal else "minecraft:air"
                        if is_solid(nm):
                            for yy in range(16):
                                for zz in range(16):
                                    for xx in range(16):
                                        solid.add((chunk_x + xx, y0 + yy, chunk_z + zz))
                        continue
                    for i, pi in enumerate(idx):
                        if pi >= len(pal):
                            continue
                        nm = pal[pi].get("Name")
                        if is_solid(nm):
                            xx = i & 15
                            zz = (i >> 4) & 15
                            yy = i >> 8
                            solid.add((chunk_x + xx, y0 + yy, chunk_z + zz))
    return solid, loaded


def dda(origin, direction, solid, max_dist=256.0, step_max=2048):
    """Amanatides-Woo DDA。返回首次命中的距离，未命中返回 max_dist。"""
    ox, oy, oz = origin
    dx, dy, dz = direction
    x, y, z = math.floor(ox), math.floor(oy), math.floor(oz)
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    sz = 1 if dz > 0 else -1
    tdx = abs(1.0 / dx) if dx != 0 else float("inf")
    tdy = abs(1.0 / dy) if dy != 0 else float("inf")
    tdz = abs(1.0 / dz) if dz != 0 else float("inf")
    tx = ((x + (1 if dx > 0 else 0)) - ox) / dx if dx != 0 else float("inf")
    ty = ((y + (1 if dy > 0 else 0)) - oy) / dy if dy != 0 else float("inf")
    tz = ((z + (1 if dz > 0 else 0)) - oz) / dz if dz != 0 else float("inf")
    t = 0.0
    for _ in range(step_max):
        if (x, y, z) in solid:
            return t
        if tx < ty and tx < tz:
            x += sx
            t = tx
            tx += tdx
        elif ty < tz:
            y += sy
            t = ty
            ty += tdy
        else:
            z += sz
            t = tz
            tz += tdz
        if t > max_dist:
            return max_dist
    return max_dist


def main():
    row = read_pose(CSV, 0)
    print("=== A-3 相机位姿（含天空版 CSV 第 0 帧） ===")
    for k in ("frame_index", "x", "y", "z", "yaw_deg", "pitch_deg",
              "quat_x", "quat_y", "quat_z", "quat_w",
              "right_x", "right_y", "right_z", "up_x", "up_y", "up_z",
              "back_x", "back_y", "back_z", "fov_deg", "width", "height"):
        print("  %-12s %s" % (k, row.get(k)))

    px, py, pz = float(row["x"]), float(row["y"]), float(row["z"])
    R = np.array([float(row["right_x"]), float(row["right_y"]), float(row["right_z"])])
    U = np.array([float(row["up_x"]), float(row["up_y"]), float(row["up_z"])])
    B = np.array([float(row["back_x"]), float(row["back_y"]), float(row["back_z"])])
    print("\n  相机位置 = (%.2f, %.2f, %.2f)" % (px, py, pz))
    print("  right = %s" % np.round(R, 4))
    print("  up    = %s" % np.round(U, 4))
    print("  back  = %s" % np.round(B, 4))
    print("  观察方向 -back = %s" % np.round(-B, 4))
    print("  正交性 r·u=%.6f r·b=%.6f u·b=%.6f" % (R @ U, R @ B, U @ B))

    print("\n=== A-2 载入相机附近体素 ===")
    solid, nchunk = load_voxels(px, pz, radius_chunks=4)
    print("  载入区块 %d 个，实体方块 %d 个" % (nchunk, len(solid)))
    if not solid:
        print("  !! 没载到方块，检查区块坐标筛选"); return

    W, H = int(row["width"]), int(row["height"])
    fov_v = float(row["fov_deg"])
    f = (H / 2.0) / math.tan(math.radians(fov_v) / 2.0)
    cx, cy = W / 2.0, H / 2.0
    print("  FOV_v=%.1f°  f=%.2f px  图像 %dx%d" % (fov_v, f, W, H))

    print("\n=== A-4 DDA 光线投射（自检分辨率 %dx%d） ===" % (W // 4, H // 4))
    sw, sh = W // 4, H // 4
    depth = np.zeros((sh, sw), dtype=np.float32)
    origin = (px, py, pz)
    for j in range(sh):
        v = (j + 0.5) * 4.0
        for i in range(sw):
            u = (i + 0.5) * 4.0
            xc = (u - cx) / f
            yc = (cy - v) / f
            d = xc * R + yc * U + (-1.0) * B
            n = np.linalg.norm(d)
            if n == 0:
                continue
            depth[j, i] = dda(origin, (d[0] / n, d[1] / n, d[2] / n), solid)

    hit = (depth < 255.0).mean()
    print("  命中率 %.1f%%   距离范围 %.2f .. %.2f" % (hit * 100, depth.min(), depth.max()))

    # 归一化存图：近白远黑（与 Flashback 一致的取向）
    vis = np.zeros((sh, sw), dtype=np.uint8)
    m = depth < 255.0
    if m.any():
        lo, hi = np.percentile(depth[m], 2), np.percentile(depth[m], 98)
        nn = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        vis = ((1.0 - nn) * 255).astype(np.uint8)
    Image.fromarray(vis).resize((W, H), Image.NEAREST).save(os.path.join(OUT, "raycast_depth.png"))
    np.save(os.path.join(OUT, "raycast_depth.npy"), depth)
    print("  已存 %s" % os.path.join(OUT, "raycast_depth.png"))

    # 取 Flashback 深度帧并排
    try:
        import cv2
        cap = cv2.VideoCapture(DEPTH_MP4)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, fr = cap.read()
        cap.release()
        if ok:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            Image.fromarray(g).save(os.path.join(OUT, "flashback_depth.png"))
            print("  已存 %s" % os.path.join(OUT, "flashback_depth.png"))
    except Exception as exc:
        print("  深度视频读取失败: %r" % (exc,))

    print("\nRAYCAST_CHECK_DONE")


if __name__ == "__main__":
    main()
