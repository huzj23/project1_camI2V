#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gt_reproject.py — A-5 真值对应表（token 级，v2 优化版）

v1 超时原因：全量加载 17 个 region（最多 17408 个区块）+ Python set of tuples。
v2 三项优化：
  1. 先读全部位姿算包围盒，只加载必要区块
  2. numpy 三维占据数组代替 set of tuples
  3. 关 stdout 缓冲，可见进度

坐标链（VACE 侧 size=(480,832)）：
  原始 1280x720，fov_v=70  -> f_orig = 360/tan(35)
  cover 缩放到 (480,832) 再中心裁剪：s = max(480/1280, 832/720)
  token 网格 52x30，token(r,c) 中心在裁剪图 = (c*16+7.5, r*16+7.5)
  射线 d = xc*right + yc*up + (-1)*back
"""
import csv
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mca_probe import read_chunk                      # noqa: E402

MC = r"D:\Mc\.minecraft\versions\26.1.2-V2DiT-Data"
REGION = os.path.join(MC, r"saves\新的世界\dimensions\minecraft\overworld\region")
CSV = os.path.join(MC, r"v2dit_capture\trajectory\20260921-154140-flashback_export.csv")
OUT = r"C:\Users\12447\Desktop\plan\ShortTerm_WeeksCounted\project4_CAMI2V\tmp\raycast"
os.makedirs(OUT, exist_ok=True)

HP, WP, STRIDE = 52, 30, 16
ORIG_W, ORIG_H, FOV_V = 1280, 720, 70.0
CROP_W, CROP_H = 480, 832
VIEW_MARGIN = 140.0          # 包围盒外扩（方块）
Y_LO, Y_HI = -64, 200        # 关注的竖直范围

EMPTY = {
    "minecraft:air", "minecraft:cave_air", "minecraft:void_air",
    "minecraft:water", "minecraft:short_grass", "minecraft:tall_grass",
    "minecraft:grass", "minecraft:fern", "minecraft:large_fern",
    "minecraft:dandelion", "minecraft:poppy", "minecraft:torch",
    "minecraft:wall_torch", "minecraft:lantern", "minecraft:vine",
    "minecraft:snow", "minecraft:light", "minecraft:barrier",
    "minecraft:oak_leaves", "minecraft:spruce_leaves", "minecraft:birch_leaves",
    "minecraft:glass", "minecraft:glass_pane", "minecraft:flower_pot",
    "minecraft:player_head", "minecraft:wall_sign", "minecraft:oak_sign",
}
SKIP_PREFIX = ("minecraft:potted_", "minecraft:white_", "minecraft:red_",
               "minecraft:yellow_", "minecraft:orange_", "minecraft:lime_",
               "minecraft:cyan_", "minecraft:purple_", "minecraft:magenta_",
               "minecraft:pink_", "minecraft:gray_", "minecraft:black_",
               "minecraft:light_blue_", "minecraft:light_gray_", "minecraft:brown_",
               "minecraft:green_", "minecraft:blue_")


def is_solid(name):
    if name is None or name in EMPTY:
        return False
    for p in SKIP_PREFIX:
        if name.startswith(p) and (name.endswith("_banner") or name.endswith("_carpet")):
            return False
    return True


def load_all_poses(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)]
    return rows


def prep_geometry():
    s = max(CROP_W / ORIG_W, CROP_H / ORIG_H)
    up_w, up_h = int(round(ORIG_W * s)), int(round(ORIG_H * s))
    f_orig = (ORIG_H / 2.0) / math.tan(math.radians(FOV_V) / 2.0)
    return dict(s=s, up_w=up_w, up_h=up_h,
                off_x=(up_w - CROP_W) / 2.0, off_y=(up_h - CROP_H) / 2.0,
                f_up=f_orig * s)


def pose_arrays(p):
    R = np.array([float(p["right_x"]), float(p["right_y"]), float(p["right_z"])])
    U = np.array([float(p["up_x"]), float(p["up_y"]), float(p["up_z"])])
    B = np.array([float(p["back_x"]), float(p["back_y"]), float(p["back_z"])])
    o = np.array([float(p["x"]), float(p["y"]), float(p["z"])])
    return o, R, U, B


def build_occupancy(poses, geo):
    xs = np.array([float(p["x"]) for p in poses])
    ys = np.array([float(p["y"]) for p in poses])
    zs = np.array([float(p["z"]) for p in poses])
    x0 = int(math.floor(xs.min() - VIEW_MARGIN)); x1 = int(math.ceil(xs.max() + VIEW_MARGIN))
    z0 = int(math.floor(zs.min() - VIEW_MARGIN)); z1 = int(math.ceil(zs.max() + VIEW_MARGIN))
    y0, y1 = Y_LO, Y_HI
    nx, ny, nz = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1
    print("  包围盒 X[%d,%d] Y[%d,%d] Z[%d,%d] -> 数组 %dx%dx%d = %.1f MB"
          % (x0, x1, y0, y1, z0, z1, nx, ny, nz, nx * ny * nz / 1048576), flush=True)
    occ = np.zeros((nx, ny, nz), dtype=bool)

    cx0, cx1 = x0 >> 4, x1 >> 4
    cz0, cz1 = z0 >> 4, z1 >> 4
    print("  需要区块 x[%d,%d] z[%d,%d] = %d 个"
          % (cx0, cx1, cz0, cz1, (cx1 - cx0 + 1) * (cz1 - cz0 + 1)), flush=True)

    # 按 region 归组
    need = {}
    for ccx in range(cx0, cx1 + 1):
        for ccz in range(cz0, cz1 + 1):
            rx, rz = ccx >> 5, ccz >> 5
            need.setdefault((rx, rz), []).append((ccx, ccz))

    nchunk = 0
    nblock = 0
    t0 = time.time()
    for (rx, rz), chunks in need.items():
        fname = "r.%d.%d.mca" % (rx, rz)
        path = os.path.join(REGION, fname)
        if not os.path.isfile(path):
            continue
        for (ccx, ccz) in chunks:
            try:
                nbt = read_chunk(path, ccx, ccz)
            except Exception:
                continue
            if nbt is None:
                continue
            nchunk += 1
            bx, bz = ccx * 16, ccz * 16
            for sec in nbt.get("sections", []):
                sy = sec.get("Y", 0) * 16
                if sy + 15 < y0 or sy > y1:
                    continue
                bs = sec.get("block_states")
                if not bs:
                    continue
                pal = bs.get("palette", [])
                data = bs.get("data")
                solid_idx = {i for i, e in enumerate(pal) if is_solid(e.get("Name"))}
                if not solid_idx:
                    continue
                if data is None:
                    if 0 in solid_idx:
                        for yy in range(16):
                            gy = sy + yy
                            if not (y0 <= gy <= y1):
                                continue
                            occ[bx - x0:bx - x0 + 16, gy - y0, bz - z0:bz - z0 + 16] = True
                        nblock += 256
                    continue
                nbits = max(4, (len(pal) - 1).bit_length())
                per_long = 64 // nbits
                mask = (1 << nbits) - 1
                idx = np.empty(4096, dtype=np.int32)
                p_ = 0
                for lv in data:
                    v = lv & 0xFFFFFFFFFFFFFFFF
                    for k in range(per_long):
                        if p_ >= 4096:
                            break
                        idx[p_] = (v >> (k * nbits)) & mask
                        p_ += 1
                    if p_ >= 4096:
                        break
                sel = np.isin(idx, list(solid_idx))
                if not sel.any():
                    continue
                li = np.nonzero(sel)[0]
                lx = li & 15
                lz = (li >> 4) & 15
                ly = li >> 8
                gy = sy + ly
                gx = (bx - x0) + lx
                gz = (bz - z0) + lz
                keep = ((gy >= y0) & (gy <= y1) &
                        (gx >= 0) & (gx < nx) & (gz >= 0) & (gz < nz))
                if keep.any():
                    occ[gx[keep], gy[keep] - y0, gz[keep]] = True
                    nblock += int(keep.sum())
        if nchunk % 200 == 0 and nchunk:
            print("    ...已解析 %d 区块，%.0fs" % (nchunk, time.time() - t0), flush=True)
    print("  区块 %d 个，实体方块 %d 个，用时 %.0fs" % (nchunk, nblock, time.time() - t0), flush=True)
    return occ, (x0, y0, z0), (nx, ny, nz)


def token_dirs(pose, geo):
    o, R, U, B = pose_arrays(pose)
    dirs = np.zeros((HP, WP, 3))
    for r in range(HP):
        vc = r * STRIDE + STRIDE / 2.0 - 0.5
        for c in range(WP):
            uc = c * STRIDE + STRIDE / 2.0 - 0.5
            u_up = uc + geo["off_x"]
            v_up = vc + geo["off_y"]
            xc = (u_up - geo["up_w"] / 2.0) / geo["f_up"]
            yc = (geo["up_h"] / 2.0 - v_up) / geo["f_up"]
            d = xc * R + yc * U - B
            dirs[r, c] = d / np.linalg.norm(d)
    return o, dirs, R, U, B


def dda(origin, d, occ, org, shape, max_dist=200.0):
    x0, y0, z0 = org
    nx, ny, nz = shape
    ox, oy, oz = origin
    dx, dy, dz = d
    x, y, z = math.floor(ox), math.floor(oy), math.floor(oz)
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    sz = 1 if dz > 0 else -1
    tdx = abs(1.0 / dx) if dx else float("inf")
    tdy = abs(1.0 / dy) if dy else float("inf")
    tdz = abs(1.0 / dz) if dz else float("inf")
    tx = ((x + (1 if dx > 0 else 0)) - ox) / dx if dx else float("inf")
    ty = ((y + (1 if dy > 0 else 0)) - oy) / dy if dy else float("inf")
    tz = ((z + (1 if dz > 0 else 0)) - oz) / dz if dz else float("inf")
    t = 0.0
    for _ in range(1500):
        ix, iy, iz = x - x0, y - y0, z - z0
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz and occ[ix, iy, iz]:
            return t, True
        if tx < ty and tx < tz:
            x += sx; t = tx; tx += tdx
        elif ty < tz:
            y += sy; t = ty; ty += tdy
        else:
            z += sz; t = tz; tz += tdz
        if t > max_dist:
            break
    return max_dist, False


def main():
    t_start = time.time()
    geo = prep_geometry()
    print("=== 预处理几何 ===")
    print("  scale=%.5f  上采样 %dx%d  裁剪偏移 (%.1f, %.1f)  f_up=%.2f"
          % (geo["s"], geo["up_w"], geo["up_h"], geo["off_x"], geo["off_y"], geo["f_up"]), flush=True)

    poses = load_all_poses(CSV)
    print("\n=== 位姿 ===")
    print("  共 %d 帧" % len(poses), flush=True)

    # 选首尾帧：从第 0 帧出发，找相机位移最大的帧作为尾帧
    p0 = np.array([float(poses[0]["x"]), float(poses[0]["y"]), float(poses[0]["z"])])
    disp = np.array([np.linalg.norm(
        np.array([float(p["x"]), float(p["y"]), float(p["z"])]) - p0) for p in poses])
    cand = [(i, float(disp[i])) for i in (20, 40, 60, 80, 120, 160, 240, 360)]
    print("  与第 0 帧的相机位移（方块）：")
    for i, d in cand:
        if i < len(poses):
            print("    frame %3d : %.3f" % (i, d))
    # 在 VACE 可用的 81 帧范围内挑位移最大者；若都很小则放宽到全片
    best = int(np.argmax(disp[:81])) if len(poses) >= 81 else int(np.argmax(disp))
    if disp[best] < 0.5:
        # 前 81 帧静止：改用「给定基线」策略 —— 取位移落在目标区间的帧
        target = float(os.environ.get("GT_BASELINE", "2.5"))
        best = int(np.argmin(np.abs(disp - target)))
        print("  !! 前 81 帧几乎无位移；按目标基线 %.2f 方块 选取 frame %d（实际 %.3f）"
              % (target, best, disp[best]))
    LAST_IDX = best if best > 0 else 80

    p_first, p_last = poses[0], poses[LAST_IDX]
    print("  选定：首帧 frame 0   尾帧 frame %d   位移 %.3f 方块"
          % (LAST_IDX, disp[LAST_IDX]))
    print("  首帧 (%.2f, %.2f, %.2f)" % (float(p_first["x"]), float(p_first["y"]), float(p_first["z"])))
    print("  尾帧 (%.2f, %.2f, %.2f)" % (float(p_last["x"]), float(p_last["y"]), float(p_last["z"])), flush=True)

    print("\n=== 载入体素（仅包围盒内） ===", flush=True)
    occ, org, shape = build_occupancy(poses, geo)
    print("  占据率 %.3f%%" % (100.0 * occ.sum() / occ.size), flush=True)

    print("\n=== 首帧 token 投射（%dx%d） ===" % (HP, WP), flush=True)
    o1, d1, _, _, _ = token_dirs(p_first, geo)
    dist1 = np.full((HP, WP), 200.0, dtype=np.float32)
    hit1 = np.zeros((HP, WP), dtype=bool)
    for r in range(HP):
        for c in range(WP):
            t, h = dda(o1, d1[r, c], occ, org, shape)
            dist1[r, c] = t
            hit1[r, c] = h
    print("  命中 %d/%d (%.1f%%)  距离 %.2f..%.2f"
          % (hit1.sum(), HP * WP, 100.0 * hit1.mean(), dist1[hit1].min(), dist1[hit1].max()), flush=True)

    print("\n=== A-5 逆投影到尾帧 + 可见性 ===", flush=True)
    o2, R2, U2, B2 = pose_arrays(p_last)
    gt = np.full((HP, WP, 2), -1, dtype=np.int32)
    vis = np.zeros((HP, WP), dtype=bool)
    n_behind = n_oob = n_occ = 0
    for r in range(HP):
        for c in range(WP):
            if not hit1[r, c]:
                continue
            p3 = o1 + d1[r, c] * float(dist1[r, c])
            rel = p3 - o2
            # back 轴正向指向相机后方 => 前方点的 (rel·B) 为负，深度 = -(rel·B)
            depth = -float(rel @ B2)
            if depth <= 1e-6:
                n_behind += 1
                continue
            u_up = float(rel @ R2) / depth * geo["f_up"] + geo["up_w"] / 2.0
            v_up = geo["up_h"] / 2.0 - float(rel @ U2) / depth * geo["f_up"]
            u_c, v_c = u_up - geo["off_x"], v_up - geo["off_y"]
            if not (0 <= u_c < CROP_W and 0 <= v_c < CROP_H):
                n_oob += 1
                continue
            rr, cc = int(v_c // STRIDE), int(u_c // STRIDE)
            if not (0 <= rr < HP and 0 <= cc < WP):
                n_oob += 1
                continue
            dv = p3 - o2
            dl = float(np.linalg.norm(dv))
            _, blocked = dda(o2, dv / dl, occ, org, shape, max_dist=dl - 0.05)
            if blocked:
                n_occ += 1
                continue
            gt[r, c] = (rr, cc)
            vis[r, c] = True
    print("  可见 %d/%d (%.1f%%)   剔除：相机后方 %d / 出画 %d / 被遮挡 %d"
          % (vis.sum(), HP * WP, 100.0 * vis.mean(), n_behind, n_oob, n_occ), flush=True)

    np.save(os.path.join(OUT, "gt_first_to_last.npy"), gt)
    np.save(os.path.join(OUT, "gt_visible.npy"), vis)
    np.save(os.path.join(OUT, "depth_first_tokens.npy"), dist1)
    np.save(os.path.join(OUT, "hit_first_tokens.npy"), hit1)
    print("  已存 gt_first_to_last.npy / gt_visible.npy / depth_first_tokens.npy", flush=True)

    if vis.sum() > 10:
        rr = np.arange(HP)[:, None].repeat(WP, 1)[vis]
        cc = np.arange(WP)[None, :].repeat(HP, 0)[vis]
        dr = gt[..., 0][vis] - rr
        dc = gt[..., 1][vis] - cc
        print("\n=== 位移场统计（token 单位） ===")
        print("  dr mean=%.2f std=%.2f min=%d max=%d" % (dr.mean(), dr.std(), dr.min(), dr.max()))
        print("  dc mean=%.2f std=%.2f min=%d max=%d" % (dc.mean(), dc.std(), dc.min(), dc.max()))
        print("  注：相机近似平移，位移场应大体一致；std 过大说明有问题")

    print("\n总用时 %.0fs" % (time.time() - t_start))
    print("GT_REPROJECT_DONE", flush=True)


if __name__ == "__main__":
    main()
