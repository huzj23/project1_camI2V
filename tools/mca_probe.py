#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mca_probe.py — 最小 Anvil/NBT 读取器（无第三方依赖）

用途：A 方案第一步 —— 确认 MC 26.x 存档的区块格式与方块调色板。

region 文件格式：
  头 8 KB：1024 个 4 字节 location（前 3 字节 = 4KB 扇区偏移，第 4 字节 = 扇区数）
           + 1024 个 4 字节 timestamp
  区块：4 字节大端长度 + 1 字节压缩类型(1=gzip,2=zlib,3=none) + 压缩后的 NBT

NBT：tag(1B) + nameLen(2B 大端) + name + payload，根为 TAG_Compound
"""
import gzip
import io
import os
import struct
import sys
import zlib

TAG_END, TAG_BYTE, TAG_SHORT, TAG_INT, TAG_LONG = 0, 1, 2, 3, 4
TAG_FLOAT, TAG_DOUBLE, TAG_BYTE_ARRAY, TAG_STRING = 5, 6, 7, 8
TAG_LIST, TAG_COMPOUND, TAG_INT_ARRAY, TAG_LONG_ARRAY = 9, 10, 11, 12


class Reader:
    def __init__(self, data):
        self.d = data
        self.p = 0

    def u1(self):
        v = self.d[self.p]
        self.p += 1
        return v

    def i1(self):
        v = struct.unpack_from(">b", self.d, self.p)[0]
        self.p += 1
        return v

    def i2(self):
        v = struct.unpack_from(">h", self.d, self.p)[0]
        self.p += 2
        return v

    def u2(self):
        v = struct.unpack_from(">H", self.d, self.p)[0]
        self.p += 2
        return v

    def i4(self):
        v = struct.unpack_from(">i", self.d, self.p)[0]
        self.p += 4
        return v

    def i8(self):
        v = struct.unpack_from(">q", self.d, self.p)[0]
        self.p += 8
        return v

    def f4(self):
        v = struct.unpack_from(">f", self.d, self.p)[0]
        self.p += 4
        return v

    def f8(self):
        v = struct.unpack_from(">d", self.d, self.p)[0]
        self.p += 8
        return v

    def s(self):
        n = self.u2()
        v = self.d[self.p:self.p + n].decode("utf-8", "replace")
        self.p += n
        return v

    def payload(self, t):
        if t == TAG_BYTE:
            return self.i1()
        if t == TAG_SHORT:
            return self.i2()
        if t == TAG_INT:
            return self.i4()
        if t == TAG_LONG:
            return self.i8()
        if t == TAG_FLOAT:
            return self.f4()
        if t == TAG_DOUBLE:
            return self.f8()
        if t == TAG_BYTE_ARRAY:
            n = self.i4()
            v = self.d[self.p:self.p + n]
            self.p += n
            return v
        if t == TAG_STRING:
            return self.s()
        if t == TAG_LIST:
            et = self.u1()
            n = self.i4()
            return [self.payload(et) for _ in range(n)]
        if t == TAG_COMPOUND:
            out = {}
            while True:
                t2 = self.u1()
                if t2 == TAG_END:
                    return out
                name = self.s()
                out[name] = self.payload(t2)
        if t == TAG_INT_ARRAY:
            n = self.i4()
            v = list(struct.unpack_from(">%di" % n, self.d, self.p))
            self.p += 4 * n
            return v
        if t == TAG_LONG_ARRAY:
            n = self.i4()
            v = list(struct.unpack_from(">%dq" % n, self.d, self.p))
            self.p += 8 * n
            return v
        raise ValueError("未知 TAG %d @%d" % (t, self.p))


def read_nbt(raw):
    r = Reader(raw)
    t = r.u1()
    if t != TAG_COMPOUND:
        raise ValueError("根不是 Compound，而是 %d" % t)
    r.s()
    return r.payload(TAG_COMPOUND)


def read_chunk(path, cx, cz):
    with open(path, "rb") as fh:
        header = fh.read(4096)
        idx = (cx % 32) + (cz % 32) * 32
        off = (header[idx * 4] << 16) | (header[idx * 4 + 1] << 8) | header[idx * 4 + 2]
        cnt = header[idx * 4 + 3]
        if off == 0 or cnt == 0:
            return None
        fh.seek(off * 4096)
        ln = struct.unpack(">I", fh.read(4))[0]
        comp = fh.read(1)[0]
        body = fh.read(ln - 1)
    if comp == 1:
        body = gzip.decompress(body)
    elif comp == 2:
        body = zlib.decompress(body)
    elif comp != 3:
        raise ValueError("未知压缩类型 %d" % comp)
    return read_nbt(body)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else \
        r"D:\Mc\.minecraft\versions\26.1.2-V2DiT-Data\saves\新的世界\dimensions\minecraft\overworld\region"
    files = sorted(f for f in os.listdir(root) if f.endswith(".mca"))
    print("region 目录: %s" % root)
    print("文件数: %d" % len(files))
    print("前 5 个: %s" % files[:5])

    path = os.path.join(root, files[0])
    print("\n=== 解析 %s ===" % os.path.basename(path))
    m = files[0].split(".")
    bx, bz = int(m[1]), int(m[2])

    found = 0
    for dx in range(4):
        for dz in range(4):
            cx, cz = bx * 32 + dx, bz * 32 + dz
            try:
                nbt = read_chunk(path, cx, cz)
            except Exception as exc:
                print("  区块 (%d,%d) 解析失败: %r" % (cx, cz, exc))
                continue
            if nbt is None:
                continue
            found += 1
            print("\n--- 区块 (%d,%d) 顶层键 ---" % (cx, cz))
            print(sorted(nbt.keys()))
            print("  DataVersion = %s" % nbt.get("DataVersion"))
            print("  xPos=%s zPos=%s yPos=%s" % (nbt.get("xPos"), nbt.get("zPos"), nbt.get("yPos")))
            secs = nbt.get("sections") or nbt.get("block_states") or []
            print("  sections 数 = %d" % len(secs))
            for s in secs[:3]:
                print("   section Y=%s 键=%s" % (s.get("Y"), sorted(s.keys())))
                bs = s.get("block_states", s)
                if isinstance(bs, dict) and "palette" in bs:
                    pal = bs["palette"]
                    print("    调色板条目数 = %d" % len(pal))
                    for e in pal[:12]:
                        print("      %s" % (e,))
                    print("    data 长度 = %s" % (len(bs.get("data", [])) if bs.get("data") else 0))
            if found >= 1:
                return
    print("\n未找到可解析区块（共尝试 %d 个）" % 16)


if __name__ == "__main__":
    main()
