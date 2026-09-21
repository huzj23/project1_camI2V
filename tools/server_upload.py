#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server_upload.py — 受 AGENTS.md R1/R2 约束保护的文件上传器。

与 server_ssh.py 同一套守卫逻辑：
  · 目的路径必须落在 ALLOWED_PREFIXES 内（R1/R2）
  · 拒绝任何越界目的路径

用法：
    python tools/server_upload.py --src <本地文件> --dst <服务器绝对路径>
    python tools/server_upload.py --manifest tmp/upload_manifest.json
"""
import argparse
import io
import json
import os
import posixpath
import re
import sys

try:
    import paramiko
except ImportError:
    sys.exit("需要 paramiko")

HERE = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.normpath(os.path.join(HERE, os.pardir, "log", "key.txt"))

ALLOWED_PREFIXES = (
    "/data/raw/huzijian/project1_camI2V",
    "/home/chenliang/.conda/envs/huzj_camI2V_VACE",
)


def parse_key_file(path):
    creds = {}
    if not os.path.isfile(path):
        return creds
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = re.search(r"ssh\s+([\w.\-]+)@([\w.\-]+)\s*密码\s*(\S+)", line)
            if m:
                creds[(m.group(2), m.group(1))] = m.group(3)
    return creds


def check_dst(dst):
    dst = posixpath.normpath(dst)
    if not any(dst == p or dst.startswith(p + "/") for p in ALLOWED_PREFIXES):
        return "R1/R2 越界目的路径：%s" % dst
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--dst")
    ap.add_argument("--manifest", help="JSON: [{src, dst}, ...]")
    ap.add_argument("--host", default="172.16.30.12")
    ap.add_argument("--user", default="chenliang")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = []
    if args.manifest:
        with io.open(args.manifest, encoding="utf-8") as fh:
            jobs = [(j["src"], j["dst"]) for j in json.load(fh)]
    else:
        jobs = [(args.src, args.dst)]

    print("=" * 72)
    for s, d in jobs:
        exist = os.path.isfile(s)
        print("SRC %-70s %s" % (s, "OK %.1f MB" % (os.path.getsize(s) / 1048576) if exist else "!! NOT FOUND"))
        print("DST %s" % d)
    print("=" * 72)

    blockers = []
    for s, d in jobs:
        if not os.path.isfile(s):
            blockers.append("本地文件不存在：%s" % s)
        err = check_dst(d)
        if err:
            blockers.append(err)
    if blockers:
        print("[BLOCKED]")
        for b in blockers:
            print("  - %s" % b)
        return 2
    if args.dry_run:
        print("[DRY-RUN] 检查通过，未上传。")
        return 0

    creds = parse_key_file(KEY_FILE)
    password = creds.get((args.host, args.user))
    if not password:
        print("[ERROR] 未找到凭据")
        return 3

    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(args.host, port=22, username=args.user, password=password, timeout=30)
    sftp = cli.open_sftp()

    ok = 0
    for s, d in jobs:
        posixpath_dir = posixpath.dirname(d)
        try:
            sftp.stat(posixpath_dir)
        except IOError:
            cli.exec_command("mkdir -p %s" % posixpath_dir)
        size = os.path.getsize(s)
        print("[up] %s -> %s  (%.1f MB)" % (os.path.basename(s), d, size / 1048576), flush=True)
        sftp.put(s, d)
        try:
            remote = sftp.stat(d).st_size
        except IOError:
            remote = -1
        flag = "OK" if remote == size else "MISMATCH(%d vs %d)" % (remote, size)
        print("     -> %s" % flag, flush=True)
        if flag == "OK":
            ok += 1

    sftp.close()
    cli.close()
    print("=" * 72)
    print("上传完成 %d/%d" % (ok, len(jobs)))
    return 0 if ok == len(jobs) else 4


if __name__ == "__main__":
    sys.exit(main())
