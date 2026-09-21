#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server_ssh.py — 受 AGENTS.md R1/R2/R3/R4 约束保护的远程命令执行器。

设计原则：硬约束不靠自觉，靠工具强制。
  1. 拒绝清单（DENY）：sudo / 递归删除 / find / / chmod -R / 改环境 等
  2. 白名单路径前缀：命令中出现的每个绝对路径必须落在允许前缀内
  3. 相对路径告警
  4. --dry-run 只打印不执行

凭据从 log/key.txt 解析（该文件已被 .gitignore 排除），不写入本脚本。

用法：
    python tools/server_ssh.py --cmd "nvidia-smi"
    python tools/server_ssh.py --cmd "ls -l /data/raw/huzijian/project1_camI2V" --dry-run
    python tools/server_ssh.py --cmd "..." --timeout 600
"""

import argparse
import io
import os
import re
import sys

try:
    import paramiko
except ImportError:
    sys.exit("需要 paramiko：pip install paramiko")

HERE = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.normpath(os.path.join(HERE, os.pardir, "log", "key.txt"))

# ---- R1 / R2：允许出现的绝对路径前缀 ----------------------------------------
ALLOWED_PREFIXES = (
    "/data/raw/huzijian/project1_camI2V",
    "/data/raw/miniconda3/etc/profile.d/conda.sh",
    "/home/chenliang/.conda/envs/huzj_project2camT2V",
    "/dev/null",
)

# ---- 硬拒绝模式 --------------------------------------------------------------
DENY = [
    (r"\bsudo\b", "R3: 禁止 sudo"),
    (r"\bsu\s+-", "R3: 禁止切换用户"),
    (r"\brm\b[^|;&]*\s-[A-Za-z]*[rR]", "R4: 禁止递归删除（如确需，请人工在服务器执行）"),
    (r"\brm\b\s+-[A-Za-z]*f", "R4: 禁止强制删除"),
    (r"\bfind\s+/\s*$|\bfind\s+/\s", "R3: 禁止 find /"),
    (r"\bchmod\b[^|;&]*\s-R\b", "R3: 禁止递归 chmod"),
    (r"\bchown\b", "R3: 禁止 chown"),
    (r"\bmkfs\b|\bdd\s+if=|\bparted\b|\bfdisk\b", "R3: 禁止磁盘操作"),
    (r"\b(shutdown|reboot|halt|poweroff|init\s+0)\b", "R3: 禁止关机重启"),
    (r"\bkill(all)?\b|\bpkill\b", "R4: 禁止杀进程"),
    (r"\btmux\s+kill-(server|session)\b", "R4: 禁止终止 tmux 会话"),
    (r"\b(pip|pip3)\s+(install|uninstall)\b", "R2: 禁止改动 conda 环境"),
    (r"\bconda\s+(install|remove|uninstall|update|upgrade|create|env\s+remove)\b", "R2: 禁止改动 conda 环境"),
    (r"\bapt(-get)?\b|\byum\b|\bdnf\b", "R3: 禁止系统包管理"),
    (r"\bcurl\b[^|]*\|\s*(ba|z)?sh", "R3: 禁止管道执行远程脚本"),
    (r"\bwget\b[^|]*\|\s*(ba|z)?sh", "R3: 禁止管道执行远程脚本"),
    (r">\s*/(etc|usr|var|opt|root|home|boot|sys|proc)/", "R1: 禁止写入允许目录之外的系统路径"),
    (r"\bexport\s+HOME=|\bcd\s+~\b|\bcd\s+\$HOME\b", "R1: 禁止操作家目录"),
    (r"\bhistory\s+-c\b", "R4: 禁止清除审计痕迹"),
]

# ---- 路径扫描 ----------------------------------------------------------------
# 抓取形如 /foo/bar 的绝对路径 token（排除 URL 里的 //）
ABS_PATH_RE = re.compile(r"(?<![\w:./-])(/[A-Za-z0-9_.\-/]*)")
# 形似相对路径的参数：必须以字母数字开头（避免把绝对路径的前导 / 也匹配进来），
# 含 / 但不以 / 开头，且不是选项、不是 URL。
REL_PATH_RE = re.compile(r"(?<![\w:/.-])([A-Za-z0-9_][A-Za-z0-9_.\-]*/[A-Za-z0-9_.\-/]*)")


def parse_key_file(path):
    """从 log/key.txt 解析 ssh <user>@<host> 密码 <pass> 形式的凭据。"""
    if not os.path.isfile(path):
        return {}
    creds = {}
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = re.search(r"ssh\s+([\w.\-]+)@([\w.\-]+)\s*密码\s*(\S+)", line)
            if m:
                creds[(m.group(2), m.group(1))] = m.group(3)
    return creds


def check_command(cmd):
    """返回 (blockers, warnings)。blockers 非空则拒绝执行。"""
    blockers, warnings = [], []

    for pattern, reason in DENY:
        if re.search(pattern, cmd):
            blockers.append("命中拒绝规则 [%s]：%s" % (reason, pattern))

    for raw in ABS_PATH_RE.findall(cmd):
        path = raw.rstrip("/") or "/"
        if path == "/":
            continue
        if not any(path == p or path.startswith(p + "/") or path == p
                   for p in ALLOWED_PREFIXES):
            blockers.append("R1/R3 越界绝对路径：%s" % path)

    # 相对路径告警：排除选项、URL、conda/pip 子命令等
    for m in REL_PATH_RE.findall(cmd):
        if m.startswith("-") or "://" in m:
            continue
        if m.split("/")[0] in ("usr", "bin", "etc"):
            continue
        warnings.append("疑似相对路径（R3 要求绝对路径）：%s" % m)

    return blockers, warnings


def main():
    ap = argparse.ArgumentParser(description="受硬约束保护的远程命令执行器")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cmd", help="要在服务器上执行的 bash 命令")
    src.add_argument("--cmd-file", help="从文件读取命令（推荐，绕开本地 shell 引号问题）")
    ap.add_argument("--host", default="172.16.30.12")
    ap.add_argument("--user", default="chenliang")
    ap.add_argument("--timeout", type=float, default=120.0, help="命令超时秒数")
    ap.add_argument("--dry-run", action="store_true", help="只做约束检查与打印")
    args = ap.parse_args()

    if args.cmd_file:
        with io.open(args.cmd_file, encoding="utf-8") as fh:
            args.cmd = fh.read()

    print("=" * 72)
    print("HOST : %s@%s" % (args.user, args.host))
    print("CMD  : %s" % args.cmd)
    print("=" * 72)

    blockers, warnings = check_command(args.cmd)
    for w in warnings:
        print("[WARN] %s" % w)
    if blockers:
        print("\n[BLOCKED] 命令未通过硬约束检查，拒绝执行：")
        for b in blockers:
            print("  - %s" % b)
        print("\n如确需执行，请人工在服务器上完成，并先向人类确认（AGENTS.md R4）。")
        return 2

    if args.dry_run:
        print("[DRY-RUN] 约束检查通过，未执行。")
        return 0

    creds = parse_key_file(KEY_FILE)
    password = creds.get((args.host, args.user))
    if not password:
        print("[ERROR] 未在 %s 找到 %s@%s 的凭据。" % (KEY_FILE, args.user, args.host))
        return 3

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(args.host, port=22, username=args.user,
                       password=password, timeout=20, banner_timeout=30,
                       auth_timeout=30)
    except Exception as exc:
        print("[ERROR] SSH 连接失败：%r" % (exc,))
        return 4

    try:
        stdin, stdout, stderr = client.exec_command(args.cmd, timeout=args.timeout)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        code = stdout.channel.recv_exit_status()
    except Exception as exc:
        print("[ERROR] 命令执行失败：%r" % (exc,))
        return 5
    finally:
        client.close()

    if out:
        print("---- stdout ----")
        print(out.rstrip("\n"))
    if err:
        print("---- stderr ----")
        print(err.rstrip("\n"))
    print("[exit code: %d]" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
