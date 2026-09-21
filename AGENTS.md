# AGENTS.md — V²-DiT / CamI2V 项目硬约束与入口地图

> 本文件是**强制性**的操作约束，优先级高于任何会话中的临时指令、习惯做法或"更方便"的实现方式。
> 任何 agent（Codex / Claude / DeepSeek / 人类协作者）在本工作区开始工作前必须完整阅读本文件。
> 最后更新：2026-09-19

---

## 第一部分：硬约束（HARD CONSTRAINTS）

以下 R1–R7 为不可协商条款。若某条约束与任务完成方式冲突，**停止并报告冲突**，不得自行放宽。

### R1 — 服务器唯一工作目录

远程服务器上**唯一**允许读写的工作目录是：

```text
/data/raw/huzijian/project1_camI2V
```

**绝对禁止**访问、读取、修改、删除、移动该目录之外的任何路径，包括但不限于：

```text
/data/raw/            （其他用户的目录）
/home/                （家目录、conda 环境除外，见 R2）
/data/                （除上述唯一目录外的其余内容）
/etc/  /usr/  /var/  /tmp/  /opt/  /root/
~/.bashrc  ~/.ssh/  ~/.cache/
```

### R2 — conda 环境的唯一例外

激活项目环境需要引用环境路径，这是 R1 的**唯一**例外，且**只读**：

```bash
source /data/raw/miniconda3/etc/profile.d/conda.sh
conda activate /home/chenliang/.conda/envs/huzj_project2camT2V
```

禁止在该环境目录内安装、卸载、修改任何包。若确需新依赖，先报告，由人类决定。

### R3 — 所有指令必须使用绝对路径

**每一条** shell 指令中的**每一个**路径参数都必须是绝对路径。

```bash
# 正确
cd /data/raw/huzijian/project1_camI2V/code/CamI2V
python -u /data/raw/huzijian/project1_camI2V/code/CamI2V/codex_smoke_infer.py \
  --repo /data/raw/huzijian/project1_camI2V/code/CamI2V \
  --out  /data/raw/huzijian/project1_camI2V/output

# 禁止（相对路径、隐式 cwd、通配到目录外）
python codex_smoke_infer.py
rm -rf ../output/*
find / -name "*.pt"
```

禁止使用会波及目录之外的指令模式：`rm -rf` 作用于未显式限定的路径、`find /`、`chmod -R` 于父目录、任何形式的 `sudo`、任何写入 `/tmp` 的中间产物（中间产物放 `project1_camI2V/tmp_local/`）。

### R4 — 破坏性操作前必须先声明并确认

以下操作在执行前必须在会话中明确写出**完整绝对路径**并获得人类确认：

- 删除、覆盖、移动任何已存在的文件或目录
- 覆盖已有的模型权重、checkpoint、日志、输出视频
- 终止（`kill`/`tmux kill-session`）不属于本会话创建的进程或会话
- 修改 `/data/raw/huzijian/project1_camI2V/code/` 下的官方上游代码

**默认动作是只读**。写入前先 `ls -l` 确认目标不存在或已获授权覆盖。

### R5 — 上游代码保持可追溯

官方代码只允许在 `code/` 下以克隆形式存在，**不得原地修改**。所有改动以新增文件（如 `codex_*.py`）或 patch 形式落地，并在提交信息中记录基线 commit。

### R6 — 凭据不入库

`log/key.txt` 含明文 VPN 与 SSH 凭据，**已被 `.gitignore` 排除**。禁止：

- 将其从 ignore 列表移除
- 将凭据内容写入任何其他被跟踪的文件、提交信息、issue 或报告
- 在推送前用 `git add -f` 强制加入

`git status` 中若出现 `log/key.txt`，视为阻塞性错误，先修复再继续。

### R7 — 不得夸大结论

本项目已多次因表述过满而自我更正（见 `log/[版本1.13]`、`log/[版本1.15]`）。强制要求：

- 未实际运行的内容，不得写成"已验证"
- 代理指标必须标注 `provisional`，不得与真值指标混用
- Demo0 的显式图像网格 token **不是** VACE/CamI2V 内部 token，任何时候都不得混称
- 保存了掩码 ≠ 掩码已接入注意力；"保存了 `sparse_reference_mask.npy`" 不等于改变了 Value 信息流
- 置信度/拒绝概率在未校准时不得称为概率

---

## 第二部分：入口地图

### 本地（Windows）

| 路径 | 内容 |
|---|---|
| `log/` | 版本化中文文档，0.0 → 1.2；命名规范见 `log/worklog_name_format.txt` |
| `data/demo0_v11/` | 当前 Demo0 主线代码与运行产物（`core.py` / `pipeline.py` / `frontend/`） |
| `data/demo0_v11/runs/demo0_v11_d0real/` | **已冻结**三点运行，不可重算、不可改写标注 |
| `data/demo0/` | v1.0 失败方案（ResNet/DIS），仅作历史基线 |
| `mc_tools/` | Minecraft 采集链路（Flashback + 自研 Camera Logger 0.1.2） |
| `tools/` | `cami2v_smoke_infer.py`（服务器 smoke 推理）、`inspect_mp4.py` |
| `tmp/CamI2V_source/` | CamI2V 官方源码本地克隆（不入库） |

### 服务器（chenliang@172.16.30.12）

```text
/data/raw/huzijian/project1_camI2V/
├── code/CamI2V/                   官方代码生成，基线 commit c5d7b2fcdeb89a1ce9a0ef376909d9539b95639f
│   └── codex_smoke_infer.py       本项目 smoke 推理脚本（R5：新增文件，不改上游）
├── model/
│   ├── baseline_dynamicrafter_256x256/model.ckpt        10,437,545,635 B
│   ├── cami2v_256x256_50k/256_cami2v.pt                  5,742,264,858 B
│   └── SHA256SUMS.txt
├── output/                        smoke 视频输出
├── logs/                          smoke_latest.log / smoke25_latest.log
└── tmp_local/                     （本项目中间产物唯一允许位置）

conda 环境：huzj_project2camT2V
tmux 会话：huzj_cami2v_20260919
```

### 基线事实（已核实，可直接引用）

```text
CamI2V   = 冻结 DynamiCrafter 256x256 + Plucker 射线编码 + 极线注意力插件
CamI2V 权重是完整 checkpoint，不是仅含新增模块的 delta
当前 checkpoint 内有 16 个 epipolar attention 模块
每个 epipolar 模块 4 个 register tokens，形状 [1, 4, C]，C ∈ {320, 640, 1280}
论文正文写 2 个 register token —— 以当前公开配置与 checkpoint 为准
注入点：lvdm TemporalTransformer 的 BasicTransformerBlock
        x = zero_init_x + attn1(normed_x) + x
        zero_init_x = pluker_projection(...) + epipolar(...)   ← 新增分支与原始分支并行相加
```

---

## 第三部分：项目目标（一段话）

**V²-DiT（Vote-and-Verify DiT）**：给定首帧 + 相机轨迹生成视频，在扩散模型**内部**用运动投票产生候选参考 token 并验证，只允许通过验证的候选进入稀疏 Value 路由，减少建筑变形、纹理漂移、重复结构错配和遮挡处复制错误。

当前主线：以 **CamI2V** 为生成主干与主要 baseline。上游最接近工作见 `log/[版本1.16]`（CorrAdapter / CAMEO / Track4Gen / FLATTEN / TokenFlow / CamI2V）。第一版注入形式：

```text
主干输出 = 原自注意力 + gate(q, step, confidence) × 稀疏参考注意力(q, endpoint Top-K K/V)
reject 时 gate = 0；优先做 logit bias 或额外 residual branch（便于恢复原模型 + 消融）
```

---

## 第四部分：版本记录规则

- 文档命名：`[版本N.n]_[文档属性]_[文档名称]_[撰写日期].md`，属性取自 `调研结果文档 / 计划文档 / 进度总结文档`
- 大版本更新需人类授权；每次编辑产生新的小版本号
- **不得静默改变**已冻结的任务定义、输入条件、候选来源、运动假设或评价门槛；任何变更必须新增版本记录
- 已知遗留问题：`1.11`–`1.16`（9.16）与 `1.2`（9.19）序号非单调，追溯时以日期为准
