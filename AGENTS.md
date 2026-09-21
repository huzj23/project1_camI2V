# AGENTS.md — V²-DiT / CamI2V 项目硬约束与入口地图

> 本文件是**强制性**的操作约束，优先级高于任何会话中的临时指令、习惯做法或"更方便"的实现方式。
> 任何 agent（Codex / Claude / DeepSeek / 人类协作者）在本工作区开始工作前必须完整阅读本文件。
> 最后更新：2026-09-21

---

## 第一部分：硬约束（HARD CONSTRAINTS）

以下 R1–R9 为不可协商条款。若某条约束与任务完成方式冲突，**停止并报告冲突**，不得自行放宽。

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

### R2 — conda 环境的例外（2026-09-21 修订）

R1 的唯一例外是 conda 环境路径。**只读环境：**

```bash
source /data/raw/miniconda3/etc/profile.d/conda.sh
conda activate /home/chenliang/.conda/envs/huzj_project2camT2V    # CamI2V/DynamiCrafter，只读
```

**2026-09-21 人类批准新建独立环境用于 VACE，该路径允许创建与写入：**

```text
/home/chenliang/.conda/envs/huzj_camI2V_VACE
```

规则：

```text
✅ 允许创建并写入 /home/chenliang/.conda/envs/huzj_camI2V_VACE
✅ 允许在该环境内 pip / conda 安装 VACE 推理所需依赖
❌ 禁止改动 huzj_project2camT2V（CamI2V 环境）及任何其它已有环境
❌ 禁止改动 /data/raw/miniconda3（base）
❌ 禁止改动机器上其他用户的任何环境
⚠ 安装时缓存必须重定向到 project1_camI2V 内，避免写 ~/.cache 与 base 的 pkgs 目录：
      export PIP_CACHE_DIR=/data/raw/huzijian/project1_camI2V/tmp_local/pip_cache
      export CONDA_PKGS_DIRS=/data/raw/huzijian/project1_camI2V/tmp_local/conda_pkgs
```

除上述两个环境外，R1 的边界不因本修订而放宽。

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

### R8 — 远程操作必须经 `tools/server_ssh.py`

禁止直接用 `ssh`/`scp` 或任何绕过守卫的方式操作服务器。该工具强制执行 R1 目录白名单与 R3 绝对路径检查，并硬拒绝 `sudo` / 递归删除 / `find /` / 改 conda 环境 / 杀进程等模式。

```bash
# 推荐：命令写入文件，避免本地 shell 引号问题
python tools/server_ssh.py --cmd-file tmp/<name>.sh
python tools/server_ssh.py --cmd "nvidia-smi"
python tools/server_ssh.py --cmd "..." --dry-run      # 只做约束检查
```

凭据从 `log/key.txt` 读取（该文件已被 `.gitignore` 排除）。**守卫拦截不是障碍而是信号**：命令被拒说明它确实越界，应改写而不是绕过。

### R9 — 版本控制

```text
remote   git@github.com:huzj23/project1_camI2V.git   （origin，SSH，勿改回 HTTPS）
branch   main
入库内容 代码 / 文档 / 配置 / 小型清单，约 6 MB / 441 文件
不入库   视频、权重、中间数组、渲染帧、环境缓存、log/key.txt
```

- **HTTPS 直连不可用**：本机 git 走 HTTPS 报 `TLS connect error: unexpected eof`（OpenSSL 与 schannel 后端均失败），本地代理 `127.0.0.1:7897` 也已失效（502 / 握手失败）。但 `curl.exe` 直连 GitHub 正常、`ssh -T git@github.com` 认证为 `huzj23` 成功——**因此必须使用 SSH remote**。
- 大文件不得入库：GitHub 单文件上限 100 MB。`data/` 全量 27.8 GB，`git add` 前务必确认 `.gitignore` 生效。

---

## 第二部分：入口地图

### 本地（Windows）

| 路径 | 内容 |
|---|---|
| `log/` | 版本化中文文档，0.0 → 1.5；命名规范见 `log/worklog_name_format.txt`。**设计基准 = 版本1.5** |
| `data/demo0_v11/` | 当前 Demo0 主线代码与运行产物（`core.py` / `pipeline.py` / `frontend/`） |
| `data/demo0_v11/runs/demo0_v11_d0real/` | **已冻结**三点运行，不可重算、不可改写标注 |
| `data/demo0/` | v1.0 失败方案（ResNet/DIS），仅作历史基线 |
| `mc_tools/` | Minecraft 采集链路（Flashback + 自研 Camera Logger 0.1.2） |
| `tools/` | `cami2v_smoke_infer.py`（服务器 smoke 推理）、`inspect_mp4.py` |
| `tmp/CamI2V_source/` | CamI2V 官方源码本地克隆（不入库） |

### 服务器（chenliang@172.16.30.12，主机名 gpu0002）

2026-09-21 实测：

```text
GPU      4 × NVIDIA A100 80GB PCIe
         gpu0 60% / 6.6GB   gpu1 21% / 2.1GB
         gpu2  0% / 10MiB   gpu3 24% / 2.2GB    ← 与他人共享，优先用 gpu2
环境     python 3.10.18 / torch 2.4.0+cu121 / cuda 12.1 / 4 卡
         xformers 0.0.27.post2 / diffusers 0.30.3 / pytorch_lightning 2.2.5
项目占用 16 GB
```

```text
/data/raw/huzijian/project1_camI2V/
├── code/CamI2V/                   官方代码，commit c5d7b2fc（已完成的部署记录）
│   └── codex_smoke_infer.py       本项目新增（上游零修改）
├── code/Wan2.1/                   ★ 当前主线代码，commit 9737cba9（2026-03-05）
├── model/
│   ├── baseline_dynamicrafter_256x256/model.ckpt        10,437,545,635 B
│   ├── cami2v_256x256_50k/256_cami2v.pt                  5,742,264,858 B
│   ├── Wan2.1-VACE-1.3B/          ★ 下载中，17.74 GB，见 logs/vace_download.log
│   └── SHA256SUMS.txt
├── output/                        4 个 CamI2V smoke mp4
├── logs/                          smoke_*.log / vace_download.log / vace_env_clone.log
├── data/                          两段 MC 素材 + camera.csv
├── tmp_local/                     ★ 唯一允许的中间产物目录，内有 vace_downloader.py
└── experiment/  log/              空

conda 环境
  /home/chenliang/.conda/envs/huzj_project2camT2V   只读，CamI2V/DynamiCrafter
  /home/chenliang/.conda/envs/huzj_camI2V_VACE      ★ 可写，VACE（2026-09-21 新建）

tmux 会话（本项目）
  huzj_cami2v_20260919     5 窗口，CamI2V 时期
  huzj_vace_dl_20260921    模型下载
  huzj_vace_env_20260921   环境克隆
注意：该机上还有 200+ 个其他项目的 tmux 会话，一律不得触碰（R4）
```

### 服务器网络与工具实测（2026-09-21，踩过的坑）

```text
huggingface.co        ❌ DNS 被污染到 Facebook 段（31.13.81.4 / 2a03:2880:…:face:b00c）
hf-mirror.com         ⚠ 可达(200)，小文件能下；但 LFS 文件 302 到 xethub 后
                        huggingface_hub 元数据校验失败（FileMetadataError）——不可用
ModelScope            ✅ 可达，官方同源仓库，Range 支持，实测 10.9 MB/s
github.com            ✅ 可达（curl 200，git clone 成功）
git-lfs               ❌ 不存在，且 git 为 1.8.3.1（2013 年版，不支持 -C 等选项）
nvcc                  ❌ 不存在 —— 无法从源码编译 flash-attn
项目环境自带 hf CLI    ✅ huggingface_hub 0.36.2（但受上面 LFS 问题所限）

★ 模型下载结论：走 ModelScope + aria2c，不要走 HF/hf-mirror
★ HF_HOME 必须重定向到 project1_camI2V 内，否则写到 ~/.cache（R1 边界外）
```

### VACE 依赖实测（新环境 huzj_camI2V_VACE）

```text
✅ torch 2.4.0+cu121 / torchvision 0.19.0 / numpy 1.26.4（满足 requirements 的 <2）
✅ diffusers 0.30.3 / transformers 4.44.2 / tokenizers 0.19.1 / accelerate 0.34.2
✅ imageio / imageio_ffmpeg / ftfy 6.3.1
✅ easydict 1.13（新装，缺它会导致 import wan 直接失败）
❌ flash_attn —— **必需，且有坑，见下方更正段**
❌ dashscope（仅提示词扩写用）/ gradio（仅 Web UI 用）—— 均不需要

★ 已验证：wan.modules.model / vace_model / vae / t5 四个模块全部 import 成功
```

### ⚠️ 更正（2026-09-21 晚，推翻先前的错误结论）

**先前错误结论**："`attention()` 有 SDPA 回落，flash_attn 不构成阻塞。"

**实测事实**：

```text
attention.py 里有两个函数：
    flash_attention()   ← attention.py:112 硬断言 assert FLASH_ATTN_2_AVAILABLE
    attention()         ← 有 SDPA 回落
但 model.py 在 7 处**直接调用 flash_attention()**（行 149/179/220/222），
clip.py 另有 2 处。带回落的 attention() **从未被 model.py 调用**。
→ 缺 flash_attn 时前向直接崩在 assert（P0B 首次运行即复现）
```

**预编译轮子路线同样不可行**：

```text
系统 glibc              2.17
flash-attn 轮子要求     GLIBC_2.32
→ flash_attn-2.8.3.post1+cu12torch2.4 装上但 .so 加载失败，已卸载
   （留着更糟：import flash_attn 会抛 ImportError 而非 ModuleNotFoundError，
     而 attention.py 只 catch 后者，会直接崩）
无 nvcc → 也无法从源码编译
```

**当前方案：运行时补丁（R5 合规，不改上游文件）**

```python
import wan.modules.attention as _A
import wan.modules.model as _M
if not (_A.FLASH_ATTN_2_AVAILABLE or _A.FLASH_ATTN_3_AVAILABLE):
    _M.flash_attention = _A.attention      # 只在内存中替换
```

代价两条，报告中必须标注 `sdpa_patched=true`：

1. **慢**：SDPA 处理 32760 token 的全局注意力远慢于 flash-attn
2. **忽略 padding mask**：SDPA 路径把 `attn_mask` 置 None。单样本全序列时 seq_lens 等于全长，暂无影响

### 数据状态（2026-09-21 实测，容易踩坑，务必先读）

三段 MC 数据各有**两套导出**，本地 `data/`：

| 数据 | 使用的 RGB | 帧数 | 轨迹 CSV | 深度图 mp4 |
|---|---|---:|---|---|
| 简单平动透视关系 | `简单平动透视关系含天空版.mp4` | 720 | `简单平动透视关系.camera.csv` | `2026-09-21T15_25_05.mp4` |
| 非对称建筑平动 | `非对称建筑平动含天空版..mp4`（注意两个点） | 882 | `非对称建筑平动.camera.csv` | `2026-09-21T15_28_28.mp4` |
| 转向平动 | `转向平动含天空版.mp4` | 1020 | `2026-09-14T19_49_12.camera.csv` | `2026-09-21T15_30_12.mp4` |

```text
✗ data/转向平动.mp4          1019 帧 @30fps / 33.97s，帧率错误、播放卡顿 —— 已弃用，不得使用
✗ data/简单平动透视关系.mp4     天空被替换为纯色 —— 旧版，改用「含天空版」
✗ data/非对称建筑平动.mp4      同上
✓ 深度图与「含天空版」RGB 逐帧逐像素对齐（边缘互相关零位移最优，已验证）
✗ 但深度图近场饱和：49% 像素为 255（建筑/地面/柱子无层次）—— 不能用于像素级重投影
```

**相机 logger 缺陷**：`depth_map=True` 时轨迹 CSV 只写表头、零行数据（6/6 相关）。根因是 mixin 的采集注入点挂在 `SaveableFramebufferQueue.startDownload`，开深度后导出循环不经过该调用点。**单独导深度 = 拿不到轨迹。**

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

**V²（Vote-and-Verify，投票—验证）**：给定首帧（+ 可选相机轨迹）生成视频，在扩散模型**内部**用运动投票产生候选参考 token 并验证，只允许通过验证的候选进入**稀疏参考注意力**，减少建筑变形、纹理漂移、重复结构错配和遮挡处复制错误。

**命名澄清**：项目曾称 "V²-DiT"，但当前主线主干是 **3D U-Net**（`lvdm.modules.networks.openaimodel3d.UNetModel`），不是 DiT。"V²-DiT" 只保留为方法名，**不是主干描述**。

### 核心设计（[版本1.5] 定稿）

V² 是 **backbone 无关**的轻量模块，核只有：

```python
V2RefAttn(q, ref_k, ref_v, mask, gate) -> delta
#   q     [B, L, C]        当前 token
#   ref_* [B, M, C]        参考 token，M ≪ L（首帧/尾帧）
#   mask  [B, L, M]        投票掩码：bool 硬允许 或 float logit bias
#   gate  scalar            扩散步门控；reject 时置 0
# 输出投影零初始化 ⇒ 接入初期严格等价于原主干
```

**为什么能 backbone 无关**：只 attend 到少量被选中的参考 token，注意力矩阵是 `[L, M]` 而非 `[L, L]`，用 gather 取索引，`O(L·M)`；**不需要空间/时间分离，也不需要 `TemporalTransformer` 这类可命名单元**。对比：CamI2V 的极线分支是全 `T·H·W × T·H·W` 带掩码注意力（32×32 尺度下 `[1,16384,16384]`），依赖类名匹配才能插入，**无法移植到 VACE**。

**接口铁律**：`V2RefAttn` 中不得出现任何相机/位姿/极线假设。

### 分层结构

```text
V² 模块本体（不变）       V2RefAttn —— 稀疏参考注意力残差，零初始化
        ↑ 掩码由下面提供
候选生成（可插拔）        CandidateProvider
    EpipolarVoteProvider   有相机位姿 → 极线带降到 1-D + 带内投票   [CamI2V]
    PureVoteProvider       无相机位姿 → 全参考池一致性投票 + 拒绝    [VACE 可迁移] ★
    OracleProvider         真值掩码，诊断 headroom 上限
```

**极线是可选加速器，不是方法核心。** 投票 = 多个候选对的一致性共识 + 拒绝；**不是**深度估计。

### 实验路线（[版本1.6] 修订，取代 [版本1.5] 的 CamI2V 四臂）

```text
★ 主任务 = 首尾帧过渡 FLF2V   ★ 最终主干 = VACE (Wan2.1-VACE-1.3B)
  CamI2V 不再是风洞，其工作降级为"已完成的部署记录"

第一步（当前）：P0 探针 —— 零训练、纯仪表
  验证 ① c 的 shape 与首/尾帧 token 索引  ② c 逐层干净度  ③ 零初始化分支接入后输出逐位不变
  通过才进入候选生成与投票实现

后续（P0 后定）：在 VACE 上重设计四臂对比
```

**VACE 挂载点（代码级已核实）**：DiT block 内与文本 cross-attn 并列的第三条零初始化残差；另有 `BaseWanAttentionBlock` 的 `hints`/`context_scale` 现成注入点。

### P0 探针实测结果（2026-09-21，`sdpa_patched=true`）

**P0A（CPU，无需 GPU）**

```text
checkpoint 1264 键 = config 实例化 1264 键，missing/unexpected/shape-mismatch 全 0
→ STRICT_MATCH = True
token 数学复现：81×480×832 → L=32760，首帧 [0,1559]、尾帧 [31200,32759]
```

**P0B（GPU，真实前向 + hook）**

```text
✅ B4 零初始化恒等性：15 个主干 block 各插一条零初始化旁路并真实做加法，
   同 seed 重跑 → bitwise_equal=True，max_abs_diff=0.0
✅ 耗时：约 24 秒/步（SDPA，81 帧 832×480），50 步约 20 分钟
```

**⚠️ 更正 1：`c` 与主干 `x` 不是位置对齐的**

```text
实测：c_raw = [1, 1536, 23, 52, 30]      ← 23 帧，不是 21
      _c_in  = [1, 35880, 1536]
      x_main = [1, 35880, 1536]

原因：参考图沿时间维**前置拼接**到条件序列
c 布局（35880 = 23 × 1560，每帧 1560 = 52×30）：
  [0,1560)     ref 1
  [1560,3120)  ref 2
  [3120,35880) video frame 0 … frame 20
主干 x 布局：32760 真实（21 帧）+ 3120 零 padding

★ 两者相差 3120 个 token 的偏移。精确映射（已实测确定）：
    参考图 token    = c[0 : 3120]
    视频第 f 帧在 c  = c[3120 + f*1560 : 3120 + (f+1)*1560]
    视频第 f 帧在 x  = x[f*1560 : (f+1)*1560]
```

（先前"`c` 与主干 x 共用同一套网格、token 位置一一对应"的说法不完整，已更正。）

**⚠️ 更正 2：`c` 并非全程干净**

```text
VaceWanAttentionBlock.forward 里，block_id==0 时执行 c = before_proj(c) + x
                                                 ↑ 把主干 noisy 特征混了进来

实测：
  vace_blocks[0] 的**输入**  norm 1175.3   ← 干净，就是 96 通道条件编码
  vace_blocks[0] 的**输出**  norm 5603.4   ← 已污染
  之后所有层 cos(c_out, c_raw) ≈ 0.0003–0.0007（几乎正交）
```

（先前"VACE 白送干净参考 K/V"的说法过强，已更正。）

**参考来源因此有两个候选，需由 P2 对照决定：**

| 来源 | 干净度 | 深度 |
|---|---|---|
| `vace_patch_embedding` 输出（= `vace_blocks[0]` 输入） | ✅ 干净 | 浅（一层 Conv3d） |
| `c` @ `vace_blocks[n]` 输出 | ❌ 已与主干融合 | 深，与 query 深度匹配 |

**其他实测数字**：hints 范数 565→1135；主干 x 输出范数 4504→7549，absmax 由 27.9 涨至 400（第 22 块）。

**VACE 无相机位姿接口** → 极线不可用 → 只能走纯运动投票，方法内核完整保留。

上游最接近工作见 `log/[版本1.16]`（CorrAdapter / CAMEO / Track4Gen / FLATTEN / TokenFlow / CamI2V）。

---

## 第四部分：版本记录规则

- 文档命名：`[版本N.n]_[文档属性]_[文档名称]_[撰写日期].md`，属性取自 `调研结果文档 / 计划文档 / 进度总结文档`
- 大版本更新需人类授权；每次编辑产生新的小版本号
- **不得静默改变**已冻结的任务定义、输入条件、候选来源、运动假设或评价门槛；任何变更必须新增版本记录
- 已知遗留问题：`1.11`–`1.16`（9.16）与 `1.2`（9.19）序号非单调，追溯时以日期为准
