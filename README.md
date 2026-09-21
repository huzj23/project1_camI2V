# project1_camI2V — V²-DiT：把运动投票—验证注入视频扩散模型注意力

> 本地工作目录：`project4_CAMI2V`
> 服务器工作目录：`/data/raw/huzijian/project1_camI2V`
> 状态更新时间：2026-09-19

---

## 一句话

给定一张静态场景首帧 + 一条相机轨迹，让视频扩散模型生成相机沿该轨迹运动的视频；在模型**内部**加入"投票—验证"模块，使当前帧 token 只从首帧/尾帧中读取**与同一运动规律相符且通过验证**的少量参考 token，减少建筑变形、纹理漂移、重复结构错配与遮挡处复制错误。

**V²-DiT（Vote-and-Verify DiT）是本项目提出的研究方案名称，不是已成熟的现成模型。**

---

## ⚠️ 先读这个

**[AGENTS.md](AGENTS.md)** 是本仓库的强制性操作约束（R1–R7），优先级高于任何会话中的临时指令。任何 agent 或协作者在开始工作前必须完整阅读。

最关键的两条：

```text
R1  服务器上唯一允许读写的工作目录是 /data/raw/huzijian/project1_camI2V
R3  所有 shell 指令的每一个路径参数都必须是绝对路径
```

---

## 当前进度（截至 2026-09-19）

| 工作线 | 状态 |
|---|---|
| 方向定型（V²-DiT，主任务 = 首帧 I2V） | ✅ 版本 1.0 / 1.1 |
| Minecraft 数据采集链路（Flashback + 自研 Camera Logger） | ✅ 可用，已产出 3 段素材 |
| Demo0 v1.1 本地实验（D0-Real，外部网格描述子） | ✅ 跑完，**结论：暂不进入注入** |
| VACE 内部机理核对 + 文献定位（1.11–1.16） | ✅ 完成 |
| **CamI2V 服务器部署（主线）** | ✅ strict load 全键匹配，2/25 步推理跑通 |
| **CamI2V 真实 token 上的候选召回诊断** | ⬜ 下一步 |

### Demo0 v1.1 的核心结论

在 24 条人工冻结 FL 查询上（[RESULTS.md](data/demo0_v11/runs/demo0_v11_d0real/RESULTS.md)）：

| | P0（原始 Top-16 余弦） | P5（完整方法） |
|---|---:|---:|
| Hit@1（19 条可匹配） | 0.789 | 0.842 |
| Hit@8 | 1.000 | 1.000 |
| MRR | 0.853 | 0.912 |

**但不足以进入注入**：遮挡窗 token 541/492 的树干漏召回且被 P5 错误接受；P5 未稳定优于 P3；Hit@8=1.000 存在选择偏差。

**必须说清的边界**：Demo0 的 1296 个 token 是工程自建的图像网格 token，**不是 VACE/CamI2V 内部 token**；运行清单 `vace_checkpoint_used=false`；稀疏掩码只保存成 `.npy`，**从未接入任何 Value 信息流**。

### CamI2V 基线事实（已核实）

```text
CamI2V = 冻结 DynamiCrafter 256x256 + Plucker 射线编码 + 极线注意力插件
权重是完整 checkpoint，不是仅含新增模块的 delta
当前 checkpoint 内有 16 个 epipolar attention 模块
每个模块 4 个 register token，形状 [1, 4, C]，C ∈ {320, 640, 1280}
论文正文写 2 个 —— 以当前公开配置与 checkpoint 为准
注入点：lvdm TemporalTransformer 的 BasicTransformerBlock
        x = zero_init_x + attn1(normed_x) + x
        zero_init_x = pluker_projection(...) + epipolar(...)
```

上游代码基线 commit：`c5d7b2fcdeb89a1ce9a0ef376909d9539b95639f`（[ZGCTroy/CamI2V](https://github.com/ZGCTroy/CamI2V)）

---

## 目录结构

```text
AGENTS.md              硬约束与入口地图（先读）
log/                   版本化文档 0.0 → 1.2，命名规范见 log/worklog_name_format.txt
data/
  *.mp4 *.camera.csv   MC 原始录屏与逐帧相机轨迹
  demo0_v11/           当前 Demo0 主线代码（core.py / pipeline.py / frontend/）
  demo0/               v1.0 失败方案（ResNet/DIS），仅作历史基线
mc_tools/              MC 采集链路：v2dit-camera-logger (Fabric mod) + 校验脚本
tools/                 cami2v_smoke_infer.py / inspect_mp4.py
tmp/CamI2V_source/     CamI2V 官方源码本地克隆（不入库）
```

---

## 本仓库不含什么

为保证仓库可克隆，以下内容**不入库**（见 [.gitignore](.gitignore)）：

| 排除项 | 体积 |
|---|---|
| 渲染帧 PNG / 预览图 | 19.3 GB / 68,545 文件 |
| 中间数组 `.npy` | 7.2 GB |
| 原始视频 `.mp4` | 1.0 GB |
| 模型权重 / JDK / Gradle 环境 | 3.3 GB |
| 依赖缓存、node_modules、vendor | 250 MB |
| `log/key.txt`（明文凭据） | — |

入库内容约 **6 MB / 441 文件**。数据与权重按 [AGENTS.md](AGENTS.md) 的服务器路径获取。

---

## 上游与参考

- CamI2V 论文：<https://arxiv.org/html/2410.15957>
- CamI2V 代码：<https://github.com/ZGCTroy/CamI2V>
- DynamiCrafter 权重：<https://huggingface.co/Doubiiu/DynamiCrafter>
- CamI2V 权重：<https://huggingface.co/MuteApo/CamI2V>
- 最接近的相关工作（见 `log/[版本1.16]`）：CorrAdapter、CAMEO、Track4Gen、FLATTEN、TokenFlow、CamI2V
