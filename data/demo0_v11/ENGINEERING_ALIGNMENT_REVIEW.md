# Demo0 v1.1 工程与计划对齐审计

审计基准：[版本 1.1 Demo0 修订计划](../../log/[版本1.1]_[计划文档]_[V²-DiT Demo0验证性实验修订计划]_[9.15].md)；当前冻结运行：`demo0_v11_d0real`。结论先说清楚：**D0-Real 的局部候选和可视化已经有价值，但当前工程没有按计划完成“真实 VACE Token + 单步扩散 + 稀疏 Value 路由”的验证。** 它是一条可解释的外部网格描述子试验分支，不能改名为 E1–E3 VACE 主实验，更不能据此决定注入时机。冻结旧成果是为了下一次会议复查，不是冻结“通过 VACE 验证”的结论。

## 当前 Token 到底怎样生成

代码在 [core.py](core.py) 的 `dense_token_descriptor` 和 [pipeline.py](pipeline.py) 的特征提取环节，把 MC 视频帧缩到 768×432；每 16 像素设一个中心，形成 48×27 个位置，按 `id=y×48+x` 展平。对各中心的 24/40 像素局部支撑域计算 RootSIFT 梯度加 RGB/Lab 统计并归一化；匹配分数是这些向量的余弦点积。这**不是随手把聚类色块当 Token**：候选、Top16、槽后验和 P5 掩码均有真正的离散索引。但这些索引是工程自建“图片网格 Token”，**不是 VACE 网络内部的 Token**。中间图的 16×16 实线方框只是名义网格格子，虚线 40 像素是该描述子最大局部支撑；方框无法证明相应 VACE Token 恰好只看这一片像素。

原版 [Wan-VACE 模型代码](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/vace_model.py) 与 [Diffusers Wan-VACE Transformer 源码](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan_vace.py) 都把 **VAE 视频 latent** 经 `patch_size=(1,2,2)` 的 3D patch embedding 再按时间—行—列展平，进入 Transformer；[Diffusers Wan-VACE Pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_vace.py) 则先用 Wan VAE 编码视频/条件、组织 latent 与去噪步骤。以公开 [Wan2.1-VACE 1.3B VAE 配置](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers/blob/main/vae/config.json) 的空间约 8 倍、时间约 4 倍压缩和 DiT 空间 patch 2 为例，**名义空间步长可能也是 16 像素**。这是尺寸层面的巧合/可映射起点，**不是特征同一性**：实际 VAE 是时空编码，patch embedding 从 16 通道 latent 取值，DiT hidden/Q/K/V 还受层、头、条件、位置和扩散噪声影响。需以选定的真实 checkpoint、预处理、padding、latent 网格和层/头记录核实，不能只凭 `stride=16` 宣称一致。

| 对齐项 | 当前 D0-Real | 计划中的真实 VACE 验证 | 判定 |
|---|---|---|---|
| 空间索引 | 48×27 图像网格、每格名义 16×16 | VAE 压缩后由 DiT `(1,2,2)` patch 形成的 latent Token 网格 | **只有名义步长可能对应**，尚未逐位置验证 |
| 时间索引 | 每个视频原始帧各有 1,296 个独立描述子 | VAE 有时间压缩；一个 DiT 时刻不必对应单个原始帧 | **不一致**；需明确帧→latent 时间位置及特殊首帧/padding |
| 特征/相似度 | RootSIFT+颜色、L2/余弦 | VACE VAE latent、指定层 hidden、实际每头 QK/`√d` | **不一致**；P0 只是外部描述子基线 |
| 感受野 | 24/40 像素明确局部支持 | 时空 VAE 与多层 DiT 的有效感受域更广、随层变化 | **不能把当前方框当 VACE 像素真值** |
| 展平顺序 | `y×48+x`，逐帧另存 | patch 后 `T,H,W` 顺序展平，含完整时间维 | 单帧空间顺序类似；**时空序列定义不一致** |
| 候选 | 首/首尾同表示 Top16 | 真实层/头/噪声步的 QK Top8/16 | 逻辑目标相似，候选来源不一致 |
| “注意力” | Top16 权重与 P5 可读掩码数组 | 对真实 VACE QK logits 掩码且限制对应参考 V 信息流 | **没有执行**；当前没有修改生成网络 |

当前视频的标框在视觉上常接近正确物理区域，说明在这个16像素网格上描述子对应**大致可用**，但不是像素级光流真值，也不保证 VACE 编码后仍保留同样局部可辨性。SIFT 的 40 像素邻域会跨树干/房屋或不同深度边界，因此遮挡窗口错匹配不应只归咎于投票。

## 从输入到输出，与计划不同的环节

| 计划阶段 | 实际进度 | 核心差距 |
|---|---|---|
| D0.1 归档 1.0 | 旧 `data/demo0` 保留；新三窗结果已经聚合哈希冻结 | 旧 ResNet/DIS 可回顾，但不应与本分支 P0 混叫“baseline” |
| D0.2 短窗与轨迹 | 三窗完成；扩展 14 段另跑，合计 17 段覆盖两视频；轨迹用于选窗/验证 | 个别启动/控制段有轻微视角变化；尚缺逐窗口**人工可见重叠确认** |
| D0.3 三栏 UI | 首/当前/尾、F/FL、Top8/16、Raw/投票、固定热标尺、局部裁剪、槽白框和 MP4 已有 | 没有 VACE 层/头、scheduler 噪声控件，因为对应实验根本未跑 |
| **D0.4 真实 VACE Token** | **未做**；运行清单明确 `vace_checkpoint_used=false` | 未加载 VAE/DiT、未录 latent/hidden/QKV、未核实 Token↔像素/时间 |
| D0.5 人工核验 | FL 手选 24 条冻结，三窗各 8 条；明确发现树干两例错误接受 | F 未单独冻标签；仅尾可见/两端不可见覆盖不足；原“可匹配19条”排除了两条可见但漏召回的关键反例 |
| D0.6 H0/H1/空间 | 六槽加权投票与 P1–P3 已实现 | H1 是简单位置相关速度近似；多深度局部透视、动态槽和边界负证据未做 |
| D0.7 时间与稀疏 | P4/P5 邻帧 DIS 槽先验、Hungarian 对齐、后验平滑和稀疏掩码文件已有 | **不等于**持续追踪同一物体；遮挡消失/重现、成员裂合与跨窗口身份未实现；Value 掩码尚未用于网络 |
| **D0.8 噪声/单步** | **未做** | 没有 scheduler、干净 latent 加噪、冻结 VACE 前向、早/中/后层 QKV、低/中/高噪声对照或后期门控 |
| D0.9 任务分支决策 | **尚不能决定** | F/FL 只有当前外部特征与不可靠代理对照；不能据此改变首帧 I2V 主目标 |

计划 1.1 的 P0–P5 明确要求**真实 VACE QK 同候选池**，而当前 P0–P5 用外部 RootSIFT+RGB；这是最大的实验定义偏差。计划允许 D0-Real 先隔离生成误差，但又要求 D0.4 接入干净 VACE latent/无噪前向，再进入 D0-Step。现在没有加噪是合理的受控起点；**“在噪声之外步骤一致”却不能说成立**，因为 VAE 编码、DiT patch/层/头、实际 QK 排名、V 路由、条件组织与时序 Token 压缩也都不同。

关于“泄露”的严格界限：F 模式参考端点数组实测只有首帧，FL 模式才有首帧和尾帧，候选检索没有读相机轨迹或人工正确位置；蓝天高相似度在不投票的 P0 也存在，因此不是尾帧真值偷偷进入 F。但 D0-Real 本来就观察**真实当前视频帧**；P4/P5 的时间先验还用真实相邻帧的 DIS 光流。这是公开的验证性辅助信息，**若把同一输入直接搬到“只给首帧的生成时”就会成为不可获得的 oracle**。D0-Step/主实验必须改为仅从当前生成 latent、已生成帧或模型可用条件推断时间状态，再以相同信息边界做对照。

此外，扩展窗口不一定都满足真实 VAE 希望的时间长度对齐（例如 31、82 帧窗口）；转入 VACE 时须按真实 pipeline 验证可接受帧数、padding/裁剪和首尾参考 Token 的时间位置，不能悄悄插帧后仍把 UI 原始帧 ID 当 DiT Token ID。未来需保存一张明确的 `original_frame → vae_latent_t → patch_token_t` 映射与空间坐标映射；真实网络感受域应通过受控扰动/梯度敏感性或解码测试描述，而不是只画 16 像素框。

## 复位路线，不偷改主任务

1. **先把当前 run 定名外部 D0-Real 先导**并维持只读；14 段扩展同算法、新 run 单独标注，包含重复纹理、透视、深度和遮挡压力测试。重新按“端点物理可见但未入 Top8/16”记录召回失败，不从指标分母中藏掉树干反例。
2. **D0.4 独立 run**：固定具体 Wan2.1-VACE checkpoint、VAE/Transformer 配置、输入缩放/帧数对齐；先做真实视频 VAE 干净 latent 与冻结 VACE 无噪声/受控干净前向，采集 E1/E2 各层/头 hidden/Q/K/V。将真实 Token ID、视频帧区间和名义图像范围投回三栏 UI；P0 改为真实 QK Top8/16，再与外部描述子 P0 分开报告。
3. **D0-Step 独立 run**：用模型实际 scheduler 的明确噪声强度加噪，记录低/中/高噪声、尤其低噪后期步的候选召回、投票收益与拒绝。先验证掩码在**实际参考 Attention Value** 上改变信息流；不要以“保存了 sparse_reference_mask.npy”替代实证。
4. **生成因果对照**：同 checkpoint/seed/条件、首帧 I2V 主任务，比较原 VACE/全参考 Attention、P0 原始 TopK 限制、投票/拒绝 TopK 限制。区分参考子通路与文本/全局上下文，不无条件关闭后者；量化同物体纹理/结构一致性、跨帧身份、运动自然度及失败接受率，并给人工视频盲评。FL 首尾过渡作为**另一个输入设定**独立做对照，不自动取代首帧任务。

因此，你在网页上感到“注意力方向多数正确”是推进真实 VACE 实验的充分**动机**，却还不是“多数真实 VACE Attention 正确”或“生成一致性会提高”的证明。下一步最关键的工程不是更多上色，而是让当前参考建议、像素映射和冻结 VACE 的真实 Token/QKV/Value 接口逐个对齐，并在同候选池下证明它优于 P0。
