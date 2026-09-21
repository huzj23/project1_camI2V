# V²-DiT Demo 0（计划 1.0 的可执行实现）

本目录只处理 `data` 下的两段 Minecraft 实录：`简单平动透视关系` 和 `非对称建筑平动`。它不会调用 VACE，也不会生成新视频。

## 固定实验口径

- 冻结 ImageNet ResNet-18 `layer3` 特征，输入 512×288，对应 32×18 个 stride-16 token。
- 每个查询 token 保留 `L=8` 个首帧候选。
- V² 使用 `K=4` 个二维平移槽和一个拒绝槽。
- 比较 B0 最近邻、B1 普通全局 cross-attention、B2 独立 token 运动、B3 soft k-means、B4 等参数普通候选 adapter、B5 完整投票—验证。
- 因现有视频没有 depth、visibility 或 entity mask，DIS 双向光流仅作为弱代理；由它计算的 PCK、EPE、Top-8 Recall 和拒绝 F1 全部标记为 **provisional**，不会与计划中的几何真值 `Demo0-Score` 混用。

## 运行

```bash
python demo0_pipeline.py validate --config config.json
python demo0_pipeline.py self-test --config config.json
python demo0_pipeline.py run --config config.json --device cuda
```

若只做本机冒烟测试：

```bash
python demo0_pipeline.py run --config config.json --device cpu --max-frames 17 --skip-training
```

完整结果位于 `runs/<run_id>/`。每个 clip 至少包含：

```text
arrays/topl_index.npy
arrays/topl_weight.npy
arrays/slot_assignment.npy
arrays/reject_prob.npy
arrays/motion_vectors.npy
arrays/token_confidence.npy
arrays/slot_id.npy
arrays/slot_prob.npy
arrays/pred_ref_xy.npy
overlays/000000.png ...
overlays/flow/000000.png ...
overlay_preview.mp4
overlay_flow_preview.mp4
legend.json
metrics.json
metrics.csv
manifest.json
```

交互查看器由 `backend/app.py` 和 `frontend/` 提供。启动后可选择 run、clip、帧、token，区分原始最相似候选与验证后允许读取的候选，并查看运动槽、拒绝概率与置信度。

本机已在 `vendor/` 放置查看器依赖并构建 `frontend/dist/`，可直接运行：

```bash
python serve_viewer.py
```

然后打开 `http://127.0.0.1:8765`。

## 完成门槛

`audit_run.py` 会逐项验证：两个输入视频都被处理；每个 overlay PNG 序列与原视频帧数完全相同；两个 MP4 预览帧数一致；六个方法和要求的消融均有指标；所有稀疏张量形状与 schema 一致；指标明确保留 provisional 标签。
