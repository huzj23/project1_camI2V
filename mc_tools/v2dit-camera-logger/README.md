# V2DiT Camera Logger

当前版本：`0.1.2`。该版本允许 Flashback 导出任务名称为空，并只在 Flashback 把画面提交给输出队列时记录相机，跳过 0.43.3 固定执行但不会写进视频的 60 次预热渲染。

Minecraft 26.1.2 / Fabric 客户端模组。它与 Flashback 0.43.3 配合，在 Flashback 导出每个画面后自动写出一行相机轨迹。

## 自动模式

用 Flashback 导出视频、PNG 或深度图时，模组自动在：

```text
<实例目录>/v2dit_capture/trajectory/
```

生成同一 session id 的 `.csv` 和 `.metadata.json`。在普通 Perspective 投影、关闭 SSAA、关闭 Depth Map 的配置下，CSV 每一行对应一个实际写入视频的画面。导出后仍应比较视频帧数和 CSV 行数，验证为一一对应后再收入数据集。

## 手动模式

在正常游戏或自由视角中按 `F8` 开始，再按一次停止。手动模式按实际渲染帧记录，可与 OBS/系统录屏配合，但它不像 Flashback 离线导出那样天然保证恒定帧率。

## 坐标

- Minecraft 世界坐标：`+X` 向东、`+Y` 向上、`+Z` 向南；
- CSV 同时保存位置、yaw/pitch、完整四元数和相机坐标基向量；
- camera-to-world 的旋转列为 `right, up, back`，观看方向为 `-back`；
- 后续训练优先使用四元数或基向量，yaw/pitch 仅用于人眼检查。
