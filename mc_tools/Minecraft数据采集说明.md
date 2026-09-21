# Minecraft 视频与相机轨迹采集说明

## 已配置的实例

在 HMCL 中启动 **`26.1.2-V2DiT-Data`**。这是从原版 `26.1.2` 派生的独立实例，使用自己的 `mods`、`config`、`flashback` 和输出目录；原实例没有被加模组。两个现有单人世界已经复制到新实例。

实例中有四个组件：

- Fabric Loader `0.19.5`
- Fabric API `0.155.3+26.1.2`
- Flashback `0.43.3`
- 项目自制的 V2DiT Camera Logger `0.1.2`

Flashback 负责先把一次游戏过程录成可回放的记录，再从回放中离线渲染平滑视频。Camera Logger 在 Flashback 每输出一帧后，自动保存这一帧对应的相机位置和朝向。

## 一次完整采集

### 1. 录下场景变化

1. 在 HMCL 启动 `26.1.2-V2DiT-Data`，进入单人世界。
2. 按 `Esc`，点击 Flashback 的 **“开始录制 / Start Recording”**。
3. 按计划移动、让生物自然运动，录制 5–10 秒即可。
4. 再按 `Esc`，点击 **“完成录制 / Finish Recording”**，为回放命名并保存。

这里录下的是世界状态和物体变化，不要求手动走出平滑相机轨迹。平滑轨迹在下一步的回放编辑器里制作。

### 2. 在回放里制作平滑相机

1. 回到主菜单，打开 **Flashback → 选择回放 / Select Replay**。
2. 打开刚才的回放，暂停在要作为开头的时刻，把相机移到起点。
3. 在 Camera 轨道添加第一个相机关键帧。
4. 移到结束时刻，把相机移到终点，添加第二个相机关键帧。轨迹复杂时可加少量中间关键帧。
5. 关键帧插值选 **Smooth**。Flashback 0.43.3 的默认插值已经是 Smooth。
6. 用 `I` 和 `O` 标出要导出的开始、结束时间，并先播放预览一次，确认不会穿墙。

这一步将“手抖的采集过程”和“训练视频中的平滑相机运动”分开，因此录制时走得不稳不会教坏模型。

### 3. 导出训练用帧和轨迹

在回放编辑器中选择 **File → Export Video**，Demo0 统一使用：

- 输出：**PNG Sequence**，不要先压成有损 MP4
- 分辨率：`1280 × 720`
- 帧率：`60 fps`
- 录制 → 第一人称更新：`60`，与导出帧率保持一致
- Projection：`Perspective`
- SSAA：关闭
- Dummy Render Frames：`0`（Flashback 默认就是 0）
- No GUI：开启
- Record Audio：关闭
- Depth Map：第一轮关闭；需要深度实验时另导出一份

开始导出时不需要按 `F8`。Camera Logger 会自动开始，导出结束后自动停止。

建议把每个样本的 PNG 放到：

```text
D:\Mc\.minecraft\versions\26.1.2-V2DiT-Data\v2dit_capture\rgb\样本名\
```

轨迹自动写到：

```text
D:\Mc\.minecraft\versions\26.1.2-V2DiT-Data\v2dit_capture\trajectory\
```

每次导出会生成同名的：

- `.csv`：每个画面一行相机姿态，`frame_index` 从 0 连续递增
- `.metadata.json`：帧率、分辨率、导出区间、坐标定义等

CSV 已同时保存位置、yaw/pitch、四元数，以及 `right/up/back` 三个相机坐标轴。训练转换优先使用四元数或坐标轴。Minecraft 世界坐标为 `+X` 向东、`+Y` 向上、`+Z` 向南；camera-to-world 的旋转列为 `right, up, back`，观察方向是 `-back`。

### 4. 校验图片与轨迹是否一一对应

在 PowerShell 中运行：

```powershell
& "C:\Users\12447\Desktop\plan\ShortTerm_WeeksCounted\project4_CAMI2V\mc_tools\validate_mc_capture.ps1" `
  -FramesPath "PNG图片文件夹" `
  -CsvPath "轨迹CSV文件"
```

输出中 `valid: true` 才收进数据集。脚本会检查图片数与轨迹行数、`frame_index` 连续性、四元数归一化、相机坐标轴正交性、导出分辨率和 SSAA 设置。

## F8 手动模式

正常游玩或自由相机中按一次 `F8` 开始记录轨迹，再按一次停止。它按照屏幕实际渲染帧记录，适合临时排查和观察；它与 OBS 的掉帧没有严格的一一对应关系，所以正式数据使用上面的 Flashback 离线导出流程。

## Demo0 推荐的第一批镜头

每段 5–10 秒，保留静态背景，可让生物自然移动：

1. 建筑近景横移：前景柱子、中景窗户、远景天空，检查视差。
2. 建筑环绕：绕一栋有不对称细节的小建筑走约 30–60 度。
3. 前进或后退：沿道路接近建筑，避免整屏都是重复草地方块。
4. 干扰对照：保留一小片重复草地或重复窗户，观察投票是否出现多峰和误匹配。

相似方块不是必须消除的坏数据，但不要让所有画面都只有同一种草地。重复纹理应当作为少量、可控的困难样本。
