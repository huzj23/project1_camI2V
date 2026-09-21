# Demo0 v1.1 本地实验入口

本目录另有动态槽运行 `demo0_v11_dynamic32`：槽数由票证据决定，支持新增、暂时消失/重现、终止、分裂和合并；32 是本次运行的容量上限，不是固定槽数。`dynamic_config_32.json`、`dynamic_config_128.json`、`dynamic_config_1024.json` 是三个可运行配置，128/1024 配置继承同一算法并使用稀疏槽存储。当前只有 32 容量配置已经对 121 帧多深度短窗完成真实推理，切换容量必须用新 run_id 重新推理，网页不会现场伪造结果。

原有两条独立运行仍在：冻结三窗口 `demo0_v11_d0real`（会议展示基准）及两段完整 MC 录屏其余 14 个短窗口的 `demo0_v11_fullcoverage_d0real`。这些运行与动态槽运行都是**真实 RGB 视频上的 D0-Real 外部网格描述子验证**，未使用 VACE 编码器、真实 QKV 或任何扩散步。旧 `data/demo0` ResNet/DIS 结果保留为历史失败方案，不是本分支同表示 P0 baseline。

在 Windows 双击 [start_demo0.cmd](start_demo0.cmd)，本地页面为 <http://127.0.0.1:8767/>；运行选择“全部短窗”可合并浏览两个 run 中已完成的窗口。详细控件与视频路径见 [网页与启动说明](WEB_AND_LAUNCHER_GUIDE.md)，公式、baseline 和三段失败诊断见 [算法报告](VOTING_ALGORITHM_REPORT.md)，相对[版本 1.1 计划](../../log/[版本1.1]_[计划文档]_[V²-DiT Demo0验证性实验修订计划]_[9.15].md)的结构性偏差见[工程对齐审计](ENGINEERING_ALIGNMENT_REVIEW.md)。

冻结运行在 `runs/demo0_v11_d0real/`，不可重算、不可改写标注；`freeze_run.py verify` 可核验内容哈希。扩展运行进度在 `runs/demo0_v11_fullcoverage_d0real/coverage_progress.json`。如推理中断，`python run_fullcoverage.py` 按窗口阶段续跑；新增产物绝不写入冻结运行。全部 14 段完成后，运行 `python audit_fullcoverage.py` 核查两原视频全帧覆盖、每段 F/FL、P1–P5 和四类 Overlay MP4。
