from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core import write_json


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "demo0_v11_d0real"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def round4(value: Any) -> Any:
    return round(float(value), 4) if isinstance(value, (int, float)) else value


def collect_proxy() -> dict[str, Any]:
    windows: dict[str, Any] = {}
    for directory in sorted(path.parent for path in RUN.glob("*/window_manifest.json")):
        candidate = read_json(directory / "metrics" / "candidate_metrics.json")
        voting = read_json(directory / "metrics" / "voting_metrics.json")
        temporal = read_json(directory / "metrics" / "temporal_metrics.json")
        modes: dict[str, Any] = {}
        for mode in ("F", "FL"):
            p0 = candidate["modes"][mode]["P0"]
            result: dict[str, Any] = {
                "P0": {"MRR": round4(p0["MRR"]), "Hit@8": round4(p0["Hit@8"]), "Hit@16": round4(p0["Hit@16"])},
            }
            for method in ("P3", "P4", "P5"):
                key = f"{mode}_{method}"
                row = voting["methods"][key]
                result[method] = {
                    "MRR": round4(row["verified"]["MRR"]),
                    "Hit@8": round4(row["verified"]["Hit@8"]),
                    "delta_correct_weight_vs_P0": round4(row["delta_correct_weight_vs_P0"]),
                    "slot_flicker_rate": round4(temporal["methods"][key]["slot_flicker_rate"]),
                    "accepted_fraction": round4(row["accepted_fraction"]),
                    "mean_value_candidates": round4(row["mean_value_candidates"]),
                }
            modes[mode] = result
        windows[directory.name] = modes
    return {
        "warning": "Dense metrics use disclosed adjacent-flow composition only as an auxiliary proxy; manual review showed that its visibility labels are often wrong.",
        "windows": windows,
    }


def markdown_table(proxy: dict[str, Any]) -> str:
    rows = [
        "| Window | Mode | P0 Hit@8 | P0 MRR | P3 MRR | P4/P5 MRR | P3→P4 flicker | P5 accept | P5 refs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window, modes in proxy["windows"].items():
        for mode, values in modes.items():
            rows.append(
                f"| {window} | {mode} | {values['P0']['Hit@8']:.3f} | {values['P0']['MRR']:.3f} | "
                f"{values['P3']['MRR']:.3f} | {values['P4']['MRR']:.3f} | "
                f"{values['P3']['slot_flicker_rate']:.3f}→{values['P4']['slot_flicker_rate']:.3f} | "
                f"{values['P5']['accepted_fraction']:.3f} | {values['P5']['mean_value_candidates']:.2f} |"
            )
    return "\n".join(rows)


def main() -> None:
    manual = read_json(RUN / "annotations" / "manual_metrics.json")
    proxy = collect_proxy()
    overall = manual["overall"]
    gates = {
        "G1_candidate_hit8": {
            "status": "pass_on_small_FL_manual_set",
            "evidence": f"P0 Hit@8={overall['P0_raw']['Hit@8']:.4f} on 19 matchable manually reviewed FL queries.",
            "limitation": "F mode was not independently annotated and the set is small.",
        },
        "G2_vote_improves_rank_or_weight": {
            "status": "pass_on_small_FL_manual_set",
            "evidence": f"Manual MRR {overall['P0_raw']['MRR']:.4f}→{overall['P5_verified']['MRR']:.4f}; mean correct weight={overall['P5_verified']['mean_correct_candidate_weight']:.4f}.",
            "limitation": "Dense auxiliary proxy often shows rank loss, so this is not yet a stable full-frame result.",
        },
        "G3_P5_outperforms_P3": {
            "status": "fail",
            "evidence": "P4/P5 greatly reduce slot flicker, but do not consistently improve P3 MRR or correct-candidate weight in dense proxy metrics; P5 ranking equals P4 by design.",
        },
        "G4_FL_tail_coverage_without_harm": {
            "status": "inconclusive",
            "evidence": "Only one genuinely tail-only manual query was confirmed. Dense proxy shows FL helps some windows but hurts the occlusion window and its visibility labels are unreliable.",
        },
        "G5_overlay_temporal_stability": {
            "status": "partial",
            "evidence": "P4 lowers dense slot flicker from 0.498–0.573 to 0.244–0.388 in FL, but residual switching and occlusion-boundary errors remain.",
        },
        "G6_real_sparse_value_route": {
            "status": "engineering_pass_D0_real_only",
            "evidence": "P5 saves and serves a real Top-8-limited sparse_reference_mask; accepted tokens retain about 2.3–2.8 references on average.",
            "limitation": "This is not yet a VACE Q/K/V Value route.",
        },
    }
    report = {
        "schema_version": "demo0-1.1-experiment-report",
        "run_id": "demo0_v11_d0real",
        "scope": "D0-Real only; real short-video frames with explicit image-grid tokens",
        "vace_checkpoint_used": False,
        "scientific_decision": "do_not_integrate_into_vace_yet",
        "reason": "Candidate retrieval and vote re-ranking are promising on the small manual set, but P5 fails the pre-registered P3 superiority gate and still false-accepts occlusion-boundary tokens.",
        "manual_metrics": manual,
        "dense_auxiliary_proxy": proxy,
        "gates": gates,
        "observed_failure_modes": [
            "Foreground trunk tokens can retrieve roof/eave candidates and remain falsely accepted.",
            "Composed adjacent DIS is unsuitable as visibility ground truth; several visibly persistent surfaces were labelled invisible.",
            "Motion-slot colour is a motion/depth-surface cue, not a semantic object mask.",
            "FL is not uniformly better than F in the occlusion window.",
        ],
        "next_algorithm_revision": [
            "Replace scalar low-texture rejection with an explicit visibility/occlusion state learned or calibrated from local forward-backward evidence.",
            "Make endpoint support explicit per hypothesis so an FL candidate cannot borrow confidence from an incompatible endpoint.",
            "Add boundary-aware negative evidence for candidates crossing strong foreground/background edges.",
            "Keep P4 identity propagation, but decouple temporal smoothing strength from candidate re-ranking to preserve good P3 ranks.",
            "Repeat the frozen audit with independent F and FL annotations before any VACE integration.",
        ],
        "plan_stage_status": {
            "D0.1": "complete",
            "D0.2": "complete",
            "D0.3": "complete",
            "D0.4": "not_executed_no_VACE_token_in_D0_real",
            "D0.5": "complete_for_FL_small_audit; F_manual_split_pending",
            "D0.6": "complete_D0_real",
            "D0.7": "partial; sparse mask and temporal slots complete, explicit split/merge lifecycle not proven",
            "D0.8": "not_executed",
            "D0.9": "decision_reached_stop_before_VACE_integration",
        },
        "interpretation_boundary": "This run validates operations on explicit image-grid tokens and a sparse reference-token mask. It does not claim operation on VACE internal tokens or a diffusion step.",
    }
    write_json(RUN / "experiment_report.json", report)

    md = f"""# Demo0 v1.1 — D0-Real 结果

## 结论

**当前结论：暂不进入 VACE 注入。**

本轮不是单纯给视频分色：P5 已实际输出每个当前 token 的稀疏参考 token 掩码、候选权重、端点来源、运动槽概率和拒绝概率。人工冻结的 24 个 FL 查询中，19 个可匹配查询的 P0 Hit@8 为 {overall['P0_raw']['Hit@8']:.3f}，MRR 从 P0 的 {overall['P0_raw']['MRR']:.3f} 提升到 P5 的 {overall['P5_verified']['MRR']:.3f}。但是遮挡窗口内两个树干 token 没有找回树干，P5 仍错误接受屋檐候选；同时 P5 没有在密集代理指标上稳定优于 P3。因此版本 1.1 的注入前门槛尚未全部通过。

## 人工核验

- 冻结查询：24；可匹配：19；应拒绝：5。
- P0：Hit@1={overall['P0_raw']['Hit@1']:.3f}，Hit@8={overall['P0_raw']['Hit@8']:.3f}，MRR={overall['P0_raw']['MRR']:.3f}。
- P5：Hit@1={overall['P5_verified']['Hit@1']:.3f}，Hit@8={overall['P5_verified']['Hit@8']:.3f}，MRR={overall['P5_verified']['MRR']:.3f}，拒绝准确率={overall['P5_verified']['reject_accuracy']:.3f}。
- 明确的错误接受：遮挡窗口 token 541、492。
- 三个由光流代理挑出的“两端不可见”样本，人工检查后都仍在端点可见；当前短窗口没有确认到真正的“两端不可见”样本，这是核验集覆盖缺口。

## 密集辅助代理（不可当真值）

{markdown_table(proxy)}

密集代理的主要可用信号是相对趋势：P4 明显降低槽闪烁；P3/P4/P5 都提高了正确候选总权重，但排序 MRR 并未稳定提高。其可见性标签经人工检查频繁出错，所以不能用这些数字单独判定算法成败。

## 为什么此前视觉效果差

此前结果差并非单一原因。旧版先在长距离候选阶段丢失正确位置，又用不可靠的长距离 DIS 判断可见性，并把每帧独立的少量恒定位移槽画成“物体追踪”。这会同时造成灰暗、碎片、换色和跨物体合并。当前短窗口、H1 平移—尺度、时间身份传播和真实稀疏掩码已经修复其中一部分；剩余主要瓶颈是遮挡边界的候选与拒绝机制，而不是天空是否被替换。

## Token 边界

当前操作对象是 48×27、stride 16 的显式图像网格 token；候选、投票、聚类和稀疏 Value 掩码都发生在 token 索引上。它不是仅对色块集合做后处理，但也不是 VACE 内部 token。颜色 overlay 只是读取这些数组的可视化。

## 下一轮算法修改

1. 增加显式可见性/遮挡状态，并以局部前后向一致性校准，不再把组合光流当真值。
2. 对首、尾端点分别累计假设支持，避免一个端点的错误候选借用另一端点的置信度。
3. 在强前景/背景边界加入负证据，重点压制树干→屋檐这类跨边界误配。
4. 保留 P4 的槽身份传播，但降低其对候选排名的直接平滑，避免稳定性换来 MRR 损失。
5. 独立冻结 F 与 FL 人工查询后，再决定是否接入 VACE Q/K/V。

## 已完成与未完成

D0-Real 的三段短窗口、F/FL、P0–P5、四类全长 overlay、三栏交互查询、24 个冻结人工样本和结构审计均已完成。D0-Step、VACE Q/K/V 与噪声步实验没有在本 run 中执行；本结果不能被表述为 VACE token 或扩散步实验。
"""
    (RUN / "RESULTS.md").write_text(md, encoding="utf-8")

    manifest_path = RUN / "run_manifest.json"
    manifest = read_json(manifest_path)
    manifest["status"] = "d0_real_complete_scientific_gate_failed"
    manifest["manual_annotations_frozen"] = True
    manifest["manual_query_count"] = int(overall["query_count"])
    manifest["scientific_decision"] = report["scientific_decision"]
    manifest["experiment_report"] = "experiment_report.json"
    write_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "decision": report["scientific_decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
