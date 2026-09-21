from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core import write_json


def check(condition: bool, message: str, evidence: dict[str, Any], failures: list[str]) -> None:
    evidence[message] = bool(condition)
    if not condition:
        failures.append(message)


def video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    value = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else -1
    capture.release()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Demo0 v1.1 outputs")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_dir = config_path.parent
    root = (config_dir / config["output_root"] / config["run_id"]).resolve()
    expected_ids = {item["window_id"] for item in config["windows"]}
    failures: list[str] = []
    evidence: dict[str, Any] = {}

    check(root.is_dir(), "run directory exists", evidence, failures)
    run_manifest_path = root / "run_manifest.json"
    check(run_manifest_path.is_file(), "run_manifest.json exists", evidence, failures)
    model_path = root / "model_and_scheduler.json"
    token_path = root / "token_metadata.json"
    annotation_path = root / "annotations" / "manual_queries.json"
    manual_metrics_path = root / "annotations" / "manual_metrics.json"
    experiment_report_path = root / "experiment_report.json"
    results_path = root / "RESULTS.md"
    for path, label in (
        (model_path, "model_and_scheduler.json exists"),
        (token_path, "token_metadata.json exists"),
        (annotation_path, "manual annotation schema exists"),
        (manual_metrics_path, "manual metrics exist"),
        (experiment_report_path, "experiment report exists"),
        (results_path, "human-readable results exist"),
    ):
        check(path.is_file(), label, evidence, failures)
    if model_path.is_file():
        model = json.loads(model_path.read_text(encoding="utf-8"))
        check(model.get("execution_mode") == "D0-Real", "execution mode is D0-Real", evidence, failures)
        check(model.get("vace_checkpoint_used") is False, "VACE checkpoint is explicitly unused", evidence, failures)
    if annotation_path.is_file():
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        records = annotations.get("queries", [])
        check(annotations.get("frozen") is True, "manual annotations are frozen", evidence, failures)
        check(len(records) == 24, "manual annotation count is 24", evidence, failures)
        check(all(item.get("review_status") == "visually_confirmed" for item in records), "manual annotations are visually confirmed", evidence, failures)
        check({item.get("window_id") for item in records} == expected_ids, "manual annotations cover all windows", evidence, failures)
    if manual_metrics_path.is_file():
        manual_metrics = json.loads(manual_metrics_path.read_text(encoding="utf-8"))
        check(manual_metrics.get("overall", {}).get("query_count") == 24, "manual metrics cover 24 queries", evidence, failures)
    if experiment_report_path.is_file():
        experiment_report = json.loads(experiment_report_path.read_text(encoding="utf-8"))
        check(experiment_report.get("scientific_decision") == "do_not_integrate_into_vace_yet", "scientific gate decision is explicit", evidence, failures)

    actual_ids = {path.parent.name for path in root.glob("*/window_manifest.json")}
    if args.allow_partial:
        check(bool(actual_ids), "at least one window exists in partial audit", evidence, failures)
        check(actual_ids <= expected_ids, "partial windows are configured", evidence, failures)
    else:
        check(actual_ids == expected_ids, "all configured windows exist", evidence, failures)

    window_reports: dict[str, Any] = {}
    top_l = int(config["processing"]["top_l_max"])
    slot_count = int(config["voting"]["slot_count"])
    for window_id in sorted(actual_ids):
        directory = root / window_id
        manifest = json.loads((directory / "window_manifest.json").read_text(encoding="utf-8"))
        total = int(manifest["frame_count"])
        token_count = int(manifest["token_grid"]["count"])
        report: dict[str, Any] = {}
        local_failures: list[str] = []
        check(video_frame_count(directory / "window.mp4") == total, "prepared video frame count matches", report, local_failures)
        check((directory / "contact_sheet.jpg").is_file(), "contact sheet exists", report, local_failures)
        descriptor = np.load(directory / "features" / "descriptor.npy", mmap_mode="r")
        check(descriptor.shape[:2] == (total, token_count), "descriptor shape matches", report, local_failures)
        check(np.isfinite(np.asarray(descriptor[[0, total - 1]], dtype=np.float32)).all(), "endpoint descriptors are finite", report, local_failures)

        for mode in config["reference_modes"]:
            mode_dir = directory / "modes" / mode
            raw_index = np.load(mode_dir / "raw_topl_index.npy", mmap_mode="r")
            endpoint = np.load(mode_dir / "reference_endpoint.npy", mmap_mode="r")
            local_id = np.load(mode_dir / "reference_local_id.npy", mmap_mode="r")
            check(raw_index.shape == (total, token_count, top_l), f"{mode} raw candidate shape", report, local_failures)
            check(local_id.max() < token_count, f"{mode} local reference ids in range", report, local_failures)
            check(set(np.unique(endpoint).tolist()) <= ({0} if mode == "F" else {0, 1}), f"{mode} endpoint ids valid", report, local_failures)
            if mode == "FL":
                check(set(np.unique(endpoint).tolist()) == {0, 1}, "FL candidates use both endpoints", report, local_failures)

            for method, method_spec in config["ablation_methods"].items():
                method_dir = mode_dir / method
                check((method_dir / "method_manifest.json").is_file(), f"{mode}/{method} method manifest exists", report, local_failures)
                verified = np.load(method_dir / "verified_topl_weight.npy", mmap_mode="r")
                assignment = np.load(method_dir / "motion_slot_assignment.npy", mmap_mode="r")
                token_slot = np.load(method_dir / "token_slot_probability.npy", mmap_mode="r")
                parameters = np.load(method_dir / "motion_slot_parameters.npy", mmap_mode="r")
                sparse = np.load(method_dir / "sparse_reference_mask.npy", mmap_mode="r")
                confidence = np.load(method_dir / "token_confidence.npy", mmap_mode="r")
                rejected_endpoint = np.load(method_dir / "selected_reference_endpoint.npy", mmap_mode="r")
                reject_probability = np.load(method_dir / "reject_probability.npy", mmap_mode="r")
                check(verified.shape == (total, token_count, top_l), f"{mode}/{method} verified shape", report, local_failures)
                check(assignment.shape == (total, token_count, top_l, slot_count), f"{mode}/{method} assignment shape", report, local_failures)
                check(token_slot.shape == (total, token_count, slot_count), f"{mode}/{method} token-slot shape", report, local_failures)
                check(parameters.shape == (total, slot_count, 3), f"{mode}/{method} parameter shape", report, local_failures)
                check(sparse.shape == verified.shape, f"{mode}/{method} Value-mask shape", report, local_failures)
                check(np.allclose(np.asarray(verified, dtype=np.float32).sum(-1), 1.0, atol=2e-3), f"{mode}/{method} weights sum to one", report, local_failures)
                if method_spec["use_sparse_value_mask"]:
                    check(np.asarray(sparse[..., 8:]).sum() == 0, f"{mode}/{method} sparse mask is limited to primary Top-8", report, local_failures)
                else:
                    check(np.asarray(sparse).all(), f"{mode}/{method} does not apply P5 Value masking", report, local_failures)
                if not method_spec["use_rejection"]:
                    check(not (np.asarray(rejected_endpoint) == 2).any(), f"{mode}/{method} does not apply rejection", report, local_failures)
                    check(np.asarray(reject_probability, dtype=np.float32).max() == 0.0, f"{mode}/{method} rejection probability disabled", report, local_failures)
                check(np.isfinite(np.asarray(confidence, dtype=np.float32)).all(), f"{mode}/{method} confidence finite", report, local_failures)
                for kind in ("motion_slots", "reference_source", "confidence", "motion_vectors"):
                    png_count = len(list((directory / "overlays" / mode / method / kind).glob("*.png")))
                    check(png_count == total, f"{mode}/{method}/{kind} PNG count", report, local_failures)
                    preview = directory / "overlays" / mode / method / f"overlay_{kind}.mp4"
                    check(video_frame_count(preview) == total, f"{mode}/{method}/{kind} MP4 frame count", report, local_failures)
        for kind in ("candidate", "voting", "temporal"):
            check((directory / "metrics" / f"{kind}_metrics.json").is_file(), f"{kind} metrics exist", report, local_failures)
        candidate_path = directory / "metrics" / "candidate_metrics.json"
        voting_path = directory / "metrics" / "voting_metrics.json"
        if candidate_path.is_file():
            candidate_metrics = json.loads(candidate_path.read_text(encoding="utf-8"))
            check(all("P0" in candidate_metrics.get("modes", {}).get(mode, {}) for mode in config["reference_modes"]), "P0 metrics exist for F and FL", report, local_failures)
        if voting_path.is_file():
            voting_metrics = json.loads(voting_path.read_text(encoding="utf-8"))
            expected_methods = {
                f"{mode}_{method}"
                for mode in config["reference_modes"]
                for method in config["ablation_methods"]
            }
            check(expected_methods <= set(voting_metrics.get("methods", {})), "P1-P5 metrics exist for F and FL", report, local_failures)
        report["failures"] = local_failures
        failures.extend(f"{window_id}: {message}" for message in local_failures)
        window_reports[window_id] = report

    result = {
        "schema_version": "demo0-1.1-audit",
        "status": "passed" if not failures else "failed",
        "partial": bool(args.allow_partial),
        "root_evidence": evidence,
        "windows": window_reports,
        "failures": failures,
    }
    if not (root / ".frozen.json").is_file():
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "audit_report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
