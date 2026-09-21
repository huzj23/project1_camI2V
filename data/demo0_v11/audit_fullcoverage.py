from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE=Path(__file__).resolve().parent
RUNS=HERE/"runs"
MODES=("F","FL")
METHODS=("P1","P2","P3","P4","P5")
OVERLAYS=("motion_slots","reference_source","confidence","motion_vectors")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    plan=read(HERE/"fullcoverage_windows.json")
    assert plan["segment_count"]==17 and plan["new_segment_count"]==14
    assert (RUNS/"demo0_v11_d0real"/".frozen.json").is_file()
    coverage=defaultdict(list)
    for window in plan["windows"]:
        coverage[window["video"]].append(window)
        directory=RUNS/window["run_id"]/window["window_id"]
        info=read(directory/"window_manifest.json")
        assert info["frame_count"]==window["frame_count"], window["window_id"]
        assert info["status"]=="complete_d0_real", window["window_id"]
        assert (directory/"window.mp4").is_file(), window["window_id"]
        for mode in MODES:
            mode_dir=directory/"modes"/mode
            raw=np.load(mode_dir/"raw_topl_index.npy",mmap_mode="r")
            assert raw.shape[:2]==(window["frame_count"],1296), (window["window_id"],mode)
            for method in METHODS:
                method_dir=mode_dir/method
                sparse=np.load(method_dir/"sparse_reference_mask.npy",mmap_mode="r")
                assert sparse.shape==raw.shape, (window["window_id"],mode,method)
                for kind in OVERLAYS:
                    assert (directory/"overlays"/mode/method/f"overlay_{kind}.mp4").is_file(), (window["window_id"],mode,method,kind)
    for value in plan["source_videos"].values():
        segments=coverage[value["video"]]
        count=value["frame_count"]
        covered=[False]*count
        for segment in segments:
            start,end=segment["source_start_frame"],segment["source_end_frame"]
            assert 0<=start<=end<count
            for frame in range(start,end+1):
                covered[frame]=True
        assert all(covered), value["video"]
    print(json.dumps({
        "status":"passed",
        "frozen_windows":plan["frozen_segment_count"],
        "new_windows":plan["new_segment_count"],
        "total_windows":plan["segment_count"],
        "source_frame_counts":{key:value["frame_count"] for key,value in plan["source_videos"].items()},
        "methods_per_mode":list(METHODS),
        "overlay_kinds":list(OVERLAYS),
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
