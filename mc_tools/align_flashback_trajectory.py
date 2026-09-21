"""Align a Flashback 0.43.3 camera CSV with an exported MP4.

Logger 0.1.1 observed ExportJob.render(), which also receives Flashback's 60
unencoded warm-up renders. This tool only corrects that known, structurally
verified case; it refuses other row-count mismatches instead of guessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path


def _boxes(data: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            return
        yield kind, pos + header, pos + size
        pos += size


def _first(data: bytes, span: tuple[int, int], kind: bytes):
    return next((x for k, *x in _boxes(data, *span) if k == kind), None)


def video_sample_count(path: Path) -> tuple[int, float]:
    data = path.read_bytes()
    moov = _first(data, (0, len(data)), b"moov")
    if not moov:
        raise ValueError("MP4 has no moov box")
    for kind, start, end in _boxes(data, *moov):
        if kind != b"trak":
            continue
        mdia = _first(data, (start, end), b"mdia")
        if not mdia:
            continue
        hdlr = _first(data, mdia, b"hdlr")
        if not hdlr or data[hdlr[0] + 8 : hdlr[0] + 12] != b"vide":
            continue
        mdhd = _first(data, mdia, b"mdhd")
        minf = _first(data, mdia, b"minf")
        stbl = _first(data, minf, b"stbl") if minf else None
        stsz = _first(data, stbl, b"stsz") if stbl else None
        if not mdhd or not stsz:
            continue
        version = data[mdhd[0]]
        offset = mdhd[0] + (20 if version == 1 else 12)
        if version == 1:
            timescale, duration = struct.unpack_from(">IQ", data, offset)
        else:
            timescale, duration = struct.unpack_from(">II", data, offset)
        samples = struct.unpack_from(">I", data, stsz[0] + 8)[0]
        return samples, duration / timescale
    raise ValueError("MP4 has no readable video track")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames
    if not fields:
        raise ValueError("CSV has no header")

    samples, duration = video_sample_count(args.video)
    extra = len(rows) - samples
    if extra == 0:
        aligned = rows
        method = "already_one_row_per_encoded_frame"
        note = "Logger output already has one camera row per encoded video frame; no rows were removed."
    elif extra == 60:
        aligned = rows[60:]
        method = "drop_60_leading_flashback_0.43.3_warmup_renders"
        note = "Flashback 0.43.3 performs a forced 60-render warm-up before encoding; logger 0.1.1 observed those renders."
    else:
        raise ValueError(
            f"Refusing to guess alignment: CSV rows={len(rows)}, video frames={samples}, difference={extra}"
        )

    for index, row in enumerate(aligned):
        row["frame_index"] = str(index)
        row["source"] = "flashback_export_aligned"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_out = args.output_dir / f"{args.video.stem}.camera.csv"
    meta_out = args.output_dir / f"{args.video.stem}.camera.metadata.json"
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(aligned)

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    metadata["csv"] = csv_out.name
    metadata["video"] = args.video.name
    if "flashback_export" in metadata:
        metadata["flashback_export"]["output"] = str(args.video.resolve())
    metadata["alignment"] = {
        "status": "verified",
        "method": method,
        "raw_csv": str(args.csv),
        "raw_rows": len(rows),
        "dropped_leading_rows": extra,
        "video_frame_count": samples,
        "aligned_csv_rows": len(aligned),
        "video_duration_seconds": duration,
        "first_aligned_raw_frame_index": int(aligned[0]["frame_index"]) if extra == 0 else 60,
        "note": note,
    }
    meta_out.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_out), "metadata": str(meta_out), **metadata["alignment"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
