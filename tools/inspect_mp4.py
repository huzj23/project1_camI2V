"""Print basic MP4 video-track timing without external dependencies."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path


CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


def boxes(data: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            if pos + 16 > end:
                return
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            return
        yield kind, pos + header, pos + size
        pos += size


def children(data: bytes, start: int, end: int, kind: bytes):
    for child_kind, child_start, child_end in boxes(data, start, end):
        if child_kind == kind:
            yield child_start, child_end


def first(data: bytes, start: int, end: int, kind: bytes):
    return next(children(data, start, end, kind), None)


def mdhd(data: bytes, start: int):
    version = data[start]
    offset = start + (20 if version == 1 else 12)
    if version == 1:
        return struct.unpack_from(">IQ", data, offset)
    return struct.unpack_from(">II", data, offset)


def main(path: Path):
    data = path.read_bytes()
    moov = first(data, 0, len(data), b"moov")
    if not moov:
        raise SystemExit("No moov box")
    tracks = []
    for trak in children(data, *moov, b"trak"):
        mdia = first(data, *trak, b"mdia")
        if not mdia:
            continue
        hdlr = first(data, *mdia, b"hdlr")
        time_box = first(data, *mdia, b"mdhd")
        if not hdlr or not time_box:
            continue
        handler = data[hdlr[0] + 8 : hdlr[0] + 12]
        timescale, duration = mdhd(data, time_box[0])
        item = {
            "handler": handler.decode("ascii", "replace"),
            "timescale": timescale,
            "duration_ticks": duration,
            "duration_seconds": duration / timescale,
        }
        minf = first(data, *mdia, b"minf")
        stbl = first(data, *minf, b"stbl") if minf else None
        stsz = first(data, *stbl, b"stsz") if stbl else None
        if stsz:
            item["sample_count"] = struct.unpack_from(">I", data, stsz[0] + 8)[0]
            item["average_fps"] = item["sample_count"] / item["duration_seconds"]
        tracks.append(item)
    print(json.dumps({"path": str(path), "tracks": tracks}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
