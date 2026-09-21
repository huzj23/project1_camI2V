from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from core import sha256, write_json


HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "demo0_v11_d0real"
MARKER = RUN / ".frozen.json"


def material_files() -> list[Path]:
    return sorted(
        (path for path in RUN.rglob("*") if path.is_file() and path != MARKER),
        key=lambda path: path.relative_to(RUN).as_posix(),
    )


def manifest() -> dict:
    aggregate = hashlib.sha256()
    files = material_files()
    total_bytes = 0
    by_extension: dict[str, int] = {}
    for path in files:
        relative = path.relative_to(RUN).as_posix()
        file_hash = sha256(path)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(file_hash.encode("ascii"))
        aggregate.update(b"\n")
        total_bytes += path.stat().st_size
        extension = path.suffix.lower() or "(none)"
        by_extension[extension] = by_extension.get(extension, 0) + 1
    return {
        "schema_version": "demo0-v1.1-freeze",
        "run_id": RUN.name,
        "policy": "Read-only conference baseline. New windows and algorithm changes must use a different run_id; pipeline refuses writes here.",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "by_extension": by_extension,
        "aggregate_sha256": aggregate.hexdigest(),
        "config_sha256": sha256(HERE / "config.json"),
        "scientific_status": "D0-Real baseline only; no VACE encoder or diffusion step",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze or verify the existing Demo0 conference run")
    parser.add_argument("command", choices=("freeze", "verify"))
    args = parser.parse_args()
    if not RUN.is_dir():
        raise RuntimeError(f"Run does not exist: {RUN}")
    if args.command == "freeze":
        if MARKER.exists():
            raise RuntimeError("Run is already frozen; use verify")
        current = manifest()
        write_json(MARKER, current)
        print(json.dumps(current, ensure_ascii=False, indent=2))
    else:
        if not MARKER.exists():
            raise RuntimeError("Run has no freeze marker")
        expected = json.loads(MARKER.read_text(encoding="utf-8"))
        current = manifest()
        if current != expected:
            raise SystemExit("Frozen run content or config has changed")
        print(f"Frozen run verified: {current['file_count']} files; {current['aggregate_sha256']}")


if __name__ == "__main__":
    main()
