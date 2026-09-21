from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CLIPS = ("简单平动透视关系", "非对称建筑平动")


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    for clip in CLIPS:
        capture = cv2.VideoCapture(str(ROOT / f"{clip}.mp4"))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        samples = [0, frame_count // 4, frame_count // 2, 3 * frame_count // 4, frame_count - 1]
        tiles = []
        for frame_id in samples:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Failed to read {clip} frame {frame_id}")
            frame = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
            cv2.putText(
                frame,
                f"frame {frame_id}/{frame_count - 1}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            tiles.append(frame)
        sheet = np.concatenate(tiles, axis=1)
        target = out_dir / f"{clip}.jpg"
        ok, encoded = cv2.imencode(".jpg", sheet)
        if not ok:
            raise RuntimeError(f"Failed to encode {target}")
        encoded.tofile(target)
        print(f"{clip}: frames={frame_count}, preview={target}")
        capture.release()


if __name__ == "__main__":
    main()
