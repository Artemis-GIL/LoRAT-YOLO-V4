"""Persistent portable YOLO worker; receives only current RGB frame bytes."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np


def _required_path(environment: str, *, directory: bool = False) -> Path:
    value = os.environ.get(environment)
    if not value:
        raise RuntimeError(f"{environment} is required")
    path = Path(value).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{environment} does not name a {kind}: {path}")
    return path


def send(message: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(message, allow_nan=False) + "\n")
    sys.stdout.flush()


def exact_read(stream, count: int) -> bytes:
    parts = []
    remaining = count
    while remaining:
        part = stream.read(remaining)
        if not part:
            raise EOFError("Incomplete RGB frame")
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)


def main() -> None:
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        import ultralytics

        scripts = _required_path("OSDTRACK_YOLO_SCRIPTS", directory=True)
        weight = _required_path("OSDTRACK_YOLO_WEIGHT")
        sys.path.insert(0, str(scripts))
        from yolo26_adapter import YOLO26Detector

        random.seed(20261002)
        np.random.seed(20261002)
        torch.manual_seed(20261002)
        torch.set_num_threads(4)
        detector = YOLO26Detector(weight, max_det=12)
    send(
        {
            "status": "ready",
            "python": sys.version,
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        }
    )
    stream = sys.stdin.buffer
    while True:
        line = stream.readline()
        if not line:
            break
        request = json.loads(line)
        if request.get("op") == "close":
            break
        if request.get("op") != "detect":
            raise ValueError("Unknown operation")
        shape = request.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(type(value) is not int for value in shape)
            or shape[2] != 3
            or min(shape) <= 0
        ):
            raise ValueError("Expected positive H,W,3 RGB shape")
        count = shape[0] * shape[1] * shape[2]
        if count > 256 * 1024 * 1024:
            raise ValueError("Frame exceeds 256 MiB")
        payload = exact_read(stream, count)
        rgb = np.frombuffer(payload, dtype=np.uint8).reshape(shape)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            candidates = detector.infer(np.ascontiguousarray(rgb[:, :, ::-1]))
        torch.cuda.synchronize()
        send(
            {
                "status": "ok",
                "request_id": request.get("request_id"),
                "rgb_sha256": hashlib.sha256(payload).hexdigest(),
                "candidates": candidates,
                "seconds": time.perf_counter() - started,
            }
        )


if __name__ == "__main__":
    main()
