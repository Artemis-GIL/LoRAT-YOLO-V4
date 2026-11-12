"""Run a short RGB sequence through the portable static-D0 runtime.

Only the supplied first-frame box is consumed.  An optional archived prediction
file is used after inference solely for parity reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osdtrack import build_tracker, load_config


def _frames(folder: Path, limit: int) -> list[Path]:
    candidates = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        candidates.extend(folder.glob(pattern))
    def natural_key(path: Path) -> tuple[tuple[int, object], ...]:
        return tuple(
            (1, int(part)) if part.isdigit() else (0, part.casefold())
            for part in re.split(r"(\d+)", path.name)
        )

    paths = sorted(set(path.resolve() for path in candidates), key=natural_key)
    if not paths:
        raise RuntimeError(f"No image frames found in {folder}")
    return paths[:limit]


def _rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_expected(path: Path, count: int) -> np.ndarray:
    with path.open("r", encoding="utf-8") as stream:
        first = stream.readline()
    values = np.loadtxt(path, delimiter="," if "," in first else None, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1, 4)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Expected predictions contain NaN or infinity: {path}")
    if len(values) < count:
        raise RuntimeError(f"Expected file has {len(values)} rows; smoke needs {count}")
    return values[:count]


def _verify_frame_manifest(path: Path, paths: list[Path], initial_bbox: list[float]) -> None:
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1 or manifest.get("variant") != "D0" or manifest.get("temporal") is not False:
        raise RuntimeError(f"Unsupported D0 smoke manifest: {path}")
    if manifest.get("frame_files") != [frame.name for frame in paths]:
        raise RuntimeError("Frame names/order differ from the frozen smoke manifest")
    frozen_bbox = np.asarray(manifest.get("initial_bbox_xywh"), dtype=np.float64)
    if frozen_bbox.shape != (4,) or not np.array_equal(frozen_bbox, np.asarray(initial_bbox, dtype=np.float64)):
        raise RuntimeError("Initial bbox differs from the frozen smoke manifest")
    expected_hashes = manifest.get("rgb_sha256")
    if not isinstance(expected_hashes, list) or len(expected_hashes) != len(paths):
        raise RuntimeError("Smoke manifest has an invalid RGB hash inventory")
    for frame, expected in zip(paths, expected_hashes):
        actual = hashlib.sha256(_rgb(frame).tobytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"RGB pixels differ from smoke manifest: {frame} ({actual} != {expected})")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--initial-bbox", type=float, nargs=4, metavar=("X", "Y", "W", "H"), required=True)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--frame-manifest", type=Path, help="optional frozen filename/RGB SHA inventory")
    parser.add_argument("--expected", type=Path, help="optional archived xywh predictions, loaded only after inference")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--yolo-python", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.max_frames < 2:
        raise SystemExit("--max-frames must be at least 2")
    if not np.isfinite(args.initial_bbox).all() or min(args.initial_bbox[2:]) <= 0:
        raise SystemExit("--initial-bbox must be finite xywh with positive width and height")
    if not np.isfinite(args.atol) or args.atol < 0:
        raise SystemExit("--atol must be finite and non-negative")

    paths = _frames(args.frames_dir.resolve(), args.max_frames)
    if args.frame_manifest is not None:
        _verify_frame_manifest(args.frame_manifest.resolve(), paths, args.initial_bbox)
    config = load_config(
        weights_dir=args.weights_dir,
        yolo_python=args.yolo_python,
        output_dir=args.output_dir,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[list[float]] = []
    states: list[dict[str, object]] = []
    started = time.perf_counter()
    with build_tracker(config) as tracker:
        initial_template = None
        initial_mask = None
        for index, path in enumerate(paths, 1):
            image = _rgb(path)
            if index == 1:
                result = tracker.initialize(image, args.initial_bbox)
                initial_template = tracker.core.native.adapter.z.clone()
                initial_mask = tracker.core.native.adapter.mask.clone()
            else:
                result = tracker.track(image)
            bbox = result.get("bbox")
            exported = [0.0, 0.0, 0.0, 0.0] if bbox is None else [float(value) for value in bbox]
            predictions.append(exported)
            states.append(
                {
                    "frame": index,
                    "state": result.get("state"),
                    "reason": result.get("reason"),
                    "localization_status": result.get("localization_status"),
                }
            )
        assert initial_template is not None and initial_mask is not None
        template_preserved = bool(
            torch.equal(initial_template, tracker.core.native.adapter.z)
            and torch.equal(initial_mask, tracker.core.native.adapter.mask)
        )

    array = np.asarray(predictions, dtype=np.float64)
    formatted = "".join(",".join(f"{value:.9f}" for value in row) + "\n" for row in array)
    summary: dict[str, object] = {
        "status": "PASS",
        "variant": config.variant,
        "temporal": config.temporal,
        "frames": len(paths),
        "template_preserved": template_preserved,
        "prediction_sha256": hashlib.sha256(formatted.encode("ascii")).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
        "states": states,
        "predictions": predictions,
    }
    if args.frame_manifest is not None:
        summary["frame_manifest"] = str(args.frame_manifest.resolve())
        summary["input_rgb_sha256_matched"] = True
    if not template_preserved:
        summary["status"] = "FAIL"
    if args.expected is not None:
        expected = _load_expected(args.expected.resolve(), len(paths))
        error = np.abs(array - expected)
        matched = bool(np.allclose(array, expected, rtol=0, atol=args.atol))
        summary.update(
            expected=str(args.expected.resolve()),
            parity_atol=args.atol,
            parity_matched=matched,
            max_abs_error=float(error.max(initial=0.0)),
        )
        if not matched:
            summary["status"] = "FAIL"
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
