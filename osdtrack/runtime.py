"""Validated construction of the archived OSDTrack B + static-D0 tracker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from .config import OSDTrackConfig, load_config


def prepare_environment(config: OSDTrackConfig) -> dict[str, str]:
    """Publish resolved paths before importing any archived model modules."""

    values = config.environment()
    os.environ.update(values)
    for source_root in (config.lorat_source_root, config.geometry_scripts, config.yolo_scripts):
        text = str(source_root)
        if text not in sys.path:
            sys.path.insert(0, text)
    return values


def _validate_repository_inputs(config: OSDTrackConfig) -> tuple[Path, ...]:
    required_files = (
        config.adapter_path,
        config.geometry_scripts / "policy.py",
        config.yolo_scripts / "yolo26_adapter.py",
        config.yolo_python,
    )
    required_directories = (config.lorat_source_root,)
    missing = [path for path in required_files if not path.is_file()]
    missing.extend(path for path in required_directories if not path.is_dir())
    if missing:
        raise RuntimeError(
            "Repository dependencies are incomplete:\n- " + "\n- ".join(map(str, missing))
        )
    return required_files + required_directories


def preflight(config: OSDTrackConfig | None = None) -> dict[str, object]:
    """Perform CPU-only path/provenance checks, including all five hashes."""

    resolved = load_config() if config is None else config
    repository_inputs = _validate_repository_inputs(resolved)
    verified = resolved.validate_weights()
    if resolved.variant != "D0" or resolved.temporal is not False:
        raise RuntimeError("Preflight refuses anything except variant D0 with temporal=false")
    return {
        "status": "PASS",
        "model": "OSDTrack",
        "variant": resolved.variant,
        "temporal": resolved.temporal,
        "weights": [item.as_dict() for item in verified],
        "repository_inputs": [str(path.resolve()) for path in repository_inputs],
        "dataset_root": str(resolved.dataset_root),
        "output_dir": str(resolved.output_dir),
        "yolo_python": str(resolved.yolo_python),
    }


def build_tracker(
    config: OSDTrackConfig | None = None,
    *,
    audit_path: os.PathLike[str] | str | None = None,
):
    """Build the frozen D0 tracker without changing its algorithmic classes.

    Checkpoints are verified before CUDA/PyTorch model allocation.  The legacy
    builder is called with ``variant='D0'`` and its constructed action head is
    checked again for ``temporal is False`` before being returned.
    """

    resolved = load_config(audit_path=audit_path) if config is None else config
    if audit_path is not None and config is not None:
        requested = Path(os.path.expandvars(os.path.expanduser(os.fspath(audit_path))))
        if not requested.is_absolute():
            requested = config.repo_root / requested
        requested = requested.resolve()
        if requested != config.audit_path.resolve():
            raise ValueError("audit_path conflicts with the supplied OSDTrackConfig")
    _validate_repository_inputs(resolved)
    resolved.validate_weights()
    if resolved.device.lower() != "cuda":
        raise RuntimeError(
            "The frozen runtime currently supports only device='cuda' (default GPU 0); "
            "CPU and explicit CUDA indices are supported only after device plumbing is audited"
        )
    prepare_environment(resolved)

    # Imports stay here so config inspection and hash verification work on a
    # CPU-only host without importing torch or allocating a CUDA context.
    from .compat import install_portable_runtime

    install_portable_runtime()
    from research.de_r1 import common as de_common
    from research.de_r1 import tracker_de

    # The archived modules historically captured an absolute B path at import.
    # Rebinding that input path changes no model/controller logic and keeps old
    # research entry points intact while the portable builder is explicit.
    de_common.B_WEIGHTS = resolved.b_weight
    tracker_de.B_WEIGHTS = resolved.b_weight
    tracker = tracker_de.build(
        variant="D0",
        audit=resolved.audit_path,
        checkpoint=resolved.d0_weight,
    )
    try:
        model = tracker.controller.action_model
        if getattr(model, "temporal", None) is not False:
            raise RuntimeError("Constructed D0 action head is not explicitly temporal=False")
    except BaseException:
        close = getattr(tracker, "close", None)
        if callable(close):
            close()
        raise
    return tracker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("show-config", "preflight"),
        help="print resolved paths, or strictly verify repository inputs and all weights",
    )
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--yolo-python", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(
        weights_dir=args.weights_dir,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        yolo_python=args.yolo_python,
    )
    result = config.as_dict() if args.command == "show-config" else preflight(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
