"""Portable entry points for the frozen OSDTrack B + static-D0 model.

Importing :mod:`osdtrack` is intentionally lightweight.  CUDA, PyTorch and the
archived research implementation are imported only by ``build_tracker``.
"""

from __future__ import annotations

from .config import ConfigError, OSDTrackConfig, load_config
from .weights import (
    WeightManifestError,
    WeightValidationError,
    WeightVerification,
    sha256_file,
)


def validate_weights(config: OSDTrackConfig | None = None) -> tuple[WeightVerification, ...]:
    """Convenience entry point that strictly verifies the resolved five files."""

    resolved = load_config() if config is None else config
    return resolved.validate_weights()


def prepare_environment(config: OSDTrackConfig) -> dict[str, str]:
    from .runtime import prepare_environment as _prepare_environment

    return _prepare_environment(config)


def preflight(config: OSDTrackConfig | None = None) -> dict[str, object]:
    from .runtime import preflight as _preflight

    return _preflight(config)


def build_tracker(config: OSDTrackConfig | None = None, *, audit_path=None):
    from .runtime import build_tracker as _build_tracker

    return _build_tracker(config, audit_path=audit_path)


__all__ = [
    "ConfigError",
    "OSDTrackConfig",
    "WeightManifestError",
    "WeightValidationError",
    "WeightVerification",
    "build_tracker",
    "load_config",
    "preflight",
    "prepare_environment",
    "sha256_file",
    "validate_weights",
]

__version__ = "0.1.0"
