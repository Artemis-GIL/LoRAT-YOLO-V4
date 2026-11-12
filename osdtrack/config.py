"""Repository-relative, environment-overridable OSDTrack D0 configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Mapping

from .weights import WeightSpec, WeightVerification, load_weight_manifest, validate_weights


DEFAULT_VARIANT = "D0"
DEFAULT_TEMPORAL = False


class ConfigError(ValueError):
    """A configuration value violates the frozen OSDTrack D0 contract."""


def _path(value: os.PathLike[str] | str, base: Path) -> Path:
    text = os.path.expandvars(os.path.expanduser(os.fspath(value))).strip()
    if not text:
        raise ConfigError("A configured path cannot be empty")
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _false_only(value: str | bool, name: str) -> bool:
    if isinstance(value, bool):
        parsed = value
    else:
        normalized = value.strip().lower()
        if normalized in {"0", "false", "no", "off"}:
            parsed = False
        elif normalized in {"1", "true", "yes", "on"}:
            parsed = True
        else:
            raise ConfigError(f"{name} must be an explicit boolean, found {value!r}")
    if parsed is not False:
        raise ConfigError("The frozen D0 head requires temporal=false; temporal D/GRU is unsupported")
    return False


@dataclass(frozen=True)
class OSDTrackConfig:
    """Resolved paths and immutable algorithm identity for native execution."""

    repo_root: Path
    weights_dir: Path
    weight_specs: tuple[WeightSpec, ...]
    weight_paths: Mapping[str, Path]
    dataset_root: Path
    output_dir: Path
    audit_path: Path
    yolo_python: Path
    variant: str = DEFAULT_VARIANT
    temporal: bool = DEFAULT_TEMPORAL
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.variant != DEFAULT_VARIANT:
            raise ConfigError(
                f"This portability layer freezes variant={DEFAULT_VARIANT}; found {self.variant!r}"
            )
        if self.temporal is not False:
            raise ConfigError("D0 must be constructed with temporal=False")
        if set(self.weight_paths) != {spec.key for spec in self.weight_specs}:
            raise ConfigError("Resolved weight paths do not cover the complete five-weight manifest")

    @property
    def dino_weight(self) -> Path:
        return self.weight_paths["dinov2"]

    @property
    def lorat_weight(self) -> Path:
        return self.weight_paths["lorat"]

    @property
    def b_weight(self) -> Path:
        return self.weight_paths["b"]

    @property
    def d0_weight(self) -> Path:
        return self.weight_paths["d0"]

    @property
    def yolo_weight(self) -> Path:
        return self.weight_paths["yolo"]

    @property
    def dependency_root(self) -> Path:
        return self.repo_root / "dependencies"

    @property
    def adapter_path(self) -> Path:
        return (
            self.dependency_root
            / "LOCAL_TRACKER_SELECTION_R1_20261002"
            / "scripts"
            / "adapters.py"
        )

    @property
    def lorat_source_root(self) -> Path:
        return (
            self.dependency_root
            / "LOCAL_TRACKER_SELECTION_R1_20261002"
            / "external"
            / "LoRAT"
        )

    @property
    def geometry_scripts(self) -> Path:
        return (
            self.dependency_root
            / "LORAT_YOLO_GEOMETRY_R2_20261002"
            / "scripts"
        )

    @property
    def yolo_scripts(self) -> Path:
        return (
            self.dependency_root
            / "MVLM_V2_GLOBAL_PROPOSAL_BACKBONE_R1_20261001"
            / "scripts"
        )

    def validate_weights(self) -> tuple[WeightVerification, ...]:
        """Hash all five required checkpoints against the frozen pins."""

        return validate_weights(self.weight_paths, self.weight_specs)

    def environment(self) -> dict[str, str]:
        """Return the variables consumed by the portable archived modules."""

        values = {
            spec.environment: str(self.weight_paths[spec.key]) for spec in self.weight_specs
        }
        values.update(
            {
                "OSDTRACK_REPO_ROOT": str(self.repo_root),
                "OSDTRACK_WEIGHTS_DIR": str(self.weights_dir),
                "OSDTRACK_DATASET_ROOT": str(self.dataset_root),
                "OSDTRACK_OUTPUT_DIR": str(self.output_dir),
                "OSDTRACK_AUDIT_PATH": str(self.audit_path),
                "OSDTRACK_YOLO_PYTHON": str(self.yolo_python),
                "OSDTRACK_VARIANT": self.variant,
                "OSDTRACK_TEMPORAL": "false",
                "OSDTRACK_DEVICE": self.device,
                "OSDTRACK_ADAPTER": str(self.adapter_path),
                "OSDTRACK_ADAPTER_PATH": str(self.adapter_path),
                "OSDTRACK_LORAT_SOURCE_ROOT": str(self.lorat_source_root),
                "OSDTRACK_GEOMETRY_SCRIPTS": str(self.geometry_scripts),
                "OSDTRACK_YOLO_SCRIPTS": str(self.yolo_scripts),
            }
        )
        return values

    def as_dict(self) -> dict[str, object]:
        return {
            "repo_root": str(self.repo_root),
            "weights_dir": str(self.weights_dir),
            "weights": {
                spec.key: {
                    "path": str(self.weight_paths[spec.key]),
                    "sha256": spec.sha256,
                    "bytes": spec.bytes,
                    "environment": spec.environment,
                }
                for spec in self.weight_specs
            },
            "dataset_root": str(self.dataset_root),
            "output_dir": str(self.output_dir),
            "audit_path": str(self.audit_path),
            "yolo_python": str(self.yolo_python),
            "variant": self.variant,
            "temporal": self.temporal,
            "device": self.device,
        }


def load_config(
    *,
    repo_root: os.PathLike[str] | str | None = None,
    weights_dir: os.PathLike[str] | str | None = None,
    weight_overrides: Mapping[str, os.PathLike[str] | str] | None = None,
    dataset_root: os.PathLike[str] | str | None = None,
    output_dir: os.PathLike[str] | str | None = None,
    audit_path: os.PathLike[str] | str | None = None,
    yolo_python: os.PathLike[str] | str | None = None,
    variant: str | None = None,
    temporal: str | bool | None = None,
    device: str | None = None,
    environ: Mapping[str, str] | None = None,
    validate: bool = False,
) -> OSDTrackConfig:
    """Resolve a portable configuration.

    Precedence is explicit arguments, then environment variables, then paths
    relative to this checkout.  ``weight_overrides`` keys are the canonical
    names ``dinov2``, ``lorat``, ``b``, ``d0`` and ``yolo``.
    """

    env = os.environ if environ is None else environ
    inferred_repo = Path(__file__).resolve().parents[1]
    root = _path(repo_root if repo_root is not None else inferred_repo, inferred_repo)
    specs = load_weight_manifest(root)

    directory_value = (
        weights_dir
        if weights_dir is not None
        else env.get("OSDTRACK_WEIGHTS_DIR", root / "weights")
    )
    resolved_weights_dir = _path(directory_value, root)
    overrides = {} if weight_overrides is None else dict(weight_overrides)
    unknown = set(overrides) - {spec.key for spec in specs}
    if unknown:
        raise ConfigError(f"Unknown weight override keys: {sorted(unknown)}")
    resolved_weights: dict[str, Path] = {}
    for spec in specs:
        value = overrides.get(spec.key)
        if value is None:
            value = env.get(spec.environment)
        resolved_weights[spec.key] = _path(
            value if value is not None else resolved_weights_dir / spec.filename,
            root,
        )

    resolved_output = _path(
        output_dir if output_dir is not None else env.get("OSDTRACK_OUTPUT_DIR", root / "runs"),
        root,
    )
    resolved_dataset = _path(
        dataset_root
        if dataset_root is not None
        else env.get("OSDTRACK_DATASET_ROOT", root / "data" / "Test"),
        root,
    )
    resolved_audit = _path(
        audit_path
        if audit_path is not None
        else env.get("OSDTRACK_AUDIT_PATH", resolved_output / "MODEL.json"),
        root,
    )
    resolved_yolo_python = _path(
        yolo_python
        if yolo_python is not None
        else env.get("OSDTRACK_YOLO_PYTHON", sys.executable),
        root,
    )
    resolved_variant = variant if variant is not None else env.get("OSDTRACK_VARIANT", DEFAULT_VARIANT)
    resolved_temporal = _false_only(
        temporal if temporal is not None else env.get("OSDTRACK_TEMPORAL", "false"),
        "OSDTRACK_TEMPORAL",
    )
    resolved_device = device if device is not None else env.get("OSDTRACK_DEVICE", "cuda")
    config = OSDTrackConfig(
        repo_root=root,
        weights_dir=resolved_weights_dir,
        weight_specs=specs,
        weight_paths=MappingProxyType(resolved_weights),
        dataset_root=resolved_dataset,
        output_dir=resolved_output,
        audit_path=resolved_audit,
        yolo_python=resolved_yolo_python,
        variant=resolved_variant,
        temporal=resolved_temporal,
        device=resolved_device,
    )
    if validate:
        config.validate_weights()
    return config
