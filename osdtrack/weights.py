"""Pinned checkpoint metadata and strict, dependency-free verification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


MANIFEST_RELATIVE_PATH = Path("provenance") / "OSDTRACK_D0.json"

# These pins deliberately live in code as well as provenance.  Editing a JSON
# manifest alone must never make an unrelated checkpoint pass verification.
PINNED_WEIGHT_METADATA: Mapping[str, Mapping[str, Any]] = {
    "dinov2": {
        "filename": "dinov2_vitl14_pretrain.pth",
        "environment": "OSDTRACK_DINO_WEIGHT",
        "sha256": "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428",
        "bytes": 1_217_586_395,
    },
    "lorat": {
        "filename": "LoRAT_large-378.bin",
        "environment": "OSDTRACK_LORAT_WEIGHT",
        "sha256": "601529735439da5d7ad0136b63e894ee02923d4694ce17d3568e333c298a06b0",
        "bytes": 130_101_948,
    },
    "b": {
        "filename": "LoRAT_B_UAV_final.safetensors",
        "environment": "OSDTRACK_B_WEIGHT",
        "sha256": "e62112dd6582552d1668130b58b0a0c3382775670ccc6c714e9b00e94dbd58bc",
        "bytes": 130_101_948,
    },
    "d0": {
        "filename": "D0_static_final.safetensors",
        "environment": "OSDTRACK_D0_WEIGHT",
        "sha256": "0a27352804839714f7e691f60ae288ef06dc0b358209e379de156831d24871bd",
        "bytes": 1_002_996,
    },
    "yolo": {
        "filename": "yolo26_uav_best.pt",
        "environment": "OSDTRACK_YOLO_WEIGHT",
        "sha256": "efa76597c9912fbda8ff150208996a88597544fb5ba2e52a6ad02e4dc964b433",
        "bytes": 60_285_155,
    },
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class WeightManifestError(RuntimeError):
    """The checked-in D0 provenance manifest is absent or inconsistent."""


class WeightValidationError(RuntimeError):
    """One or more required checkpoint files failed strict verification."""


@dataclass(frozen=True)
class WeightSpec:
    """One immutable checkpoint identity from the checked-in manifest."""

    key: str
    filename: str
    environment: str
    sha256: str
    bytes: int
    role: str
    release_asset: str | None


@dataclass(frozen=True)
class WeightVerification:
    """A successful size and SHA-256 verification result."""

    key: str
    path: Path
    bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "path": str(self.path),
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError as exc:
        raise WeightManifestError(f"Weight provenance manifest is missing: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WeightManifestError(f"Cannot read weight provenance manifest {path}: {exc}") from exc


def load_weight_manifest(repo_root: Path) -> tuple[WeightSpec, ...]:
    """Load and cross-check the repository's five frozen checkpoint records."""

    manifest_path = Path(repo_root) / MANIFEST_RELATIVE_PATH
    document = _read_json(manifest_path)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise WeightManifestError(f"Unsupported weight manifest schema: {manifest_path}")
    if document.get("model") != "OSDTrack" or document.get("variant") != "D0":
        raise WeightManifestError("The weight manifest is not the frozen OSDTrack D0 variant")
    if document.get("temporal") is not False:
        raise WeightManifestError("OSDTrack D0 must explicitly declare temporal=false")
    records = document.get("weights")
    if not isinstance(records, list) or len(records) != len(PINNED_WEIGHT_METADATA):
        raise WeightManifestError("OSDTrack D0 must declare exactly five required weights")

    by_key: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("key"), str):
            raise WeightManifestError("Each weight manifest entry must be an object with a string key")
        key = record["key"]
        if key in by_key:
            raise WeightManifestError(f"Duplicate weight key in manifest: {key}")
        by_key[key] = record
    if set(by_key) != set(PINNED_WEIGHT_METADATA):
        raise WeightManifestError(
            "Weight keys differ from the five code-pinned checkpoints: "
            f"expected {sorted(PINNED_WEIGHT_METADATA)}, found {sorted(by_key)}"
        )

    specs: list[WeightSpec] = []
    for key, pinned in PINNED_WEIGHT_METADATA.items():
        record = by_key[key]
        for field in ("filename", "environment", "sha256", "bytes"):
            if record.get(field) != pinned[field]:
                raise WeightManifestError(
                    f"Manifest value for {key}.{field} differs from the code pin"
                )
        digest = record["sha256"]
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise WeightManifestError(f"Invalid SHA-256 for weight {key}")
        role = record.get("role")
        release_asset = record.get("release_asset")
        if not isinstance(role, str) or not role.strip():
            raise WeightManifestError(f"Missing role for weight {key}")
        if release_asset is not None and not isinstance(release_asset, str):
            raise WeightManifestError(f"Invalid release_asset for weight {key}")
        specs.append(
            WeightSpec(
                key=key,
                filename=record["filename"],
                environment=record["environment"],
                sha256=digest,
                bytes=record["bytes"],
                role=role,
                release_asset=release_asset,
            )
        )
    return tuple(specs)


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 of *path* without loading a large weight into RAM."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_weights(
    paths: Mapping[str, Path],
    specs: tuple[WeightSpec, ...],
) -> tuple[WeightVerification, ...]:
    """Strictly validate all five sizes and SHA-256 digests.

    All entries are checked before an exception is raised so a new machine gets
    one actionable report instead of a sequence of single-file failures.
    """

    expected_keys = {spec.key for spec in specs}
    if set(paths) != expected_keys:
        raise WeightValidationError(
            f"Weight path keys must be exactly {sorted(expected_keys)}; found {sorted(paths)}"
        )
    failures: list[str] = []
    verified: list[WeightVerification] = []
    for spec in specs:
        path = Path(paths[spec.key])
        try:
            if not path.is_file():
                failures.append(f"{spec.key}: missing regular file: {path}")
                continue
            size = path.stat().st_size
            digest = sha256_file(path)
            valid = True
            if size != spec.bytes:
                failures.append(
                    f"{spec.key}: size mismatch for {path} "
                    f"(expected {spec.bytes}, found {size})"
                )
                valid = False
            if digest != spec.sha256:
                failures.append(
                    f"{spec.key}: SHA-256 mismatch for {path} "
                    f"(expected {spec.sha256}, found {digest})"
                )
                valid = False
            if not valid:
                continue
            verified.append(
                WeightVerification(spec.key, path.resolve(), size, digest)
            )
        except OSError as exc:
            failures.append(f"{spec.key}: cannot read {path}: {exc}")
    if failures:
        raise WeightValidationError("Weight validation failed:\n- " + "\n- ".join(failures))
    if len(verified) != len(PINNED_WEIGHT_METADATA):
        raise WeightValidationError("Internal error: not all five weights were verified")
    return tuple(verified)
