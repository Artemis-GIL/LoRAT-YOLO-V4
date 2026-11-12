"""Narrow runtime shims around the byte-frozen Ubuntu research snapshot.

The archived files remain unchanged so ``tools/verify_snapshot.py`` continues
to prove their original hashes.  These adapters redirect only machine-local
weight/process paths; model and controller methods stay inherited from the
frozen implementation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
from typing import Any

import numpy as np
import torch


_ADAPTER_LOAD_LOCK = threading.Lock()
_LEGACY_DINO_PATH = "/home/xu/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth"
_DEFAULT_YOLO_TIMEOUT_SECONDS = 60.0


def _required_path(environment: str) -> Path:
    value = os.environ.get(environment)
    if not value:
        raise RuntimeError(f"{environment} is required by the portable runtime")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{environment} does not name a file: {path}")
    return path


def _load_frozen_adapter(audit_path: os.PathLike[str] | str):
    """Instantiate the frozen LoRAT adapter with two path-only redirects."""

    adapter_path = _required_path("OSDTRACK_ADAPTER_PATH")
    spec = importlib.util.spec_from_file_location("_osdtrack_frozen_lorat_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen LoRAT adapter: {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dino_path = _required_path("OSDTRACK_DINO_WEIGHT")
    lorat_path = _required_path("OSDTRACK_LORAT_WEIGHT")
    legacy_lorat_path = (module.ROOT / "weights" / "LoRAT_large-378.bin").resolve()

    from safetensors import torch as safetensors_torch

    original_torch_load = torch.load
    original_safetensors_load = safetensors_torch.load_file

    def redirected_torch_load(source: Any, *args: Any, **kwargs: Any):
        try:
            legacy_source = os.fspath(source) == _LEGACY_DINO_PATH
        except TypeError:
            legacy_source = False
        redirected = dino_path if legacy_source else source
        return original_torch_load(redirected, *args, **kwargs)

    def redirected_safetensors_load(source: Any, *args: Any, **kwargs: Any):
        try:
            candidate = Path(source).expanduser().resolve()
        except (TypeError, ValueError, OSError):
            candidate = None
        redirected = lorat_path if candidate == legacy_lorat_path else source
        return original_safetensors_load(str(redirected), *args, **kwargs)

    # The frozen constructor imports load_file inside __init__, so patching the
    # module attribute here redirects that one load without editing the archive.
    # Startup is single-threaded; the lock also protects embedded callers.
    with _ADAPTER_LOAD_LOCK:
        torch.load = redirected_torch_load
        safetensors_torch.load_file = redirected_safetensors_load
        try:
            return module.LoRATLocal(audit_path)
        finally:
            safetensors_torch.load_file = original_safetensors_load
            torch.load = original_torch_load


def _portable_lorat_core_class():
    from tracking.lorat_yolo import LoRATCore as FrozenLoRATCore

    class PortableLoRATCore(FrozenLoRATCore):
        """Frozen LoRATCore methods with a repository-local adapter constructor."""

        def __init__(self, audit_path):
            self.adapter = _load_frozen_adapter(audit_path)
            self.initialized = False

    PortableLoRATCore.__name__ = "PortableLoRATCore"
    PortableLoRATCore.__qualname__ = "PortableLoRATCore"
    return PortableLoRATCore


class PortableCurrentFrameDetector:
    """Frozen current-RGB IPC contract with portable executable/worker paths."""

    def __init__(self):
        python_executable = _required_path("OSDTRACK_YOLO_PYTHON")
        worker = Path(__file__).with_name("yolo_worker.py").resolve()
        timeout_text = os.environ.get(
            "OSDTRACK_YOLO_TIMEOUT_SECONDS", str(_DEFAULT_YOLO_TIMEOUT_SECONDS)
        )
        try:
            self.response_timeout = float(timeout_text)
        except ValueError as exc:
            raise RuntimeError("OSDTRACK_YOLO_TIMEOUT_SECONDS must be a number") from exc
        if not math.isfinite(self.response_timeout) or not 0.0 < self.response_timeout <= 3600.0:
            raise RuntimeError(
                "OSDTRACK_YOLO_TIMEOUT_SECONDS must be finite and in (0, 3600]"
            )
        self.process = subprocess.Popen(
            [str(python_executable), "-u", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        )
        self.request_id = 0
        self._responses: queue.Queue[tuple[str, object]] = queue.Queue()
        self._reader = threading.Thread(
            target=self._read_responses,
            name="osdtrack-yolo-stdout",
            daemon=True,
        )
        self._reader.start()
        try:
            self.environment = self._receive()
            if self.environment.get("status") != "ready":
                raise RuntimeError(f"Detector startup failed: {self.environment}")
        except BaseException:
            self.close(force=True)
            raise

    def _read_responses(self) -> None:
        """Read worker stdout on a thread so Windows anonymous pipes can time out."""

        process = self.process
        try:
            if process.stdout is None:
                self._responses.put(("error", RuntimeError("YOLO worker stdout is unavailable")))
                return
            while True:
                line = process.stdout.readline()
                if not line:
                    self._responses.put(("eof", process.poll()))
                    return
                self._responses.put(("line", line))
        except BaseException as exc:
            self._responses.put(("error", exc))

    def _receive(self) -> dict[str, Any]:
        try:
            kind, value = self._responses.get(timeout=self.response_timeout)
        except queue.Empty as exc:
            raise RuntimeError(
                "YOLO worker response timed out after "
                f"{self.response_timeout:g} seconds (request_id={self.request_id})"
            ) from exc
        if kind == "eof":
            raise RuntimeError(f"YOLO worker exited without a response (exit={value}); see stderr")
        if kind == "error":
            raise RuntimeError("YOLO worker response reader failed") from value  # type: ignore[arg-type]
        response = json.loads(value)  # type: ignore[arg-type]
        if not isinstance(response, dict):
            raise RuntimeError("YOLO worker response must be a JSON object")
        return response

    def infer(self, rgb: np.ndarray) -> dict[str, Any]:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected uint8 RGB image")
        if self.process.stdin is None or self.process.poll() is not None:
            raise RuntimeError("YOLO worker is not running")
        self.request_id += 1
        payload = np.ascontiguousarray(rgb).tobytes()
        request = {"op": "detect", "shape": list(rgb.shape), "request_id": self.request_id}
        try:
            self.process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
            self.process.stdin.write(payload)
            self.process.stdin.flush()
            result = self._receive()
            if result.get("status") != "ok" or result.get("request_id") != self.request_id:
                raise RuntimeError("Detector response does not match the current frame")
            if result.get("rgb_sha256") != hashlib.sha256(payload).hexdigest():
                raise RuntimeError("Detector received different RGB pixels")
            return result
        except BaseException:
            self.close(force=True)
            raise

    def close(self, *, force: bool = False) -> None:
        process = getattr(self, "process", None)
        if process is None:
            return

        def stop_hard() -> None:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
                process.wait()

        if process.poll() is None:
            if force:
                stop_hard()
            else:
                try:
                    if process.stdin is not None:
                        process.stdin.write(b'{"op":"close"}\n')
                        process.stdin.flush()
                        process.stdin.close()
                    process.wait(timeout=10)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    stop_hard()
        for stream in (process.stdin, process.stdout):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass
        reader = getattr(self, "_reader", None)
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1)
        self.process = None


def install_portable_runtime() -> dict[str, str]:
    """Patch the two constructor seams used by ``tracker_de.build``."""

    import tracking.lorat_yolo as lorat_yolo

    portable_lorat_core = _portable_lorat_core_class()
    lorat_yolo.LoRATCore = portable_lorat_core

    # Import after the LoRAT patch: online_tracker caches LoRATCore at import,
    # while tracker_de imports CurrentFrameDetector dynamically during build.
    import tracking.online_tracker as online_tracker

    online_tracker.LoRATCore = portable_lorat_core
    online_tracker.CurrentFrameDetector = PortableCurrentFrameDetector
    return {
        "lorat_core": f"{portable_lorat_core.__module__}.{portable_lorat_core.__name__}",
        "detector": f"{PortableCurrentFrameDetector.__module__}.{PortableCurrentFrameDetector.__name__}",
    }


__all__ = ["PortableCurrentFrameDetector", "install_portable_runtime"]
