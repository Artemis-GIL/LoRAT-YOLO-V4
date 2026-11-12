"""Stateful, simulator-independent request service for OSDTrack."""

from __future__ import annotations

from enum import Enum
import threading
import time
from typing import Any, BinaryIO, Mapping, Optional, Protocol, Sequence

import numpy as np

from .protocol import (
    BBox,
    Operation,
    ProtocolError,
    Request,
    Response,
    ResponseStatus,
    TrackerStatus,
    read_request,
    validate_bbox,
    write_response,
)


class TrackerBackend(Protocol):
    """Minimal adapter expected by :class:`TrackingService`.

    ``initialize`` may return ``None`` (the initial box is then echoed), an
    xywh sequence, or a mapping containing ``bbox``.  ``track`` returns an
    xywh sequence, ``None`` for a lost target, or a mapping.  A mapping may
    expose ``tracker_status`` directly or use the archived tracker's
    ``localization_status``/``state`` fields.  ``reset`` and ``close`` are
    optional at runtime to accommodate existing research trackers.
    """

    def initialize(self, image_rgb: np.ndarray, initial_bbox: BBox) -> Any:
        ...

    def track(self, image_rgb: np.ndarray) -> Any:
        ...


class SessionState(str, Enum):
    NEW = "new"
    RESET = "reset"
    INITIALIZED = "initialized"
    FAILED = "failed"
    CLOSED = "closed"


class _ServiceFailure(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class TrackingService:
    """Enforce one ordered tracking session around an arbitrary backend.

    Valid success order is ``reset -> init -> track* -> close``.  A new reset
    can replace an active (not closed) session.  Track frame IDs must increase
    by exactly one and timestamps must increase strictly.  State advances only
    after the backend call and its bbox result both succeed.
    """

    def __init__(self, backend: TrackerBackend):
        if backend is None:
            raise ValueError("backend is required")
        self.backend = backend
        self._state = SessionState.NEW
        self._session_id: Optional[str] = None
        self._last_frame_id: Optional[int] = None
        self._last_timestamp: Optional[float] = None
        self._lock = threading.RLock()

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def last_frame_id(self) -> Optional[int]:
        return self._last_frame_id

    def handle(self, request: Request) -> Response:
        """Handle one already-decoded request and always return a response."""

        started = time.perf_counter_ns()
        if not isinstance(request, Request):
            raise TypeError("request must be an osdtrack.protocol.Request")
        with self._lock:
            try:
                return self._handle_success(request, started)
            except _ServiceFailure as exc:
                if (
                    request.op in (Operation.INIT, Operation.TRACK)
                    and exc.code == "invalid_backend_result"
                ):
                    self._state = SessionState.FAILED
                return self._error_response(request, started, exc.code, str(exc))
            except ProtocolError as exc:
                return self._error_response(request, started, exc.code, str(exc))
            except Exception as exc:
                if request.op in (Operation.INIT, Operation.TRACK):
                    # The backend may have mutated before raising.  Replaying a
                    # frame would then desynchronize protocol and model state.
                    self._state = SessionState.FAILED
                message = str(exc).strip() or type(exc).__name__
                return self._error_response(request, started, "backend_error", message[:1024])

    def _handle_success(self, request: Request, started: int) -> Response:
        if self._state is SessionState.CLOSED:
            raise _ServiceFailure("session_closed", "the service session is closed")
        if request.op is Operation.RESET:
            self._call_optional("reset")
            self._session_id = request.session_id
            self._last_frame_id = None
            self._last_timestamp = None
            self._state = SessionState.RESET
            return Response(
                op=request.op,
                session_id=request.session_id,
                status=ResponseStatus.RESET,
                elapsed_ms=self._elapsed_ms(started),
            )
        if self._state is SessionState.FAILED:
            raise _ServiceFailure(
                "reset_required",
                "the backend failed after a frame began; reset and initialize again",
            )

        self._require_session(request)
        if request.op is Operation.INIT:
            if self._state is not SessionState.RESET:
                raise _ServiceFailure("invalid_state", "init requires a preceding reset")
            assert request.image is not None and request.bbox is not None
            assert request.frame_id is not None and request.timestamp is not None
            initial_bbox = self._bbox_inside_image(request.bbox, request.image)
            result = self.backend.initialize(request.image, initial_bbox)
            bbox = self._result_bbox(result, fallback=request.bbox)
            self._last_frame_id = request.frame_id
            self._last_timestamp = request.timestamp
            self._state = SessionState.INITIALIZED
            return Response(
                op=request.op,
                session_id=request.session_id,
                status=ResponseStatus.INITIALIZED,
                elapsed_ms=self._elapsed_ms(started),
                frame_id=request.frame_id,
                timestamp=request.timestamp,
                bbox=bbox,
            )

        if request.op is Operation.TRACK:
            if self._state is not SessionState.INITIALIZED:
                raise _ServiceFailure("invalid_state", "track requires a successful init")
            assert request.frame_id is not None and request.timestamp is not None and request.image is not None
            assert self._last_frame_id is not None and self._last_timestamp is not None
            expected_frame_id = self._last_frame_id + 1
            if request.frame_id != expected_frame_id:
                raise _ServiceFailure(
                    "out_of_order_frame",
                    f"expected frame_id {expected_frame_id}, received {request.frame_id}",
                )
            if request.timestamp <= self._last_timestamp:
                raise _ServiceFailure(
                    "out_of_order_timestamp",
                    f"timestamp must be greater than {self._last_timestamp}",
                )
            result = self.backend.track(request.image)
            bbox, tracker_status = self._track_result(result)
            self._last_frame_id = request.frame_id
            self._last_timestamp = request.timestamp
            return Response(
                op=request.op,
                session_id=request.session_id,
                status=ResponseStatus.TRACKING,
                elapsed_ms=self._elapsed_ms(started),
                frame_id=request.frame_id,
                timestamp=request.timestamp,
                bbox=bbox,
                tracker_status=tracker_status,
            )

        if request.op is Operation.CLOSE:
            self._call_optional("close")
            self._state = SessionState.CLOSED
            return Response(
                op=request.op,
                session_id=request.session_id,
                status=ResponseStatus.CLOSED,
                elapsed_ms=self._elapsed_ms(started),
            )
        raise _ServiceFailure("unsupported_operation", f"unsupported op: {request.op}")

    def _require_session(self, request: Request) -> None:
        if self._state is SessionState.NEW or self._session_id is None:
            raise _ServiceFailure("invalid_state", f"{request.op.value} requires a preceding reset")
        if request.session_id != self._session_id:
            raise _ServiceFailure(
                "session_mismatch",
                f"active session is {self._session_id!r}, received {request.session_id!r}",
            )

    def _call_optional(self, method_name: str) -> None:
        method = getattr(self.backend, method_name, None)
        if method is not None:
            if not callable(method):
                raise _ServiceFailure("invalid_backend", f"backend.{method_name} is not callable")
            method()

    @staticmethod
    def _result_bbox(result: Any, fallback: Optional[Sequence[float]] = None) -> BBox:
        if result is None:
            if fallback is None:
                raise _ServiceFailure("invalid_backend_result", "tracker returned no bbox")
            result = fallback
        elif isinstance(result, Mapping):
            if "bbox" not in result:
                raise _ServiceFailure("invalid_backend_result", "tracker result mapping has no 'bbox'")
            result = result["bbox"]
        try:
            return validate_bbox(result)
        except ProtocolError as exc:
            raise _ServiceFailure("invalid_backend_result", str(exc)) from exc

    @classmethod
    def _track_result(cls, result: Any) -> tuple[Optional[BBox], TrackerStatus]:
        if result is None:
            return None, TrackerStatus.LOST

        raw_bbox: Any = result
        raw_status: Any = None
        if isinstance(result, Mapping):
            if "bbox" not in result:
                raise _ServiceFailure(
                    "invalid_backend_result",
                    "tracker result mapping must contain an explicit 'bbox' field",
                )
            raw_bbox = result.get("bbox")
            if "tracker_status" in result:
                raw_status = result["tracker_status"]
            elif "localization_status" in result:
                localization = getattr(result["localization_status"], "value", result["localization_status"])
                key = str(localization).strip().lower()
                raw_status = {
                    "confirmed": TrackerStatus.TRACKING,
                    "tracking": TrackerStatus.TRACKING,
                    "tentative": TrackerStatus.TENTATIVE,
                    "verifying": TrackerStatus.TENTATIVE,
                    "unavailable": TrackerStatus.LOST,
                    "lost": TrackerStatus.LOST,
                    "searching": TrackerStatus.LOST,
                }.get(key)
                if raw_status is None:
                    raise _ServiceFailure(
                        "invalid_backend_result",
                        f"unknown localization_status: {localization!r}",
                    )
            elif "state" in result:
                state = getattr(result["state"], "value", result["state"])
                key = str(state).strip().lower()
                if key == "tracking":
                    raw_status = TrackerStatus.TRACKING
                elif key == "verifying":
                    raw_status = TrackerStatus.TENTATIVE
                elif key == "searching":
                    raw_status = TrackerStatus.LOST if raw_bbox is None else TrackerStatus.TENTATIVE
                else:
                    raise _ServiceFailure("invalid_backend_result", f"unknown tracker state: {state!r}")

        status = cls._backend_tracker_status(raw_status, has_bbox=raw_bbox is not None)
        if raw_bbox is None:
            if status is not TrackerStatus.LOST:
                raise _ServiceFailure(
                    "invalid_backend_result",
                    f"backend status {status.value!r} requires a bbox",
                )
            return None, status
        try:
            bbox = validate_bbox(raw_bbox)
        except ProtocolError as exc:
            raise _ServiceFailure("invalid_backend_result", str(exc)) from exc
        if status is TrackerStatus.LOST:
            raise _ServiceFailure("invalid_backend_result", "backend returned LOST together with a bbox")
        return bbox, status

    @staticmethod
    def _bbox_inside_image(value: Sequence[float], image: np.ndarray) -> BBox:
        bbox = validate_bbox(value)
        x, y, width, height = bbox
        image_height, image_width = image.shape[:2]
        tolerance = 1e-8
        if (
            x < -tolerance
            or y < -tolerance
            or x + width > image_width + tolerance
            or y + height > image_height + tolerance
        ):
            raise ProtocolError(
                "invalid_bbox",
                f"initial bbox must lie inside the {image_width}x{image_height} frame",
            )
        return bbox

    @staticmethod
    def _backend_tracker_status(raw_status: Any, *, has_bbox: bool) -> TrackerStatus:
        if raw_status is None:
            return TrackerStatus.TRACKING if has_bbox else TrackerStatus.LOST
        if isinstance(raw_status, TrackerStatus):
            return raw_status
        raw_status = getattr(raw_status, "value", raw_status)
        if not isinstance(raw_status, str):
            raise _ServiceFailure("invalid_backend_result", "backend tracker status must be a string")
        try:
            return TrackerStatus(raw_status.strip().lower())
        except ValueError as exc:
            raise _ServiceFailure(
                "invalid_backend_result",
                f"unknown backend tracker status: {raw_status!r}",
            ) from exc

    @staticmethod
    def _elapsed_ms(started: int) -> float:
        # perf_counter_ns is monotonic.  max protects against exotic test mocks.
        return max(0.0, (time.perf_counter_ns() - started) / 1_000_000.0)

    def _error_response(self, request: Request, started: int, code: str, message: str) -> Response:
        return Response(
            op=request.op,
            session_id=request.session_id,
            status=ResponseStatus.ERROR,
            elapsed_ms=self._elapsed_ms(started),
            frame_id=request.frame_id,
            timestamp=request.timestamp,
            error_code=code,
            error_message=message,
        )


def serve(
    service: TrackingService,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    *,
    stop_on_close: bool = True,
) -> int:
    """Serve framed requests until clean EOF or a successful close.

    Malformed wire input raises :class:`ProtocolError` and terminates the
    connection because its framing can no longer be trusted.  Valid requests
    with state/order/backend errors receive ordinary ``status=error`` replies.
    The return value is the number of responses written.
    """

    count = 0
    while True:
        try:
            request = read_request(input_stream)
        except EOFError:
            return count
        response = service.handle(request)
        write_response(output_stream, response)
        flush = getattr(output_stream, "flush", None)
        if callable(flush):
            flush()
        count += 1
        if stop_on_close and response.status is ResponseStatus.CLOSED:
            return count


__all__ = ["SessionState", "TrackerBackend", "TrackingService", "serve"]
