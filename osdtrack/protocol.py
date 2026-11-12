"""Simulator-neutral wire protocol for the native OSDTrack service.

The wire format is deliberately small and deterministic::

    +-------------------- 16-byte network-order prefix --------------------+
    | magic | version | kind | flags | JSON length | binary payload length |
    +-----------------------------------------------------------------------+
    | UTF-8 JSON metadata | packed RGB uint8 HWC bytes (requests only)       |
    +-----------------------------------------------------------------------+

JSON is used only for bounded metadata.  Image pixels are never base64 encoded.
Every image carries an exact shape and SHA-256 digest, so truncated, oversized,
or corrupted frames are rejected before reaching the tracker.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
import hashlib
import hmac
import io
import json
import math
import numbers
import re
import struct
from typing import Any, BinaryIO, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


MAGIC = b"OSD1"
VERSION = 1
PREFIX = struct.Struct("!4sBBHII")
PREFIX_SIZE = PREFIX.size

MAX_HEADER_BYTES = 64 * 1024
MAX_IMAGE_BYTES = 256 * 1024 * 1024
MAX_IMAGE_DIMENSION = 16_384
MAX_SESSION_ID_BYTES = 128


class ProtocolError(ValueError):
    """A malformed or unsupported wire message."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class MessageKind(IntEnum):
    REQUEST = 1
    RESPONSE = 2


class Operation(str, Enum):
    RESET = "reset"
    INIT = "init"
    TRACK = "track"
    CLOSE = "close"


class ResponseStatus(str, Enum):
    RESET = "reset"
    INITIALIZED = "initialized"
    TRACKING = "tracking"
    CLOSED = "closed"
    ERROR = "error"


class TrackerStatus(str, Enum):
    """Algorithm-level localization state for a successful track response."""

    TRACKING = "tracking"
    TENTATIVE = "tentative"
    LOST = "lost"


BBox = Tuple[float, float, float, float]
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]*$")
_ERROR_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _fail(code: str, message: str) -> None:
    raise ProtocolError(code, message)


def _operation(value: Any) -> Operation:
    if not isinstance(value, str):
        _fail("invalid_operation", "op must be a string")
    try:
        return Operation(value)
    except ValueError:
        _fail("invalid_operation", f"unsupported op: {value!r}")


def _status(value: Any) -> ResponseStatus:
    if not isinstance(value, str):
        _fail("invalid_status", "status must be a string")
    try:
        return ResponseStatus(value)
    except ValueError:
        _fail("invalid_status", f"unsupported status: {value!r}")


def _tracker_status(value: Any) -> TrackerStatus:
    if not isinstance(value, str):
        _fail("invalid_tracker_status", "tracker_status must be a string")
    try:
        return TrackerStatus(value)
    except ValueError:
        _fail("invalid_tracker_status", f"unsupported tracker_status: {value!r}")


def _session_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        _fail("invalid_session_id", "session_id must be a non-empty string")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        _fail("invalid_session_id", "session_id must contain ASCII characters only")
    if len(encoded) > MAX_SESSION_ID_BYTES or not _SESSION_RE.fullmatch(value):
        _fail(
            "invalid_session_id",
            "session_id must be at most 128 bytes and use letters, digits, '.', '_', ':', '@', or '-'",
        )
    return value


def _frame_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        _fail("invalid_frame_id", "frame_id must be an integer")
    result = int(value)
    if result < 0 or result > (1 << 63) - 1:
        _fail("invalid_frame_id", "frame_id must be in [0, 2^63-1]")
    return result


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        _fail(f"invalid_{field}", f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _fail(f"invalid_{field}", f"{field} must be finite and non-negative")
    return result


def validate_bbox(value: Any, field: str = "bbox") -> BBox:
    """Validate and normalize an ``xywh`` box."""

    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(f"invalid_{field}", f"{field} must be an xywh sequence")
    if len(value) != 4:
        _fail(f"invalid_{field}", f"{field} must contain exactly four values")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, numbers.Real):
            _fail(f"invalid_{field}", f"{field} values must be numbers")
        number = float(item)
        if not math.isfinite(number):
            _fail(f"invalid_{field}", f"{field} values must be finite")
        result.append(number)
    if result[2] <= 0.0 or result[3] <= 0.0:
        _fail(f"invalid_{field}", f"{field} width and height must be positive")
    return tuple(result)  # type: ignore[return-value]


def validate_rgb(image: Any) -> np.ndarray:
    """Validate an RGB uint8 HWC array without silently converting its dtype."""

    if not isinstance(image, np.ndarray):
        _fail("invalid_image", "image must be a numpy.ndarray")
    if image.dtype != np.uint8:
        _fail("invalid_image", "image dtype must be uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        _fail("invalid_image", "image shape must be HxWx3 (RGB HWC)")
    height, width, channels = (int(part) for part in image.shape)
    if height <= 0 or width <= 0:
        _fail("invalid_image", "image height and width must be positive")
    if height > MAX_IMAGE_DIMENSION or width > MAX_IMAGE_DIMENSION:
        _fail("invalid_image", f"image dimensions must not exceed {MAX_IMAGE_DIMENSION}")
    expected = height * width * channels
    if expected > MAX_IMAGE_BYTES:
        _fail("image_too_large", f"image payload exceeds {MAX_IMAGE_BYTES} bytes")
    return image


@dataclass(frozen=True)
class Request:
    op: Union[Operation, str]
    session_id: str
    frame_id: Optional[int] = None
    timestamp: Optional[float] = None
    bbox: Optional[Sequence[float]] = None
    image: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        op = _operation(self.op)
        session_id = _session_id(self.session_id)
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "session_id", session_id)

        if op in (Operation.RESET, Operation.CLOSE):
            if any(value is not None for value in (self.frame_id, self.timestamp, self.bbox, self.image)):
                _fail("unexpected_field", f"{op.value} must not contain frame metadata or pixels")
            return

        if self.frame_id is None or self.timestamp is None or self.image is None:
            _fail("missing_field", f"{op.value} requires frame_id, timestamp, and image")
        object.__setattr__(self, "frame_id", _frame_id(self.frame_id))
        object.__setattr__(self, "timestamp", _finite_nonnegative(self.timestamp, "timestamp"))
        object.__setattr__(self, "image", validate_rgb(self.image))

        if op is Operation.INIT:
            if self.bbox is None:
                _fail("missing_field", "init requires an initial xywh bbox")
            object.__setattr__(self, "bbox", validate_bbox(self.bbox))
        elif self.bbox is not None:
            _fail("unexpected_field", "track must not contain a bbox")


@dataclass(frozen=True)
class Response:
    op: Union[Operation, str]
    session_id: str
    status: Union[ResponseStatus, str]
    elapsed_ms: float
    frame_id: Optional[int] = None
    timestamp: Optional[float] = None
    bbox: Optional[Sequence[float]] = None
    tracker_status: Optional[Union[TrackerStatus, str]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        op = _operation(self.op)
        status = _status(self.status)
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "session_id", _session_id(self.session_id))
        object.__setattr__(self, "elapsed_ms", _finite_nonnegative(self.elapsed_ms, "elapsed_ms"))

        if self.frame_id is not None:
            object.__setattr__(self, "frame_id", _frame_id(self.frame_id))
        if self.timestamp is not None:
            object.__setattr__(self, "timestamp", _finite_nonnegative(self.timestamp, "timestamp"))
        if self.bbox is not None:
            object.__setattr__(self, "bbox", validate_bbox(self.bbox))
        if self.tracker_status is not None:
            object.__setattr__(self, "tracker_status", _tracker_status(self.tracker_status))

        if status is ResponseStatus.ERROR:
            if self.bbox is not None or self.tracker_status is not None:
                _fail("unexpected_field", "error responses must not contain bbox or tracker_status")
            if op in (Operation.INIT, Operation.TRACK):
                if self.frame_id is None or self.timestamp is None:
                    _fail("missing_field", f"{op.value} error requires frame_id and timestamp")
            elif self.frame_id is not None or self.timestamp is not None:
                _fail("unexpected_field", f"{op.value} error must not contain frame fields")
            if not isinstance(self.error_code, str) or not _ERROR_CODE_RE.fullmatch(self.error_code):
                _fail("invalid_error", "error_code must be a lower-case protocol identifier")
            if not isinstance(self.error_message, str) or not self.error_message or len(self.error_message) > 1024:
                _fail("invalid_error", "error_message must contain 1 to 1024 characters")
            return

        if self.error_code is not None or self.error_message is not None:
            _fail("unexpected_field", "successful responses must not contain error fields")
        expected_status = {
            Operation.RESET: ResponseStatus.RESET,
            Operation.INIT: ResponseStatus.INITIALIZED,
            Operation.TRACK: ResponseStatus.TRACKING,
            Operation.CLOSE: ResponseStatus.CLOSED,
        }[op]
        if status is not expected_status:
            _fail("invalid_status", f"{op.value} success status must be {expected_status.value!r}")
        if op is Operation.INIT:
            if self.frame_id is None or self.timestamp is None or self.bbox is None:
                _fail("missing_field", "init success requires frame_id, timestamp, and bbox")
            if self.tracker_status is not None:
                _fail("unexpected_field", "init success must not contain tracker_status")
        elif op is Operation.TRACK:
            if self.frame_id is None or self.timestamp is None or self.tracker_status is None:
                _fail("missing_field", "track success requires frame_id, timestamp, and tracker_status")
            if self.tracker_status is TrackerStatus.LOST:
                if self.bbox is not None:
                    _fail("inconsistent_tracking_result", "lost track response must use bbox=null")
            elif self.bbox is None:
                _fail("missing_field", "tracking/tentative response requires a bbox")
        elif any(
            value is not None for value in (self.frame_id, self.timestamp, self.bbox, self.tracker_status)
        ):
            _fail("unexpected_field", f"{op.value} success must not contain frame fields")


def _json_pairs(pairs: Sequence[Tuple[str, Any]]) -> Mapping[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key", f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    _fail("invalid_json", f"non-finite JSON number is forbidden: {value}")


def _encode_json(header: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            header,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _fail("invalid_header", f"header cannot be encoded: {exc}")
    if not encoded or len(encoded) > MAX_HEADER_BYTES:
        _fail("header_too_large", f"header must contain 1 to {MAX_HEADER_BYTES} bytes")
    return encoded


def _decode_json(encoded: bytes) -> Mapping[str, Any]:
    try:
        text = encoded.decode("utf-8", errors="strict")
        value = json.loads(text, object_pairs_hook=_json_pairs, parse_constant=_reject_json_constant)
    except ProtocolError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        _fail("invalid_json", f"invalid UTF-8 JSON header: {exc}")
    if not isinstance(value, Mapping):
        _fail("invalid_header", "JSON header must be an object")
    return value


def _require_keys(header: Mapping[str, Any], required: Sequence[str]) -> None:
    expected = set(required)
    actual = set(header)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        _fail("missing_field", f"missing header field(s): {', '.join(sorted(missing))}")
    if unknown:
        _fail("unknown_field", f"unknown header field(s): {', '.join(sorted(unknown))}")


def _pack(kind: MessageKind, header: Mapping[str, Any], payload: bytes = b"") -> bytes:
    encoded_header = _encode_json(header)
    if len(payload) > MAX_IMAGE_BYTES:
        _fail("payload_too_large", f"payload exceeds {MAX_IMAGE_BYTES} bytes")
    prefix = PREFIX.pack(MAGIC, VERSION, int(kind), 0, len(encoded_header), len(payload))
    return prefix + encoded_header + payload


def encode_request(request: Request) -> bytes:
    if not isinstance(request, Request):
        _fail("invalid_request", "encode_request expects a Request")
    header = {"op": request.op.value, "session_id": request.session_id}
    payload = b""
    if request.op in (Operation.INIT, Operation.TRACK):
        assert request.image is not None and request.frame_id is not None and request.timestamp is not None
        image = np.ascontiguousarray(request.image)
        payload = image.tobytes(order="C")
        header.update(
            frame_id=request.frame_id,
            timestamp=request.timestamp,
            shape=[int(part) for part in image.shape],
            encoding="rgb8",
            rgb_sha256=hashlib.sha256(payload).hexdigest(),
        )
        if request.op is Operation.INIT:
            header["bbox"] = list(request.bbox)  # type: ignore[arg-type]
    return _pack(MessageKind.REQUEST, header, payload)


def encode_response(response: Response) -> bytes:
    if not isinstance(response, Response):
        _fail("invalid_response", "encode_response expects a Response")
    header = {
        "op": response.op.value,
        "session_id": response.session_id,
        "status": response.status.value,
        "elapsed_ms": response.elapsed_ms,
    }
    if response.frame_id is not None:
        header["frame_id"] = response.frame_id
    if response.timestamp is not None:
        header["timestamp"] = response.timestamp
    if response.bbox is not None:
        header["bbox"] = list(response.bbox)
    if response.tracker_status is not None:
        header["tracker_status"] = response.tracker_status.value
        # A lost frame has a semantically meaningful explicit null box.  It is
        # not equivalent to a malformed response that omitted the field.
        if response.op is Operation.TRACK and response.tracker_status is TrackerStatus.LOST:
            header["bbox"] = None
    if response.error_code is not None:
        header["error_code"] = response.error_code
        header["error_message"] = response.error_message
    return _pack(MessageKind.RESPONSE, header)


def _read_exact(
    stream: BinaryIO,
    length: int,
    *,
    allow_clean_eof: bool = False,
) -> bytearray:
    """Read into one bounded buffer, accepting streams with or without readinto."""

    buffer = bytearray(length)
    view = memoryview(buffer)
    offset = 0
    readinto = getattr(stream, "readinto", None)
    while offset < length:
        remaining = length - offset
        if callable(readinto):
            count = readinto(view[offset:])
            if count is None:
                _fail("invalid_stream", "binary stream returned None")
            if isinstance(count, bool) or not isinstance(count, numbers.Integral):
                _fail("invalid_stream", "binary stream readinto() must return an integer")
            count = int(count)
            if count < 0 or count > remaining:
                _fail("invalid_stream", "binary stream reported an invalid read length")
        else:
            chunk = stream.read(remaining)
            if chunk is None:
                _fail("invalid_stream", "binary stream returned None")
            if not isinstance(chunk, (bytes, bytearray)):
                _fail("invalid_stream", "binary stream must return bytes")
            count = len(chunk)
            if count > remaining:
                _fail("invalid_stream", "binary stream returned more bytes than requested")
            if count:
                view[offset : offset + count] = chunk
        if count == 0:
            if allow_clean_eof and offset == 0:
                raise EOFError("end of stream")
            _fail("truncated_message", f"stream ended with {remaining} byte(s) still required")
        offset += count
    return buffer


def _prevalidate_payload_length(
    kind: MessageKind,
    header: Mapping[str, Any],
    payload_length: int,
) -> None:
    """Reject impossible payload lengths before reading their potentially large body."""

    if kind is MessageKind.RESPONSE:
        if payload_length:
            _fail("unexpected_payload", "responses must not contain a binary payload")
        return
    if "op" not in header:
        _fail("missing_field", "missing header field: op")
    op = _operation(header["op"])
    if op in (Operation.RESET, Operation.CLOSE):
        if payload_length:
            _fail("unexpected_payload", f"{op.value} must not contain a binary payload")
        return
    if header.get("encoding") != "rgb8":
        _fail("unsupported_encoding", "only packed rgb8 frames are supported")
    shape = header.get("shape")
    if not isinstance(shape, list) or len(shape) != 3:
        _fail("invalid_shape", "shape must be a JSON array [height, width, 3]")
    height = _positive_dimension(shape[0], "height")
    width = _positive_dimension(shape[1], "width")
    if isinstance(shape[2], bool) or not isinstance(shape[2], numbers.Integral) or shape[2] != 3:
        _fail("invalid_shape", "RGB channel dimension must equal 3")
    expected_length = height * width * 3
    if expected_length > MAX_IMAGE_BYTES:
        _fail("image_too_large", f"image payload exceeds {MAX_IMAGE_BYTES} bytes")
    if payload_length != expected_length:
        _fail(
            "payload_size_mismatch",
            f"RGB shape requires {expected_length} bytes, envelope declares {payload_length}",
        )


def _read_envelope(stream: BinaryIO) -> Tuple[MessageKind, Mapping[str, Any], bytes]:
    prefix = _read_exact(stream, PREFIX_SIZE, allow_clean_eof=True)
    magic, version, raw_kind, flags, header_length, payload_length = PREFIX.unpack(prefix)
    if magic != MAGIC:
        _fail("bad_magic", "message magic does not match OSDTrack protocol")
    if version != VERSION:
        _fail("unsupported_version", f"protocol version {version} is not supported")
    try:
        kind = MessageKind(raw_kind)
    except ValueError:
        _fail("invalid_kind", f"unknown message kind: {raw_kind}")
    if flags != 0:
        _fail("unsupported_flags", "reserved message flags must be zero")
    if header_length < 2 or header_length > MAX_HEADER_BYTES:
        _fail("invalid_header_length", f"header length must be in [2, {MAX_HEADER_BYTES}]")
    if payload_length > MAX_IMAGE_BYTES:
        _fail("payload_too_large", f"payload exceeds {MAX_IMAGE_BYTES} bytes")
    header = _decode_json(_read_exact(stream, header_length))
    _prevalidate_payload_length(kind, header, payload_length)
    payload = _read_exact(stream, payload_length) if payload_length else b""
    return kind, header, payload


def _positive_dimension(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        _fail("invalid_shape", f"{name} must be an integer")
    result = int(value)
    if result <= 0 or result > MAX_IMAGE_DIMENSION:
        _fail("invalid_shape", f"{name} must be in [1, {MAX_IMAGE_DIMENSION}]")
    return result


def _decode_request(header: Mapping[str, Any], payload: bytes) -> Request:
    if "op" not in header:
        _fail("missing_field", "missing header field: op")
    op = _operation(header["op"])
    if op in (Operation.RESET, Operation.CLOSE):
        _require_keys(header, ("op", "session_id"))
        if payload:
            _fail("unexpected_payload", f"{op.value} must not contain a binary payload")
        return Request(op=op, session_id=header["session_id"])

    required = [
        "op",
        "session_id",
        "frame_id",
        "timestamp",
        "shape",
        "encoding",
        "rgb_sha256",
    ]
    if op is Operation.INIT:
        required.append("bbox")
    _require_keys(header, required)
    if header["encoding"] != "rgb8":
        _fail("unsupported_encoding", "only packed rgb8 frames are supported")
    shape = header["shape"]
    if not isinstance(shape, list) or len(shape) != 3:
        _fail("invalid_shape", "shape must be a JSON array [height, width, 3]")
    height = _positive_dimension(shape[0], "height")
    width = _positive_dimension(shape[1], "width")
    if isinstance(shape[2], bool) or not isinstance(shape[2], numbers.Integral) or shape[2] != 3:
        _fail("invalid_shape", "RGB channel dimension must equal 3")
    expected_length = height * width * 3
    if expected_length > MAX_IMAGE_BYTES:
        _fail("image_too_large", f"image payload exceeds {MAX_IMAGE_BYTES} bytes")
    if len(payload) != expected_length:
        _fail(
            "payload_size_mismatch",
            f"RGB shape requires {expected_length} bytes, received {len(payload)}",
        )
    digest = header["rgb_sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        _fail("invalid_digest", "rgb_sha256 must be 64 lower-case hexadecimal characters")
    actual_digest = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(digest, actual_digest):
        _fail("digest_mismatch", "RGB payload SHA-256 does not match the header")
    image = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()
    return Request(
        op=op,
        session_id=header["session_id"],
        frame_id=header["frame_id"],
        timestamp=header["timestamp"],
        bbox=header.get("bbox"),
        image=image,
    )


def _decode_response(header: Mapping[str, Any], payload: bytes) -> Response:
    if payload:
        _fail("unexpected_payload", "responses must not contain a binary payload")
    base = {"op", "session_id", "status", "elapsed_ms"}
    if not base.issubset(header):
        missing = base - set(header)
        _fail("missing_field", f"missing header field(s): {', '.join(sorted(missing))}")
    status = _status(header["status"])
    op = _operation(header["op"])
    allowed = set(base)
    if status is ResponseStatus.ERROR:
        allowed.update(("error_code", "error_message"))
        required = base | {"error_code", "error_message"}
        if op in (Operation.INIT, Operation.TRACK):
            allowed.update(("frame_id", "timestamp"))
            required |= {"frame_id", "timestamp"}
    else:
        required = base
        if op is Operation.INIT:
            allowed.update(("frame_id", "timestamp", "bbox"))
            required = base | {"frame_id", "timestamp", "bbox"}
        elif op is Operation.TRACK:
            allowed.update(("frame_id", "timestamp", "bbox", "tracker_status"))
            required = base | {"frame_id", "timestamp", "bbox", "tracker_status"}
    missing = required - set(header)
    unknown = set(header) - allowed
    if missing:
        _fail("missing_field", f"missing header field(s): {', '.join(sorted(missing))}")
    if unknown:
        _fail("unknown_field", f"unknown header field(s): {', '.join(sorted(unknown))}")
    return Response(
        op=header["op"],
        session_id=header["session_id"],
        status=status,
        elapsed_ms=header["elapsed_ms"],
        frame_id=header.get("frame_id"),
        timestamp=header.get("timestamp"),
        bbox=header.get("bbox"),
        tracker_status=header.get("tracker_status"),
        error_code=header.get("error_code"),
        error_message=header.get("error_message"),
    )


def read_request(stream: BinaryIO) -> Request:
    kind, header, payload = _read_envelope(stream)
    if kind is not MessageKind.REQUEST:
        _fail("unexpected_kind", "expected a request message")
    return _decode_request(header, payload)


def read_response(stream: BinaryIO) -> Response:
    kind, header, payload = _read_envelope(stream)
    if kind is not MessageKind.RESPONSE:
        _fail("unexpected_kind", "expected a response message")
    return _decode_response(header, payload)


def decode_request(data: bytes) -> Request:
    stream = io.BytesIO(data)
    request = read_request(stream)
    if stream.read(1):
        _fail("trailing_bytes", "request contains bytes after its framed message")
    return request


def decode_response(data: bytes) -> Response:
    stream = io.BytesIO(data)
    response = read_response(stream)
    if stream.read(1):
        _fail("trailing_bytes", "response contains bytes after its framed message")
    return response


def _write_all(stream: BinaryIO, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = stream.write(view)
        if written is None:
            _fail("invalid_stream", "binary stream returned None while writing")
        if isinstance(written, bool) or not isinstance(written, numbers.Integral) or written <= 0:
            _fail("invalid_stream", "binary stream failed to make progress while writing")
        if written > len(view):
            _fail("invalid_stream", "binary stream reported writing too many bytes")
        view = view[int(written) :]


def write_request(stream: BinaryIO, request: Request) -> None:
    _write_all(stream, encode_request(request))


def write_response(stream: BinaryIO, response: Response) -> None:
    _write_all(stream, encode_response(response))


__all__ = [
    "BBox",
    "MAGIC",
    "MAX_HEADER_BYTES",
    "MAX_IMAGE_BYTES",
    "MessageKind",
    "Operation",
    "PREFIX",
    "PREFIX_SIZE",
    "ProtocolError",
    "Request",
    "Response",
    "ResponseStatus",
    "TrackerStatus",
    "VERSION",
    "decode_request",
    "decode_response",
    "encode_request",
    "encode_response",
    "read_request",
    "read_response",
    "validate_bbox",
    "validate_rgb",
    "write_request",
    "write_response",
]
