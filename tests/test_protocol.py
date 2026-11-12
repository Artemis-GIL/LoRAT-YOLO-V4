"""Pure-CPU tests for the native wire protocol and session state machine."""

from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pytest

import osdtrack.protocol as protocol_module
from osdtrack.protocol import (
    MAGIC,
    PREFIX,
    VERSION,
    MessageKind,
    Operation,
    ProtocolError,
    Request,
    Response,
    ResponseStatus,
    TrackerStatus,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
    read_response,
)
from osdtrack.service import SessionState, TrackingService, serve
from osdtrack.server import serve_tcp


def rgb(height: int = 4, width: int = 5, offset: int = 0) -> np.ndarray:
    values = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    return values + np.uint8(offset)


def raw_message(kind: MessageKind, header_text: str, payload: bytes = b"") -> bytes:
    encoded = header_text.encode("utf-8")
    return PREFIX.pack(MAGIC, VERSION, int(kind), 0, len(encoded), len(payload)) + encoded + payload


class FakeTracker:
    def __init__(self) -> None:
        self.events = []

    def reset(self) -> None:
        self.events.append(("reset",))

    def initialize(self, image: np.ndarray, bbox):
        self.events.append(("init", image.copy(), tuple(bbox)))
        return None

    def track(self, image: np.ndarray):
        self.events.append(("track", image.copy()))
        index = sum(event[0] == "track" for event in self.events)
        return {"bbox": np.asarray([index, index + 1, 20, 10], dtype=np.float32)}

    def close(self) -> None:
        self.events.append(("close",))


def test_init_round_trip_preserves_noncontiguous_rgb_and_metadata() -> None:
    # Slicing makes a legal HWC view whose rows are not C contiguous.
    frame = rgb(4, 10)[:, ::2, :]
    assert not frame.flags.c_contiguous
    original = Request(
        op="init",
        session_id="sim-run:001",
        frame_id=17,
        timestamp=12.5,
        bbox=[1, 2, 3, 4],
        image=frame,
    )

    decoded = decode_request(encode_request(original))

    assert decoded.op is Operation.INIT
    assert decoded.session_id == "sim-run:001"
    assert decoded.frame_id == 17
    assert decoded.timestamp == 12.5
    assert decoded.bbox == (1.0, 2.0, 3.0, 4.0)
    assert decoded.image is not None and decoded.image.flags.c_contiguous
    np.testing.assert_array_equal(decoded.image, frame)


@pytest.mark.parametrize("op", ["reset", "close"])
def test_control_request_round_trip_has_no_payload(op: str) -> None:
    encoded = encode_request(Request(op=op, session_id="s1"))
    _, _, _, _, _, payload_length = PREFIX.unpack(encoded[: PREFIX.size])
    assert payload_length == 0
    assert decode_request(encoded) == Request(op=op, session_id="s1")


def test_track_rejects_wrong_dtype_shape_and_extra_bbox() -> None:
    with pytest.raises(ProtocolError, match="dtype"):
        Request("track", "s", 1, 1.0, image=np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(ProtocolError, match="HxWx3"):
        Request("track", "s", 1, 1.0, image=np.zeros((2, 2, 4), dtype=np.uint8))
    with pytest.raises(ProtocolError, match="must not contain a bbox"):
        Request("track", "s", 1, 1.0, bbox=[0, 0, 1, 1], image=rgb())


def test_init_rejects_invalid_xywh_and_nonfinite_timestamp() -> None:
    with pytest.raises(ProtocolError, match="width and height"):
        Request("init", "s", 0, 0.0, bbox=[1, 2, 0, 4], image=rgb())
    with pytest.raises(ProtocolError, match="finite"):
        Request("init", "s", 0, float("nan"), bbox=[1, 2, 3, 4], image=rgb())


def test_corrupted_rgb_payload_is_rejected_by_digest() -> None:
    wire = bytearray(encode_request(Request("track", "s", 1, 1.0, image=rgb())))
    wire[-1] ^= 0xFF
    with pytest.raises(ProtocolError) as caught:
        decode_request(bytes(wire))
    assert caught.value.code == "digest_mismatch"


def test_shape_payload_size_mismatch_is_rejected() -> None:
    payload = b"\x00" * 5
    header = {
        "op": "track",
        "session_id": "s",
        "frame_id": 1,
        "timestamp": 1.0,
        "shape": [1, 2, 3],
        "encoding": "rgb8",
        "rgb_sha256": hashlib.sha256(payload).hexdigest(),
    }
    wire = raw_message(MessageKind.REQUEST, json.dumps(header), payload)
    with pytest.raises(ProtocolError) as caught:
        decode_request(wire)
    assert caught.value.code == "payload_size_mismatch"


def test_bad_magic_truncation_and_trailing_data_are_rejected() -> None:
    valid = encode_request(Request("reset", "s"))
    with pytest.raises(ProtocolError) as bad_magic:
        decode_request(b"NOPE" + valid[4:])
    assert bad_magic.value.code == "bad_magic"

    with pytest.raises(ProtocolError) as truncated:
        decode_request(valid[:-1])
    assert truncated.value.code == "truncated_message"

    with pytest.raises(ProtocolError) as trailing:
        decode_request(valid + b"x")
    assert trailing.value.code == "trailing_bytes"


def test_duplicate_and_unknown_json_keys_are_rejected() -> None:
    duplicate = raw_message(
        MessageKind.REQUEST,
        '{"op":"reset","op":"close","session_id":"s"}',
    )
    with pytest.raises(ProtocolError) as caught:
        decode_request(duplicate)
    assert caught.value.code == "duplicate_json_key"

    unknown = raw_message(
        MessageKind.REQUEST,
        '{"op":"reset","session_id":"s","surprise":true}',
    )
    with pytest.raises(ProtocolError) as caught:
        decode_request(unknown)
    assert caught.value.code == "unknown_field"


def test_success_and_error_response_round_trip() -> None:
    success = Response(
        op="track",
        session_id="s",
        status="tracking",
        elapsed_ms=2.25,
        frame_id=9,
        timestamp=0.9,
        bbox=[1, 2, 30, 40],
        tracker_status="tracking",
    )
    assert decode_response(encode_response(success)) == success

    error = Response(
        op="track",
        session_id="s",
        status="error",
        elapsed_ms=0.1,
        frame_id=9,
        timestamp=0.9,
        error_code="out_of_order_frame",
        error_message="expected frame 8",
    )
    assert decode_response(encode_response(error)) == error


def test_service_enforces_full_lifecycle_and_returns_bbox() -> None:
    tracker = FakeTracker()
    service = TrackingService(tracker)

    reset = service.handle(Request("reset", "session-a"))
    init = service.handle(Request("init", "session-a", 100, 10.0, [1, 1, 2, 2], rgb()))
    tracked = service.handle(Request("track", "session-a", 101, 10.1, image=rgb(offset=1)))
    closed = service.handle(Request("close", "session-a"))

    assert reset.status is ResponseStatus.RESET
    assert init.status is ResponseStatus.INITIALIZED and init.bbox == (1.0, 1.0, 2.0, 2.0)
    assert tracked.status is ResponseStatus.TRACKING and tracked.bbox == (1.0, 2.0, 20.0, 10.0)
    assert tracked.tracker_status is TrackerStatus.TRACKING
    assert closed.status is ResponseStatus.CLOSED
    assert all(response.elapsed_ms >= 0 for response in (reset, init, tracked, closed))
    assert service.state is SessionState.CLOSED
    assert [event[0] for event in tracker.events] == ["reset", "init", "track", "close"]


def test_service_rejects_state_session_frame_and_timestamp_order() -> None:
    tracker = FakeTracker()
    service = TrackingService(tracker)
    before_reset = service.handle(Request("track", "s", 1, 1.0, image=rgb()))
    assert before_reset.status is ResponseStatus.ERROR
    assert before_reset.error_code == "invalid_state"

    service.handle(Request("reset", "s"))
    wrong_session = service.handle(Request("init", "other", 7, 7.0, [0, 0, 2, 2], rgb()))
    assert wrong_session.error_code == "session_mismatch"
    service.handle(Request("init", "s", 7, 7.0, [0, 0, 2, 2], rgb()))

    skipped = service.handle(Request("track", "s", 9, 8.0, image=rgb()))
    assert skipped.error_code == "out_of_order_frame"
    same_timestamp = service.handle(Request("track", "s", 8, 7.0, image=rgb()))
    assert same_timestamp.error_code == "out_of_order_timestamp"
    assert [event[0] for event in tracker.events] == ["reset", "init"]

    valid = service.handle(Request("track", "s", 8, 8.0, image=rgb()))
    assert valid.status is ResponseStatus.TRACKING
    assert service.last_frame_id == 8


def test_backend_failure_requires_reset_before_more_frames() -> None:
    class FailOnceTracker(FakeTracker):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        def track(self, image: np.ndarray):
            if not self.failed:
                self.failed = True
                raise RuntimeError("GPU worker unavailable")
            return super().track(image)

    service = TrackingService(FailOnceTracker())
    service.handle(Request("reset", "s"))
    service.handle(Request("init", "s", 0, 0.0, [0, 0, 2, 2], rgb()))
    failed = service.handle(Request("track", "s", 1, 0.1, image=rgb()))
    rejected = service.handle(Request("track", "s", 1, 0.1, image=rgb()))

    assert failed.status is ResponseStatus.ERROR and failed.error_code == "backend_error"
    assert service.state is SessionState.FAILED and service.last_frame_id == 0
    assert rejected.error_code == "reset_required"

    service.handle(Request("reset", "s2"))
    service.handle(Request("init", "s2", 10, 1.0, [0, 0, 2, 2], rgb()))
    recovered = service.handle(Request("track", "s2", 11, 1.1, image=rgb()))
    assert recovered.status is ResponseStatus.TRACKING
    assert service.last_frame_id == 11


def test_service_rejects_initial_bbox_outside_frame_without_calling_backend() -> None:
    tracker = FakeTracker()
    service = TrackingService(tracker)
    service.handle(Request("reset", "s"))

    response = service.handle(Request("init", "s", 0, 0.0, [4, 3, 2, 2], rgb()))

    assert response.status is ResponseStatus.ERROR and response.error_code == "invalid_bbox"
    assert service.state is SessionState.RESET
    assert [event[0] for event in tracker.events] == ["reset"]


def test_backend_mapping_requires_explicit_bbox_and_forces_reset() -> None:
    class EmptyMappingTracker(FakeTracker):
        def track(self, image: np.ndarray):
            self.events.append(("track", image.copy()))
            return {}

    service = TrackingService(EmptyMappingTracker())
    service.handle(Request("reset", "s"))
    service.handle(Request("init", "s", 0, 0.0, [0, 0, 2, 2], rgb()))

    response = service.handle(Request("track", "s", 1, 0.1, image=rgb()))

    assert response.error_code == "invalid_backend_result"
    assert service.state is SessionState.FAILED


def test_lost_track_is_success_with_explicit_null_bbox() -> None:
    class LostTracker(FakeTracker):
        def track(self, image: np.ndarray):
            self.events.append(("track", image.copy()))
            return {
                "bbox": None,
                "state": "SEARCHING",
                "localization_status": "unavailable",
            }

    service = TrackingService(LostTracker())
    service.handle(Request("reset", "s"))
    service.handle(Request("init", "s", 0, 0.0, [0, 0, 2, 2], rgb()))
    lost = service.handle(Request("track", "s", 1, 0.1, image=rgb()))

    assert lost.status is ResponseStatus.TRACKING
    assert lost.tracker_status is TrackerStatus.LOST
    assert lost.bbox is None and lost.error_code is None
    wire = encode_response(lost)
    _, _, _, _, header_length, payload_length = PREFIX.unpack(wire[: PREFIX.size])
    header = json.loads(wire[PREFIX.size : PREFIX.size + header_length])
    assert payload_length == 0 and "bbox" in header and header["bbox"] is None
    assert decode_response(wire) == lost


def test_track_status_and_bbox_must_be_consistent() -> None:
    common = dict(
        op="track",
        session_id="s",
        status="tracking",
        elapsed_ms=0.1,
        frame_id=1,
        timestamp=0.1,
    )
    with pytest.raises(ProtocolError, match="bbox=null"):
        Response(**common, bbox=[0, 0, 2, 2], tracker_status="lost")
    with pytest.raises(ProtocolError, match="requires a bbox"):
        Response(**common, bbox=None, tracker_status="tentative")


def test_reset_can_replace_an_open_session_but_close_is_terminal() -> None:
    tracker = FakeTracker()
    service = TrackingService(tracker)
    service.handle(Request("reset", "first"))
    service.handle(Request("init", "first", 4, 4.0, [0, 0, 2, 2], rgb()))
    replaced = service.handle(Request("reset", "second"))
    assert replaced.status is ResponseStatus.RESET
    assert service.session_id == "second" and service.last_frame_id is None
    service.handle(Request("close", "second"))
    after_close = service.handle(Request("reset", "third"))
    assert after_close.error_code == "session_closed"


def test_serve_processes_concatenated_messages_over_binary_streams() -> None:
    requests = [
        Request("reset", "s"),
        Request("init", "s", 0, 0.0, [0, 0, 2, 2], rgb()),
        Request("track", "s", 1, 0.1, image=rgb(offset=1)),
        Request("close", "s"),
    ]
    input_stream = io.BytesIO(b"".join(encode_request(request) for request in requests))
    output_stream = io.BytesIO()

    assert serve(TrackingService(FakeTracker()), input_stream, output_stream) == 4
    output_stream.seek(0)
    responses = [read_response(output_stream) for _ in requests]
    assert [response.status.value for response in responses] == [
        "reset",
        "initialized",
        "tracking",
        "closed",
    ]
    assert output_stream.read() == b""


def test_json_value_error_is_normalized_to_protocol_error(monkeypatch) -> None:
    def reject(*args, **kwargs):
        raise ValueError("integer conversion limit")

    monkeypatch.setattr(protocol_module.json, "loads", reject)
    with pytest.raises(ProtocolError) as caught:
        decode_request(encode_request(Request("reset", "s")))
    assert caught.value.code == "invalid_json"


def test_invalid_tcp_options_close_backend_before_socket_allocation() -> None:
    tracker = FakeTracker()
    service = TrackingService(tracker)

    with pytest.raises(ValueError, match="non-loopback"):
        serve_tcp(service, "0.0.0.0", 8765)

    assert [event[0] for event in tracker.events] == ["close"]
