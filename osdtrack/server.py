"""Launch the validated OSDTrack backend behind the simulator wire protocol."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
from pathlib import Path
import socket
import sys
from typing import Sequence

from .config import load_config
from .runtime import build_tracker
from .service import SessionState, TrackingService, serve


def _log(event: str, **fields: object) -> None:
    message = {"event": event, **fields}
    print(json.dumps(message, ensure_ascii=False, allow_nan=False), file=sys.stderr, flush=True)


def _close_backend(service: TrackingService) -> None:
    if service.state is SessionState.CLOSED:
        return
    close = getattr(service.backend, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:  # best-effort cleanup after a broken connection
            _log("backend_close_failed", error=type(exc).__name__, message=str(exc)[:1024])


def serve_stdio(service: TrackingService) -> int:
    """Serve one binary session on stdin/stdout without text-mode translation."""

    _log("ready", transport="stdio", protocol="OSD1", version=1)
    try:
        return serve(service, sys.stdin.buffer, sys.stdout.buffer)
    finally:
        _close_backend(service)


def _is_loopback(host: str) -> bool:
    try:
        addresses = socket.getaddrinfo(host, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    return bool(addresses) and all(ipaddress.ip_address(item[4][0]).is_loopback for item in addresses)


def _validate_tcp_options(host: str, port: int, allow_remote: bool, io_timeout: float) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    if not math.isfinite(io_timeout) or io_timeout <= 0.0:
        raise ValueError("io_timeout must be a finite positive number of seconds")
    if not allow_remote and not _is_loopback(host):
        raise ValueError("refusing a non-loopback bind without --allow-remote")


def serve_tcp(
    service: TrackingService,
    host: str,
    port: int,
    *,
    allow_remote: bool = False,
    io_timeout: float = 30.0,
) -> int:
    """Serve one TCP client and then exit.

    The wire protocol is intentionally unauthenticated and unencrypted, so a
    non-loopback bind requires an explicit opt-in.
    """

    listener: socket.socket | None = None
    try:
        _validate_tcp_options(host, port, allow_remote, io_timeout)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(1)
        bound_host, bound_port = listener.getsockname()[:2]
        _log(
            "ready",
            transport="tcp",
            host=bound_host,
            port=bound_port,
            protocol="OSD1",
            version=1,
            remote_access=bool(allow_remote),
        )
        connection, address = listener.accept()
        with connection:
            connection.settimeout(io_timeout)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            _log("client_connected", address=address[0], port=address[1])
            with connection.makefile("rb", buffering=0) as input_stream:
                with connection.makefile("wb", buffering=0) as output_stream:
                    return serve(service, input_stream, output_stream)
    finally:
        if listener is not None:
            listener.close()
        _close_backend(service)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("tcp", "stdio"), default="tcp")
    parser.add_argument("--host", default="127.0.0.1", help="TCP bind address")
    parser.add_argument("--port", type=int, default=8765, help="TCP port")
    parser.add_argument(
        "--io-timeout",
        type=float,
        default=30.0,
        help="maximum seconds for each client socket read/write operation",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="allow an unauthenticated, unencrypted non-loopback TCP bind",
    )
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--yolo-python", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.transport == "stdio" and args.allow_remote:
        raise SystemExit("--allow-remote is valid only with --transport tcp")
    if args.transport == "tcp":
        # Validate the endpoint before allocating CUDA models or launching YOLO.
        _validate_tcp_options(args.host, args.port, args.allow_remote, args.io_timeout)
    config = load_config(
        weights_dir=args.weights_dir,
        output_dir=args.output_dir,
        yolo_python=args.yolo_python,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tracker = build_tracker(config)
    service = TrackingService(tracker)
    if args.transport == "stdio":
        count = serve_stdio(service)
    else:
        count = serve_tcp(
            service,
            args.host,
            args.port,
            allow_remote=args.allow_remote,
            io_timeout=args.io_timeout,
        )
    _log("stopped", responses=count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "serve_stdio", "serve_tcp"]
