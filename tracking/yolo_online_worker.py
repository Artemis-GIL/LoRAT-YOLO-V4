"""Persistent detector: one RGB request, one response; no dataset/file access."""
import contextlib
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np


def send(message):
    sys.stdout.write(json.dumps(message, allow_nan=False) + '\n')
    sys.stdout.flush()


def exact_read(stream, count):
    parts = []
    remaining = count
    while remaining:
        part = stream.read(remaining)
        if not part:
            raise EOFError('Incomplete RGB frame')
        parts.append(part)
        remaining -= len(part)
    return b''.join(parts)


def main():
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        import ultralytics
        sys.path.insert(0, '/home/xu/zyx/send/MVLM_V2_GLOBAL_PROPOSAL_BACKBONE_R1_20261001/scripts')
        from yolo26_adapter import YOLO26Detector
        random.seed(20261002); np.random.seed(20261002); torch.manual_seed(20261002)
        torch.set_num_threads(4)
        detector = YOLO26Detector(Path('/home/xu/zyx/send/MVLM_V2_GLOBAL_PROPOSAL_BACKBONE_R1_20261001/training/yolo26_uav/weights/best.pt'), max_det=12)
    send(dict(status='ready', python=sys.version, torch=torch.__version__, ultralytics=ultralytics.__version__))
    stream = sys.stdin.buffer
    while True:
        line = stream.readline()
        if not line:
            break
        request = json.loads(line)
        if request['op'] == 'close':
            break
        if request['op'] != 'detect':
            raise ValueError('Unknown operation')
        shape = request['shape']
        if len(shape) != 3 or shape[2] != 3 or min(shape) <= 0:
            raise ValueError('Expected H,W,3 RGB image')
        count = int(np.prod(shape))
        if count > 256 * 1024 * 1024:
            raise ValueError('Frame exceeds 256 MiB')
        payload = exact_read(stream, count)
        rgb = np.frombuffer(payload, dtype=np.uint8).reshape(shape)
        torch.cuda.synchronize(); start = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            candidates = detector.infer(np.ascontiguousarray(rgb[:, :, ::-1]))
        torch.cuda.synchronize()
        send(dict(status='ok', request_id=request['request_id'], rgb_sha256=hashlib.sha256(payload).hexdigest(),
                  candidates=candidates, seconds=time.perf_counter()-start))


if __name__ == '__main__':
    main()
