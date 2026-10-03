"""Current RGB frame -> detection + local tracking + committed output.

The detector stays in its validated Python environment. Synchronous IPC carries
only the current RGB array; the worker receives no paths or future frames.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

import numpy as np
import torch

if __package__:
    from .lorat_yolo import LoRATCore
else:
    from lorat_yolo import LoRATCore

sys.path.insert(0, '/home/xu/zyx/send/LORAT_YOLO_GEOMETRY_R2_20261002/scripts')
from policy import match

YOLO_PYTHON = '/home/xu/miniconda3/envs/yolo26_rtx5090/bin/python'


class CurrentFrameDetector:
    def __init__(self):
        self.process = subprocess.Popen(
            [YOLO_PYTHON, '-u', str(Path(__file__).with_name('yolo_online_worker.py'))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
        self.request_id = 0
        try:
            self.environment = self._receive()
            if self.environment.get('status') != 'ready':
                raise RuntimeError(f'Detector startup failed: {self.environment}')
        except BaseException:
            self.close()
            raise

    def _receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'YOLO worker exited without a response (exit={self.process.poll()}); see console log')
        return json.loads(line)

    def infer(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Expected uint8 RGB image')
        self.request_id += 1
        payload = np.ascontiguousarray(rgb).tobytes()
        request = dict(op='detect', shape=list(rgb.shape), request_id=self.request_id)
        self.process.stdin.write((json.dumps(request)+'\n').encode())
        self.process.stdin.write(payload)
        self.process.stdin.flush()
        result = self._receive()
        if result.get('status') != 'ok' or result['request_id'] != self.request_id:
            raise RuntimeError('Detector response does not match current frame')
        if result['rgb_sha256'] != hashlib.sha256(payload).hexdigest():
            raise RuntimeError('Detector received different RGB pixels')
        return result

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write(b'{"op":"close"}\n')
                self.process.stdin.flush()
                self.process.stdin.close()
                self.process.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait()
        self.process.stdout.close()


class OnlineLoRATYOLO:
    def __init__(self, audit_path):
        self.core = LoRATCore(audit_path)
        self.detector = CurrentFrameDetector()
        self.frame = 0

    def initialize(self, image_rgb, initial_box):
        self.core.initialize(image_rgb, initial_box)
        self.frame = 1

    def track(self, image_rgb):
        if self.frame < 1:
            raise RuntimeError('Initialize the first frame before tracking')
        start = time.perf_counter()
        torch.cuda.synchronize(); local_start = time.perf_counter()
        local = self.core.track(image_rgb)
        torch.cuda.synchronize(); local_seconds = time.perf_counter()-local_start
        # Exactly one synchronous detector request for this current frame.
        detection = self.detector.infer(image_rgb)
        box, index, overlap = match(local['box'], detection['candidates'])
        if index is not None:
            self.core.commit(box, image_rgb.shape)
        self.frame += 1
        return dict(frame=self.frame, bbox=box, local_box=local['box'], score=local['score'], roi=local['roi'],
                    candidates=detection['candidates'], matched_index=index, association_iou=overlap,
                    detector_request_id=detection['request_id'], rgb_sha256=detection['rgb_sha256'],
                    tracker_seconds=local_seconds, detector_seconds=detection['seconds'],
                    online_seconds=time.perf_counter()-start)

    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
