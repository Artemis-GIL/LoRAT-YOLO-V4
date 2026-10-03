"""Real-model adapter and single-frame execution for framework V3.

The frozen R2 adapter is reused, but all inference runs on a temporary search
provider. Only a controller commit can move the authoritative search anchor.
"""
from pathlib import Path
import hashlib
import math
import time
import numpy as np

if __package__:
    from .recovery_controller import Config, RecoveryController, box_array, clip_box
else:
    from recovery_controller import Config, RecoveryController, box_array, clip_box


class TransactionalLoRAT:
    def __init__(self, audit_path):
        if __package__:
            from .lorat_yolo import LoRATCore
        else:
            from lorat_yolo import LoRATCore
        self.native = LoRATCore(audit_path)

    @property
    def anchor(self):
        a = self.native.adapter.provider.cached_bbox.copy()
        a[2:] -= a[:2]
        return a.tolist()

    def initialize(self, rgb, bbox):
        self.native.initialize(rgb, bbox)

    def propose(self, rgb, anchor=None, *, purpose='local'):
        adapter = self.native.adapter
        original = adapter.provider
        before = original.cached_bbox.copy()
        temporary = adapter.Provider(original.area_factor, tuple(original.min_object_size))
        if anchor is None:
            box = before.copy()
        else:
            box = np.asarray(clip_box(anchor, rgb.shape)); box[2:] += box[:2]
        temporary.initialize(box)
        adapter.provider = temporary
        try:
            return self.native.track(rgb)
        finally:
            adapter.provider = original
            if not np.array_equal(original.cached_bbox, before):
                original.cached_bbox[:] = before
                raise RuntimeError('Read-only proposal modified the authoritative search anchor')

    def commit(self, bbox, shape):
        self.native.commit(bbox, shape)


def validate_image(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError('Expected a nonempty uint8 RGB frame')


def checked_proposal(proposal, shape):
    """Represent unusable numeric outputs explicitly, keeping JSON finite."""
    try:
        box = clip_box(proposal['box'], shape)
        score = float(proposal['score'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Nonfinite or out-of-range model score')
        roi = np.asarray(proposal['roi'], float)
        if roi.shape != (4,) or not np.isfinite(roi).all():
            raise ValueError('Invalid ROI')
        return dict(box=box, score=score, roi=roi.tolist(), valid=True)
    except (ValueError, TypeError, KeyError):
        return dict(box=None, score=None, roi=None, valid=False)


def finite_trace(value):
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite_trace(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite_trace(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    return value


class CausalTrackerV3:
    """Dependency-injected core makes the complete control path testable on CPU."""
    def __init__(self, core, detector, config=None):
        self.core = core
        self.detector = detector
        self.config = config or Config()
        self.controller = RecoveryController(self.config)
        self.frame = 0
        self.failed = False
        self.image_shape = None
        self.tentative_anchor = None

    def initialize(self, rgb, bbox):
        validate_image(rgb)
        # An initialization partly outside the image is a caller error.
        box = box_array(bbox).tolist()
        clipped = clip_box(box, rgb.shape)
        if not np.allclose(box, clipped, rtol=0, atol=1e-8):
            raise ValueError('The initialization box must lie inside the image')
        self.core.initialize(rgb, box)
        self.controller.reset(); self.frame = 1; self.failed = False
        self.tentative_anchor = None
        self.image_shape = rgb.shape
        return dict(frame=1, previous_state=None, state='TRACKING', reason='INITIALIZED',
            bbox=box, output_source='initialization', proposed_commit=box,
            committed_bbox=self.core.anchor, state_committed=True,
            localization_status='confirmed', physical_presence='unknown', physical_absence_inferred=False,
            rgb_sha256=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
            probe_count=0, detector_request_id=None, detector_seconds=0., online_seconds=0.)

    def track(self, rgb):
        validate_image(rgb)
        if self.frame == 0 or self.failed:
            raise RuntimeError('Initialize before tracking; reinitialize after an inference error')
        if rgb.shape != self.image_shape:
            raise ValueError('Frame dimensions changed; reinitialize the tracker')
        start = time.perf_counter(); before = self.core.anchor
        local_anchor = self.tentative_anchor
        try:
            local = checked_proposal(self.core.propose(rgb, local_anchor, purpose='local'), rgb.shape)
            if not np.array_equal(self.core.anchor, before):
                raise RuntimeError('Local proposal changed the authoritative state')
            detection = self.detector.infer(rgb)
            digest = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
            if detection.get('rgb_sha256') != digest:
                raise RuntimeError('Detector response does not belong to the current RGB frame')

            def probe(image, anchor):
                proposal = checked_proposal(self.core.propose(image, anchor, purpose='candidate'), image.shape)
                if not np.array_equal(self.core.anchor, before):
                    raise RuntimeError('Candidate probe changed the authoritative state')
                return proposal

            result = self.controller.decide(rgb, local, detection['candidates'], self.frame + 1, probe)
            # Output choice and state write are independent fields. No rejected
            # or merely tentative box can reach commit().
            anchor = result['proposed_commit']
            if anchor is not None:
                self.core.commit(anchor, rgb.shape)
                self.tentative_anchor = None
            elif not np.array_equal(self.core.anchor, before):
                raise RuntimeError('Unconfirmed frame changed the search anchor')
            elif local['valid']:
                # Continue a private local hypothesis through uncertainty. It
                # never becomes the authoritative state without confirmation.
                self.tentative_anchor = list(local['box'])
            self.frame += 1
            result.update(local_proposal=local, search_anchor_before=before,
                local_search_anchor=local_anchor if local_anchor is not None else before,
                tentative_local_anchor=self.tentative_anchor,
                committed_bbox=self.core.anchor, state_committed=anchor is not None,
                rgb_sha256=digest, detector_request_id=detection.get('request_id'),
                detector_seconds=float(detection['seconds']), raw_candidates=finite_trace(detection['candidates']),
                online_seconds=time.perf_counter() - start)
            return result
        except Exception:
            self.failed = True
            raise

    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def build_tracker(audit_path, config=None):
    """Load each model once. The detector receives current RGB bytes only."""
    import torch
    if __package__:
        from .online_tracker import CurrentFrameDetector
        from .test420_pipeline import setup_torch
    else:
        from online_tracker import CurrentFrameDetector
        from test420_pipeline import setup_torch
    setup_torch()
    core = TransactionalLoRAT(Path(audit_path))
    try:
        detector = CurrentFrameDetector()
    except Exception:
        del core
        torch.cuda.empty_cache()
        raise
    return CausalTrackerV3(core, detector, config)
