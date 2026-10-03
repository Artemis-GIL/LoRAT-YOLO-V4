"""V4 reuses the frozen causal engine with an isolated reverse-reference check."""
from copy import copy
from pathlib import Path
import hashlib
import numpy as np
import torch

if __package__:
    from .causal_tracker_v3 import TransactionalLoRAT, CausalTrackerV3, checked_proposal, validate_image
    from .recovery_controller import Config, clip_box, iou
    from .recovery_controller_v4 import RecoveryControllerV4
else:
    from causal_tracker_v3 import TransactionalLoRAT, CausalTrackerV3, checked_proposal, validate_image
    from recovery_controller import Config, clip_box, iou
    from recovery_controller_v4 import RecoveryControllerV4


class ReferenceCycleLoRAT(TransactionalLoRAT):
    def initialize(self, rgb, bbox):
        super().initialize(rgb, bbox)
        self.reference_rgb = np.ascontiguousarray(rgb).copy()
        self.reference_rgb.flags.writeable = False
        self.reference_bbox = list(bbox)
        self.reference_digest = hashlib.sha256(self.reference_rgb.tobytes()).hexdigest()

    @torch.inference_mode()
    def verify_reference(self, rgb, bbox):
        """Current hypothesis as temporary template -> initial RGB localization.

        This adapter owns fresh z/mask/mean/provider tensors. Only the frozen
        model and immutable crop/postprocess helpers are shared with tracking.
        No rolling reference, template update, or GT after initialization.
        """
        validate_image(rgb)
        box = clip_box(bbox, rgb.shape)
        reference = copy(self.native.adapter)
        try:
            reference.initialize(rgb, box, '')
        except (ValueError, AssertionError) as exc:
            return dict(valid=False, score=None, box=None, reference_iou=0.,
                reason='CANDIDATE_TEMPLATE_INVALID', error=type(exc).__name__)
        anchor = np.asarray(self.reference_bbox, dtype=float).copy(); anchor[2:] += anchor[:2]
        reference.provider.initialize(anchor)
        pred, trace = reference.track(self.reference_rgb.copy())
        result = checked_proposal(dict(box=pred, score=trace['score'], roi=trace['roi']), self.reference_rgb.shape)
        result['reference_iou'] = iou(result['box'], self.reference_bbox) if result['valid'] else 0.
        result['reference_frame'] = 1
        return result


class CausalTrackerV4(CausalTrackerV3):
    def __init__(self, core, detector, config=None):
        super().__init__(core, detector, config or Config())
        self.controller = RecoveryControllerV4(self.config, self._verify_read_only)

    def _verify_read_only(self, rgb, bbox):
        before = self.core.anchor
        try:
            return self.core.verify_reference(rgb, bbox)
        finally:
            if not np.array_equal(self.core.anchor, before):
                raise RuntimeError('Reference verification modified the authoritative anchor')

    def initialize(self, rgb, bbox):
        result = super().initialize(rgb, bbox)
        result.update(commit_source='initialization', output_confirmed=True, reference_checks=0,
            verified_spatial_groups=1, identity_guaranteed=False)
        return result


def build_tracker(audit_path, config=None):
    if __package__:
        from .online_tracker import CurrentFrameDetector
        from .test420_pipeline import setup_torch
    else:
        from online_tracker import CurrentFrameDetector
        from test420_pipeline import setup_torch
    setup_torch()
    core = ReferenceCycleLoRAT(Path(audit_path))
    try:
        detector = CurrentFrameDetector()
    except Exception:
        del core; torch.cuda.empty_cache(); raise
    return CausalTrackerV4(core, detector, config)
