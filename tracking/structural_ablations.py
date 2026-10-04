"""Fixed structural ablations; all evaluated V3/V4 source files stay frozen.

The V3 variant changes output only. The V4 factorial changes reverse checking
and the ability of competing candidates to veto a healthy TRACKING local
proposal. Candidate inference and recovery competition remain enabled.
"""
from pathlib import Path

from .causal_tracker_v3 import CausalTrackerV3, TransactionalLoRAT
from .causal_tracker_v4 import CausalTrackerV4, ReferenceCycleLoRAT
from .recovery_controller import Config, Mode, RecoveryController, proposal_valid
from .recovery_controller_v4 import RecoveryControllerV4
from .reliability_v4 import usable


VARIANTS = {
    'v3_output_only': dict(family='v3', reverse=False, tracking_competition=None),
    'v4_full': dict(family='v4', reverse=True, tracking_competition=True),
    'v4_no_reverse': dict(family='v4', reverse=False, tracking_competition=True),
    'v4_no_tracking_veto': dict(family='v4', reverse=True, tracking_competition=False),
    'v4_no_reverse_no_tracking_veto': dict(family='v4', reverse=False, tracking_competition=False),
}


class OutputOnlyControllerV3(RecoveryController):
    def decide(self, image, local, detections, frame, probe):
        result = super().decide(image, local, detections, frame, probe)
        result.update(v3_original_bbox=result['bbox'], v3_original_output_source=result['output_source'],
                      v3_original_localization_status=result['localization_status'],
                      commit_source=result['output_source'], reference_checks=0,
                      output_confirmed=result['proposed_commit'] is not None, identity_guaranteed=False)
        if result['bbox'] is None and usable(local):
            result.update(bbox=list(local['box']), output_source='local_tentative', localization_status='tentative')
        # No candidate fallback, changed health, mode, streak, or commit rule.
        return result


class OutputOnlyTrackerV3(CausalTrackerV3):
    def __init__(self, core, detector, config=None):
        super().__init__(core, detector, config)
        self.controller = OutputOnlyControllerV3(self.config)

    def initialize(self, rgb, bbox):
        result = super().initialize(rgb, bbox)
        result.update(v3_original_bbox=result['bbox'], v3_original_output_source='initialization',
                      v3_original_localization_status='confirmed', commit_source='initialization',
                      reference_checks=0, output_confirmed=True, identity_guaranteed=False)
        return result


class ForwardOnlyVerifier:
    def __init__(self, config):
        self.cfg = config
        self.reset_frame()

    def reset_frame(self):
        self.calls = 0

    def assess(self, image, proposal):
        if not usable(proposal):
            return dict(passed=False, reason='INVALID_PROPOSAL', reference=None)
        passed = proposal_valid(proposal, self.cfg)
        return dict(passed=passed,
                    reason='FORWARD_ONLY_ACCEPTED' if passed else 'FORWARD_RESPONSE_UNRELIABLE', reference=None)


class AblationControllerV4(RecoveryControllerV4):
    def __init__(self, config, reference_check, *, reverse=True, tracking_competition=True):
        super().__init__(config, reference_check)
        self.reverse_enabled = reverse
        self.tracking_competition_enabled = tracking_competition
        if not reverse:
            self.verifier = ForwardOnlyVerifier(config)

    def decide(self, image, local, detections, frame, probe):
        result = super().decide(image, local, detections, frame, probe)
        if not self.reverse_enabled and result['reason'] == 'LOCAL_REFERENCE_CONFIRMED':
            result['reason'] = 'LOCAL_FORWARD_CONFIRMED'
        # Keep candidate probes, verification, association and streak updates.
        # Only remove the edge "verified rival vetoes currently trusted local".
        bypass = (not self.tracking_competition_enabled
                  and result['previous_state'] == Mode.TRACKING.value
                  and result['local_health']['accepted']
                  and result['proposed_commit'] is None)
        if bypass:
            result['competition_shadow_decision'] = dict(state=result['state'], reason=result['reason'],
                                                        proposed_commit=result['proposed_commit'])
            self.mode = Mode.TRACKING
            result.update(state=Mode.TRACKING.value, reason='TRACKING_RIVAL_VETO_DISABLED',
                          bbox=list(local['box']), output_source='local', proposed_commit=list(local['box']),
                          commit_source='local', localization_status='confirmed', output_confirmed=True)
        result.update(reverse_enabled=self.reverse_enabled,
                      tracking_competition_enabled=self.tracking_competition_enabled,
                      tracking_competition_bypassed=bypass)
        return result


class AblationTrackerV4(CausalTrackerV4):
    def __init__(self, core, detector, config=None, *, reverse=True, tracking_competition=True):
        super().__init__(core, detector, config)
        self.controller = AblationControllerV4(self.config, self._verify_read_only,
                                             reverse=reverse, tracking_competition=tracking_competition)


def build_tracker(variant, audit_path, config=None):
    from .online_tracker import CurrentFrameDetector
    from .test420_pipeline import setup_torch
    import torch
    if variant not in VARIANTS:
        raise ValueError(f'Unknown structural variant: {variant}')
    setup_torch(); cfg=config or Config(); spec=VARIANTS[variant]
    core = (TransactionalLoRAT if spec['family']=='v3' else ReferenceCycleLoRAT)(Path(audit_path))
    try:
        detector = CurrentFrameDetector()
    except Exception:
        del core; torch.cuda.empty_cache(); raise
    if spec['family']=='v3':return OutputOnlyTrackerV3(core, detector, cfg)
    # The full reference is exactly the original, evaluated V4 class.
    if variant=='v4_full':return CausalTrackerV4(core, detector, cfg)
    return AblationTrackerV4(core, detector, cfg, reverse=spec['reverse'],
                            tracking_competition=spec['tracking_competition'])
