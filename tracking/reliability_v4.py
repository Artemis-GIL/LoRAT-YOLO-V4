"""One reference-based reliability test for local and global hypotheses.

This is a consistency test, not an identity probability. A forward response or
an unverified detector rival alone can neither establish nor refute identity.
"""
import math

if __package__:
    from .recovery_controller import proposal_valid, box_array
else:
    from recovery_controller import proposal_valid, box_array


def usable(proposal):
    try:
        box_array(proposal['box'])
        score = float(proposal['score'])
        return math.isfinite(score) and 0 <= score <= 1 and proposal.get('valid', True)
    except (KeyError, TypeError, ValueError):
        return False


class SharedReferenceVerifier:
    def __init__(self, config, reference_check):
        self.cfg = config
        self.reference_check = reference_check
        self.reset_frame()

    def reset_frame(self):
        self.cache = {}
        self.calls = 0

    def assess(self, image, proposal):
        if not usable(proposal):
            return dict(passed=False, reason='INVALID_PROPOSAL', reference=None)
        if not proposal_valid(proposal, self.cfg):
            return dict(passed=False, reason='FORWARD_RESPONSE_UNRELIABLE', reference=None)
        key = tuple(proposal['box'])
        if key not in self.cache:
            self.cache[key] = self.reference_check(image, list(key))
            self.calls += 1
        reference = self.cache[key]
        try:
            valid = bool(reference.get('valid', False)) and all(
                math.isfinite(float(reference[k])) and 0 <= float(reference[k]) <= 1
                for k in ['score', 'reference_iou'])
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            reason = 'REFERENCE_CHECK_INVALID'
        elif reference['score'] < self.cfg.local_score_min:
            reason = 'REFERENCE_RESPONSE_UNRELIABLE'
        elif reference['reference_iou'] < self.cfg.verification_iou_min:
            reason = 'REFERENCE_TARGET_MISMATCH'
        else:
            reason = 'BIDIRECTIONAL_REFERENCE_CONSISTENT'
        return dict(passed=reason == 'BIDIRECTIONAL_REFERENCE_CONSISTENT', reason=reason, reference=reference)
