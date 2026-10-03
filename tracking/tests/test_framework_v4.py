import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tracking.causal_tracker_v4 import CausalTrackerV4
from tracking.recovery_controller import Config
from tracking.reliability_v4 import SharedReferenceVerifier
from tracking.tests.test_framework_v3 import FakeCore, FakeDetector, proposal, detection, image, A, B


def reference(good=True, **extra):
    return dict(valid=True, box=list(A), score=.9, reference_iou=.9 if good else 0., **extra)


class ReferenceCore(FakeCore):
    def __init__(self, local=None, probe=None, verify=None):
        super().__init__(local, probe)
        self.verify_fn = verify or (lambda i, box: reference())
        self.checks = []

    def verify_reference(self, rgb, box):
        self.checks.append((int(rgb[0, 0, 0]), list(box)))
        return self.verify_fn(int(rgb[0, 0, 0]), box)


class FrameworkV4Tests(unittest.TestCase):
    def make(self, local=None, candidates=None, probe=None, verify=None):
        core = ReferenceCore(local, probe, verify)
        tracker = CausalTrackerV4(core, FakeDetector(candidates))
        tracker.initialize(image(1), A)
        return tracker

    def test_low_response_still_outputs_current_local_without_commit(self):
        box = [32., 30., 10., 10.]
        t = self.make(local=lambda i: proposal(box, .1))
        r = t.track(image(2))
        self.assertEqual(r['bbox'], box)
        self.assertEqual(r['localization_status'], 'tentative')
        self.assertIsNone(r['proposed_commit'])
        self.assertEqual(t.core.anchor, A)
        self.assertEqual(t.tentative_anchor, box)
        self.assertFalse(r['physical_absence_inferred'])

    def test_bad_reference_blocks_high_forward_response_commit(self):
        t = self.make(verify=lambda i, box: reference(False))
        r = t.track(image(2))
        self.assertEqual(r['bbox'], A)
        self.assertEqual(r['local_health']['reason'], 'REFERENCE_TARGET_MISMATCH')
        self.assertFalse(r['state_committed'])

    def test_unverified_remote_detection_cannot_veto_good_local(self):
        t = self.make(candidates=lambda i: [detection(B)], verify=lambda i, box: reference(box == A))
        r = t.track(image(2))
        self.assertEqual(r['state'], 'TRACKING')
        self.assertEqual(r['proposed_commit'], A)
        self.assertEqual(r['probe_count'], 1)
        self.assertFalse(r['verification_branches'][0]['passed'])

    def test_verified_competitor_checked_even_in_tracking(self):
        t = self.make(candidates=lambda i: [detection(B)])
        r = t.track(image(2))
        self.assertEqual(r['bbox'], A)
        self.assertIsNone(r['proposed_commit'])
        self.assertEqual(r['reason'], 'AMBIGUOUS_VERIFIED_HYPOTHESES')
        self.assertEqual(r['verified_spatial_groups'], 2)
        self.assertEqual(r['reference_checks'], 2)

    def test_distant_recovery_keeps_output_while_waiting(self):
        t = self.make(local=lambda i: proposal(A, .1), candidates=lambda i: [detection(B)])
        for i in [2, 3]:
            r = t.track(image(i))
            self.assertEqual(r['bbox'], A)
            self.assertFalse(r['state_committed'])
            self.assertEqual(t.core.anchor, A)
        r = t.track(image(4))
        self.assertEqual(r['bbox'], B)
        self.assertEqual(r['proposed_commit'], B)
        self.assertEqual(r['commit_source'], 'global_verification')

    def test_global_and_local_share_identical_verification(self):
        t = self.make(candidates=lambda i: [detection(B)], verify=lambda i, box: reference(False))
        r = t.track(image(2))
        self.assertEqual(r['local_health']['reliability'], r['verification_branches'][0]['reliability'])
        self.assertFalse(r['state_committed'])

    def test_duplicate_local_candidate_reuses_local_forward_and_reverse(self):
        t = self.make(candidates=lambda i: [detection(A), detection(A)])
        r = t.track(image(2))
        self.assertEqual(r['probe_count'], 0)
        self.assertEqual(r['reference_checks'], 1)
        self.assertEqual(len(r['verification_branches']), 1)
        self.assertTrue(r['verification_branches'][0]['proposal_reused_from_local'])

    def test_reconfirmation_never_blanks_usable_local(self):
        t = self.make(verify=lambda i, box: reference(i > 2))
        for i in [2, 3, 4]:
            r = t.track(image(i))
            self.assertEqual(r['bbox'], A)
            self.assertFalse(r['state_committed'])
        r = t.track(image(5))
        self.assertTrue(r['state_committed'])
        self.assertEqual(r['commit_source'], 'local_reconfirmation')

    def test_disappearing_candidate_does_not_inherit_confirmation(self):
        t = self.make(local=lambda i: proposal(A, .1), candidates=lambda i: [] if i == 3 else [detection(B)])
        for i in [2, 3, 4, 5]: self.assertFalse(t.track(image(i))['state_committed'])
        self.assertTrue(t.track(image(6))['state_committed'])

    def test_invalid_local_may_output_unique_tentative_candidate(self):
        t = self.make(local=lambda i: proposal(A, float('nan')), candidates=lambda i: [detection(B)])
        r = t.track(image(2))
        self.assertEqual(r['bbox'], B)
        self.assertEqual(r['output_source'], 'candidate_tentative')
        self.assertIsNone(r['proposed_commit'])
        self.assertEqual(t.core.anchor, A)

    def test_invalid_local_and_no_verified_alternative_is_unavailable(self):
        t = self.make(local=lambda i: proposal(A, float('nan')))
        r = t.track(image(2))
        self.assertIsNone(r['bbox'])
        self.assertEqual(r['localization_status'], 'unavailable')
        self.assertEqual(r['physical_presence'], 'unknown')

    def test_reference_exception_fails_closed_and_requires_reset(self):
        def fail(i, box): raise RuntimeError('reference failure')
        t = self.make(verify=fail)
        with self.assertRaisesRegex(RuntimeError, 'reference failure'): t.track(image(2))
        self.assertTrue(t.failed)
        self.assertEqual(t.core.anchor, A)
        with self.assertRaisesRegex(RuntimeError, 'reinitialize'): t.track(image(3))

    def test_reference_cannot_mutate_authoritative_state(self):
        t = self.make()
        def mutate(i, box):
            t.core.commit(B, image(i).shape)
            return reference()
        t.core.verify_fn = mutate
        with self.assertRaisesRegex(RuntimeError, 'modified the authoritative'): t.track(image(2))
        self.assertTrue(t.failed)

    def test_nonfinite_reference_cannot_pass(self):
        verifier = SharedReferenceVerifier(Config(), lambda rgb, box: dict(valid=True, score=float('nan'), reference_iou=.9))
        result = verifier.assess(image(2), proposal(A))
        self.assertFalse(result['passed'])
        self.assertEqual(result['reason'], 'REFERENCE_CHECK_INVALID')

    def test_all_numeric_parameters_preserved(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(json.loads((root/'framework_v3.json').read_text())['policy'], json.loads((root/'framework_v4.json').read_text())['policy'])

    def test_sequence_reset_discards_confirmation_and_tentative_path(self):
        t = self.make(local=lambda i: proposal(A, .1), candidates=lambda i: [detection(B)])
        t.track(image(2));t.track(image(3))
        t.initialize(image(1), A)
        r = t.track(image(2))
        self.assertFalse(r['state_committed'])
        self.assertEqual(r['verification_branches'][0]['unique_streak'], 1)


if __name__ == '__main__': unittest.main()
