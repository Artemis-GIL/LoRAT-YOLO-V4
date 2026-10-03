import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tracking.causal_tracker_v3 import CausalTrackerV3, TransactionalLoRAT
from tracking.recovery_controller import Config, CandidateBank, spatial_groups

A = [30., 30., 10., 10.]
B = [130., 30., 10., 10.]
C = [130., 130., 10., 10.]


def proposal(box, score=.9):
    return dict(box=list(box), score=score, roi=[0., 0., 256., 256.])


def detection(box, score=.9):
    return dict(box=list(box), score=score, cls=0)


def image(index):
    return np.full((256, 256, 3), index, np.uint8)


class FakeCore:
    def __init__(self, local=None, probe=None, events=None):
        self.local_fn = local or (lambda i: proposal(A))
        self.probe_fn = probe or (lambda i, a: proposal(a))
        self.events = events if events is not None else []
        self.writes = []; self._anchor = None

    @property
    def anchor(self):return list(self._anchor)

    def initialize(self, rgb, box):self._anchor = list(box);self.writes = []

    def propose(self, rgb, anchor=None, *, purpose='local'):
        i = int(rgb[0, 0, 0]);self.events.append(('local' if purpose == 'local' else 'probe', i))
        return self.local_fn(i) if purpose == 'local' else self.probe_fn(i, anchor)

    def commit(self, box, shape):self._anchor = list(box);self.writes.append(list(box))


class FakeDetector:
    def __init__(self, fn=None, events=None):
        self.fn = fn or (lambda i: [])
        self.calls = 0;self.closed = False;self.events = events if events is not None else []

    def infer(self, rgb):
        self.calls += 1;i = int(rgb[0, 0, 0]);self.events.append(('detect', i))
        return dict(candidates=self.fn(i), request_id=self.calls, seconds=0., rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())

    def close(self):self.closed = True


class FrameworkTests(unittest.TestCase):
    def make(self, local=None, candidates=None, probe=None):
        core = FakeCore(local, probe);detector = FakeDetector(candidates)
        tracker = CausalTrackerV3(core, detector);tracker.initialize(image(1), A)
        return tracker

    def test_healthy_output_keeps_native_box(self):
        tracker = self.make(candidates=lambda i: [detection([29., 29., 12., 12.])])
        r = tracker.track(image(2))
        self.assertEqual(r['bbox'], A);self.assertEqual(tracker.core.anchor, A)
        self.assertEqual(r['probe_count'], 0);self.assertEqual(r['state'], 'TRACKING')

    def test_no_remote_overlap_needed_for_confirmed_recovery(self):
        tracker = self.make(local=lambda i: proposal(A, .01), candidates=lambda i: [detection(B)])
        for i in [2, 3]:
            r = tracker.track(image(i));self.assertIsNone(r['bbox']);self.assertEqual(tracker.core.anchor, A)
            self.assertEqual(r['state'], 'VERIFYING');self.assertFalse(r['state_committed'])
        r = tracker.track(image(4))
        self.assertEqual(r['bbox'], B);self.assertEqual(tracker.core.anchor, B)
        self.assertEqual(r['reason'], 'RECOVERY_CONFIRMED');self.assertEqual(len(tracker.core.writes), 1)

    def test_high_response_local_conflict_triggers_verification(self):
        tracker = self.make(local=lambda i: proposal(A, .99), candidates=lambda i: [detection(B)])
        r = tracker.track(image(2))
        self.assertEqual(r['local_health']['reason'], 'LOCAL_DETECTOR_CONFLICT')
        self.assertIsNone(r['bbox']);self.assertEqual(tracker.core.anchor, A)

    def test_no_candidates_is_unconfirmed_not_physical_absence(self):
        tracker = self.make(local=lambda i: proposal(A, .01))
        for i in range(2, 8):
            r = tracker.track(image(i));self.assertIsNone(r['bbox'])
            self.assertEqual(r['state'], 'SEARCHING');self.assertEqual(r['physical_presence'], 'unknown')
            self.assertFalse(r['physical_absence_inferred']);self.assertEqual(tracker.core.anchor, A)

    def test_missing_observation_resets_confirmation_and_retires_id(self):
        tracker = self.make(local=lambda i: proposal(A, .01), candidates=lambda i: [] if i == 3 else [detection(B)])
        first = tracker.track(image(2));old_id = first['verification_branches'][0]['id']
        r = tracker.track(image(3));self.assertIn(old_id, r['retired_branch_ids'])
        for i in [4, 5]:self.assertIsNone(tracker.track(image(i))['bbox'])
        r = tracker.track(image(6));self.assertEqual(r['bbox'], B)
        self.assertNotEqual(r['verification_branches'][0]['id'], old_id)

    def test_alternating_distant_candidates_never_accumulate_identity(self):
        tracker = self.make(local=lambda i: proposal(A, .01), candidates=lambda i: [detection(B if i % 2 else C)])
        for i in range(2, 10):self.assertIsNone(tracker.track(image(i))['bbox'])
        self.assertEqual(tracker.core.writes, [])

    def test_highest_response_cannot_resolve_distinct_rivals(self):
        tracker = self.make(local=lambda i: proposal(A, .01), candidates=lambda i: [detection(B), detection(C, .8)],
                            probe=lambda i, a: proposal(a, .99 if a == B else .65))
        for i in range(2, 8):
            r = tracker.track(image(i));self.assertIsNone(r['bbox']);self.assertEqual(r['reason'], 'AMBIGUOUS_CANDIDATES')
        self.assertEqual(tracker.core.writes, [])

    def test_unverified_spatial_rival_is_not_silently_discarded(self):
        tracker = self.make(local=lambda i: proposal(A, .01), candidates=lambda i: [detection(B), detection(C, .8)],
                            probe=lambda i, a: proposal(a, .9 if a == B else .01))
        for i in range(2, 8):self.assertIsNone(tracker.track(image(i))['bbox'])

    def test_ambiguity_resolution_requires_fresh_unique_evidence(self):
        tracker = self.make(local=lambda i: proposal(A, .01), candidates=lambda i: [detection(B), detection(C)] if i < 5 else [detection(B)])
        for i in range(2, 7):self.assertIsNone(tracker.track(image(i))['bbox'])
        self.assertEqual(tracker.track(image(7))['bbox'], B)

    def test_duplicate_candidates_do_not_count_as_distinct_targets(self):
        duplicate = [131., 30., 10., 10.]
        tracker = self.make(local=lambda i: proposal(A, .01), candidates=lambda i: [detection(B), detection(duplicate, .8)])
        for i in [2, 3]:self.assertIsNone(tracker.track(image(i))['bbox'])
        self.assertIsNotNone(tracker.track(image(4))['bbox'])

    def test_local_path_can_reconfirm_without_detector(self):
        tracker = self.make(local=lambda i: proposal(A, .01 if i == 2 else .9))
        for i in [2, 3, 4]:self.assertIsNone(tracker.track(image(i))['bbox'])
        r = tracker.track(image(5));self.assertEqual(r['bbox'], A);self.assertEqual(r['output_source'], 'local_reconfirmation')

    def test_tentative_local_motion_does_not_write_trusted_anchor(self):
        tracker = self.make(local=lambda i: proposal([30.+i, 30., 10., 10.], .01))
        r = tracker.track(image(2));self.assertEqual(r['tentative_local_anchor'], [32.,30.,10.,10.])
        r = tracker.track(image(3));self.assertEqual(r['local_search_anchor'], [32.,30.,10.,10.])
        self.assertEqual(tracker.core.anchor, A);self.assertIsNone(r['bbox'])

    def test_exact_duplicate_observations_preserve_one_confirmation_history(self):
        tracker = self.make(local=lambda i: proposal(A, .01), candidates=lambda i: [detection(B), detection(B, .8)])
        for i in [2,3]:self.assertIsNone(tracker.track(image(i))['bbox'])
        r = tracker.track(image(4));self.assertEqual(r['bbox'], B);self.assertEqual(len(r['active_candidates']),1)

    def test_probe_exception_never_commits_and_requires_reinitialize(self):
        def broken(i, a):raise RuntimeError('injected model error')
        tracker = self.make(local=lambda i: proposal(A, .01), candidates=lambda i: [detection(B)], probe=broken)
        with self.assertRaisesRegex(RuntimeError, 'injected'):tracker.track(image(2))
        self.assertEqual(tracker.core.anchor, A);self.assertEqual(tracker.frame, 1)
        with self.assertRaisesRegex(RuntimeError, 'reinitialize'):tracker.track(image(3))

    def test_invalid_candidate_and_model_numbers_do_not_reach_output(self):
        tracker = self.make(local=lambda i: proposal(A, float('nan')), candidates=lambda i: [detection(B, float('nan'))])
        r = tracker.track(image(2));self.assertIsNone(r['bbox']);self.assertEqual(r['rejected_invalid_candidates'], 1)
        json.dumps(r, allow_nan=False)

    def test_reset_discards_all_old_branches_and_confirmation(self):
        tracker = self.make(local=lambda i: proposal(A, .01), candidates=lambda i: [detection(B)])
        tracker.track(image(2));tracker.track(image(3));tracker.initialize(image(1), A)
        self.assertEqual(tracker.controller.bank.branches, [])
        self.assertIsNone(tracker.track(image(2))['bbox'])

    def test_one_detector_call_per_current_frame_and_no_future_dependence(self):
        events = [];core = FakeCore(events=events);detector = FakeDetector(events=events)
        tracker = CausalTrackerV3(core, detector);tracker.initialize(image(1), A)
        tracker.track(image(2));tracker.track(image(3))
        self.assertEqual(events, [('local', 2), ('detect', 2), ('local', 3), ('detect', 3)])
        self.assertEqual(detector.calls, 2)

    def test_wrong_frame_response_is_rejected(self):
        tracker = self.make();tracker.detector.infer = lambda rgb: dict(rgb_sha256='wrong')
        with self.assertRaisesRegex(RuntimeError, 'current RGB'):tracker.track(image(2))
        self.assertEqual(tracker.core.writes, [])

    def test_complete_link_does_not_merge_overlap_chain(self):
        cfg = Config(same_object_iou=.3)
        rows = [dict(id=i, box=[i*4., 0., 10., 10.]) for i in range(3)]
        self.assertEqual(len(spatial_groups(rows, cfg)), 2)

    def test_configuration_and_initialization_validate(self):
        for values in [dict(confirmation_frames=1), dict(confirmation_frames=2.5), dict(local_score_min=float('nan')), dict(max_candidates=13)]:
            with self.assertRaises(ValueError):Config(**values)
        tracker = CausalTrackerV3(FakeCore(), FakeDetector())
        with self.assertRaises(ValueError):tracker.initialize(image(1), [-1, 0, 10, 10])
        with self.assertRaises(RuntimeError):tracker.track(image(2))


class TransactionTests(unittest.TestCase):
    def test_temporary_provider_restored_even_when_native_track_raises(self):
        class Provider:
            def __init__(self, factor, minimum):self.area_factor=factor;self.min_object_size=np.array(minimum);self.cached_bbox=np.zeros(4)
            def initialize(self, b):self.cached_bbox[:]=b
        class Adapter:
            def __init__(self):self.provider=Provider(5., (10,10));self.provider.initialize([30,30,40,40]);self.Provider=Provider
        class Native:
            def __init__(self):self.adapter=Adapter()
            def track(self, rgb):self.adapter.provider.cached_bbox[:]=[130,30,140,40];raise RuntimeError('model failed')
        wrapper=object.__new__(TransactionalLoRAT);wrapper.native=Native();provider=wrapper.native.adapter.provider
        with self.assertRaisesRegex(RuntimeError, 'model failed'):wrapper.propose(image(1), B)
        self.assertIs(wrapper.native.adapter.provider, provider);self.assertEqual(wrapper.anchor, A)


class RunnerTests(unittest.TestCase):
    def test_frame_stream_decodes_only_when_requested(self):
        import cv2
        from tracking.run_tracker_v3 import frame_stream
        with tempfile.TemporaryDirectory() as folder:
            for i in [1,2]:cv2.imwrite(str(Path(folder)/f'{i:06d}.png'),image(i))
            args=SimpleNamespace(frames=Path(folder),video=None,max_frames=None)
            with patch('tracking.run_tracker_v3.cv2.imread',wraps=cv2.imread) as read:
                stream=frame_stream(args)
                self.assertEqual(read.call_count,0)
                self.assertEqual(next(stream)[0],1);self.assertEqual(read.call_count,1)
                self.assertEqual(next(stream)[0],2);self.assertEqual(read.call_count,2)
                stream.close()


if __name__ == '__main__':unittest.main()
