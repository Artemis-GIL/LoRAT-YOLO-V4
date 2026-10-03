"""Launcher regression checks: current-frame ordering, resume, null scoring."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tracking import test420_v4 as launch


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.row = dict(sequence='UAV-Anti-UAV_Test_000001', initial_bbox=[2., 2., 4., 4.], frames=['1', '2', '3'])
        self.adapter = SimpleNamespace(z=torch.ones(1), mask=torch.zeros(1))

    def tracker(self, fail=False):
        owner = self
        class FakeTracker:
            core = SimpleNamespace(native=SimpleNamespace(adapter=owner.adapter))
            def initialize(self, rgb, box): return self.result(rgb, box)
            def track(self, rgb):
                if fail: raise RuntimeError('injected interruption')
                return self.result(rgb, None)
            def result(self, rgb, box):
                return dict(frame=int(rgb[0, 0, 0]), rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                    bbox=box, state='TRACKING' if box else 'SEARCHING', reason='INITIALIZED' if box else 'UNCONFIRMED',
                    probe_count=0, online_seconds=.01, detector_seconds=.002, state_committed=box is not None,
                    localization_status='confirmed' if box is not None else 'unavailable', reference_checks=0)
        return FakeTracker()

    def image_at(self, path):
        frame = int(path)
        # The previous prediction and trace must already be flushed before the
        # launcher can read the next image; this catches accidental preloading.
        if frame > 1:
            pred = self.out/'predictions'/f'{self.row["sequence"]}.partial.txt'
            trace = self.out/'traces'/self.row['sequence']/'frames.partial.jsonl'
            self.assertEqual(len(pred.read_text().splitlines()), frame-1)
            self.assertEqual(len(trace.read_text().splitlines()), frame-1)
        rgb = np.full((12, 12, 3), frame, np.uint8)
        return rgb, rgb, hashlib.sha256(rgb.tobytes()).hexdigest()

    def run_fake(self, fail=False):
        with patch.object(launch.common, 'image_at', self.image_at):
            launch.run_sequence(self.tracker(fail), torch, self.out, self.row, 1, 1)

    def test_causal_flush_null_export_and_completed_hashes(self):
        self.run_fake()
        done = launch.completed(self.out, self.row)
        self.assertEqual(done['frames'], 3)
        self.assertEqual(done['no_output_frames'], 2)
        self.assertEqual(done['detector_requests'], 2)
        boxes = np.loadtxt(self.out/'predictions'/f'{self.row["sequence"]}.txt', delimiter=',')
        np.testing.assert_array_equal(boxes[1:], np.zeros((2, 4)))

    def test_interrupted_sequence_archived_and_restarted(self):
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            self.run_fake(fail=True)
        self.assertIsNone(launch.completed(self.out, self.row))
        self.run_fake()
        attempts = list((self.out/'interrupted_attempts'/self.row['sequence']).iterdir())
        self.assertEqual(len(attempts), 1)
        self.assertTrue((attempts[0]/'INTERRUPTED.json').exists())
        self.assertEqual(launch.completed(self.out, self.row)['frames'], 3)

    def test_tentative_output_and_deferred_commit_are_separate_counts(self):
        tracker = self.tracker()
        def tentative(rgb):
            r = tracker.result(rgb, [3., 3., 4., 4.])
            r.update(state_committed=False, localization_status='tentative', state='VERIFYING')
            return r
        tracker.track = tentative
        with patch.object(launch.common, 'image_at', self.image_at):
            launch.run_sequence(tracker, torch, self.out, self.row, 1, 1)
        done = launch.completed(self.out, self.row)
        self.assertEqual(done['no_output_frames'], 0)
        self.assertEqual(done['tentative_output_frames'], 2)
        self.assertEqual(done['commit_deferred_frames'], 2)

    def test_completed_result_corruption_rejected(self):
        self.run_fake()
        (self.out/'predictions'/f'{self.row["sequence"]}.txt').write_text('corrupted\n')
        with self.assertRaisesRegex(RuntimeError, 'Corrupt completed'):
            launch.completed(self.out, self.row)

    def test_null_scoring_keeps_legacy_metrics_and_explicit_failure_diagnostics(self):
        sys.path.insert(0, str(launch.common.TOOLKIT))
        from got10k.experiments import ExperimentUAVAntiUAV_Overall
        evaluator = ExperimentUAVAntiUAV_Overall.__new__(ExperimentUAVAntiUAV_Overall)
        evaluator.nbins_iou = evaluator.nbins_iou_complete = 21
        evaluator.nbins_ce = evaluator.nbins_ce_norm = 51
        spec = launch.importlib.util.spec_from_file_location('launcher_test_accuracy', launch.common.TOOLKIT/'accuracy/accuracy.py')
        accuracy = launch.importlib.util.module_from_spec(spec); spec.loader.exec_module(accuracy)
        gt = np.asarray([[1., 1., 4., 4.], [1., 1., 4., 4.], [0., 0., 0., 0.]])
        boxes = gt.copy(); boxes[1:] = 0.
        values = launch.score_arrays(evaluator, accuracy, boxes, gt, np.asarray([0, 0, 1]))
        expected = launch.curve_values(evaluator._calc_curves(*evaluator._calc_metrics(boxes.copy(), gt.copy())))
        for key, value in expected.items(): self.assertEqual(values[key], value)
        self.assertEqual(values['visible_frames'], 1)
        self.assertEqual(values['visible_no_output_frames'], 1)
        self.assertEqual(values['visible_hit50'], 0.)
        self.assertAlmostEqual(values['mACC'], 1/3)
        self.assertAlmostEqual(values['null_failure_Precision20'], 1/3)
        self.assertGreater(values['Precision20'], values['null_failure_Precision20'])


if __name__ == '__main__': unittest.main()
