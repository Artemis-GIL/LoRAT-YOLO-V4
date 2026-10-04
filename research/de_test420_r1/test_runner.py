"""CPU-only checks of causal image order, resume integrity and official scoring."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np
import torch
from . import test420 as runner

class FakeTracker:
    def __init__(self,events,fail_at=None,mutate=False):
        self.events=events;self.fail_at=fail_at;self.mutate=mutate;self.frame=0
        adapter=SimpleNamespace(z=torch.zeros(2),mask=torch.zeros(2))
        self.core=SimpleNamespace(native=SimpleNamespace(adapter=adapter))
    def initialize(self,rgb,box):
        self.box=list(box);self.events.append('init');return self.result(rgb)
    def track(self,rgb):
        self.events.append('track');return self.result(rgb)
    def result(self,rgb):
        self.frame+=1
        if self.frame==self.fail_at:raise RuntimeError('simulated interruption')
        if self.mutate and self.frame==2:self.core.native.adapter.z.add_(1)
        return dict(frame=self.frame,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),bbox=self.box,
            reference_checks=0,physical_absence_inferred=False,online_seconds=0 if self.frame==1 else .01,
            detector_seconds=0 if self.frame==1 else .005,probe_count=0,reason='LOCAL',localization_status='confirmed',
            state_committed=True,state='tracking')

class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.out=self.root/'output';images=self.root/'images';images.mkdir()
        paths=[]
        for i in range(1,4):
            path=images/f'{i:08d}.png';cv2.imwrite(str(path),np.full((20,20,3),i,np.uint8));paths.append(str(path))
        self.row=dict(sequence='UAV-Anti-UAV_Test_000001',initial_bbox=[1,2,3,4],frames=paths)
        self.events=[]
    def run_fake(self,tracker):
        original=runner.image_at
        def observed(path):self.events.append('read');return original(path)
        with patch.object(runner,'image_at',side_effect=observed):
            return runner.infer_sequence(tracker,self.out,self.row,1,1,'selection-digest')
    def test_current_frame_only_and_resume_corruption(self):
        done=self.run_fake(FakeTracker(self.events))
        self.assertEqual(self.events,['read','init','read','track','read','track'])
        self.assertEqual(done['detector_requests'],2)
        self.assertEqual(runner.completed(self.out,self.row,'selection-digest'),done)
        with self.assertRaises(RuntimeError):runner.completed(self.out,self.row,'different-selection')
        with self.assertRaises(RuntimeError):self.run_fake(FakeTracker(self.events))
        pred=self.out/'predictions'/f"{self.row['sequence']}.txt";pred.write_text(pred.read_text()+'0,0,0,0\n')
        with self.assertRaises(RuntimeError):runner.completed(self.out,self.row,'selection-digest')
    def test_interruption_archived_and_restart_from_first_frame(self):
        with self.assertRaisesRegex(RuntimeError,'simulated'):self.run_fake(FakeTracker(self.events,fail_at=2))
        self.assertIsNone(runner.completed(self.out,self.row,'selection-digest'))
        self.events.clear();done=self.run_fake(FakeTracker(self.events))
        self.assertEqual(done['frames'],3);self.assertEqual(self.events[1],'init')
        self.assertEqual(len(list((self.out/'interrupted_attempts'/self.row['sequence']).glob('*'))),1)
    def test_template_change_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'template changed'):self.run_fake(FakeTracker(self.events,mutate=True))
        self.assertIsNone(runner.completed(self.out,self.row,'selection-digest'))
    def test_official_functions_score_test_root_after_complete(self):
        dataset=self.root/'Test';folder=dataset/self.row['sequence'];folder.mkdir(parents=True)
        np.savetxt(folder/'groundtruth_rect.txt',np.tile(self.row['initial_bbox'],(3,1)),delimiter=',')
        np.savetxt(folder/'absent.txt',np.zeros(3))
        selection=dict(variant='B',checkpoint=None,checkpoint_sha256=None,baseline_checkpoint='frozen-B',baseline_sha256='sha',
            selection_rule='validation only',validation_auc=.7,validation_np_auc=.8)
        with patch.object(runner,'SEQUENCES',1),patch.object(runner,'FRAMES',3):
            with self.assertRaisesRegex(RuntimeError,'must finish'):runner.score_test(self.out,[self.row],dataset,selection,'selection-digest')
            self.run_fake(FakeTracker(self.events))
            result=runner.score_test(self.out,[self.row],dataset,selection,'selection-digest')
        self.assertEqual(result['scope'],'TEST420');self.assertEqual(result['counts']['detector_requests'],2)
        self.assertEqual(result['visible_no_output_frames'],0);self.assertEqual(result['sequence_macro']['visible_hit50'],1)
        self.assertAlmostEqual(result['online_tracker_fps'],100.)
        self.assertFalse((self.out/'COMPLETE.json').exists())

if __name__=='__main__':unittest.main()
