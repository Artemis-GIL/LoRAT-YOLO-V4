import hashlib, unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from research.bc_r1.model_c import CandidateComparator,comparison_loss
from research.bc_r1.mine_c import labels
from research.bc_r1.tracker_c import ResearchTracker,PoolController
from tracking.structural_ablations import OutputOnlyTrackerV3

class Core:
    def initialize(self,rgb,box):self.anchor=list(box)
    def begin_frame(self):pass
    def propose(self,rgb,anchor=None,**kwargs):
        return dict(box=list(self.anchor if anchor is None else anchor),score=.9,roi=[0.,0.,100.,100.],valid=True)
    def commit(self,box,shape):self.anchor=list(box)
    def descriptor(self,proposal):return np.full((4,1024),proposal['box'][0],dtype=np.float16)
class Detector:
    def infer(self,rgb):
        return dict(rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),request_id=1,seconds=0.,
            candidates=[dict(box=[10.,10.,10.,10.],score=.9),dict(box=[70.,70.,10.,10.],score=.8)])
    def close(self):pass
class PickRight(torch.nn.Module):
    def forward(self,f,m):
        scores=f[:,:,2,0]
        return torch.cat((scores,torch.zeros(len(f),1,device=f.device)),1),torch.zeros_like(scores)
class PickNone(torch.nn.Module):
    def forward(self,f,m):
        scores=torch.zeros(f.shape[:2],device=f.device)
        return torch.cat((scores,torch.ones(len(f),1,device=f.device)*100),1),scores

class Tests(unittest.TestCase):
    def test_comparator_permutation_equivariance(self):
        torch.manual_seed(5);model=CandidateComparator().eval()
        f=torch.randn(1,4,4,1024);m=torch.randn(1,4,4);p=torch.tensor([2,0,3,1])
        a,q=model(f,m);b,r=model(f[:,p],m[:,p])
        self.assertTrue(torch.allclose(a[:,:4][:,p],b[:,:4],atol=1e-6))
        self.assertTrue(torch.allclose(a[:,-1],b[:,-1],atol=1e-6))
        self.assertTrue(torch.allclose(q[:,p],r,atol=1e-6))
    def test_padding_cannot_change_valid_choice(self):
        torch.manual_seed(3);model=CandidateComparator().eval()
        f=torch.randn(1,1,4,1024);m=torch.randn(1,1,4);a,_=model(f,m)
        b,_=model(torch.cat((f,torch.randn(1,2,4,1024)),1),torch.cat((m,torch.randn(1,2,4)),1),torch.tensor([[False,True,True]]))
        self.assertTrue(torch.allclose(a,b[:,[0,3]],atol=1e-6))
    def test_empty_pool_selects_unmatched(self):
        m=CandidateComparator();a,q=m(torch.zeros(1,0,4,1024),torch.zeros(1,0,4))
        self.assertEqual(tuple(a.shape),(1,1));self.assertEqual(q.numel(),0)
    def test_low_quality_target_not_automatically_negative(self):
        y,q,valid,_=labels([[7.,11.,26.,2.],[70.,70.,10.,10.]],np.array([10.,10.,20.,4.]),True)
        self.assertTrue(valid);self.assertEqual(y.tolist(),[1.,0.,0.]);self.assertLess(q[0],.5)
    def test_unmatched_is_pool_relative(self):
        a=labels([[70.,70.,10.,10.]],np.array([10.,10.,10.,10.]),True)
        b=labels([[70.,70.,10.,10.]],np.zeros(4),False)
        self.assertEqual(a[0].tolist(),[0.,1.]);self.assertEqual(a[0].tolist(),b[0].tolist())
    def test_ambiguous_set_has_no_selection_gradient(self):
        logits=torch.randn(1,3,requires_grad=True);quality=torch.randn(1,2,requires_grad=True)
        loss,_=comparison_loss(logits,quality,torch.tensor([[1.,0.,0.]]),torch.tensor([[.3,0.]]),torch.ones(1,2),torch.zeros(1))
        loss.backward();self.assertEqual(float(logits.grad.abs().sum()),0.);self.assertGreater(float(quality.grad.abs().sum()),0.)
    @unittest.skipUnless(torch.cuda.is_available(),'training batches run on CUDA')
    def test_variable_pool_storage_and_null_padding_keep_labels(self):
        from research.bc_r1.train_c import Samples,make_batch
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'UAV_fixture';folder.mkdir()
            features=np.zeros((3,4,1024),np.float16)
            features[0]+=10;features[1]+=20;features[2]+=30
            np.savez(folder/'samples.npz',features=features,metadata=np.zeros((3,4),np.float32),
                target=np.array([0.,1.,1.,0.,0.],np.float32),iou=np.array([0.,1.,0.],np.float32),
                pointers=np.array([0,1,3]),selection_valid=np.array([True,True]),category=np.array([1,2]))
            ds=Samples(directory);rng=np.random.default_rng(7)
            a=ds.get(0,rng);b=ds.get(1,rng)
            self.assertEqual(a[2].tolist(),[0.,1.])
            self.assertEqual(float(b[0][b[2][:-1].argmax(),0,0]),20.)
            f,m,y,q,padding,mask,valid=make_batch([a,b])
            self.assertEqual(y[0].tolist(),[0.,0.,1.]);self.assertEqual(padding[0].tolist(),[False,True])
            model=CandidateComparator().cuda();logits,quality=model(f,m,padding)
            loss,_=comparison_loss(logits,quality,y,q,mask,valid);loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
    def test_extra_pool_keeps_original_state_and_output(self):
        image=np.zeros((100,100,3),np.uint8)
        a=OutputOnlyTrackerV3(Core(),Detector());b=ResearchTracker(Core(),Detector(),pool=True)
        a.initialize(image,[10,10,10,10]);b.initialize(image,[10,10,10,10])
        for _ in range(12):
            x=a.track(image);y=b.track(image)
            for k in ['bbox','state','reason','proposed_commit','committed_bbox','local_proposal','tentative_local_anchor']:
                self.assertEqual(x[k],y[k],k)
            self.assertGreater(y['extra_pool_probes'],0)
    @unittest.skipUnless(torch.cuda.is_available(),'real C inference runs on CUDA')
    def test_global_selection_changes_next_search_after_confirmation(self):
        image=np.zeros((100,100,3),np.uint8);tracker=ResearchTracker(Core(),Detector(),pool=True,comparator=PickRight())
        tracker.initialize(image,[10,10,10,10]);a=tracker.track(image);b=tracker.track(image);c=tracker.track(image)
        self.assertIsNone(a['proposed_commit']);self.assertIsNone(b['proposed_commit'])
        self.assertEqual(c['proposed_commit'],[70.,70.,10.,10.]);self.assertIsNone(c['tentative_local_anchor'])
        d=tracker.track(image);self.assertEqual(d['local_search_anchor'],[70.,70.,10.,10.])
    @unittest.skipUnless(torch.cuda.is_available(),'real C inference runs on CUDA')
    def test_unmatched_does_not_claim_absence(self):
        image=np.zeros((100,100,3),np.uint8);t=ResearchTracker(Core(),Detector(),pool=True,comparator=PickNone())
        t.initialize(image,[10,10,10,10]);r=t.track(image)
        self.assertIsNone(r['proposed_commit']);self.assertIsNotNone(r['bbox']);self.assertFalse(r['physical_absence_inferred'])

if __name__=='__main__':unittest.main()
