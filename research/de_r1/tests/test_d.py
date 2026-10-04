import unittest
import numpy as np
import torch
from research.de_r1.history import OnlineHistory,history_indices
from research.de_r1.model_d import ActionValue,action_loss
from research.de_r1.tracker_de import ActionController
from research.bc_r1.tracker_c import ResearchTracker
from research.bc_r1.tests.test_bc import Core,Detector
from tracking.structural_ablations import OutputOnlyTrackerV3

class PreferRight(torch.nn.Module):
    def forward(self,features,baseline_index,**kwargs):
        q=features[:,:,-1,2,0]/100
        return dict(quality=q,advantage=q-q.gather(1,baseline_index[:,None]))

class Tests(unittest.TestCase):
    def test_ambiguous_split_has_no_inherited_history(self):
        past=[dict(frame=8,groups=[dict(box=[0,0,1,1],member_ids=[0,7],row=4)])]
        now=[dict(member_ids=[0]),dict(member_ids=[7])]
        self.assertTrue((history_indices(now,past)==-1).all())
    def test_fixed_clock_does_not_read_current_frame(self):
        h=OnlineHistory();g=[dict(box=[10,10,10,10],member_ids=[0,1])]
        f=np.ones((1,4,1024),np.float16);m=np.ones((1,4),np.float32)
        for frame in range(2,9):
            q=h.query(g,f,m,g[0]['box'],frame)
            self.assertFalse(q['history_valid'][0,:3].any());h.observe(g,f,m,frame)
        q=h.query(g,f*2,m,g[0]['box'],9)
        self.assertEqual(q['history_valid'][0].tolist(),[False,False,True,True])
        self.assertEqual(float(q['features'][0,2,0,0]),1.)
        self.assertEqual(float(q['features'][0,3,0,0]),2.)
    def batch(self):
        return dict(features=torch.randn(2,3,4,4,1024),metadata=torch.randn(2,3,4,4),
            geometry=torch.randn(2,3,4,5),history_valid=torch.ones(2,3,4,dtype=torch.bool),baseline_index=torch.tensor([1,0]),
            padding=torch.zeros(2,3,dtype=torch.bool))
    def test_zero_head_exact_keep_and_loss_gradients(self):
        m=ActionValue();b=self.batch();out=m(**b)
        self.assertTrue(torch.equal(out['advantage'],torch.zeros(2,3)))
        target=torch.rand(2,3);delta=target-target.gather(1,b['baseline_index'][:,None])
        loss,_=action_loss(out,target,delta,b['padding'],b['baseline_index']);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(m.value[-1].weight.grad.abs().sum()),0.)
    def test_static_does_not_consume_history(self):
        m=ActionValue(temporal=False);torch.nn.init.normal_(m.value[-1].weight)
        b=self.batch();q=m(**b)['quality'];b['features'][:,:,:3]=torch.randn_like(b['features'][:,:,:3])*100
        b['geometry'][:,:,:3]=torch.randn_like(b['geometry'][:,:,:3])*100
        self.assertTrue(torch.equal(q,m(**b)['quality']))
    def test_temporal_head_can_consume_history(self):
        m=ActionValue();torch.nn.init.normal_(m.value[-1].weight)
        b=self.batch();q=m(**b)['quality'];b['features'][:,:,:3]=torch.randn_like(b['features'][:,:,:3])*100
        self.assertFalse(torch.allclose(q,m(**b)['quality']))
    def test_candidate_permutation_preserves_actions(self):
        m=ActionValue();torch.nn.init.normal_(m.value[-1].weight);b=self.batch();q=m(**b)['quality']
        order=torch.tensor([2,0,1]);inverse=torch.argsort(order)
        changed={k:(inverse[v] if k=='baseline_index' else v[:,order]) for k,v in b.items()}
        self.assertTrue(torch.allclose(q[:,order],m(**changed)['quality'],atol=1e-6))
    @unittest.skipUnless(torch.cuda.is_available(),'runtime is CUDA')
    def test_zero_controller_keeps_b_decisions_and_anchors(self):
        image=np.zeros((100,100,3),np.uint8);a=OutputOnlyTrackerV3(Core(),Detector())
        b=ResearchTracker(Core(),Detector());b.controller=ActionController(b.config,b.core,ActionValue().cuda().eval())
        for t in [a,b]:t.initialize(image,[10,10,10,10])
        for _ in range(24):
            x=a.track(image);y=b.track(image)
            for k in ['bbox','state','reason','proposed_commit','committed_bbox','local_search_anchor','tentative_local_anchor']:
                self.assertEqual(x[k],y[k],k)
    @unittest.skipUnless(torch.cuda.is_available(),'runtime is CUDA')
    def test_alternative_requires_own_three_frames_and_changes_anchor(self):
        image=np.zeros((100,100,3),np.uint8);t=ResearchTracker(Core(),Detector())
        t.controller=ActionController(t.config,t.core,PreferRight());t.initialize(image,[10,10,10,10])
        a=t.track(image);b=t.track(image);c=t.track(image)
        self.assertEqual(a['bbox'],[10,10,10,10]);self.assertEqual(b['bbox'],[10,10,10,10])
        self.assertEqual(c['bbox'],[70,70,10,10]);self.assertEqual(c['d_action'],'REPLACE_B_CONFIRMED')
        self.assertEqual(t.track(image)['local_search_anchor'],[70,70,10,10])

if __name__=='__main__':unittest.main()
