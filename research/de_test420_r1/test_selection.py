"""CPU regression checks for selection and zero-control safeguards."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from .select import rank_summaries,exact_zero_trace,require_digest
from research.de_r1.common import sha

class SelectionTests(unittest.TestCase):
    def scores(self):
        return {v:dict(scope='TRAIN_DEVELOPMENT_VAL132',sequences=132,frames=47200,
            sequence_macro=dict(AUC=.7,nPrecision_AUC=.8)) for v in ('B','D0','D','E')}
    def test_auc_priority_and_exact_ties(self):
        s=self.scores();self.assertEqual(rank_summaries(s),['B','D0','D','E'])
        s['D']['sequence_macro']['nPrecision_AUC']=.9;self.assertEqual(rank_summaries(s)[0],'D')
        s['E']['sequence_macro'].update(AUC=.7+1e-15,nPrecision_AUC=.1)
        self.assertEqual(rank_summaries(s)[0],'E')
        # Zero controls and any historical Test result cannot be extra candidates.
        s['D_ZERO']=copy.deepcopy(s['D'])
        with self.assertRaises(AssertionError):rank_summaries(s)
    def test_missing_incomplete_wrong_scope_and_nonfinite_rejected(self):
        s=self.scores();del s['E']
        with self.assertRaises(AssertionError):rank_summaries(s)
        for key,value in [('frames',47199),('sequences',131),('scope','TEST420')]:
            s=self.scores();s['D'][key]=value
            with self.assertRaises(AssertionError):rank_summaries(s)
        for value in [float('nan'),float('inf'),-1,2,True,None]:
            s=self.scores();s['D']['sequence_macro']['AUC']=value
            with self.assertRaises(AssertionError):rank_summaries(s)
    def test_zero_output_alone_cannot_hide_state_or_rgb_change(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';b=Path(d)/'b'
            rows=[dict(frame=i,rgb_sha256=str(i),bbox=[1,2,3,4],state='tracking',local_search_anchor=[1,2,3,4]) for i in (1,2)]
            def put(path,data):path.write_text(''.join(json.dumps(r)+'\n' for r in data))
            put(a,rows);put(b,rows);self.assertEqual(exact_zero_trace(a,b),2)
            for field,new in [('local_search_anchor',[9,2,3,4]),('state','lost'),('rgb_sha256','changed'),('frame',3)]:
                changed=copy.deepcopy(rows);changed[-1][field]=new;put(b,changed)
                with self.assertRaises(AssertionError):exact_zero_trace(a,b)
            put(b,rows[:1])
            with self.assertRaises(AssertionError):exact_zero_trace(a,b)
    def test_artifact_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'SUMMARY.json';p.write_text('{"AUC":0.7}')
            digest=sha(p);require_digest(p,digest);p.write_text('{"AUC":0.9}')
            with self.assertRaises(AssertionError):require_digest(p,digest)

if __name__=='__main__':unittest.main()
