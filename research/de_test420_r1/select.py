"""Read-only completed validation audit and predetermined model selection."""
import json
import math
from itertools import zip_longest
from pathlib import Path
from research.de_r1.common import OLD, RUN, B_WEIGHTS, BASELINE, read, sha, verify, verify_baseline
from research.bc_r1.evaluate import check_done
from tracking.test420_v4 import METRIC_KEYS

CANDIDATES=('B','D0','D','E')
ZERO=('D_ZERO','E_ZERO')
STATE_FIELDS=('state','previous_state','proposed_commit','commit_source','search_anchor_before',
    'local_search_anchor','committed_bbox','tentative_local_anchor','localization_status','reason','output_source')
RULE='Highest unrounded Val132 sequence-macro AUC; exact ties use nPrecision_AUC, then fixed order B,D0,D,E. No Test metrics enter selection.'

def rank_summaries(summaries):
    assert set(summaries)==set(CANDIDATES),'All four candidates must be present; zero controls are ineligible'
    for v in CANDIDATES:
        s=summaries[v]
        assert s['scope']=='TRAIN_DEVELOPMENT_VAL132' and s['sequences']==132 and s['frames']==47200
        for metric in ('AUC','nPrecision_AUC'):
            x=s['sequence_macro'][metric]
            assert type(x) in (int,float) and math.isfinite(x) and 0<=x<=1
    return sorted(CANDIDATES,key=lambda v:(-summaries[v]['sequence_macro']['AUC'],
        -summaries[v]['sequence_macro']['nPrecision_AUC'],CANDIDATES.index(v)))

def exact_zero_trace(current,baseline):
    count=0
    with Path(current).open() as a,Path(baseline).open() as b:
        for frame,(left,right) in enumerate(zip_longest(a,b),1):
            assert left is not None and right is not None,'zero trace length mismatch'
            l=json.loads(left);r=json.loads(right)
            assert l['frame']==r['frame']==frame
            assert l['rgb_sha256']==r['rgb_sha256']
            for key in STATE_FIELDS:assert l.get(key)==r.get(key),(frame,key,'zero-state mismatch')
            count+=1
    return count

def require_digest(path,digest):
    assert sha(path)==digest,('artifact digest mismatch',str(path))

def validated_selection():
    verify_baseline()
    assert read(RUN/'STATUS.json')['stage']=='COMPLETE','Wait for the complete D/E pipeline'
    assert read(RUN/'pipeline/COMPLETE.json')['stage']=='COMPLETE'
    verify(read(RUN/'pipeline/FREEZE.json'))
    cd=read(RUN/'comparison/COMPLETE.json')
    assert cd['status']=='PASS' and cd['sequences']==132 and cd['frames_per_variant']==47200 and cd['zero_controls_exact']
    require_digest(RUN/'comparison/COMPARISON.json',cd['comparison_sha256'])
    require_digest(RUN/'comparison/REPORT.zh.md',cd['report_sha256'])
    comparison=read(RUN/'comparison/COMPARISON.json')
    assert comparison['scope']=='TRAIN_DEVELOPMENT_VAL132' and comparison['sequences']==132 and comparison['frames']==47200
    assert comparison['no_test_inference'] is True
    cases=[c for c in read(OLD/'PROTOCOL.json')['cases'] if c['split']=='val']
    assert len(cases)==132 and sum(c['frames'] for c in cases)==47200
    names={c['sequence'] for c in cases};assert len(names)==132 and all(n.startswith('UAV-Anti-UAV_Train_') for n in names)
    folders={'B':OLD/'validation/B',**{v:RUN/'validation'/v for v in ('D_ZERO','D0','D','E_ZERO','E')}}
    summaries={};markers={};provenance={}
    def record(path):provenance[str(Path(path).resolve())]=sha(path)
    for p in [BASELINE,OLD/'PROTOCOL.json',RUN/'PROTOCOL.json',RUN/'pipeline/FREEZE.json',
        RUN/'pipeline/COMPLETE.json',RUN/'comparison/COMPLETE.json',RUN/'comparison/COMPARISON.json',RUN/'comparison/REPORT.zh.md']:
        record(p)
    for v,folder in folders.items():
        verify(read(folder/'FREEZE.json'))
        done=read(folder/'COMPLETE.json')
        assert done['sequences']==132 and done['frames']==47200
        if v!='B':assert done['status']=='PASS' and done['scope']=='TRAIN_DEVELOPMENT_VAL132'
        require_digest(folder/'metrics/SUMMARY.json',done['summary_sha256'])
        require_digest(folder/'metrics/SEQUENCES.json',done['sequences_sha256'])
        s=read(folder/'metrics/SUMMARY.json');rows=read(folder/'metrics/SEQUENCES.json')
        assert s['scope']=='TRAIN_DEVELOPMENT_VAL132' and s['sequences']==132 and s['frames']==47200
        assert len(rows)==132 and {r['sequence'] for r in rows}==names and sum(r['frames'] for r in rows)==47200
        row_frames={r['sequence']:r['frames'] for r in rows}
        assert all(row_frames[c['sequence']]==c['frames'] for c in cases)
        assert s['reverse_verification'] is False and s['detector_precomputation'] is False
        for key in METRIC_KEYS:
            values=[r[key] for r in rows if r[key] is not None]
            if key in ('AUC','nPrecision_AUC'):assert len(values)==132,(v,key,'missing sequence metric')
            assert values and all(type(x) in (int,float) and math.isfinite(x) for x in values)
            assert abs(sum(values)/len(values)-s['sequence_macro'][key])<=1e-12,(v,key,'macro mismatch')
        assert s==comparison['summaries'][v] and done==comparison['input_completion_digests'][v]
        markers[v]={}
        for c in cases:
            d=check_done(folder,c,None);assert d and d['initial_template_preserved']
            assert d['detector_requests']==c['frames']-1
            markers[v][c['sequence']]=d
            record(folder/c['sequence']/'COMPLETE.json')
        for name in ['COMPLETE.json','FREEZE.json','metrics/SUMMARY.json','metrics/SEQUENCES.json']:record(folder/name)
        summaries[v]=s
    for v in ZERO:
        folder=folders[v];parity=read(folder/'PARITY_B.json');done=read(folder/'COMPLETE.json')
        require_digest(folder/'PARITY_B.json',done['parity_sha256'])
        assert parity['status']=='PASS' and parity['sequences']==132 and parity['frames']==47200
        for key in ['exact_prediction_equality','exact_state_and_anchor_equality','exact_RGB_equality']:assert parity[key] is True
        assert set(parity['state_fields_checked'])==set(STATE_FIELDS)
        pairs=parity['sequences_checked'];assert len(pairs)==132 and {p['sequence'] for p in pairs}==names
        for c in cases:
            name=c['sequence'];a=markers[v][name];b=markers['B'][name]
            assert a['prediction_sha256']==b['prediction_sha256']
            assert exact_zero_trace(folder/name/'frames.jsonl',folders['B']/name/'frames.jsonl')==c['frames']
        for key in METRIC_KEYS:assert abs(summaries[v]['sequence_macro'][key]-summaries['B']['sequence_macro'][key])<=1e-12
        assert parity==comparison['zero_controls'][v];record(folder/'PARITY_B.json')
    protocol=read(RUN/'PROTOCOL.json')
    for v in ('D0','D','E'):
        folder=RUN/v;done=read(folder/'COMPLETE.json');assert done['status']=='COMPLETE'
        expected_epochs=protocol['E' if v=='E' else 'D']['epochs']
        assert done['epochs']==expected_epochs
        assert done['updates']==(2048 if v=='E' else 49152)
        if v=='E':assert done['frozen_B_unchanged'] and done['frozen_B_gradients_absent']
        else:assert done['baseline_frozen'] and done['temporal']==(v=='D')
        require_digest(folder/'final.safetensors',done.get('checkpoint_sha256',done.get('sha256')))
        validation_frozen=read(folders[v]/'FREEZE.json')
        assert validation_frozen[str(folder/'final.safetensors')]==sha(folder/'final.safetensors')
        verify(read(folder/'FREEZE.json'))
        for file in ['COMPLETE.json','FREEZE.json','final.safetensors']:record(folder/file)
    record(B_WEIGHTS);record(OLD/'B/COMPLETE.json')
    ranking=rank_summaries({v:summaries[v] for v in CANDIDATES});winner=ranking[0]
    checkpoint=None if winner=='B' else RUN/winner/'final.safetensors'
    return dict(variant=winner,checkpoint=str(checkpoint) if checkpoint else None,checkpoint_sha256=sha(checkpoint) if checkpoint else None,
        baseline_checkpoint=str(B_WEIGHTS),baseline_sha256=sha(B_WEIGHTS),selection_rule=RULE,
        validation_auc=summaries[winner]['sequence_macro']['AUC'],validation_np_auc=summaries[winner]['sequence_macro']['nPrecision_AUC'],
        ranking=[dict(variant=v,AUC=summaries[v]['sequence_macro']['AUC'],nPrecision_AUC=summaries[v]['sequence_macro']['nPrecision_AUC']) for v in ranking],
        scope='Select one fixed validation winner for separately authorized Test420; B baseline registration is unchanged',
        provenance=provenance)
