"""Causal Train-development Val132 evaluation for independent B+D / B+E arms.

No Test entry point. Zero-effect controls must exactly reproduce frozen B.
"""
import argparse
import fcntl
from itertools import zip_longest
from .common import *
from research.bc_r1.evaluate import infer_sequence, check_done, score
from tracking.test420_pipeline import TOOLKIT

VARIANTS=('B','D_ZERO','D0','D','E_ZERO','E')


def checkpoint_for(variant,smoke):
    if variant not in ('D0','D','E'):return None
    folder=RUN/(variant+('_smoke' if smoke else ''))
    checkpoint=folder/'final.safetensors';done=read(folder/'COMPLETE.json')
    expected=done.get('checkpoint_sha256',done.get('sha256'))
    assert expected and sha(checkpoint)==expected,(variant,'checkpoint does not match completed training')
    return checkpoint


def verify_zero_parity(folder,cases,limit):
    reference=OLD/('validation_smoke' if limit else 'validation')/'B'
    pairs=[]
    state_fields=('state','previous_state','proposed_commit','commit_source',
        'search_anchor_before','local_search_anchor','committed_bbox','tentative_local_anchor',
        'localization_status','reason','output_source')
    for case in cases:
        name=case['sequence']
        current=check_done(folder,case,limit);baseline=check_done(reference,case,limit)
        assert current and baseline,(name,'missing completed zero-control or B sequence')
        assert current['prediction_sha256']==baseline['prediction_sha256'],(name,'zero control differs from frozen B')
        checked_frames=0
        with (folder/name/'frames.jsonl').open() as left,(reference/name/'frames.jsonl').open() as right:
            for frame,(line_a,line_b) in enumerate(zip_longest(left,right),1):
                assert line_a is not None and line_b is not None,(name,frame,'trace length differs')
                row_a=json.loads(line_a);row_b=json.loads(line_b)
                assert row_a['frame']==row_b['frame']==frame,(name,frame,'frame indices differ')
                assert row_a['rgb_sha256']==row_b['rgb_sha256'],(name,frame,'RGB digest differs')
                for key in state_fields:
                    assert row_a.get(key)==row_b.get(key),(name,frame,key,'zero-control state/anchor differs from B')
                checked_frames+=1
        assert checked_frames==current['frames']==baseline['frames'],(name,'trace frame count disagrees with completion')
        pairs.append(dict(sequence=name,frames=current['frames'],prediction_sha256=current['prediction_sha256']))
    result=dict(status='PASS',exact_prediction_equality=True,exact_state_and_anchor_equality=True,exact_RGB_equality=True,
        sequences=len(cases),frames=sum(x['frames'] for x in pairs),state_fields_checked=list(state_fields),
        reference=str(reference),comparison='Per-file prediction SHA256 plus per-frame exact controller state, search anchors and RGB digest equality; timing, probe counts and extra logs excluded',sequences_checked=pairs)
    save(folder/'PARITY_B.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=VARIANTS,required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    setup();verify_baseline();protocol=read(OLD/'PROTOCOL.json')
    cases=[c for c in protocol['cases'] if c['split']=='val']
    assert len(cases)==132 and sum(c['frames'] for c in cases)==47200
    assert not any('Test' in c['sequence'] for c in cases)
    limit=8 if args.smoke else None
    if args.smoke:cases=cases[:1]
    folder=RUN/('validation_smoke' if args.smoke else 'validation')/args.variant;folder.mkdir(parents=True,exist_ok=True)
    checkpoint=checkpoint_for(args.variant,args.smoke)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        files=list(Path(__file__).parent.glob('*.py'))+baseline_files()
        files+=[PROJECT/'research/bc_r1/evaluate.py',TOOLKIT/'accuracy/accuracy.py']+list((TOOLKIT/'got10k').rglob('*.py'))
        if (RUN/'PROTOCOL.json').exists():files.append(RUN/'PROTOCOL.json')
        if checkpoint:files.append(checkpoint)
        frozen=freeze_stage(str(folder.relative_to(RUN)),files);verify(read(OLD/'INPUT_SHA256.json'))
        pending=[]
        for index,case in enumerate(cases,1):
            stat=Path(case['video']).stat()
            assert stat.st_size==case['video_size'] and stat.st_mtime_ns==case['video_mtime_ns'],case['sequence']
            if not check_done(folder,case,limit):pending.append((index,case))
        if pending:
            from .tracker_de import build
            with build(args.variant,folder/'MODEL.json',checkpoint=checkpoint) as tracker:
                save(folder/'ENVIRONMENT.json',dict(python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                    detector=tracker.detector.environment,tf32=torch.backends.cuda.matmul.allow_tf32))
                for index,case in pending:infer_sequence(tracker,folder,case,index,len(cases),limit)
        verify(frozen);verify_baseline();verify(read(OLD/'INPUT_SHA256.json'))
        # Fail before score() creates a successful completion marker on parity errors.
        parity=verify_zero_parity(folder,cases,limit) if args.variant in ('D_ZERO','E_ZERO','B') else None
        score(folder,cases,args.variant,limit)
        summary_path=folder/'metrics/SUMMARY.json';summary=read(summary_path)
        summary.update(research_stage='INDEPENDENT_B_PLUS_D_AND_B_PLUS_E',
            baseline_checkpoint_sha256=sha(B_WEIGHTS),automatic_promotion=False,
            experiment_role={'B':'Frozen B reference',
                'D_ZERO':'Zero-effect temporal implementation control; must reproduce B exactly',
                'D0':'Static candidate scorer with matched training loss; control for D temporal contribution',
                'D':'Temporal candidate value experiment on existing geometric associations; compare with D0 as well as B',
                'E_ZERO':'Zero-effect spatial implementation control; must reproduce B exactly',
                'E':'Local spatial refinement experiment starting independently from frozen B'}[args.variant])
        if parity:summary['exact_prediction_parity_with_frozen_B']=True
        save(summary_path,summary)
        marker=dict(status='PASS',summary_sha256=sha(summary_path),sequences_sha256=sha(folder/'metrics/SEQUENCES.json'),
            sequences=len(cases),frames=sum(min(c['frames'],limit or c['frames']) for c in cases),
            scope='SMOKE_NOT_ACCURACY' if limit else 'TRAIN_DEVELOPMENT_VAL132',
            reverse_verification=False,detector_precomputation=False,automatic_promotion=False)
        if parity:marker['parity_sha256']=sha(folder/'PARITY_B.json')
        save(folder/'COMPLETE.json',marker)
        print('DE_VAL_COMPLETE',json.dumps(marker),flush=True)


if __name__=='__main__':main()
