"""Complete Train-development validation. This entry point has no Test option."""
import argparse,csv,fcntl,time,uuid
from collections import Counter
from types import SimpleNamespace
import torch
from .common import *
from .tracker_c import build
from tracking.run_tracker_v3 import frame_stream
from tracking.test420_v4 import score_arrays,METRIC_KEYS
from tracking.test420_pipeline import TOOLKIT
from tracking.recovery_controller import iou

def check_done(folder,case,limit):
    path=folder/case['sequence'];marker=path/'COMPLETE.json'
    if not marker.exists():return None
    done=read(marker);assert done['frames']==min(case['frames'],limit or case['frames'])
    assert sha(path/'predictions.txt')==done['prediction_sha256']
    assert sha(path/'frames.jsonl')==done['trace_sha256']
    return done

def infer_sequence(tracker,folder,case,index,total,limit):
    target=folder/case['sequence']
    if target.exists():
        archive=folder/'interrupted_attempts'/case['sequence']/uuid.uuid4().hex
        archive.parent.mkdir(parents=True,exist_ok=True);target.replace(archive)
    target.mkdir();counts=Counter();online=0.;started=time.perf_counter()
    stream=frame_stream(SimpleNamespace(video=Path(case['video']),frames=None,max_frames=limit))
    try:
        with (target/'predictions.txt').open('x',buffering=1) as pf,(target/'frames.jsonl').open('x',buffering=1) as tf:
            for frame,rgb,source in stream:
                if frame==1:
                    result=tracker.initialize(rgb,case['initial_bbox'])
                    z=tracker.core.native.adapter.z.clone();mask=tracker.core.native.adapter.mask.clone()
                else:result=tracker.track(rgb)
                assert result['frame']==frame and result['rgb_sha256']==hashlib.sha256(rgb.tobytes()).hexdigest()
                assert result.get('reference_checks',0)==0 and not result['physical_absence_inferred']
                box=[0.,0.,0.,0.] if result['bbox'] is None else result['bbox']
                assert len(box)==4 and np.isfinite(box).all() and (result['bbox'] is None or min(box[2:])>0)
                pf.write(','.join(f'{float(v):.9f}' for v in box)+'\n');tf.write(json.dumps(result,allow_nan=False)+'\n')
                counts['frames']+=1;counts['detector_requests']+=frame>1
                counts['no_output_frames']+=result['bbox'] is None
                counts['candidate_probes']+=result['probe_count'];counts['recovery_events']+=result['reason']=='RECOVERY_CONFIRMED'
                counts['tentative_output_frames']+=result['localization_status']=='tentative'
                online+=result['online_seconds']
                if frame%200==0:
                    progress=dict(sequence_index=index,sequences=total,sequence=case['sequence'],frame=frame,frames=case['frames'])
                    save(folder/'PROGRESS.json',progress);print('BC_VAL',json.dumps(progress),flush=True)
    finally:stream.close()
    assert counts['frames']==min(case['frames'],limit or case['frames'])
    assert torch.equal(z,tracker.core.native.adapter.z) and torch.equal(mask,tracker.core.native.adapter.mask)
    done=dict(**counts,online_seconds=online,loop_seconds=time.perf_counter()-started,initial_template_preserved=True,
        prediction_sha256=sha(target/'predictions.txt'),trace_sha256=sha(target/'frames.jsonl'))
    save(target/'COMPLETE.json',done);print('BC_VAL_DONE',index,total,case['sequence'],dict(counts),flush=True)
    return done

def diagnostics(boxes,gt,absent,traces):
    vis=(absent==0)&(gt[:,2:]>0).all(1);vis[0]=False
    overlaps=np.array([iou(b,g) if v and min(b[2:])>0 else 0. for b,g,v in zip(boxes,gt,vis)])
    hit=vis&(overlaps>=.5);out=Counter(visible_frames=int(vis.sum()),visible_hit50_frames=int(hit.sum()),
        severe_offset_frames=int((vis&(overlaps<.1)).sum()))
    size=np.sqrt(np.maximum(gt[:,2]*gt[:,3],0.))
    for name,subset in [('tiny_le8',size<=8),('small_8_16',(size>8)&(size<=16)),('larger_gt16',size>16)]:
        out[name+'_frames']=int((vis&subset).sum());out[name+'_hit50_frames']=int((hit&subset).sum())
    # Each annotated absent-to-visible transition defines one reappearance episode.
    starts=np.flatnonzero(vis[1:]&~vis[:-1])+1
    for start in starts:
        if start==1 and absent[0]==0:continue
        end=start
        while end<len(vis) and vis[end]:end+=1
        out['reappearance_events']+=1
        found=any(hit[j:j+3].all() for j in range(start,end-2))
        out['reappearance_no_three_hits']+=not found
    for i,row in enumerate(traces):
        if not vis[i]:continue
        pool=row.get('comparison_pool',[])
        poolhit=any(iou(p['box'],gt[i])>=.5 for p in pool)
        out['visible_pool_frames']+=bool(pool);out['pool_hit50_frames']+=poolhit
        comp=row.get('comparison')
        if comp:
            unmatched=comp['selected']==comp['unmatched_index']
            out['unmatched_with_hit50_candidate']+=unmatched and poolhit
            out['selected_wrong_with_hit50_candidate']+=not unmatched and poolhit and iou(pool[comp['selected']]['box'],gt[i])<.5
        if row.get('commit_source')=='global_verification' and row['state_committed']:
            out['global_commits']+=1;out['global_commit_hit50']+=overlaps[i]>=.5
    return {k:int(v) for k,v in out.items()}

def score(folder,cases,variant,limit):
    sys.path.insert(0,str(TOOLKIT))
    from got10k.experiments import ExperimentUAVAntiUAV_Overall
    evaluator=ExperimentUAVAntiUAV_Overall(str(DATA),'All','UAVAntiUAV',result_dir=str(folder),
        report_dir=str(folder/'metrics/mmot'),report_scenario_dir=str(folder/'metrics/scenarios'))
    spec=importlib.util.spec_from_file_location('bc_accuracy',TOOLKIT/'accuracy/accuracy.py')
    accuracy=importlib.util.module_from_spec(spec);spec.loader.exec_module(accuracy)
    rows=[];total_diag=Counter()
    for case in cases:
        done=check_done(folder,case,limit);assert done
        n=done['frames'];target=folder/case['sequence']
        boxes=np.loadtxt(target/'predictions.txt',delimiter=',').reshape(-1,4)
        gt=np.loadtxt(DATA/case['sequence']/'groundtruth_rect.txt',delimiter=',').reshape(-1,4)[:n]
        absent=np.loadtxt(DATA/case['sequence']/'absent.txt').reshape(-1)[:n]
        assert len(boxes)==len(gt)==len(absent)==n and np.allclose(boxes[0],case['initial_bbox'],rtol=0,atol=1e-8)
        traces=[json.loads(line) for line in (target/'frames.jsonl').open()]
        diag=diagnostics(boxes,gt,absent,traces);total_diag.update(diag)
        rows.append(dict(sequence=case['sequence'],**score_arrays(evaluator,accuracy,boxes,gt,absent),
            **{k:v for k,v in done.items() if k not in ('prediction_sha256','trace_sha256','initial_template_preserved')},**{'diag_'+k:v for k,v in diag.items()}))
    def macro(k):
        v=[r[k] for r in rows if r[k] is not None];return float(np.mean(v)) if v else None
    frames=sum(r['frames'] for r in rows);online=sum(r['online_seconds'] for r in rows)
    result=dict(scope='SMOKE_NOT_ACCURACY' if limit else 'TRAIN_DEVELOPMENT_VAL132',variant=variant,sequences=len(rows),frames=frames,
        sequence_macro={k:macro(k) for k in METRIC_KEYS},
        sequence_macro_null_as_localization_failure={k:macro('null_failure_'+k) for k in METRIC_KEYS[:5]},
        diagnostics=dict(total_diag),online_tracker_fps=(frames-len(rows))/online if online else None,
        sequence_loop_fps=frames/sum(r['loop_seconds'] for r in rows),
        counts={k:sum(r[k] for r in rows) for k in ['detector_requests','no_output_frames','candidate_probes','recovery_events','tentative_output_frames']},
        metric_definition='Installed official MMOT functions used in prior R2/V3/V4 runs. Primary curves include GT-initialized first frame; visible diagnostics exclude it. nPrecision_AUC is primary NP.',
        speed_definition='Synchronous current RGB detector IPC and all candidate/head computation included; model startup and image reading excluded from online FPS. Loop FPS includes reads and output writes.',
        reverse_verification=False,detector_precomputation=False,physical_absence_classifier=False,
        protocol='Fixed first-frame template, no later GT passed to tracker. Development validation already used for detector selection; not a blind test.')
    save(folder/'metrics/SUMMARY.json',result);save(folder/'metrics/SEQUENCES.json',rows)
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (folder/'metrics/SEQUENCES.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    save(folder/'COMPLETE.json',dict(summary_sha256=sha(folder/'metrics/SUMMARY.json'),sequences_sha256=sha(folder/'metrics/SEQUENCES.json'),sequences=len(cases),frames=frames))
    print('BC_VAL_COMPLETE',json.dumps(result),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['A','B','B_POOL','C'],required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    setup();protocol=read(RUN/'PROTOCOL.json');cases=[c for c in protocol['cases'] if c['split']=='val']
    assert len(cases)==132 and not any('Test' in c['sequence'] for c in cases)
    limit=8 if args.smoke else None
    if args.smoke:cases=cases[:1]
    folder=RUN/('validation_smoke' if args.smoke else 'validation')/args.variant;folder.mkdir(parents=True,exist_ok=True)
    b=RUN/'B/final.safetensors';c=RUN/('C_smoke' if args.smoke else 'C')/'final.safetensors'
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        files=[Path(__file__),Path(__file__).with_name('tracker_c.py'),Path(__file__).with_name('model_c.py'),Path(__file__).with_name('common.py'),RUN/'PROTOCOL.json']
        files+=list(PROJECT.glob('tracking/*.py'))+[TOOLKIT/'accuracy/accuracy.py']+list((TOOLKIT/'got10k').rglob('*.py'))
        if args.variant!='A':assert sha(b)==read(RUN/'B/COMPLETE.json')['sha256'];files.append(b)
        if args.variant=='C':assert sha(c)==read(c.parent/'COMPLETE.json')['checkpoint_sha256'];files.append(c)
        frozen=freeze_stage(str(folder.relative_to(RUN)),files);verify(read(RUN/'INPUT_SHA256.json'))
        pending=[]
        for index,case in enumerate(cases,1):
            s=Path(case['video']).stat();assert s.st_size==case['video_size'] and s.st_mtime_ns==case['video_mtime_ns']
            if not check_done(folder,case,limit):pending.append((index,case))
        if pending:
            with build(args.variant,folder/'MODEL.json',checkpoint_b=b,checkpoint_c=c) as tracker:
                save(folder/'ENVIRONMENT.json',dict(python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                    detector=tracker.detector.environment,tf32=torch.backends.cuda.matmul.allow_tf32))
                for index,case in pending:infer_sequence(tracker,folder,case,index,len(cases),limit)
        verify(frozen);verify(read(RUN/'INPUT_SHA256.json'));score(folder,cases,args.variant,limit)
        if args.variant=='B_POOL':
            other=folder.parent/'B'
            assert (other/'COMPLETE.json').exists(),'Evaluate B before B_POOL for the mandatory output-parity control'
            for case in cases:
                assert sha(folder/case['sequence']/'predictions.txt')==sha(other/case['sequence']/'predictions.txt'),case['sequence']
            save(folder/'PARITY_B.json',dict(status='PASS',sequences=len(cases),exact_prediction_equality=True))
if __name__=='__main__':main()
