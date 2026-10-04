"""Five fixed online structural variants on 32 complete predetermined Train videos."""
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import argparse, csv, fcntl, gc, hashlib, json, shutil, sys, time
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from tracking.structural_ablations import VARIANTS, build_tracker
from tracking.run_tracker_v3 import frame_stream, save, sha
from tracking.run_tracker_v4 import freeze_dependencies, load_config
from tracking.test420_pipeline import setup_torch, TOOLKIT
from tracking.recovery_controller import RecoveryController

DATA=Path('/home/xu/zyx/GOT_REMOTE/datasets/Train')
OLD_PROTOCOL=PROJECT/'analysis/LORAT_YOLO_V4_STRUCTURAL_20261003/PROTOCOL.json'
OLD_PREFIX=Path('/home/xu/zyx/send/LORAT_YOLO_V4_STRUCTURAL_20261003/train32')
DEFAULT_OUTPUT=Path('/home/xu/zyx/send/LORAT_YOLO_STRUCTURAL_ABLATION_R1_20261003')

def read(p):return json.loads(Path(p).read_text())
def same_or_save(p,value):
    if p.exists():
        if read(p)!=value:raise RuntimeError(f'Frozen run inputs or code changed: {p}')
    else:save(p,value)
def verify(mapping):
    for path,digest in mapping.items():
        if sha(path)!=digest:raise RuntimeError(f'Frozen dependency changed: {path}')
def spans(mask):
    diff=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(diff==1),np.flatnonzero(diff==-1)))
def csvout(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def prepare(out):
    document,cfg=load_config(PROJECT/'tracking/framework_v4.json')
    assert document['policy']==read(PROJECT/'tracking/framework_v3.json')['policy']
    old=read(OLD_PROTOCOL);cases=[];inputs={str(OLD_PROTOCOL):sha(OLD_PROTOCOL)}
    for c in old['cases']:
        folder=DATA/c['sequence'];gt=folder/'groundtruth_rect.txt';video=Path(c['video'])
        with gt.open() as f:
            first=f.readline().strip();n=1+sum(bool(line.strip()) for line in f)
        initial=[float(x) for x in first.replace(',',' ').split()]
        assert initial==c['initial_bbox'] and video.is_file()
        cases.append(dict(sequence=c['sequence'],video=str(video),initial_bbox=initial,frames=n))
        for p in [gt,video,folder/'absent.txt']:inputs[str(p)]=sha(p)
    prior_hashes=read(OLD_PREFIX/'RESULT_SHA256.json')
    inputs[str(OLD_PREFIX/'RESULT_SHA256.json')]=sha(OLD_PREFIX/'RESULT_SHA256.json')
    for path,digest in prior_hashes.items():
        assert sha(path)==digest,path
        inputs[path]=digest
    assert len(cases)==32 and sum(x['frames'] for x in cases)==10244
    protocol=dict(version='structural_ablation_r1',scope='32 complete predetermined Train videos; development diagnostic, not independent held-out or Test performance',
        selection='Same 32 sequence identities from the earlier fixed SHA256 selection; all annotated frames, no outcome-based selection',
        source_protocol=str(OLD_PROTOCOL),cases=cases,variants=VARIANTS,execution_order=list(VARIANTS),policy=document['policy'],
        reverse_off='Only disable the reverse reference assessment; retain numerical validity and unchanged forward score minimum',
        competition_off='Only remove a verified rival veto when previous mode is TRACKING and local check passes. Candidate computation, association and recovery competition are retained.',
        output_only='Run the original V3 controller and native tracker; fill only null output with usable current local proposal. No state/commit changes and no candidate fallback.',
        inference='One current RGB -> local -> current-frame YOLO -> candidate checks -> output/commit/write -> next RGB',
        detector_precomputation=False,future_gt_input=False,new_training=False,threshold_sweep=False,full_test_run=False,
        labels='Only first-frame coordinates parsed in preparation; subsequent GT coordinates and absence labels parsed after all five inference variants complete',
        limitation='These 32 Train prefixes have previously been inspected; full-sequence results are development evidence and do not establish generalization.')
    frozen=freeze_dependencies(PROJECT/'tracking/framework_v4.json')
    for p in [Path(__file__).resolve(),PROJECT/'tracking/structural_ablations.py',PROJECT/'tracking/run_structural_ablation.sh',
              PROJECT/'tracking/tests/test_structural_ablations.py',PROJECT/'tracking/framework_v3.json',
              PROJECT/'tracking/run_tracker_v3.py',PROJECT/'tracking/test420_v4.py',TOOLKIT/'accuracy/accuracy.py']:
        frozen[str(p)]=sha(p)
    for p in (TOOLKIT/'got10k').rglob('*.py'):frozen[str(p)]=sha(p)
    for name,data in [('PROTOCOL.json',protocol),('INPUT_SHA256.json',inputs),('FREEZE.json',frozen)]:same_or_save(out/name,data)
    snapshots={}
    for path,digest in frozen.items():
        if Path(path).suffix not in {'.py','.sh','.json','.yaml','.yml'}:continue
        target=out/'code_snapshot'/(hashlib.sha256(path.encode()).hexdigest()[:12]+'_'+Path(path).name)
        target.parent.mkdir(exist_ok=True)
        if not target.exists():shutil.copy2(path,target)
        assert sha(target)==digest
        snapshots[path]=dict(snapshot=str(target.relative_to(out)),sha256=digest)
    same_or_save(out/'SOURCE_SNAPSHOT.json',snapshots)
    save(out/'PREFLIGHT.json',dict(status='PASS',sequences=32,unique_frames=10244,variants=5,inference_frames=51220,no_model_inference=True))
    return protocol,cfg,frozen,inputs

def completed(folder,case):
    p=folder/'COMPLETE.json'
    if not p.exists():return None
    d=read(p);assert d['sequence']==case['sequence'] and d['frames']==case['frames']
    for name,key in [('frames.jsonl','trace_sha256'),('predictions.txt','prediction_sha256')]:
        if sha(folder/name)!=d[key]:raise RuntimeError(f'Corrupt completed artifact: {folder/name}')
    return d

def run_case(out,tracker,variant,case,vi,si):
    import torch
    folder=out/variant/case['sequence']
    if folder.exists():
        if completed(folder,case):return read(folder/'COMPLETE.json')
        archive=out/'interrupted'/f'{variant}_{case["sequence"]}_{time.time_ns()}'
        archive.parent.mkdir(exist_ok=True);folder.rename(archive)
    folder.mkdir(parents=True)
    setup_torch();args=SimpleNamespace(video=Path(case['video']),frames=None,max_frames=None)
    stream=frame_stream(args);frames=0;counts=Counter();modes=Counter();seconds=detector_seconds=0.;started=time.perf_counter()
    try:
        with (folder/'frames.partial.jsonl').open('x') as tf,(folder/'predictions.partial.txt').open('x') as pf:
            for i,rgb,source in stream:
                if i>case['frames']:raise RuntimeError('Video has more frames than annotations')
                if i==1:
                    result=tracker.initialize(rgb,case['initial_bbox'])
                    template=tracker.core.native.adapter.z.clone();mask=tracker.core.native.adapter.mask.clone()
                else:result=tracker.track(rgb)
                assert result['frame']==i
                if not result['state_committed']:assert result['committed_bbox']==result['search_anchor_before']
                if not VARIANTS[variant]['reverse']:assert result.get('reference_checks',0)==0
                result.update(ablation_variant=variant,image_shape=list(rgb.shape),source=source)
                tf.write(json.dumps(result,allow_nan=False)+'\n');tf.flush()
                pf.write(','.join(f'{x:.9f}' for x in (result['bbox'] or [0.,0.,0.,0.]))+'\n');pf.flush()
                frames=i;modes[result['state']]+=1
                counts.update(no_output_frames=int(result['bbox'] is None),tentative_output_frames=int(result['localization_status']=='tentative'),
                    commit_deferred_frames=int(not result['state_committed']),reference_checks=result.get('reference_checks',0),candidate_probes=result['probe_count'],
                    recovery_events=int(result['reason']=='RECOVERY_CONFIRMED'),detector_requests=int(i>1),tracking_veto_bypasses=int(result.get('tracking_competition_bypassed',False)))
                seconds+=result['online_seconds'];detector_seconds+=result['detector_seconds']
                if i==1 or i%200==0 or i==case['frames']:
                    save(out/'PROGRESS.json',dict(variant=variant,variant_index=vi,variants=5,sequence=case['sequence'],sequence_index=si,sequences=32,frame=i,frames=case['frames']))
                    print('ABLATION_PROGRESS',variant,si,32,case['sequence'],i,case['frames'],flush=True)
        assert frames==case['frames']
        assert torch.equal(template,tracker.core.native.adapter.z) and torch.equal(mask,tracker.core.native.adapter.mask)
        if hasattr(tracker.core,'reference_digest'):assert hashlib.sha256(tracker.core.reference_rgb.tobytes()).hexdigest()==tracker.core.reference_digest
        (folder/'frames.partial.jsonl').rename(folder/'frames.jsonl');(folder/'predictions.partial.txt').rename(folder/'predictions.txt')
        done=dict(variant=variant,sequence=case['sequence'],frames=frames,**counts,states=dict(modes),online_seconds=seconds,
            detector_seconds=detector_seconds,loop_seconds=time.perf_counter()-started,initial_template_preserved=True,
            trace_sha256=sha(folder/'frames.jsonl'),prediction_sha256=sha(folder/'predictions.txt'))
        save(folder/'COMPLETE.json',done);print('ABLATION_DONE',variant,si,32,case['sequence'],frames,flush=True)
        return done
    except BaseException as exc:
        save(folder/'INTERRUPTED.json',dict(frame=frames,error=type(exc).__name__,message=str(exc)));raise
    finally:stream.close()

def execute(out,protocol,cfg):
    summaries=[]
    for vi,variant in enumerate(VARIANTS,1):
        target=out/variant;target.mkdir(exist_ok=True)
        pending=[]
        for si,case in enumerate(protocol['cases'],1):
            d=completed(target/case['sequence'],case)
            if d:summaries.append(d)
            else:pending.append((si,case))
        if not pending:continue
        with build_tracker(variant,target/'MODEL.json',cfg) as tracker:
            import torch,cv2
            save(target/'ENVIRONMENT.json',dict(python=sys.version,torch=torch.__version__,opencv=cv2.__version__,
                gpu=torch.cuda.get_device_name(0),detector=tracker.detector.environment))
            for si,case in pending:summaries.append(run_case(out,tracker,variant,case,vi,si))
        del tracker;gc.collect();torch.cuda.empty_cache()
    save(out/'EXECUTION.json',summaries)
    save(out/'INFERENCE_COMPLETE.json',dict(variants=5,sequences_per_variant=32,unique_frames=10244,inference_frames=sum(x['frames'] for x in summaries)))

def controller_parity(trace,cfg):
    controller=RecoveryController(cfg);dummy=np.empty(trace[0]['image_shape'],dtype=np.uint8)
    count=0
    for t in trace[1:]:
        probes=iter(t['verification_branches'])
        def probe(image,anchor):
            b=next(probes);assert np.allclose(anchor,b['probe_anchor'],atol=1e-8,rtol=0)
            return b['proposal']
        r=controller.decide(dummy,t['local_proposal'],t['raw_candidates'],t['frame'],probe)
        assert next(probes,None) is None
        for k in ['proposed_commit','state','reason','local_health','verification_branches','local_confirmation_streak','probe_count']:
            assert r[k]==t[k],(t['frame'],k)
        assert r['bbox']==t['v3_original_bbox'];count+=1
    return count

def evaluate(out,protocol,cfg):
    # All variants must finish before post-initialization coordinates are parsed.
    assert (out/'INFERENCE_COMPLETE.json').exists()
    for variant in VARIANTS:
        for case in protocol['cases']:assert completed(out/variant/case['sequence'],case)
    from tracking.test420_v4 import score_arrays
    import importlib.util
    sys.path.insert(0,str(TOOLKIT))
    from got10k.experiments import ExperimentUAVAntiUAV_Overall
    evaluator=ExperimentUAVAntiUAV_Overall(str(DATA),'All','UAVAntiUAV',result_dir=str(out),report_dir=str(out/'mmot'),report_scenario_dir=str(out/'mmot_scenarios'))
    spec=importlib.util.spec_from_file_location('ablation_accuracy',TOOLKIT/'accuracy/accuracy.py')
    accuracy=importlib.util.module_from_spec(spec);spec.loader.exec_module(accuracy)
    rows=[];recoveries=[];returns=[];parity=Counter();hashes={};diagnostics=defaultdict_counter()
    for case in protocol['cases']:
        name=case['sequence'];n=case['frames'];traces={};scores={}
        g=np.loadtxt(DATA/name/'groundtruth_rect.txt',delimiter=',').reshape(-1,4);a=np.loadtxt(DATA/name/'absent.txt').reshape(-1)
        assert len(g)==len(a)==n
        visible=(a==0)&(g[:,2:]>0).all(1);visible[0]=False
        for variant in VARIANTS:
            folder=out/variant/name;done=completed(folder,case)
            trace=[json.loads(x) for x in (folder/'frames.jsonl').open()];assert len(trace)==n;traces[variant]=trace
            p=np.loadtxt(folder/'predictions.txt',delimiter=',').reshape(-1,4)
            from tracking.recovery_controller import iou
            overlap=np.array([iou(b,gt) for b,gt in zip(p,g)]);scores[variant]=overlap
            result=score_arrays(evaluator,accuracy,p,g,a)
            row=dict(variant=variant,sequence=name,frames=n,**result,**{k:done[k] for k in ['no_output_frames','tentative_output_frames','commit_deferred_frames','reference_checks','candidate_probes','tracking_veto_bypasses','online_seconds','loop_seconds']})
            rows.append(row)
            hashes[str(folder/'frames.jsonl')]=done['trace_sha256'];hashes[str(folder/'predictions.txt')]=done['prediction_sha256']
            if variant=='v3_output_only':
                parity['full_v3_controller_frames']+=controller_parity(trace,cfg)
                original=np.array([t['v3_original_bbox'] or [0]*4 for t in trace],dtype=float)
                original_score=score_arrays(evaluator,accuracy,original,g,a)
                row['v3_original_same_execution_AUC']=original_score['AUC']
            else:row['v3_original_same_execution_AUC']=None
            if variant in ['v3_output_only','v4_full']:
                oldname='v3_original' if variant=='v3_output_only' else 'v4_shared_reference'
                oldpath=OLD_PREFIX/oldname/(name+'.jsonl');oldtrace=[json.loads(x) for x in oldpath.open()]
                assert len(oldtrace)==min(96,n)
                for prior,current in zip(oldtrace,trace):
                    for k in ['rgb_sha256','proposed_commit','committed_bbox','state','reason','local_proposal','verification_branches','raw_candidates','local_search_anchor','tentative_local_anchor']:
                        assert prior.get(k)==current.get(k),(variant,name,prior['frame'],k)
                    expected=current['v3_original_bbox'] if variant=='v3_output_only' else current['bbox']
                    assert prior['bbox']==expected
                parity[variant+'_prior_prefix_frames']+=len(oldtrace)
            for i,t in enumerate(trace):
                if i==0:continue
                if t['reason']=='RECOVERY_CONFIRMED':
                    e=min(n,i+10);full=e-i==10 and bool(visible[i:e].all())
                    recoveries.append(dict(variant=variant,sequence=name,frame=i+1,source=t.get('commit_source') or t['output_source'],visible=bool(visible[i]),
                        iou=float(overlap[i]),full10_visible=full,full10_all_hit=bool(full and (overlap[i:e]>=.5).all())))
                if not visible[i]:continue
                stat=diagnostics[variant];lp=t['local_proposal'];good=lp['box'] is not None and iou(lp['box'],g[i])>=.5;accepted=t['local_health']['accepted']
                stat.update(visible=1,local_hit=int(good),local_hit_rejected=int(good and not accepted),local_wrong_accepted=int(not good and accepted),
                    output_hit=int(overlap[i]>=.5),ambiguous=int('AMBIGUOUS' in t['reason']),tentative_failure=int(t['localization_status']=='tentative' and overlap[i]<.5))
                for b in t['verification_branches']:
                    if b.get('proposal_reused_from_local',False):continue
                    goodb=b['proposal']['box'] is not None and iou(b['proposal']['box'],g[i])>=.5
                    stat.update(global_probes=1,global_hit=int(goodb),global_passed=int(b['passed']),global_passed_hit=int(b['passed'] and goodb),global_hit_rejected=int(goodb and not b['passed']))
            runs=spans(visible&(overlap<.5));diagnostics[variant]['failure_frames_ge30']+=int(sum(b-s for s,b in runs if b-s>=30))
        base=traces['v3_output_only']
        for variant,trace in traces.items():
            if variant=='v3_output_only':continue
            assert all(x['rgb_sha256']==y['rgb_sha256'] for x,y in zip(base,trace));parity['matched_rgb_pairs']+=n
            assert all(x.get('raw_candidates')==y.get('raw_candidates') for x,y in zip(base,trace));parity['matched_raw_candidates']+=n-1
        for start,end in spans(a==0):
            if start==0:continue
            rr=dict(sequence=name,frame=int(start+1),visible_window=int(end-start))
            for variant,overlap in scores.items():
                stable=[s for s,e in spans(overlap[start:end]>=.5) if e-s>=3]
                rr[variant+'_stable3_delay']=int(stable[0]) if stable else None
            returns.append(rr)
    keys=['AUC','Precision20','cAUC','nPrecision_at_0_2','nPrecision_AUC','mACC','visible_mean_iou','visible_hit50']
    variants={}
    for variant in VARIANTS:
        subset=[r for r in rows if r['variant']==variant]
        variants[variant]=dict(sequence_macro={k:float(np.mean([r[k] for r in subset if r[k] is not None])) for k in keys},
            online_fps=sum(r['frames']-1 for r in subset)/sum(r['online_seconds'] for r in subset),
            **{k:int(sum(r[k] for r in subset)) for k in ['no_output_frames','tentative_output_frames','commit_deferred_frames','reference_checks','candidate_probes','tracking_veto_bypasses']},
            visible_recoveries=sum(x['variant']==variant and x['visible'] for x in recoveries),
            visible_recovery_hits=sum(x['variant']==variant and x['visible'] and x['iou']>=.5 for x in recoveries),
            reappearance_no_stable3=sum(x[variant+'_stable3_delay'] is None for x in returns),diagnostics=dict(diagnostics[variant]))
    csvout(out/'SEQUENCES.csv',rows)
    if recoveries:csvout(out/'RECOVERY_EVENTS.csv',recoveries)
    if returns:csvout(out/'REAPPEARANCE.csv',returns)
    save(out/'RESULT_SHA256.json',hashes)
    summary=dict(status='COMPLETE',scope=protocol['scope'],sequences=32,unique_frames=10244,inference_frames=51220,
        variants=variants,engineering_parity=dict(parity),reappearance_events=len(returns),
        original_v3_same_execution_auc=float(np.mean([r['v3_original_same_execution_AUC'] for r in rows if r['variant']=='v3_output_only'])),
        real_output_only_baseline=True,parameter_sweep=False,new_training=False,test_inference=False,
        limitation=protocol['limitation'])
    assert parity['full_v3_controller_frames']==10244-32
    assert parity['matched_rgb_pairs']==10244*4
    save(out/'SUMMARY.json',summary);save(out/'COMPLETE.json',dict(summary_sha256=sha(out/'SUMMARY.json'),scope=protocol['scope']))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

def defaultdict_counter():
    from collections import defaultdict
    return defaultdict(Counter)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--evaluate-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol,cfg,frozen,inputs=prepare(out)
        if args.dry_run:
            print('ABLATION_PREFLIGHT_PASS',32,10244,5,flush=True);return
        if not args.evaluate_only:execute(out,protocol,cfg)
        verify(frozen);verify(inputs);evaluate(out,protocol,cfg)
        print('ALL_ABLATIONS_COMPLETE:',out/'SUMMARY.json',flush=True)

if __name__=='__main__':main()
