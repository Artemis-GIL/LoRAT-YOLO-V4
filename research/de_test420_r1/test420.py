"""Frozen selected B/D0/D/E Test420 inference and official-function scoring.

The parent worker freezes inputs and the validation-only selection before calling
run_test(). This file never selects a model, trains one, or reads later GT into
the tracker. The image inventory is the existing complete V4 Test420 inventory;
no V4 controller/configuration is inherited.
"""
from collections import Counter
from pathlib import Path
import csv
import fcntl
import hashlib
import importlib.util
import json
import sys
import time
import uuid

import numpy as np

from tracking.test420_pipeline import TOOLKIT, image_at, load_array
from tracking.test420_v4 import METRIC_KEYS, score_arrays
from research.bc_r1.evaluate import diagnostics

MODE='lorat_validation_selected_de_current_frame'
SEQUENCES=420
FRAMES=476221


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def check(condition,message):
    if not condition:raise RuntimeError(message)


def verify_mapping(mapping):
    check(isinstance(mapping,dict) and bool(mapping),'Empty or invalid frozen manifest')
    for source,digest in mapping.items():
        path=Path(source)
        check(path.is_absolute() and path.is_file() and sha(path)==digest,f'Frozen input changed: {source}')


def verify_inputs(parent):
    """Verify frozen manifests, all image metadata, and complete Test inventory."""
    parent=Path(parent).resolve();frozen=read(parent/'FREEZE.json');verify_mapping(frozen)
    preparation_path=parent/'PREPARATION.json'
    check(str(preparation_path) in frozen and sha(preparation_path)==frozen[str(preparation_path)],
        'PREPARATION.json is not covered by the parent freeze')
    check(str(Path(__file__).resolve()) in frozen,'Test runner is not covered by the parent freeze')
    preparation=read(preparation_path)
    check(preparation['status']=='PASS','Test preparation did not pass')
    check(preparation['sequences']==SEQUENCES and preparation['frames']==FRAMES,'Preparation is not complete Test420')
    inputs=Path(preparation['inputs_path']);config_path=Path(preparation['source_config_path'])
    check(sha(inputs)==preparation['inputs_sha256'],'Test frame manifest digest changed')
    check(sha(config_path)==preparation['source_config_sha256'],'Source Test inventory configuration digest changed')
    config=read(config_path);rows=read(inputs);dataset=Path(preparation['dataset_root']).resolve()
    check(dataset==Path(config['dataset_root']).resolve(),'Dataset root differs from the source Test manifest')
    check(config.get('partial') is False and config.get('max_frames') is None,'Source inventory is partial')
    expected_names=[f'UAV-Anti-UAV_Test_{i:06d}' for i in range(1,SEQUENCES+1)]
    check(len(rows)==SEQUENCES and [r['sequence'] for r in rows]==expected_names,'Test sequence inventory changed')
    check(config['sequences']==expected_names,'Source configuration has incompatible sequence names')
    check(sum(len(r['frames']) for r in rows)==FRAMES,'Test frame count changed')
    inventory=hashlib.sha256()
    for row in rows:
        name=row['sequence'];box=np.asarray(row['initial_bbox'],float)
        check(box.shape==(4,) and np.isfinite(box).all() and (box[2:]>0).all(),f'Invalid initialization: {name}')
        check(len(row['frames'])>1,f'Empty or single-frame Test sequence: {name}')
        for index,source in enumerate(row['frames'],1):
            path=Path(source)
            check(path.is_absolute() and path.is_file() and name in path.parts and int(path.stem)==index,
                f'Non-contiguous or missing source image: {source}')
            stat=path.stat();inventory.update(f'{path}\t{stat.st_size}\t{stat.st_mtime_ns}\n'.encode())
    digest=inventory.hexdigest()
    check(digest==preparation['frame_inventory_sha256']==config['frame_inventory_sha256'],
        'Image path/size/mtime inventory changed')
    return rows,dataset


def verify_selection(selection,parent):
    """Check the supplied selected arm and its completed validation provenance."""
    from research.de_r1.common import B_WEIGHTS
    parent=Path(parent).resolve();selection_path=parent/'SELECTION.json'
    check(read(selection_path)==selection,'Selected model differs from the frozen selection record')
    variant=selection['variant'];check(variant in ('B','D0','D','E'),'Unsupported selected arm')
    baseline=Path(selection['baseline_checkpoint']).resolve()
    check(baseline==B_WEIGHTS.resolve(),'Tracker baseline path differs from the selection record')
    check(sha(baseline)==selection['baseline_sha256'],'Selected baseline checkpoint changed')
    checkpoint=selection['checkpoint']
    if variant=='B':
        check(checkpoint is None and selection['checkpoint_sha256'] is None,'B must have no extension checkpoint')
    else:
        check(checkpoint is not None and sha(checkpoint)==selection['checkpoint_sha256'],'Selected extension checkpoint changed')
    verify_mapping(selection['provenance'])
    return sha(selection_path)


def completed(out,row,selection_sha256):
    """Return a verified completion marker, rejecting mixed-arm/corrupt resumes."""
    out=Path(out);name=row['sequence'];folder=out/'traces'/name;marker=folder/'COMPLETE.json'
    if not marker.exists():return None
    done=read(marker)
    check(done['mode']==MODE and done['selection_sha256']==selection_sha256,f'Incompatible completed selection: {name}')
    check(done['frames']==len(row['frames']) and done['initial_template_preserved'],f'Incomplete sequence marker: {name}')
    check(done['detector_requests']==len(row['frames'])-1 and done['reference_checks']==0,f'Incompatible inference protocol: {name}')
    for path,key in [(out/'predictions'/f'{name}.txt','prediction_sha256'),(folder/'frames.jsonl','trace_sha256')]:
        check(path.is_file() and sha(path)==done[key],f'Completed sequence artifact changed: {path}')
    return done


def archive_unfinished(out,row):
    out=Path(out);name=row['sequence'];folder=out/'traces'/name
    paths=[folder,out/'predictions'/f'{name}.txt',out/'predictions'/f'{name}.partial.txt']
    paths=[path for path in paths if path.exists()]
    if not paths:return
    archive=out/'interrupted_attempts'/name/uuid.uuid4().hex;archive.mkdir(parents=True)
    for path in paths:path.replace(archive/path.name)
    print('DE_TEST_RESTART_SEQUENCE',name,str(archive),flush=True)


def infer_sequence(tracker,out,row,index,total,selection_sha256):
    """One synchronous image loop; torch.equal supports real and CPU fake cores."""
    import torch
    out=Path(out);name=row['sequence']
    check(not (out/'traces'/name/'COMPLETE.json').exists(),f'Will not overwrite completed sequence: {name}')
    archive_unfinished(out,row)
    folder=out/'traces'/name;folder.mkdir(parents=True,exist_ok=True);(out/'predictions').mkdir(parents=True,exist_ok=True)
    pred=out/'predictions'/f'{name}.txt';trace=folder/'frames.jsonl'
    partial_pred=pred.with_suffix('.partial.txt');partial_trace=folder/'frames.partial.jsonl'
    counters=Counter(frames=0,detector_requests=0,no_output_frames=0,candidate_probes=0,
        recovery_events=0,tentative_output_frames=0,commit_deferred_frames=0,reference_checks=0)
    states=Counter();reasons=Counter();actions=Counter();online_seconds=detector_seconds=0.;started=time.perf_counter()
    try:
        with partial_pred.open('x',buffering=1) as pf,partial_trace.open('x',buffering=1) as tf:
            for frame,path in enumerate(row['frames'],1):
                frame_start=time.perf_counter();_,rgb,digest=image_at(path)
                if frame==1:
                    result=tracker.initialize(rgb,row['initial_bbox'])
                    initial_template=tracker.core.native.adapter.z.clone();initial_mask=tracker.core.native.adapter.mask.clone()
                else:result=tracker.track(rgb)
                check(result['frame']==frame and result['rgb_sha256']==digest,f'Current-RGB alignment failed: {name} {frame}')
                check(result.get('reference_checks',0)==0,f'Reverse verification unexpectedly ran: {name} {frame}')
                check(result.get('physical_absence_inferred') is False,f'Unexpected absence classification: {name} {frame}')
                box=[0.,0.,0.,0.] if result['bbox'] is None else result['bbox']
                check(len(box)==4 and np.isfinite(box).all() and (result['bbox'] is None or min(box[2:])>0),f'Invalid box: {name} {frame}')
                check(np.isfinite(result['online_seconds']) and result['online_seconds']>=0,f'Invalid online time: {name} {frame}')
                result['frame_wall_seconds']=time.perf_counter()-frame_start
                pf.write(','.join(f'{float(v):.9f}' for v in box)+'\n')
                tf.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
                counters['frames']+=1;counters['detector_requests']+=frame>1
                counters['no_output_frames']+=result['bbox'] is None;counters['candidate_probes']+=result['probe_count']
                counters['recovery_events']+=result['reason']=='RECOVERY_CONFIRMED'
                counters['tentative_output_frames']+=result['localization_status']=='tentative'
                counters['commit_deferred_frames']+=not result['state_committed'];counters['reference_checks']+=result.get('reference_checks',0)
                states[result['state']]+=1;reasons[result['reason']]+=1
                if 'd_action' in result:actions[result['d_action']]+=1
                online_seconds+=result['online_seconds'];detector_seconds+=result['detector_seconds']
                if frame==1 or frame%200==0 or frame==len(row['frames']):
                    progress=dict(scope='TEST420',variant_selection_sha256=selection_sha256,sequence_index=index,sequences=total,
                        sequence=name,frame=frame,frames=len(row['frames']),updated_unix=time.time())
                    save(out/'PROGRESS.json',progress)
                    print('DE_TEST_PROGRESS',index,total,name,frame,len(row['frames']),result['state'],flush=True)
        check(counters['frames']==len(row['frames']),f'Frame count mismatch: {name}')
        check(torch.equal(initial_template,tracker.core.native.adapter.z) and torch.equal(initial_mask,tracker.core.native.adapter.mask),
            f'First-frame template changed: {name}')
        partial_pred.replace(pred);partial_trace.replace(trace)
        done=dict(mode=MODE,selection_sha256=selection_sha256,**counters,initial_template_preserved=True,
            prediction_sha256=sha(pred),trace_sha256=sha(trace),online_seconds=online_seconds,
            detector_seconds=detector_seconds,loop_seconds=time.perf_counter()-started,states=dict(states),reasons=dict(reasons),d_actions=dict(actions))
        save(folder/'COMPLETE.json',done)
        print('DE_TEST_DONE',index,total,name,counters['frames'],flush=True)
        return done
    except BaseException as exc:
        save(folder/'INTERRUPTED.json',dict(completed_frames=counters['frames'],error=type(exc).__name__,message=str(exc),selection_sha256=selection_sha256))
        raise


def score_test(out,rows,dataset,selection,selection_sha256):
    """Load post-initialization Test GT only after every inference sequence ends."""
    out=Path(out);dataset=Path(dataset)
    check(len(rows)==SEQUENCES and sum(len(r['frames']) for r in rows)==FRAMES,'Refuse partial Test scoring')
    completions=[completed(out,row,selection_sha256) for row in rows]
    check(all(item is not None for item in completions),'All Test sequences must finish before scoring')
    sys.path.insert(0,str(TOOLKIT))
    from got10k.experiments import ExperimentUAVAntiUAV_Overall
    evaluator=ExperimentUAVAntiUAV_Overall(str(dataset),'All','UAVAntiUAV',result_dir=str(out),
        report_dir=str(out/'metrics/mmot'),report_scenario_dir=str(out/'metrics/scenarios'))
    spec=importlib.util.spec_from_file_location('selected_de_official_accuracy',TOOLKIT/'accuracy/accuracy.py')
    accuracy=importlib.util.module_from_spec(spec);spec.loader.exec_module(accuracy)
    metrics=[];all_diagnostics=Counter();all_actions=Counter()
    for row,done in zip(rows,completions):
        name=row['sequence'];count=len(row['frames'])
        boxes=load_array(out/'predictions'/f'{name}.txt').reshape(-1,4)
        gt=load_array(dataset/name/'groundtruth_rect.txt').reshape(-1,4)
        absent=load_array(dataset/name/'absent.txt').reshape(-1)
        check(len(boxes)==len(gt)==len(absent)==count,f'Scoring length mismatch: {name}')
        check(np.isfinite(boxes).all() and np.isfinite(gt).all() and np.isfinite(absent).all(),f'Nonfinite arrays: {name}')
        check(np.allclose(boxes[0],row['initial_bbox'],rtol=0,atol=1e-8),f'Wrong initialization: {name}')
        check(np.allclose(gt[0],row['initial_bbox'],rtol=0,atol=1e-8),f'Initial Test annotation differs from frozen input: {name}')
        with (out/'traces'/name/'frames.jsonl').open() as stream:
            traces=[json.loads(line) for line in stream]
        check(len(traces)==count and all(r['frame']==i for i,r in enumerate(traces,1)),f'Trace length/index mismatch: {name}')
        diag=diagnostics(boxes,gt,absent,traces);all_diagnostics.update(diag);all_actions.update(done.get('d_actions',{}))
        metrics.append(dict(sequence=name,frames=count,**score_arrays(evaluator,accuracy,boxes,gt,absent),
            **{k:done[k] for k in ['no_output_frames','candidate_probes','recovery_events','tentative_output_frames',
                'commit_deferred_frames','reference_checks','detector_requests','online_seconds','detector_seconds','loop_seconds']},
            **{'diag_'+k:v for k,v in diag.items()}))
    def macro(key):
        values=[row[key] for row in metrics if row[key] is not None]
        return float(np.mean(values)) if values else None
    total_frames=sum(r['frames'] for r in metrics);online=sum(r['online_seconds'] for r in metrics);loop=sum(r['loop_seconds'] for r in metrics)
    check(total_frames==FRAMES and len(metrics)==SEQUENCES,'Scoring did not cover Test420')
    check(online>0 and loop>0,'Invalid aggregate timing')
    no_output=sum(r['no_output_frames'] for r in metrics)
    summary=dict(scope='TEST420',mode=MODE,variant=selection['variant'],selected_variant=selection['variant'],
        checkpoint=selection['checkpoint'],checkpoint_sha256=selection['checkpoint_sha256'],
        baseline_checkpoint=selection['baseline_checkpoint'],baseline_sha256=selection['baseline_sha256'],
        selection_record=str(out.parent/'SELECTION.json'),selection_sha256=selection_sha256,
        selection_rule=selection['selection_rule'],validation_auc=selection['validation_auc'],validation_np_auc=selection['validation_np_auc'],
        sequences=SEQUENCES,frames=FRAMES,sequence_macro={k:macro(k) for k in METRIC_KEYS},
        sequence_macro_null_as_localization_failure={k:macro('null_failure_'+k) for k in METRIC_KEYS[:5]},
        diagnostics=dict(all_diagnostics),d_actions=dict(all_actions),no_output_frames=no_output,no_output_fraction=no_output/FRAMES,
        visible_no_output_frames=sum(r['visible_no_output_frames'] for r in metrics),
        counts={k:sum(r[k] for r in metrics) for k in ['detector_requests','candidate_probes','recovery_events','tentative_output_frames','commit_deferred_frames','reference_checks']},
        online_tracker_fps=(FRAMES-SEQUENCES)/online,sequence_loop_fps=FRAMES/loop,
        metric_definition='Installed official MMOT toolkit functions; primary curves include GT-initialized first frames; visible diagnostics exclude first frames. nPrecision_AUC is the primary NP.',
        unknown_metric_definition='Null exports as [0,0,0,0]. Additional null-as-failure curves force null center errors to infinity. mACC uses the unchanged four-value export and awards no physical-absence credit for null.',
        speed_definition='online_tracker_fps includes synchronous current-RGB YOLO IPC, LoRAT candidates, chosen-arm computation and commit; excludes image I/O and startup. sequence_loop_fps includes frame reading, initialization and output writes, excluding startup/final scoring.',
        inference_protocol='GT-initialized single-object tracking; fixed first-frame template; no subsequent GT passed to tracker',
        detector_precomputation=False,reverse_verification=False,physical_absence_classifier=False,
        automatic_candidate_recovery=True,output_semantics='Tentative output is not physical absence; missing output alone exports four zeros.',
        recovery_event_semantics='Confirmed candidate or default recovery; not a guarantee of target identity.',
        comparison_diagnostic_semantics='For D0/D, comparison.selected is a head suggestion; KEEP_B_WHILE_ALTERNATIVE_CONFIRMS retains the default output. Suggested-choice error is not necessarily executed takeover error.',
        model_selection_uses_test_metrics=False,automatic_promotion=False)
    save(out/'metrics/SUMMARY.json',summary);save(out/'metrics/SEQUENCES.json',metrics)
    keys=sorted(set().union(*(row.keys() for row in metrics)))
    with (out/'metrics/SEQUENCES.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(metrics)
    return summary


def run_test(selection:dict,out:Path)->None:
    """Run exactly the frozen validation-selected arm on complete Test420."""
    out=Path(out).resolve();parent=out.parent
    check(out.name=='test420','Use the dedicated test420 output directory')
    rows,dataset=verify_inputs(parent);selection_sha256=verify_selection(selection,parent)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=dict(read(parent/'FREEZE.json'))
        for source,digest in selection['provenance'].items():
            check(source not in frozen or frozen[source]==digest,f'Selection provenance conflicts with preparation freeze: {source}')
            frozen[source]=digest
        frozen[str(parent/'SELECTION.json')]=selection_sha256
        frozen[str(Path(selection['baseline_checkpoint']).resolve())]=selection['baseline_sha256']
        if selection['checkpoint'] is not None:
            frozen[str(Path(selection['checkpoint']).resolve())]=selection['checkpoint_sha256']
        verify_mapping(frozen)
        if (out/'FREEZE.json').exists():check(read(out/'FREEZE.json')==frozen,'Existing selected-model freeze differs')
        else:save(out/'FREEZE.json',frozen)
        config=dict(scope='TEST420',mode=MODE,selection=selection,selection_sha256=selection_sha256,dataset_root=str(dataset),
            inputs_manifest=read(parent/'PREPARATION.json')['inputs_path'],preparation_sha256=sha(parent/'PREPARATION.json'),
            sequences=SEQUENCES,frames=FRAMES,initialization='first-frame GT only; fixed template',
            execution='current RGB -> local LoRAT -> synchronous current-frame YOLO -> selected-arm decision/commit -> next RGB',
            reverse_verification=False,detector_precomputation=False,post_initialization_gt_input=False,
            resume='verify completed sequences and archive/restart incomplete sequences from frame one')
        if (out/'CONFIG.json').exists():check(read(out/'CONFIG.json')==config,'Existing Test run configuration differs')
        else:save(out/'CONFIG.json',config)
        pending=[]
        for index,row in enumerate(rows,1):
            if completed(out,row,selection_sha256) is None:pending.append((index,row))
        if pending:
            from research.de_r1.tracker_de import build
            import torch
            with build(selection['variant'],out/'MODEL.json',checkpoint=selection['checkpoint']) as tracker:
                save(out/'ENVIRONMENT.json',dict(python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                    detector=tracker.detector.environment,tf32=torch.backends.cuda.matmul.allow_tf32,
                    transport='synchronous current RGB bytes only',variant=selection['variant']))
                for index,row in pending:infer_sequence(tracker,out,row,index,SEQUENCES,selection_sha256)
        # Freeze and image inventory are verified again before loading GT for scoring.
        checked_rows,checked_dataset=verify_inputs(parent)
        check(checked_rows==rows and checked_dataset==dataset,'Test inputs changed during inference')
        check(verify_selection(selection,parent)==selection_sha256,'Selection record changed during inference')
        check(read(out/'FREEZE.json')==frozen,'Selected-model freeze changed during inference');verify_mapping(frozen)
        summary=score_test(out,rows,dataset,selection,selection_sha256)
        verify_mapping(read(parent/'FREEZE.json'))
        check(verify_selection(selection,parent)==selection_sha256,'Selection changed during final scoring')
        check(read(out/'FREEZE.json')==frozen,'Selected-model freeze changed during scoring');verify_mapping(frozen)
        save(out/'COMPLETE.json',dict(status='PASS',scope='TEST420',mode=MODE,variant=selection['variant'],
            sequences=SEQUENCES,frames=FRAMES,selection_sha256=selection_sha256,
            baseline_sha256=selection['baseline_sha256'],checkpoint_sha256=selection['checkpoint_sha256'],
            summary_sha256=sha(out/'metrics/SUMMARY.json'),sequences_sha256=sha(out/'metrics/SEQUENCES.json'),
            reverse_verification=False,detector_precomputation=False,initial_templates_preserved=True))
        print('ALL_SELECTED_DE_TEST420_COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)
