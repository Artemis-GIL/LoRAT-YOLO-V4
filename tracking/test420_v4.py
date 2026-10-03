"""Frozen V4 Test420 run: one current-frame loop, sequence resume, then scoring."""
from collections import Counter
from pathlib import Path
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import shutil
import sys
import time
import uuid

import numpy as np

if __package__:
    from . import test420_pipeline as common
    from .run_tracker_v4 import freeze_dependencies, load_config
else:
    import test420_pipeline as common
    from run_tracker_v4 import freeze_dependencies, load_config

HERE = Path(__file__).resolve().parent
MODE = 'lorat_yolo_v4_current_frame'
METRIC_KEYS = ['AUC', 'Precision20', 'cAUC', 'nPrecision_at_0_2', 'nPrecision_AUC',
               'mACC', 'visible_mean_iou', 'visible_hit50']


def prepare(args):
    document, policy = load_config(args.config)
    common.check(not (args.output/'detections').exists(), 'Use a new V4 output directory; cached detections are not accepted.')
    dataset = args.dataset_root.resolve()
    names = [args.sequence] if args.sequence else sorted(p.name for p in dataset.glob('UAV-Anti-UAV_Test_*') if p.is_dir())
    common.check(args.sequence is not None or len(names) == 420, f'Expected 420 Test sequences; found {len(names)}')
    inventory = hashlib.sha256(); rows = []; annotations = {}
    for name in names:
        common.check(Path(name).name == name and name.startswith('UAV-Anti-UAV_Test_'), 'Invalid sequence name')
        folder = args.frames_root.resolve()/name
        images = sorted((folder/'img').glob('*.jpg')) if (folder/'img').is_dir() else sorted(folder.glob('*.png'))
        ann = dataset/name/'groundtruth_rect.txt'
        with ann.open() as stream:
            first = stream.readline().strip()
            count = 1 + sum(bool(line.strip()) for line in stream)
        box = np.asarray([float(v) for v in first.replace(',', ' ').split()])
        common.check(box.shape == (4,) and np.isfinite(box).all() and (box[2:] > 0).all(), f'Invalid initialization: {name}')
        common.check(len(images) == count and count > 1, f'Frame/annotation count mismatch: {name}')
        common.check([int(p.stem) for p in images] == list(range(1, count+1)), f'Non-contiguous frames: {name}')
        if args.max_frames: images = images[:args.max_frames]
        for path in images:
            stat = path.stat()
            inventory.update(f'{path}\t{stat.st_size}\t{stat.st_mtime_ns}\n'.encode())
        rows.append(dict(sequence=name, initial_bbox=box.tolist(), frames=[str(p) for p in images]))
        for path in [ann, dataset/name/'absent.txt']: annotations[str(path)] = common.sha(path)
    config = dict(mode=MODE, framework=document, dataset_root=str(dataset), frames_root=str(args.frames_root.resolve()),
        sequences=names, max_frames=args.max_frames, partial=bool(args.sequence or args.max_frames),
        initialization='first-frame GT box; fixed template', frame_inventory_sha256=inventory.hexdigest(),
        execution='current RGB -> local LoRAT -> current-frame YOLO -> candidate verification -> output/commit -> next RGB',
        detector_precomputation=False, detector_cache_input=False, future_gt_input=False,
        detector=dict(imgsz=1280, conf=.001, iou=.60, max_det=12),
        unknown_box_export=[0., 0., 0., 0.], unknown_is_not_physical_absence=True,
        policy_selection='one fixed existing V4 configuration; no Test threshold sweep',
        resume='verify completed sequences; archive and restart unfinished sequence from first frame')
    frozen = freeze_dependencies(args.config)
    deps = [HERE/'test420_v4.py', HERE/'run_test420_v4.sh', common.TOOLKIT/'accuracy/accuracy.py']
    deps += list((common.TOOLKIT/'got10k').rglob('*.py'))
    frozen.update({str(p.resolve()): common.sha(p) for p in deps}); frozen.update(annotations)
    manifests = [('CONFIG.json', config), ('INPUTS.json', rows), ('FREEZE.json', frozen)]
    # Validate every existing manifest before writing anything used for resume.
    for name, value in manifests:
        path = args.output/name
        if path.exists():
            common.check(common.read(path) == value, f'Inputs/config/code changed: {path}; choose a new output directory.')
        else:
            common.check(not (args.output/'predictions').exists(), f'Missing manifest in an existing run: {path}')
    for name, value in manifests:
        if not (args.output/name).exists(): common.save(args.output/name, value)
    snapshots = {}
    for source, digest in frozen.items():
        if Path(source).suffix not in {'.py', '.sh', '.json', '.yaml', '.yml'}: continue
        target = args.output/'code_snapshot'/(hashlib.sha256(source.encode()).hexdigest()[:12]+'_'+Path(source).name)
        target.parent.mkdir(exist_ok=True)
        if not target.exists(): shutil.copy2(source, target)
        common.check(common.sha(target) == digest, f'Corrupt source snapshot: {target}')
        snapshots[source] = dict(snapshot=str(target.relative_to(args.output)), sha256=digest)
    common.save(args.output/'SOURCE_SNAPSHOT.json', snapshots)
    common.save(args.output/'PREFLIGHT.json', dict(status='PASS', mode=MODE, sequences=len(rows),
        frames=sum(len(r['frames']) for r in rows), partial=config['partial'], no_model_inference=True))
    print('V4_PREPARED', len(rows), 'sequences', sum(len(r['frames']) for r in rows), 'frames',
          'PARTIAL' if config['partial'] else 'TEST420', flush=True)
    return rows, policy


def completed(out, row):
    folder = out/'traces'/row['sequence']; done = folder/'COMPLETE.json'
    if not done.exists(): return None
    result = common.read(done)
    common.check(result['mode'] == MODE and result['frames'] == len(row['frames']), f'Incompatible completed sequence: {row["sequence"]}')
    for path, key in [(out/'predictions'/f'{row["sequence"]}.txt', 'prediction_sha256'), (folder/'frames.jsonl', 'trace_sha256')]:
        common.check(path.exists() and common.sha(path) == result[key], f'Corrupt completed result: {path}')
    return result


def archive_unfinished(out, row):
    name = row['sequence']; folder = out/'traces'/name
    paths = [out/'predictions'/f'{name}.txt', out/'predictions'/f'{name}.partial.txt',
             folder/'frames.jsonl', folder/'frames.partial.jsonl', folder/'INTERRUPTED.json']
    paths = [p for p in paths if p.exists()]
    if paths:
        target = out/'interrupted_attempts'/name/uuid.uuid4().hex
        target.mkdir(parents=True)
        for path in paths: path.replace(target/path.name)
        print('V4_RESTART_SEQUENCE', name, 'previous_attempt', str(target), flush=True)


def run_sequence(tracker, torch, out, row, index, total):
    name = row['sequence']; folder = out/'traces'/name
    archive_unfinished(out, row)
    folder.mkdir(parents=True, exist_ok=True); (out/'predictions').mkdir(exist_ok=True)
    pred = out/'predictions'/f'{name}.txt'; trace = folder/'frames.jsonl'
    pp = pred.with_suffix('.partial.txt'); tp = folder/'frames.partial.jsonl'
    states = Counter(); reasons = Counter(); frames = nulls = probes = recoveries = tentative = deferred = reference_checks = 0
    online_seconds = detector_seconds = 0.; started = time.perf_counter()
    try:
        with pp.open('x') as pf, tp.open('x') as tf:
            for i, path in enumerate(row['frames'], 1):
                frame_start = time.perf_counter()
                _, rgb, digest = common.image_at(path)
                if i == 1:
                    result = tracker.initialize(rgb, row['initial_bbox'])
                    template = tracker.core.native.adapter.z.clone(); mask = tracker.core.native.adapter.mask.clone()
                else:
                    result = tracker.track(rgb)
                common.check(result['frame'] == i and result['rgb_sha256'] == digest, f'Current-frame alignment failed: {name} {i}')
                box = [0., 0., 0., 0.] if result['bbox'] is None else result['bbox']
                common.check(len(box) == 4 and np.isfinite(box).all() and (result['bbox'] is None or min(box[2:]) > 0), f'Invalid box: {name} {i}')
                result['frame_wall_seconds'] = time.perf_counter()-frame_start
                pf.write(','.join(f'{v:.9f}' for v in box)+'\n'); pf.flush()
                tf.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n'); tf.flush()
                frames = i; states[result['state']] += 1; reasons[result['reason']] += 1
                nulls += result['bbox'] is None; probes += result['probe_count']
                tentative += result['localization_status'] == 'tentative'
                deferred += not result['state_committed']; reference_checks += result.get('reference_checks',0)
                recoveries += result['reason'] == 'RECOVERY_CONFIRMED'
                online_seconds += result['online_seconds']; detector_seconds += result['detector_seconds']
                if i == 1 or i % 200 == 0 or i == len(row['frames']):
                    print('V4_TRACK_PROGRESS', index, total, name, i, len(row['frames']), result['state'], flush=True)
        common.check(torch.equal(template, tracker.core.native.adapter.z) and torch.equal(mask, tracker.core.native.adapter.mask), 'Initial template changed')
        loop_seconds = time.perf_counter()-started
        pp.replace(pred); tp.replace(trace)
        common.save(folder/'COMPLETE.json', dict(mode=MODE, frames=frames, prediction_sha256=common.sha(pred),
            trace_sha256=common.sha(trace), initial_template_preserved=True, online_seconds=online_seconds,
            detector_seconds=detector_seconds, loop_seconds=loop_seconds, detector_requests=frames-1,
            no_output_frames=nulls, candidate_probes=probes, recovery_events=recoveries,
            tentative_output_frames=tentative, commit_deferred_frames=deferred, reference_checks=reference_checks,
            states=dict(states), reasons=dict(reasons)))
        print('V4_TRACK_DONE', index, total, name, frames, 'no_output', nulls, flush=True)
    except BaseException as exc:
        common.save(folder/'INTERRUPTED.json', dict(completed_frames=frames, error=type(exc).__name__, message=str(exc)))
        raise


def run(args, rows, policy):
    pending = []
    for index, row in enumerate(rows, 1):
        if completed(args.output, row): print('V4_TRACK_SKIP', index, len(rows), row['sequence'], flush=True)
        else: pending.append((index, row))
    if not pending: return
    if __package__:
        from .causal_tracker_v4 import build_tracker
    else:
        from causal_tracker_v4 import build_tracker
    with build_tracker(args.output/'MODEL.json', policy) as tracker:
        import torch
        common.save(args.output/'ENVIRONMENT.json', dict(python=sys.version, torch=torch.__version__,
            opencv=common.cv2.__version__, gpu=torch.cuda.get_device_name(0), detector=tracker.detector.environment,
            tf32=False, transport='synchronous current RGB bytes only'))
        for index, row in pending:
            common.setup_torch()
            run_sequence(tracker, torch, args.output, row, index, len(rows))


def curve_values(curves):
    return dict(AUC=float(curves[0].mean()), Precision20=float(curves[1][20]), cAUC=float(curves[2].mean()),
                nPrecision_at_0_2=float(curves[3][20]), nPrecision_AUC=float(curves[3].mean()))


def score_arrays(evaluator, accuracy, boxes, gt, absent):
    # The primary metrics retain the exact R2 toolkit functions and zero-box
    # export. Also report strict localization diagnostics: null has no center.
    metrics = evaluator._calc_metrics(boxes.copy(), gt.copy())
    result = curve_values(evaluator._calc_curves(*metrics))
    result['mACC'] = float(accuracy.eval(boxes.tolist(), gt.tolist(), np.any(gt != 0, axis=1)))
    visible = (absent == 0) & (gt[:, 2:] > 0).all(1); visible[0] = False
    null = (boxes == 0).all(1)
    result.update(visible_frames=int(visible.sum()), visible_no_output_frames=int((visible & null).sum()),
        visible_mean_iou=float(metrics[0][visible].mean()) if visible.any() else None,
        visible_hit50=float((metrics[0][visible] >= .5).mean()) if visible.any() else None)
    strict = [np.array(x, dtype=float, copy=True) for x in metrics]
    strict[0][null] = 0.; strict[1][null] = np.inf; strict[2][null] = 0.; strict[3][null] = np.inf
    result.update({'null_failure_'+key: value for key, value in curve_values(evaluator._calc_curves(*strict)).items()})
    return result


def evaluate(args, rows):
    # All sequences must be complete before any post-initialization GT is scored.
    complete = []
    for row in rows:
        done = completed(args.output, row)
        common.check(done is not None, f'Sequence is incomplete: {row["sequence"]}')
        complete.append(done)
    common.verify(args.output)
    config = common.read(args.output/'CONFIG.json')
    sys.path.insert(0, str(common.TOOLKIT))
    from got10k.experiments import ExperimentUAVAntiUAV_Overall
    evaluator = ExperimentUAVAntiUAV_Overall(config['dataset_root'], 'All', 'UAVAntiUAV',
        result_dir=str(args.output), report_dir=str(args.output/'metrics/mmot'), report_scenario_dir=str(args.output/'metrics/scenarios'))
    spec = importlib.util.spec_from_file_location('uav_v4_official_accuracy', common.TOOLKIT/'accuracy/accuracy.py')
    accuracy = importlib.util.module_from_spec(spec); spec.loader.exec_module(accuracy)
    scored = []
    for row, done in zip(rows, complete):
        name = row['sequence']; count = len(row['frames'])
        boxes = common.load_array(args.output/'predictions'/f'{name}.txt').reshape(-1, 4)
        gt = common.load_array(Path(config['dataset_root'])/name/'groundtruth_rect.txt').reshape(-1, 4)[:count]
        absent = common.load_array(Path(config['dataset_root'])/name/'absent.txt').reshape(-1)[:count]
        common.check(len(boxes) == len(gt) == len(absent) == count, f'Scoring length mismatch: {name}')
        common.check(np.allclose(boxes[0], row['initial_bbox'], rtol=0, atol=1e-8), f'Wrong initialization: {name}')
        scored.append(dict(sequence=name, frames=count, **score_arrays(evaluator, accuracy, boxes, gt, absent),
            no_output_frames=done['no_output_frames'], candidate_probes=done['candidate_probes'], recovery_events=done['recovery_events'],
            tentative_output_frames=done['tentative_output_frames'], commit_deferred_frames=done['commit_deferred_frames'], reference_checks=done['reference_checks'],
            online_seconds=done['online_seconds'], detector_seconds=done['detector_seconds'], loop_seconds=done['loop_seconds']))
    def macro(key):
        values = [r[key] for r in scored if r[key] is not None]
        return float(np.mean(values)) if values else None
    total_frames = sum(r['frames'] for r in scored)
    no_output = sum(r['no_output_frames'] for r in scored)
    summary = dict(scope='PARTIAL_SMOKE' if config['partial'] else 'TEST420', mode=MODE, sequences=len(scored), frames=total_frames,
        sequence_macro={key: macro(key) for key in METRIC_KEYS},
        sequence_macro_null_as_localization_failure={key: macro('null_failure_'+key) for key in METRIC_KEYS[:5]},
        no_output_frames=no_output, no_output_fraction=no_output/total_frames,
        visible_no_output_frames=sum(r['visible_no_output_frames'] for r in scored),
        candidate_probes=sum(r['candidate_probes'] for r in scored), recovery_events=sum(r['recovery_events'] for r in scored),
        detector_requests=sum(r['detector_requests'] for r in complete),
        tentative_output_frames=sum(r['tentative_output_frames'] for r in scored),
        commit_deferred_frames=sum(r['commit_deferred_frames'] for r in scored),
        reference_checks=sum(r['reference_checks'] for r in scored),
        online_tracker_fps=(total_frames-len(rows))/sum(r['online_seconds'] for r in scored),
        sequence_loop_fps=total_frames/sum(r['loop_seconds'] for r in scored),
        metric_definition='Primary sequence_macro uses the same installed MMOT functions as R2, including GT-initialized first frames. Visible diagnostics exclude first frames.',
        unknown_metric_definition='Null exports as [0,0,0,0]. Additional null-as-failure curves force null center errors to infinity. mACC uses the unchanged four-value export and awards no physical-absence credit for null.',
        speed_definition='online_tracker_fps includes synchronous detector IPC, candidate probes and commit, excludes image I/O. sequence_loop_fps includes frame reading, initialization and per-frame file writes; excludes model startup and final scoring.',
        inference_protocol='GT-initialized single-object tracking; fixed template; no subsequent GT input',
        detector_precomputation=False, automatic_candidate_recovery=True,
        recovery_event_semantics='RECOVERY_CONFIRMED means the shared reference protocol accepted a candidate; it does not guarantee identity.',
        output_semantics='Tentative current localization remains output when trusted-state commit is deferred; no_output counts only missing boxes.',
        physical_absence_classifier=False, fixed_policy=config['framework'])
    common.save(args.output/'metrics/SUMMARY.json', summary)
    with (args.output/'metrics/SEQUENCES.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scored[0])); writer.writeheader(); writer.writerows(scored)
    common.save(args.output/'COMPLETE.json', dict(mode=MODE, sequences=len(scored), frames=total_frames,
        summary_sha256=common.sha(args.output/'metrics/SUMMARY.json')))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=HERE/'framework_v4.json')
    parser.add_argument('--dataset-root', type=Path, default=Path('/home/xu/zyx/GOT_REMOTE/datasets/Test'))
    parser.add_argument('--frames-root', type=Path, default=Path('/home/xu/zyx/GOT_REMOTE/datasets/UAV-Anti-UAV_LaSOT_LINUX_REEXTRACT_20260911'))
    parser.add_argument('--sequence'); parser.add_argument('--max-frames', type=int)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--dry-run', action='store_true'); actions.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args(); args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    common.check(args.max_frames is None or args.max_frames >= 2, '--max-frames must be at least 2')
    with (args.output/'.run.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another V4 process is using this output directory')
        rows, policy = prepare(args)
        if args.dry_run:
            print('V4_PREFLIGHT_COMPLETE: no model inference started.', flush=True)
            return
        if not args.evaluate_only: run(args, rows, policy)
        evaluate(args, rows)
        print('ALL_V4_COMPLETE:', str(args.output/'metrics/SUMMARY.json'), flush=True)


if __name__ == '__main__': main()
