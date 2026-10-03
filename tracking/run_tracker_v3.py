"""Run V3 causally on one video or image directory, with an explicit initial box."""
from pathlib import Path
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import shutil
import sys
import time

import cv2
import numpy as np

if __package__:
    from .recovery_controller import Config, clip_box
else:
    from recovery_controller import Config, clip_box

HERE = Path(__file__).resolve().parent
SEND = Path('/home/xu/zyx/send')
LOCAL_ROOT = SEND / 'LOCAL_TRACKER_SELECTION_R1_20261002'
DETECTOR_ROOT = SEND / 'MVLM_V2_GLOBAL_PROPOSAL_BACKBONE_R1_20261001'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def load_config(path):
    document = json.loads(Path(path).read_text())
    if document.get('version') != 'lorat_yolo_framework_v3':
        raise ValueError('Expected a framework V3 config')
    return document, Config(**document['policy'])


def image_paths(directory):
    directory = Path(directory).resolve()
    if (directory/'img').is_dir():directory /= 'img'
    def key(p):return (0, int(p.stem), p.name) if p.stem.isdigit() else (1, p.name)
    paths = sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}), key=key)
    if not paths:raise ValueError(f'No frames found: {directory}')
    return paths


def frame_stream(args):
    """Decode one frame only when the consumer requests the next result."""
    if args.frames:
        paths = image_paths(args.frames)
        for i, path in enumerate(paths, 1):
            if args.max_frames and i > args.max_frames:break
            bgr = cv2.imread(str(path))
            if bgr is None:raise RuntimeError(f'Cannot decode frame {path}')
            yield i, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), str(path)
    else:
        capture = cv2.VideoCapture(str(args.video.resolve()))
        if not capture.isOpened():raise RuntimeError(f'Cannot open video: {args.video}')
        try:
            i = 0
            while args.max_frames is None or i < args.max_frames:
                ok, bgr = capture.read()
                if not ok:break
                i += 1
                yield i, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), f'{args.video.resolve()}#frame={i}'
        finally:
            capture.release()


def freeze_dependencies(config_path):
    paths = [HERE/n for n in ['run_tracker_v3.py','recovery_controller.py','causal_tracker_v3.py',
             'lorat_yolo.py','online_tracker.py','yolo_online_worker.py','test420_pipeline.py']]
    paths += [Path(config_path), LOCAL_ROOT/'scripts/adapters.py', DETECTOR_ROOT/'scripts/yolo26_adapter.py',
              DETECTOR_ROOT/'training/yolo26_uav/weights/best.pt',
              SEND/'LORAT_YOLO_GEOMETRY_R2_20261002/scripts/policy.py']
    inherited = json.loads((LOCAL_ROOT/'provenance/FREEZE.json').read_text())['sha256']
    frozen = {}
    for path, digest in inherited.items():
        if '/external/LoRAT/' in path or path.endswith(('dinov2_vitl14_pretrain.pth','LoRAT_large-378.bin')):
            actual = sha(path)
            if actual != digest:raise RuntimeError(f'Frozen LoRAT dependency changed: {path}')
            frozen[path] = actual
    for path in paths:frozen[str(path.resolve())] = sha(path)
    return frozen


def prepare(args):
    document, policy = load_config(args.config)
    if args.max_frames is not None and args.max_frames < 1:raise ValueError('--max-frames must be positive')
    if args.frames:
        paths = image_paths(args.frames)
        if args.max_frames:paths = paths[:args.max_frames]
        h = hashlib.sha256()
        for p in paths:
            stat = p.stat();h.update(f'{p}\t{stat.st_size}\t{stat.st_mtime_ns}\n'.encode())
        source = dict(type='frames', path=str(args.frames.resolve()), frames=len(paths), size_mtime_inventory_sha256=h.hexdigest())
    else:
        source = dict(type='video', path=str(args.video.resolve()), sha256=sha(args.video))
    stream = frame_stream(args)
    try:
        _, rgb, _ = next(stream)
    except StopIteration:
        raise ValueError('Input has no decodable frames')
    finally:
        stream.close()
    clipped = clip_box(args.init_box, rgb.shape)
    if not np.allclose(args.init_box, clipped, atol=1e-8, rtol=0):raise ValueError('Initialization box must be inside the first frame')
    config = dict(framework=document, source=source, initial_bbox=list(args.init_box), max_frames=args.max_frames,
                  image_shape=list(rgb.shape), execution='strict current-frame LoRAT -> YOLO -> verification -> output/commit',
                  detector_precomputation=False, detector_cache_input=False, future_gt_input=False,
                  unknown_box_export=[0.,0.,0.,0.], unknown_is_not_physical_absence=True)
    frozen = freeze_dependencies(args.config)
    for name, value in [('CONFIG.json', config), ('FREEZE.json', frozen)]:
        path = args.output/name
        if path.exists() and json.loads(path.read_text()) != value:
            raise RuntimeError(f'Inputs/config/code changed after preflight: {path}; choose a new output directory')
        save(path, value)
    snapshots = {}
    for source_path, digest in frozen.items():
        if Path(source_path).suffix not in {'.py','.json','.yaml','.yml'}:
            continue
        snapshot = args.output/'code_snapshot'/(hashlib.sha256(source_path.encode()).hexdigest()[:12]+'_'+Path(source_path).name)
        snapshot.parent.mkdir(exist_ok=True)
        if not snapshot.exists():shutil.copy2(source_path, snapshot)
        if sha(snapshot) != digest:raise RuntimeError(f'Corrupt source snapshot: {snapshot}')
        snapshots[source_path] = dict(snapshot=str(snapshot.relative_to(args.output)), sha256=digest)
    save(args.output/'SOURCE_SNAPSHOT.json', snapshots)
    save(args.output/'PREFLIGHT.json', dict(status='PASS', policy_validation='engineering_defaults_only', **source))
    return policy


def run(args, config):
    if __package__:
        from .causal_tracker_v3 import build_tracker
    else:
        from causal_tracker_v3 import build_tracker
    trace_path = args.output/'frames.partial.jsonl';pred_path = args.output/'predictions.partial.txt'
    if any((args.output/p).exists() for p in ['frames.partial.jsonl','frames.jsonl','predictions.partial.txt','predictions.txt','SUMMARY.json']):
        raise RuntimeError('An execution already exists; choose a fresh output directory. Partial runs are not silently resumed.')
    with build_tracker(args.output/'MODEL.json', config) as tracker:
        import torch
        save(args.output/'ENVIRONMENT.json', dict(python=sys.version, torch=torch.__version__, opencv=cv2.__version__,
             gpu=torch.cuda.get_device_name(0), detector=tracker.detector.environment))
        modes = Counter();reasons = Counter();nulls = probes = detector_calls = 0
        started = time.perf_counter();frames = 0
        with trace_path.open('x') as tf, pred_path.open('x') as pf:
            try:
                for i, rgb, source in frame_stream(args):
                    result = tracker.initialize(rgb, args.init_box) if i == 1 else tracker.track(rgb)
                    result['source'] = source
                    tf.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n');tf.flush()
                    bbox = result['bbox'] if result['bbox'] is not None else [0.,0.,0.,0.]
                    pf.write(','.join(f'{v:.9f}' for v in bbox)+'\n');pf.flush()
                    frames = i;modes[result['state']] += 1;reasons[result['reason']] += 1
                    nulls += result['bbox'] is None;probes += result['probe_count'];detector_calls += i > 1
                    if i == 1 or i % 25 == 0 or result['previous_state'] != result['state']:
                        print('FRAMEWORK_V3',i,result['state'],result['reason'],'probes',result['probe_count'],flush=True)
            except BaseException as exc:
                save(args.output/'INTERRUPTED.json', dict(completed_frames=frames,error=type(exc).__name__,message=str(exc)))
                raise
        if frames == 0:raise RuntimeError('No frames processed')
        elapsed = time.perf_counter() - started
        trace_path.replace(args.output/'frames.jsonl');pred_path.replace(args.output/'predictions.txt')
        summary = dict(status='COMPLETE',scope='single_sequence_execution_not_accuracy_evaluation',frames=frames,
            unconfirmed_frames=nulls, unconfirmed_fraction=nulls/frames, states=dict(modes), reasons=dict(reasons),
            candidate_probes=probes, detector_requests=detector_calls, loop_seconds=elapsed, loop_fps=frames/elapsed,
            timing='Includes frame reading, initialization, IPC, proposals, commits and output writing; excludes model loading.',
            predictions_sha256=sha(args.output/'predictions.txt'),traces_sha256=sha(args.output/'frames.jsonl'))
        save(args.output/'SUMMARY.json', summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', type=Path);source.add_argument('--frames', type=Path)
    parser.add_argument('--init-box', nargs=4, type=float, required=True, metavar=('X','Y','W','H'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=HERE/'framework_v3.json')
    parser.add_argument('--max-frames', type=int);parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args();args.output = args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.run.lock').open('a') as lock:
        try:fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another process is using this output directory')
        policy = prepare(args)
        if args.dry_run:
            print('FRAMEWORK_V3_PREFLIGHT_COMPLETE: no model inference started.',flush=True)
        else:
            run(args, policy)


if __name__ == '__main__':main()
