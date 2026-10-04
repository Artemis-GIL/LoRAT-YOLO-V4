"""Run the tracker in one causal frame loop; no detection precomputation."""
from pathlib import Path
import argparse
import contextlib
import io
import json
import sys
import time

import numpy as np

import test420_pipeline as common

HERE = Path(__file__).resolve().parent


def prepare(args):
    common.check(not (args.output/'detections').exists(), 'Choose a new output directory for the online run; cached detection runs cannot be resumed here.')
    common.prepare(args)
    config = dict(mode='online_current_frame', execution='synchronous current RGB request; output and commit before reading next frame',
                  detector_precomputation=False, detector_cache_input=False, initial_template='fixed')
    freeze = {str(HERE/name):common.sha(HERE/name) for name in
              ['test420_online.py','online_tracker.py','yolo_online_worker.py','run_test420_online.sh']}
    for name,value in [('ONLINE_CONFIG.json',config),('ONLINE_FREEZE.json',freeze)]:
        path = args.output/name
        if path.exists():
            common.check(common.read(path) == value, f'Online code/config changed: {path}; use a new output directory')
        else:
            common.save(path,value)


def verify(args):
    rows = common.verify(args.output)
    for path,digest in common.read(args.output/'ONLINE_FREEZE.json').items():
        common.check(common.sha(path) == digest, f'Online source changed: {path}')
    return rows


def run(args):
    rows = verify(args)
    torch = common.setup_torch()
    from online_tracker import OnlineLoRATYOLO
    with OnlineLoRATYOLO(args.output/'MODEL.json') as tracker:
        common.save(args.output/'ONLINE_ENV.json',dict(python=sys.version,torch=torch.__version__,opencv=common.cv2.__version__,
                    detector=tracker.detector.environment,transport='one current RGB frame over a blocking pipe',
                    tf32=False,gpu=torch.cuda.get_device_name(0)))
        for n,row in enumerate(rows,1):
            name = row['sequence']; folder = args.output/'traces'/name
            pred = args.output/'predictions'/f'{name}.txt'; done = folder/'COMPLETE.json'
            if done.exists():
                old = common.read(done)
                common.check(common.sha(pred) == old['prediction_sha256'] and common.sha(folder/'frames.json') == old['trace_sha256'], f'Corrupt online result: {name}')
                print('ONLINE_SKIP',n,len(rows),name,flush=True)
                continue
            common.setup_torch(); folder.mkdir(parents=True,exist_ok=True); pred.parent.mkdir(exist_ok=True)
            partial_pred = pred.with_suffix('.partial.txt'); partial_trace = folder/'frames.partial.jsonl'
            trace = []; loop_start = time.perf_counter()
            with partial_pred.open('w') as pf, partial_trace.open('w') as tf:
                for i,path in enumerate(row['frames']):
                    # Read exactly this frame, produce its result, then advance.
                    frame_start = time.perf_counter()
                    _,rgb,digest = common.image_at(path)
                    if i == 0:
                        tracker.initialize(rgb,row['initial_bbox'])
                        template = tracker.core.adapter.z.clone(); mask = tracker.core.adapter.mask.clone()
                        result = dict(frame=1,bbox=row['initial_bbox'],rgb_sha256=digest,
                                      tracker_seconds=0.,detector_seconds=0.,online_seconds=0.)
                    else:
                        result = tracker.track(rgb)
                        common.check(result['rgb_sha256'] == digest and result['frame'] == i+1, 'Current-frame alignment failed')
                    box = result['bbox']
                    common.check(np.isfinite(box).all() and min(box[2:]) > 0, f'Invalid online box: {name} {i+1}')
                    pf.write(','.join(f'{v:.9f}' for v in box)+'\n'); pf.flush()
                    result['frame_wall_seconds'] = time.perf_counter()-frame_start
                    tf.write(json.dumps(result,allow_nan=False)+'\n'); tf.flush()
                    trace.append(result)
                    if i == 0 or (i+1) % 100 == 0 or i+1 == len(row['frames']):
                        print('ONLINE_FRAME',n,len(rows),name,i+1,len(row['frames']),flush=True)
            common.check(torch.equal(template,tracker.core.adapter.z) and torch.equal(mask,tracker.core.adapter.mask), 'Initial template changed')
            loop_seconds = time.perf_counter()-loop_start
            partial_pred.replace(pred); partial_trace.replace(folder/'frames.jsonl')
            common.save(folder/'frames.json',trace)
            common.save(done,dict(mode='online_current_frame',frames=len(trace),prediction_sha256=common.sha(pred),trace_sha256=common.sha(folder/'frames.json'),
                                 initial_template_preserved=True,tracker_seconds=sum(t['tracker_seconds'] for t in trace[1:]),
                                 detector_seconds=sum(t['detector_seconds'] for t in trace[1:]),
                                 online_seconds=sum(t['online_seconds'] for t in trace[1:]),loop_seconds=loop_seconds,
                                 detector_requests=len(trace)-1))
            print('ONLINE_DONE',n,len(rows),name,len(trace),flush=True)


def evaluate(args):
    rows = verify(args)
    # Reuse exactly the metric definitions of the previous launcher.
    with contextlib.redirect_stdout(io.StringIO()):
        common.evaluate(args)
    result = common.read(args.output/'metrics/SUMMARY.json')
    complete = [common.read(args.output/'traces'/r['sequence']/'COMPLETE.json') for r in rows]
    result.update(execution='online_current_frame',detector_precomputation=False,
                  online_tracker_fps=sum(c['frames']-1 for c in complete)/sum(c['online_seconds'] for c in complete),
                  sequence_loop_fps=sum(c['frames'] for c in complete)/sum(c['loop_seconds'] for c in complete),
                  speed_definition='online_tracker_fps includes synchronous detector IPC and state commit, excludes image I/O; sequence_loop_fps includes image I/O, per-frame output writes and per-sequence initialization, excludes model startup.')
    common.save(args.output/'metrics/SUMMARY.json',result)
    common.save(args.output/'COMPLETE.json',dict(execution='online_current_frame',sequences=len(rows),frames=result['frames'],summary_sha256=common.sha(args.output/'metrics/SUMMARY.json')))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--dataset-root',type=Path,default=Path('/home/xu/zyx/GOT_REMOTE/datasets/Test'))
    parser.add_argument('--frames-root',type=Path,default=Path('/home/xu/zyx/GOT_REMOTE/datasets/UAV-Anti-UAV_LaSOT_LINUX_REEXTRACT_20260911'))
    parser.add_argument('--sequence')
    parser.add_argument('--max-frames',type=int)
    args = parser.parse_args(); args.output = args.output.resolve(); args.output.mkdir(parents=True,exist_ok=True)
    common.check(args.max_frames is None or args.max_frames >= 2, '--max-frames must be at least 2')
    prepare(args)
    if args.dry_run:
        print('ONLINE_PREFLIGHT_COMPLETE: no inference started.',flush=True)
        return
    run(args)
    evaluate(args)
    print('ALL_ONLINE_COMPLETE',str(args.output/'metrics/SUMMARY.json'),flush=True)


if __name__ == '__main__':
    main()
