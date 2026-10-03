"""Frozen LoRAT/YOLO geometry rule; staged inference, resumable per sequence.

Preparation reads initialization annotations and counts only. Detection/tracking
receive frame paths and the initial box; full GT is loaded only by evaluate().
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SEND = Path('/home/xu/zyx/send')
R1 = SEND / 'LOCAL_TRACKER_SELECTION_R1_20261002'
GEOMETRY = SEND / 'LORAT_YOLO_GEOMETRY_R2_20261002'
YOLO_ROOT = SEND / 'MVLM_V2_GLOBAL_PROPOSAL_BACKBONE_R1_20261001'
TOOLKIT = Path('/home/xu/zyx/GOT_REMOTE/_external/Awesome-Multimodal-Object-Tracking/MMOT_Evaluation_Toolkit')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_array(path):
    with Path(path).open() as f:
        first = f.readline()
    return np.loadtxt(path, delimiter=',' if ',' in first else None)


def prepare(args):
    dataset = args.dataset_root.resolve()
    names = [args.sequence] if args.sequence else sorted(p.name for p in dataset.glob('UAV-Anti-UAV_Test_*') if p.is_dir())
    check(args.sequence is not None or len(names) == 420, f'Expected 420 Test sequences; found {len(names)}')
    rows = []
    sources = {}
    inventory = hashlib.sha256()
    for name in names:
        folder = args.frames_root.resolve() / name
        images = sorted((folder / 'img').glob('*.jpg')) if (folder / 'img').is_dir() else sorted(folder.glob('*.png'))
        ann = dataset / name / 'groundtruth_rect.txt'
        with ann.open() as f:
            first = f.readline().strip()
            count = 1 + sum(bool(line.strip()) for line in f)
        box = np.asarray([float(x) for x in first.replace(',', ' ').split()])
        check(box.shape == (4,) and np.isfinite(box).all() and (box[2:] > 0).all(), f'Invalid first-frame box: {name}')
        check(len(images) == count and count > 1, f'Frame/annotation count mismatch: {name}: {len(images)} / {count}')
        check([int(p.stem) for p in images] == list(range(1, count + 1)), f'Non-contiguous frames: {name}')
        if args.max_frames:
            images = images[:args.max_frames]
        for p in images:
            st = p.stat()
            inventory.update(f'{p}\t{st.st_size}\t{st.st_mtime_ns}\n'.encode())
        rows.append(dict(sequence=name,initial_bbox=box.tolist(),frames=[str(p) for p in images]))
        for p in [ann, dataset / name / 'absent.txt']:
            sources[str(p)] = sha(p)
    config = dict(dataset_root=str(dataset),frames_root=str(args.frames_root.resolve()),sequences=names,
                  max_frames=args.max_frames,initialization='first-frame GT box; fixed template',
                  rule='maximum positive IoU; output raw YOLO box and commit next search anchor',
                  detector=dict(imgsz=1280,conf=.001,iou=.60,max_det=12),
                  partial=bool(args.sequence or args.max_frames),frame_inventory_sha256=inventory.hexdigest())
    deps = [HERE/'test420_pipeline.py',HERE/'run_test420.sh',HERE/'lorat_yolo.py',R1/'scripts/adapters.py',
            GEOMETRY/'scripts/policy.py',YOLO_ROOT/'scripts/yolo26_adapter.py',YOLO_ROOT/'training/yolo26_uav/weights/best.pt']
    # Keep the validated native LoRAT source/weights unchanged.
    inherited = read(R1/'provenance/FREEZE.json')['sha256']
    for p,h in inherited.items():
        if '/external/LoRAT/' in p or p.endswith(('dinov2_vitl14_pretrain.pth','LoRAT_large-378.bin')):
            check(sha(p) == h, f'Validated LoRAT dependency changed: {p}')
            deps.append(Path(p))
    deps += list((TOOLKIT/'got10k').rglob('*.py')) + [TOOLKIT/'accuracy/accuracy.py']
    for p in deps:
        sources[str(p)] = sha(p)
    for name,obj in [('CONFIG.json',config),('INPUTS.json',rows),('FREEZE.json',sources)]:
        path = args.output / name
        if path.exists():
            check(read(path) == obj, f'Resume inputs or code changed: {path}. Use a new output directory.')
        else:
            check(not (args.output/'detections').exists(), 'Incomplete manifest in an existing run; use a new output directory.')
            save(path,obj)
    print('PREPARED',len(rows),'sequences',sum(len(r['frames']) for r in rows),'frames', 'PARTIAL' if config['partial'] else 'TEST420',flush=True)


def verify(out):
    for p,h in read(out/'FREEZE.json').items():
        check(sha(p) == h, f'Frozen dependency changed: {p}')
    return read(out/'INPUTS.json')


def setup_torch(native_detector=False):
    import torch
    random.seed(20261002)
    np.random.seed(20261002)
    torch.manual_seed(20261002)
    torch.cuda.manual_seed_all(20261002)
    torch.set_num_threads(4)
    if not native_detector:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    check(torch.cuda.is_available(), 'CUDA is unavailable in this environment')
    return torch


def image_at(path):
    bgr = cv2.imread(path)
    check(bgr is not None, f'Cannot read frame: {path}')
    rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    return bgr,rgb,hashlib.sha256(rgb.tobytes()).hexdigest()


def detect(args):
    rows = verify(args.output)
    # Match R2's YOLO environment defaults; LoRAT has its own precision settings.
    torch = setup_torch(native_detector=True)
    import ultralytics
    sys.path.insert(0,str(YOLO_ROOT/'scripts'))
    from yolo26_adapter import YOLO26Detector
    detector = YOLO26Detector(YOLO_ROOT/'training/yolo26_uav/weights/best.pt',max_det=12)
    save(args.output/'DETECTOR_ENV.json',dict(python=sys.version,torch=torch.__version__,opencv=cv2.__version__,ultralytics=ultralytics.__version__))
    for n,row in enumerate(rows,1):
        name = row['sequence']
        target = args.output/'detections'/f'{name}.json'
        done = target.with_suffix('.complete.json')
        if done.exists():
            check(target.exists() and sha(target) == read(done)['sha256'], f'Corrupt detection cache: {name}')
            print('DETECT_SKIP',n,len(rows),name,flush=True)
            continue
        start_sequence = time.perf_counter()
        events = []
        for i,path in enumerate(row['frames']):
            bgr,rgb,digest = image_at(path)
            torch.cuda.synchronize();start = time.perf_counter()
            candidates = detector.infer(bgr) if i else []
            torch.cuda.synchronize()
            events.append(dict(frame=i+1,rgb_sha256=digest,candidates=candidates,seconds=time.perf_counter()-start))
            if i and i % 200 == 0:
                print('DETECT_PROGRESS',n,len(rows),name,i+1,len(row['frames']),flush=True)
        save(target,events)
        save(done,dict(frames=len(events),sha256=sha(target),stage_seconds=time.perf_counter()-start_sequence))
        print('DETECT_DONE',n,len(rows),name,len(events),flush=True)


def track(args):
    rows = verify(args.output)
    torch = setup_torch()
    from lorat_yolo import LoRATCore
    sys.path.insert(0,str(GEOMETRY/'scripts'))
    from policy import match
    core = LoRATCore(args.output/'MODEL.json')
    save(args.output/'TRACKER_ENV.json',dict(python=sys.version,torch=torch.__version__,opencv=cv2.__version__,gpu=torch.cuda.get_device_name(0),tf32=False))
    for n,row in enumerate(rows,1):
        name = row['sequence']
        folder = args.output/'traces'/name
        done = folder/'COMPLETE.json'
        pred = args.output/'predictions'/f'{name}.txt'
        dp = args.output/'detections'/f'{name}.json'
        digest = sha(dp)
        check(digest == read(dp.with_suffix('.complete.json'))['sha256'], f'Corrupt detections: {name}')
        if done.exists():
            old = read(done)
            check(sha(pred) == old['prediction_sha256'] and digest == old['detection_sha256'] and sha(folder/'frames.json') == old['trace_sha256'], f'Corrupt completed result: {name}')
            print('TRACK_SKIP',n,len(rows),name,flush=True)
            continue
        setup_torch()
        dets = read(dp)
        check(len(dets) == len(row['frames']), f'Detection length mismatch: {name}')
        boxes,trace = [],[]
        start_sequence = time.perf_counter()
        for i,(path,d) in enumerate(zip(row['frames'],dets)):
            bgr,rgb,digest_rgb = image_at(path)
            check(d['frame'] == i+1 and digest_rgb == d['rgb_sha256'], f'Detector/tracker pixel mismatch: {name} {i+1}')
            torch.cuda.synchronize();start = time.perf_counter()
            if i == 0:
                core.initialize(rgb,row['initial_bbox']);box = row['initial_bbox'];extra = {}
                template = core.adapter.z.clone();mask = core.adapter.mask.clone()
            else:
                local = core.track(rgb)
                box,index,overlap = match(local['box'],d['candidates'])
                if index is not None:
                    core.commit(box,rgb.shape)
                extra = dict(local_box=local['box'],score=local['score'],roi=local['roi'],matched_index=index,association_iou=overlap)
            torch.cuda.synchronize();elapsed = time.perf_counter()-start
            check(np.isfinite(box).all() and min(box[2:]) > 0, f'Invalid output: {name} {i+1}')
            boxes.append(box)
            trace.append(dict(frame=i+1,bbox=box,rgb_sha256=digest_rgb,seconds=elapsed,**extra))
            if i and i % 200 == 0:
                print('TRACK_PROGRESS',n,len(rows),name,i+1,len(row['frames']),flush=True)
        check(torch.equal(template,core.adapter.z) and torch.equal(mask,core.adapter.mask), 'Initial template changed')
        pred.parent.mkdir(parents=True,exist_ok=True)
        temp = pred.with_suffix('.tmp');np.savetxt(temp,boxes,delimiter=',',fmt='%.9f');temp.replace(pred)
        save(folder/'frames.json',trace)
        save(done,dict(frames=len(boxes),prediction_sha256=sha(pred),trace_sha256=sha(folder/'frames.json'),detection_sha256=digest,
                       initial_template_preserved=True,tracker_seconds=sum(t['seconds'] for t in trace[1:]),
                       detector_seconds=sum(t['seconds'] for t in dets[1:]),stage_seconds=time.perf_counter()-start_sequence))
        print('TRACK_DONE',n,len(rows),name,len(boxes),flush=True)


def evaluate(args):
    rows = verify(args.output)
    config = read(args.output/'CONFIG.json')
    sys.path.insert(0,str(TOOLKIT))
    from got10k.experiments import ExperimentUAVAntiUAV_Overall
    evaluator = ExperimentUAVAntiUAV_Overall(config['dataset_root'],'All','UAVAntiUAV',result_dir=str(args.output),report_dir=str(args.output/'metrics/mmot'),report_scenario_dir=str(args.output/'metrics/scenarios'))
    spec = importlib.util.spec_from_file_location('uav_official_accuracy',TOOLKIT/'accuracy/accuracy.py')
    accuracy = importlib.util.module_from_spec(spec);spec.loader.exec_module(accuracy)
    scored = []
    for row in rows:
        name = row['sequence'];count = len(row['frames'])
        folder = args.output/'traces'/name;done = read(folder/'COMPLETE.json')
        pred = args.output/'predictions'/f'{name}.txt'
        check(sha(pred) == done['prediction_sha256'] and sha(folder/'frames.json') == done['trace_sha256'], f'Result hash mismatch: {name}')
        boxes = load_array(pred).reshape(-1,4)
        gt = load_array(Path(config['dataset_root'])/name/'groundtruth_rect.txt').reshape(-1,4)[:count]
        absent = load_array(Path(config['dataset_root'])/name/'absent.txt').reshape(-1)[:count]
        check(len(boxes) == len(gt) == len(absent) == count == done['frames'], f'Scoring length mismatch: {name}')
        check(np.allclose(boxes[0],row['initial_bbox'],rtol=0,atol=1e-8), f'Wrong initialization: {name}')
        metrics = evaluator._calc_metrics(boxes.copy(),gt.copy())
        curves = evaluator._calc_curves(*metrics)
        vis = (absent == 0) & (gt[:,2:] > 0).all(1);vis[0] = False
        scored.append(dict(sequence=name,frames=count,AUC=float(curves[0].mean()),Precision20=float(curves[1][20]),cAUC=float(curves[2].mean()),
                           nPrecision_at_0_2=float(curves[3][20]),nPrecision_AUC=float(curves[3].mean()),
                           mACC=float(accuracy.eval(boxes.tolist(),gt.tolist(),np.any(gt != 0,axis=1))),
                           visible_frames=int(vis.sum()),visible_mean_iou=float(metrics[0][vis].mean()) if vis.any() else None,
                           visible_hit50=float((metrics[0][vis] >= .5).mean()) if vis.any() else None,
                           tracker_seconds=done['tracker_seconds'],detector_seconds=done['detector_seconds']))
    keys = ['AUC','Precision20','cAUC','nPrecision_at_0_2','nPrecision_AUC','mACC','visible_mean_iou','visible_hit50']
    summary = dict(scope='PARTIAL_SMOKE' if config['partial'] else 'TEST420',sequences=len(scored),frames=sum(r['frames'] for r in scored),
                   sequence_macro={k:float(np.mean([r[k] for r in scored if r[k] is not None])) for k in keys},
                   serial_compute_fps=sum(r['frames']-1 for r in scored)/sum(r['tracker_seconds']+r['detector_seconds'] for r in scored),
                   metric_definition='Installed MMOT toolkit functions; all frames including GT-initialized first frame. Visible diagnostics exclude first frame. FPS excludes image I/O and startup.',
                   inference_protocol='GT-initialized single-object tracking; not language-only/target-free initialization',
                   automatic_global_identity_recovery=False)
    save(args.output/'metrics/SUMMARY.json',summary)
    with (args.output/'metrics/SEQUENCES.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(scored[0]));writer.writeheader();writer.writerows(scored)
    save(args.output/'COMPLETE.json',dict(sequences=len(scored),frames=summary['frames'],summary_sha256=sha(args.output/'metrics/SUMMARY.json')))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','detect','track','evaluate'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset-root',type=Path,default=Path('/home/xu/zyx/GOT_REMOTE/datasets/Test'))
    parser.add_argument('--frames-root',type=Path,default=Path('/home/xu/zyx/GOT_REMOTE/datasets/UAV-Anti-UAV_LaSOT_LINUX_REEXTRACT_20260911'))
    parser.add_argument('--sequence')
    parser.add_argument('--max-frames',type=int)
    args = parser.parse_args();args.output = args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    check(args.max_frames is None or args.max_frames >= 2, '--max-frames must be at least 2')
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
