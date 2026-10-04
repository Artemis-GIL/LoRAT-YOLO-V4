"""Prepare a new frozen follow-on job without changing the active D/E run."""
import datetime
import hashlib
from pathlib import Path
from research.de_r1.common import RUN, PROJECT, read, save, sha, verify, verify_baseline
from tracking.test420_pipeline import TOOLKIT

OUT=Path('/home/xu/zyx/send/LORAT_BEST_DE_TEST420_20261004')
V4=Path('/home/xu/zyx/send/LORAT_YOLO_TEST420_V4_20261003')

def prepare():
    verify_baseline();verify(read(RUN/'pipeline/FREEZE.json'))
    authorization=read(OUT/'AUTHORIZATION.json')
    assert authorization['test420_authorized'] is True and authorization['selection_candidates']==['B','D0','D','E']
    config=read(V4/'CONFIG.json');rows=read(V4/'INPUTS.json');old_frozen=read(V4/'FREEZE.json')
    dataset=Path(config['dataset_root']);assert dataset.name=='Test' and not config['partial']
    assert len(rows)==420 and len({r['sequence'] for r in rows})==420 and sum(len(r['frames']) for r in rows)==476221
    inventory=hashlib.sha256();annotations={}
    for index,row in enumerate(rows,1):
        assert row['sequence']==f'UAV-Anti-UAV_Test_{index:06d}'
        assert [int(Path(p).stem) for p in row['frames']]==list(range(1,len(row['frames'])+1))
        ann=dataset/row['sequence']/'groundtruth_rect.txt'
        with ann.open() as stream:
            first=stream.readline();count=1+sum(bool(line.strip()) for line in stream)
        assert [float(x) for x in first.replace(',',' ').split()]==row['initial_bbox'] and count==len(row['frames'])
        for path in [ann,ann.with_name('absent.txt')]:
            digest=sha(path);assert digest==old_frozen[str(path)],path
            annotations[str(path)]=digest
        for p in row['frames']:
            stat=Path(p).stat();inventory.update(f'{p}\t{stat.st_size}\t{stat.st_mtime_ns}\n'.encode())
    assert inventory.hexdigest()==config['frame_inventory_sha256'],'Test image inventory changed since V4'
    preparation=dict(status='PASS',inputs_path=str(V4/'INPUTS.json'),inputs_sha256=sha(V4/'INPUTS.json'),
        source_config_path=str(V4/'CONFIG.json'),source_config_sha256=sha(V4/'CONFIG.json'),dataset_root=str(dataset),
        frame_inventory_sha256=inventory.hexdigest(),sequences=420,frames=476221,
        input_protocol='Exact previously verified V4 JPEG frame list; never substitute raw MP4',
        source_decoder_note='Test 000316 original MP4 has 39 frames while official aligned image list and GT have 31; preserve canonical image list',
        detector_precomputation=False,reference_run=str(V4),annotations=annotations)
    path=OUT/'PREPARATION.json'
    if path.exists():assert read(path)==preparation
    else:save(path,preparation)
    artifacts=read(RUN/'pipeline/FREEZE.json').copy()
    paths=list(Path(__file__).parent.rglob('*.py'))+[OUT/'AUTHORIZATION.json',path,OUT/'ENGINEERING_CHECKS.json',V4/'INPUTS.json',V4/'CONFIG.json']
    paths+=[TOOLKIT/'accuracy/accuracy.py']+list((TOOLKIT/'got10k').rglob('*.py'))
    artifacts.update({str(p.resolve()):sha(p) for p in paths});artifacts.update(annotations)
    if (OUT/'FREEZE.json').exists():assert read(OUT/'FREEZE.json')==artifacts
    else:save(OUT/'FREEZE.json',artifacts)
    verify(artifacts)
    for source,digest in artifacts.items():
        p=Path(source)
        if p.suffix not in ('.py','.json','.sh','.yaml','.yml'):continue
        target=OUT/'code_snapshot'/(hashlib.sha256(source.encode()).hexdigest()[:12]+'_'+p.name)
        target.parent.mkdir(exist_ok=True)
        if not target.exists():target.write_bytes(p.read_bytes())
        assert sha(target)==digest
    print('TEST420_FOLLOWON_PREPARED',len(rows),sum(len(r['frames']) for r in rows),flush=True)

if __name__=='__main__':prepare()
