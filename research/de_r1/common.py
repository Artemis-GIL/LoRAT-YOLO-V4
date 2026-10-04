from pathlib import Path
import hashlib,json,sys,datetime
import numpy as np
import torch
from research.bc_r1.common import PROJECT,DATA,ORIGINAL,LORAT,PYTHON,read,save,sha,setup,verify,load_adapter

OLD=Path('/home/xu/zyx/send/LORAT_BC_R1_20261003')
RUN=Path('/home/xu/zyx/send/LORAT_B_DE_R1_20261004')
B_WEIGHTS=OLD/'B/final.safetensors'
BASELINE=PROJECT/'research/BASELINE.json'

def freeze_stage(name,files):
    mapping={str(Path(p).resolve()):sha(p) for p in files}
    path=RUN/name/'FREEZE.json'
    if path.exists():assert read(path)==mapping,f'{name}: frozen source/input changed'
    else:save(path,mapping)
    for source,digest in mapping.items():
        p=Path(source)
        if p.suffix in {'.py','.json','.yaml','.sh'}:
            dst=RUN/name/'code_snapshot'/(hashlib.sha256(source.encode()).hexdigest()[:10]+'_'+p.name)
            dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():dst.write_bytes(p.read_bytes())
            assert sha(dst)==digest
    return mapping

def baseline_files():
    return [BASELINE,B_WEIGHTS,OLD/'PROTOCOL.json',OLD/'INPUT_SHA256.json',*map(Path,read(OLD/'pipeline/FREEZE.json'))]

def verify_baseline():
    assert sha(B_WEIGHTS)==read(OLD/'B/COMPLETE.json')['sha256']
    assert read(OLD/'COMPLETION_AUDIT.json')['status']=='PASS'
    assert read(BASELINE)['checkpoint_sha256']==sha(B_WEIGHTS)
    verify(read(OLD/'pipeline/FREEZE.json'))
