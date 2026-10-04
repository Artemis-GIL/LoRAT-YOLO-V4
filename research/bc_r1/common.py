from pathlib import Path
import hashlib, importlib.util, json, random, sys
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
RUN = Path('/home/xu/zyx/send/LORAT_BC_R1_20261003')
DATA = Path('/home/xu/zyx/GOT_REMOTE/datasets/Train')
ORIGINAL = Path('/home/xu/zyx/send/LOCAL_TRACKER_SELECTION_R1_20261002')
LORAT = ORIGINAL / 'external/LoRAT'
MANIFEST = Path('/home/xu/zyx/send/MVLM_V2_GLOBAL_PROPOSAL_BACKBONE_R1_20261001/data/yolo26_uav/manifest.csv')
PYTHON = '/home/xu/miniconda3/envs/mvlm_cvpr26_cuda128/bin/python'
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(LORAT))

def read(path): return json.loads(Path(path).read_text())
def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def setup(seed=20261003):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=True
def load_adapter(audit, weights=None):
    spec=importlib.util.spec_from_file_location('bc_native_adapter',ORIGINAL/'scripts/adapters.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    adapter=m.LoRATLocal(audit)
    if weights:
        from safetensors.torch import load_file
        state=load_file(str(weights));trainable={n for n,p in adapter.model.named_parameters() if p.requires_grad}
        assert set(state)==trainable
        r=adapter.model.load_state_dict(state,strict=False)
        assert not r.unexpected_keys and not (set(r.missing_keys)&trainable)
    return adapter
def save_trainable(model,path):
    from safetensors.torch import save_file
    path=Path(path);tmp=path.with_suffix('.partial.safetensors')
    save_file({n:p.detach().cpu().contiguous() for n,p in model.named_parameters() if p.requires_grad},str(tmp));tmp.replace(path)
def freeze_stage(name,files):
    mapping={str(Path(p).resolve()):sha(p) for p in files}
    path=RUN/name/'FREEZE.json'
    if path.exists(): assert read(path)==mapping, f'{name}: source or inputs changed'
    else:save(path,mapping)
    for source,digest in mapping.items():
        p=Path(source)
        if p.suffix in {'.py','.json','.yaml','.sh'}:
            dst=RUN/name/'code_snapshot'/(hashlib.sha256(source.encode()).hexdigest()[:10]+'_'+p.name)
            dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():dst.write_bytes(p.read_bytes())
            assert sha(dst)==digest
    return mapping
def verify(mapping):
    for p,digest in mapping.items():assert sha(p)==digest,p
