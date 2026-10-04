"""CPU-only packaging checks; no checkpoint training or actual tracker inference."""
from pathlib import Path
import os,sys,json,hashlib,unittest
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.dont_write_bytecode=True
import torch
torch.set_num_threads(1)
assert not torch.cuda.is_available()
import research.bc_r1.common as common
old=str(common.LORAT)
common.ORIGINAL=R/'dependencies/LOCAL_TRACKER_SELECTION_R1_20261002'
common.LORAT=common.ORIGINAL/'external/LoRAT'
sys.path=[str(R),str(common.LORAT)]+[p for p in sys.path if p not in (old,str(R),str(common.LORAT))]
names=['research.bc_r1.tests.test_bc','research.de_r1.tests.test_d','research.de_r1.tests.test_e','research.de_test420_r1.test_runner','research.de_test420_r1.test_selection']
suite=unittest.defaultTestLoader.loadTestsFromNames(names)
result=unittest.TextTestRunner(verbosity=2).run(suite)
from safetensors.torch import load_file
from research.de_r1.model_d import ActionValue
assets=R.parent/'release_assets'
D=ActionValue().cpu().eval();D.load_state_dict(load_file(str(assets/'GRU_D_final.safetensors'),device='cpu'),strict=True)
B=load_file(str(assets/'LoRAT_B_UAV_final.safetensors'),device='cpu')
assert len(B)==301
# Account for project dependencies loaded from original paths; verify only used files against frozen hashes.
frozen=json.loads((R/'results/test420/FREEZE.json').read_text());imports=[];conflicts=[]
for mod in list(sys.modules.values()):
    name=getattr(mod,'__file__',None)
    if name and str(Path(name).resolve()) in frozen:
        p=Path(name).resolve();h=hashlib.sha256(p.read_bytes()).hexdigest()
        imports.append(str(p))
        if h!=frozen[str(p)]:conflicts.append(str(p))
out=dict(status='PASS' if result.wasSuccessful() and not conflicts else 'FAIL',device='CPU',tests_run=result.testsRun,passed=result.testsRun-len(result.skipped)-len(result.failures)-len(result.errors),skipped=[dict(test=str(t),reason=reason) for t,reason in result.skipped],failures=len(result.failures),errors=len(result.errors),D_checkpoint_strict_load=True,D_trainable_parameters=sum(p.numel() for p in D.parameters()),B_checkpoint_keys=len(B),original_dependency_imports_verified=sorted(set(imports)),dependency_hash_conflicts=conflicts,new_model_inference=False,notes='Synthetic model/head tests and API checks only; original LoRAT model and detector were not run. Local path redirection during tests is not a changed algorithm.')
(R/'provenance/CPU_CHECKS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2));raise SystemExit(0 if out['status']=='PASS' else 1)
