"""Serial, resumable B/C research run with frozen sources and fail-closed status."""
import datetime,fcntl,os,subprocess,time
from .common import *
from tracking.run_tracker_v3 import freeze_dependencies
from tracking.test420_pipeline import TOOLKIT

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    folder=RUN/'pipeline';folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(stage='PREFLIGHT',pid=os.getpid(),updated_utc=stamp(),B_training_complete=True,
            reverse_verification=False,training_sequences=1268,validation_sequences=132,no_test_inference=True)
        save(RUN/'STATUS.json',state)
        try:
            verify(read(RUN/'B/FREEZE.json'));verify(read(RUN/'INPUT_SHA256.json'))
            assert sha(RUN/'B/final.safetensors')==read(RUN/'B/COMPLETE.json')['sha256']
            assert read(RUN/'B_smoke/COMPLETE.json')['status']=='PASS'
            assert read(RUN/'C_feature_check/RESULT.json')['status']=='PASS'
            verify(read(RUN/'C_feature_check/RESULT.json')['source_hashes'])
            assert read(RUN/'C_smoke/COMPLETE.json')['status']=='SMOKE_PASS'
            assert read(RUN/'validation_smoke/B_POOL/PARITY_B.json')['exact_prediction_equality']
            assert read(RUN/'validation_smoke/C/COMPLETE.json')['frames']==8
            assert read(RUN/'ENGINEERING_CHECKS.json')['status']=='PASS'
            inherited=freeze_dependencies(PROJECT/'tracking/framework_v3.json')
            files=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('tests/*.py'))
            files+=list(PROJECT.glob('tracking/*.py'))+[RUN/'PROTOCOL.json',RUN/'INPUT_SHA256.json',RUN/'B/final.safetensors']
            files+=list(map(Path,inherited))+list(map(Path,read(RUN/'B/FREEZE.json')))
            files+=[TOOLKIT/'accuracy/accuracy.py']+list((TOOLKIT/'got10k').rglob('*.py'))
            frozen=freeze_stage('pipeline',files)
            stages=[('C_MINING','research.bc_r1.mine_c',[]),('C_TRAINING','research.bc_r1.train_c',[])]
            stages += [('VALIDATING_'+v,'research.bc_r1.evaluate',['--variant',v]) for v in ['A','B','B_POOL','C']]
            stages += [('COMPARISON','research.bc_r1.report',[])]
            for stage,module,args in stages:
                verify(frozen)
                state.update(stage=stage,child_pid=None,stage_started_utc=stamp(),updated_utc=stamp(),log=str(folder/(stage+'.log')))
                save(RUN/'STATUS.json',state)
                with (folder/(stage+'.log')).open('a',buffering=1) as log:
                    log.write('\nBC_STAGE_START '+stamp()+'\n');log.flush()
                    child=subprocess.Popen([PYTHON,'-u','-m',module,*args],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
                    state.update(child_pid=child.pid,updated_utc=stamp());save(RUN/'STATUS.json',state)
                    try:code=child.wait()
                    except BaseException:
                        child.terminate()
                        try:child.wait(timeout=20)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        raise
                    if code:raise RuntimeError(f'{stage} exited {code}; see {log.name}')
                verify(frozen)
                print('BC_STAGE_COMPLETE',stage,stamp(),flush=True)
            verify(read(RUN/'INPUT_SHA256.json'))
            state.update(stage='COMPLETE',child_pid=None,updated_utc=stamp(),report=str(RUN/'comparison/REPORT.zh.md'))
            save(RUN/'STATUS.json',state);save(folder/'COMPLETE.json',state)
        except BaseException as exc:
            state.update(stage='FAILED',failed_stage=state['stage'],updated_utc=stamp(),error=type(exc).__name__,message=str(exc))
            save(RUN/'STATUS.json',state);raise
if __name__=='__main__':main()
