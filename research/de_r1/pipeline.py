"""Run the independent D/E arms on B with mandatory full zero-effect controls."""
import datetime,fcntl,os,subprocess
from .common import *

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    folder=RUN/'pipeline';folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(stage='PREFLIGHT',pid=os.getpid(),child_pid=None,baseline='B',updated_utc=stamp(),no_test_inference=True,no_reverse_verification=True)
        save(RUN/'STATUS.json',state)
        try:
            verify_baseline()
            assert read(RUN/'ENGINEERING_CHECKS.json')['status']=='PASS'
            verify(read(RUN/'ENGINEERING_CHECKS.json')['artifacts'])
            for stage in ['D0_smoke','D_smoke','E_smoke']:
                assert read(RUN/stage/'COMPLETE.json')['status']=='PASS'
                verify(read(RUN/stage/'FREEZE.json'))
            for variant in ['D_ZERO','D0','D','E_ZERO','E']:
                assert read(RUN/'validation_smoke'/variant/'COMPLETE.json')['status']=='PASS'
                verify(read(RUN/'validation_smoke'/variant/'FREEZE.json'))
            files=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('tests/*.py'))
            files+=[RUN/'PROTOCOL.json',RUN/'D_data/SUMMARY.json',RUN/'ENGINEERING_CHECKS.json',*baseline_files()]
            frozen=freeze_stage('pipeline',files)
            stages=[('TRAIN_D0','research.de_r1.train_d',['--variant','D0']),('TRAIN_D','research.de_r1.train_d',['--variant','D']),
                ('TRAIN_E','research.de_r1.train_e',[])]
            stages += [('VALIDATING_'+v,'research.de_r1.evaluate',['--variant',v]) for v in ['D_ZERO','D0','D','E_ZERO','E']]
            stages += [('COMPARISON','research.de_r1.report',[])]
            for stage,module,args in stages:
                verify(frozen);state.update(stage=stage,child_pid=None,stage_started_utc=stamp(),updated_utc=stamp(),log=str(folder/(stage+'.log')))
                save(RUN/'STATUS.json',state)
                with (folder/(stage+'.log')).open('a',buffering=1) as log:
                    log.write('\nDE_STAGE_START '+stamp()+'\n');log.flush()
                    child=subprocess.Popen([PYTHON,'-u','-m',module,*args],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
                    state.update(child_pid=child.pid,updated_utc=stamp());save(RUN/'STATUS.json',state)
                    try:code=child.wait()
                    except BaseException:
                        child.terminate()
                        try:child.wait(timeout=20)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        raise
                    if code:raise RuntimeError(f'{stage} exited {code}; see {log.name}')
                verify(frozen);print('DE_STAGE_COMPLETE',stage,stamp(),flush=True)
            verify_baseline();state.update(stage='COMPLETE',child_pid=None,updated_utc=stamp(),report=str(RUN/'comparison/REPORT.zh.md'))
            save(RUN/'STATUS.json',state);save(folder/'COMPLETE.json',state)
        except BaseException as exc:
            state.update(stage='FAILED',failed_stage=state['stage'],updated_utc=stamp(),error=type(exc).__name__,message=str(exc))
            save(RUN/'STATUS.json',state);raise
if __name__=='__main__':main()
