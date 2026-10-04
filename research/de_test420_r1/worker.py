"""Wait for the frozen D/E pipeline, select from validation, run Test420 once."""
import datetime
import fcntl
import os
import time
from pathlib import Path
from research.de_r1.common import RUN, read, save, sha, verify, verify_baseline
from .prepare import OUT
from .select import validated_selection

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def alive(pid):
    try:
        content=Path(f'/proc/{int(pid)}/stat').read_text()
        return content[content.rfind(')')+2:].split()[0]!='Z'
    except (FileNotFoundError,ProcessLookupError,ValueError,TypeError):return False

def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'.worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(stage='WAITING_FOR_DE',pid=os.getpid(),updated_utc=stamp(),source_run=str(RUN),test_output=str(OUT/'test420'))
        try:
            verify(read(OUT/'FREEZE.json'));verify_baseline()
            assert read(OUT/'ENGINEERING_CHECKS.json')['status']=='PASS'
            verify(read(OUT/'ENGINEERING_CHECKS.json')['artifacts'])
            if (OUT/'COMPLETE.json').exists():
                d=read(OUT/'COMPLETE.json');assert d['status']=='PASS'
                assert sha(OUT/'test420/COMPLETE.json')==d['test_completion_sha256']
                assert sha(OUT/'SELECTION.json')==d['selection_sha256']
                verify(read(OUT/'SELECTION.json')['provenance']);return
            save(OUT/'STATUS.json',state)
            while True:
                source=read(RUN/'STATUS.json')
                if source['stage']=='FAILED':raise RuntimeError('D/E pipeline failed: '+str(source))
                if source['stage']=='COMPLETE':break
                if not alive(source.get('pid')):raise RuntimeError('D/E pipeline exited before successful completion')
                state.update(source_stage=source['stage'],updated_utc=stamp());save(OUT/'STATUS.json',state)
                time.sleep(30)
            state.update(stage='VERIFYING_AND_SELECTING',updated_utc=stamp());save(OUT/'STATUS.json',state)
            # The completion marker may be written just before the D/E process exits.
            with (RUN/'pipeline/.lock').open('a') as source_lock:
                fcntl.flock(source_lock,fcntl.LOCK_EX)
                verify(read(OUT/'FREEZE.json'));selection=validated_selection()
                selection['authorization_sha256']=sha(OUT/'AUTHORIZATION.json')
                if (OUT/'SELECTION.json').exists():assert read(OUT/'SELECTION.json')==selection
                else:save(OUT/'SELECTION.json',selection)
            state.update(stage='RUNNING_TEST420',variant=selection['variant'],updated_utc=stamp(),selected_at_utc=stamp())
            save(OUT/'STATUS.json',state);print('TEST420_SELECTED',selection['variant'],selection['ranking'],flush=True)
            from .test420 import run_test
            run_test(selection,OUT/'test420')
            verify(read(OUT/'FREEZE.json'));verify_baseline()
            complete=read(OUT/'test420/COMPLETE.json')
            assert complete['status']=='PASS' and complete['sequences']==420 and complete['frames']==476221
            save(OUT/'COMPLETE.json',dict(status='PASS',variant=selection['variant'],sequences=420,frames=476221,
                selection_sha256=sha(OUT/'SELECTION.json'),test_completion_sha256=sha(OUT/'test420/COMPLETE.json'),completed_utc=stamp()))
            state.update(stage='COMPLETE',updated_utc=stamp());save(OUT/'STATUS.json',state)
            print('BEST_DE_TEST420_COMPLETE',OUT/'test420/metrics/SUMMARY.json',flush=True)
        except BaseException as exc:
            state.update(failed_stage=state['stage'],stage='FAILED',updated_utc=stamp(),error=type(exc).__name__,message=str(exc))
            save(OUT/'STATUS.json',state);raise

if __name__=='__main__':main()
