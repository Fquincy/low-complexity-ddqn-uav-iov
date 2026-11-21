"""Persistent local supervisor for the explicitly authorized frozen development run."""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def timestamp():return datetime.datetime.now().astimezone().isoformat()

def atomic_write(path,text):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(text,encoding='utf-8');os.replace(temporary,path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--confirm-development-run',action='store_true')
    args=parser.parse_args()
    if not args.confirm_development_run:parser.error('--confirm-development-run is required')
    control=args.control.resolve();authorization=json.loads((control/'LAUNCH_AUTHORIZATION.json').read_text())
    output=Path(authorization['output'])
    if (control/'SUPERVISOR_STARTED.json').exists():raise RuntimeError('Supervisor already started; no automatic duplicate run')
    from launch_v3_development import preflight
    preflight(control)
    if output.exists():raise FileExistsError(output)
    atomic_write(control/'SUPERVISOR_STARTED.json',json.dumps(dict(pid=os.getpid(),started_at=timestamp()),indent=2))
    command=[sys.executable,'-u',str(ROOT/'launch_v3_development.py'),'--mode','development',
        '--confirm-development-run','--control',str(control),'--output',str(output)]
    start=time.monotonic();started_at=timestamp()
    with (control/'worker_stdout.log').open('w',encoding='utf-8') as stdout, (control/'worker_stderr.log').open('w',encoding='utf-8') as stderr:
        worker=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        while True:
            code=worker.poll()
            train_files=list(output.glob('*/fit_*/train_*.json.gz')) if output.exists() else []
            checkpoints=list(output.glob('*/fit_*/final.pt')) if output.exists() else []
            events=[]
            try:events=json.loads((output/'events.json').read_text())
            except (OSError,ValueError):pass
            completed_slots=len(train_files)*40
            state='running' if code is None else ('completed_and_audited' if code==0 and (control/'COMPLETION.json').exists() else 'failed_requires_review')
            stage=events[-1] if events else {'stage':'initializing_or_training_first_fit'}
            status=dict(state=state,started_at=started_at,updated_at=timestamp(),supervisor_pid=os.getpid(),
                worker_pid=worker.pid,worker_exit_code=code,elapsed_seconds=round(time.monotonic()-start,1),
                completed_training_episodes=len(train_files),completed_training_slots=completed_slots,
                total_training_slots=440000,training_progress_percent=round(100*completed_slots/440000,3),
                final_checkpoints=len(checkpoints),total_final_checkpoints=55,last_runner_event=stage,
                output=str(output),protocol_unchanged=True,final_phase_launched=False,thesis_changed=False)
            atomic_write(control/'LIVE_STATUS.json',json.dumps(status,indent=2))
            text=(f"VERSION 3 DEVELOPMENT RUN\nState: {state}\nUpdated: {status['updated_at']}\n"
                f"Training slots saved: {completed_slots:,} / 440,000 ({status['training_progress_percent']}%)\n"
                f"Final checkpoints: {len(checkpoints)} / 55\nLast runner event: {json.dumps(stage)}\n"
                f"Worker PID: {worker.pid}\nElapsed hours: {(time.monotonic()-start)/3600:.2f}\n"
                f"Research output: {output}\nLogs: {control}\n"
                "Progress counts fully saved training episodes; the current episode may add up to39 slots.\n"
                "After training, export checks, held-out evaluation and the saved-artifact audit still run.\n"
                "Final phase has not been launched. Word thesis unchanged.\n")
            atomic_write(control/'LIVE_STATUS.txt',text)
            if code is not None:break
            time.sleep(15)
    sys.exit(0 if state=='completed_and_audited' else 1)

if __name__=='__main__':main()
