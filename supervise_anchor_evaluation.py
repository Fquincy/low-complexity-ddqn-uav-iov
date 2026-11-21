"""Persistent monitor for the authorized evaluation-only continuation. No retries."""
import argparse,datetime,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def timestamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

def write(path,value):
    text=value if isinstance(value,str) else json.dumps(value,indent=2)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text,encoding='utf-8');os.replace(tmp,path)

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--monitor',type=Path,required=True)
    p.add_argument('--confirm-development-evaluation',action='store_true')
    a=p.parse_args()
    if not a.confirm_development_evaluation:p.error('--confirm-development-evaluation required')
    monitor=a.monitor.resolve();authorization=read(monitor/'USER_AUTHORIZATION.json')
    if not authorization['explicit_evaluation_and_full_audit']:raise ValueError('authorization missing')
    output=Path(authorization['output']);control=Path(authorization['control']);source=Path(authorization['source'])
    if output.exists() or control.exists():raise FileExistsError('No automatic duplicate, restart or overwrite')
    import hashlib
    for name,digest in authorization['executable_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('executable changed:'+name)
    with (monitor/'SUPERVISOR_STARTED.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(),started_at=timestamp()),stream,indent=2)
    cmd=[sys.executable,'-u',str(ROOT/'launch_anchor_evaluation.py'),'--mode','development',
        '--source',str(source),'--output',str(output),'--control',str(control),'--confirm-development-evaluation']
    started=timestamp();start=time.monotonic()
    with (monitor/'worker_stdout.log').open('w',encoding='utf-8') as stdout,(monitor/'worker_stderr.log').open('w',encoding='utf-8') as stderr:
        worker=subprocess.Popen(cmd,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        write(monitor/'WORKER_STARTED.json',dict(pid=worker.pid,command=cmd,started_at=started))
        while True:
            code=worker.poll();events=[]
            try:events=read(output/'events.json')
            except (OSError,ValueError):pass
            evaluated=len(list(output.glob('*/fit_*/evaluation_*.json.gz'))) if output.exists() else 0
            heuristics=len(list(output.glob('heuristic_*.json.gz'))) if output.exists() else 0
            stage=events[-1] if events else dict(stage='preflight_or_materializing_saved_inputs')
            if evaluated==4080 and heuristics==144 and stage.get('stage')=='complete' and code is None:
                stage=dict(stage='full_saved_training_and_evaluation_artifact_audit')
            completed=code==0 and (control/'COMPLETION.json').exists()
            state='running' if code is None else ('completed_and_audited' if completed else 'failed_requires_review')
            status=dict(state=state,started_at=started,updated_at=timestamp(),supervisor_pid=os.getpid(),worker_pid=worker.pid,
                worker_exit_code=code,elapsed_seconds=round(time.monotonic()-start,1),stage=stage,
                completed_controller_evaluation_episodes=evaluated,total_controller_evaluation_episodes=4080,
                completed_heuristic_episodes=heuristics,total_heuristic_episodes=144,new_training_slots=0,new_optimizer_updates=0,
                source=str(source),output=str(output),control=str(control),final_evaluation_launched=False,thesis_changed=False)
            write(monitor/'LIVE_STATUS.json',status)
            write(monitor/'LIVE_STATUS.txt',f"TERMINAL-ANCHOR DEVELOPMENT EVALUATION AND FULL AUDIT\nState: {state}\nUpdated: {status['updated_at']}\n"
                f"Stage: {json.dumps(stage)}\nController evaluation episodes saved: {evaluated:,} / 4,080\n"
                f"Heuristic episodes saved: {heuristics} / 144\nNew training slots: 0\nNew optimizer updates: 0\n"
                f"Worker PID: {worker.pid}\nElapsed hours: {(time.monotonic()-start)/3600:.2f}\n"
                f"Output: {output}\nCompletion record: {control/'COMPLETION.json'}\n"
                "Evaluation completion is followed by the full artifact audit.\nFinal evaluation not launched. Word thesis unchanged.\n")
            if code is not None:break
            time.sleep(15)
    sys.exit(0 if completed else 1)

if __name__=='__main__':main()
