"""Verify launched process and freeze startup evidence; no restarts."""
import datetime,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run_anchor_development_v4 as r
C=ROOT/'validation/anchor_development_launch_2026-11-06_01'
O=ROOT/'development_runs/terminal_anchor_development_2026-11-06_01'
status=r.read(C/'LIVE_STATUS.json');plan=r.read(O/'runtime_plan.json')
assert status['state']=='running' and status['worker_exit_code'] is None
assert plan['kind']=='RESEARCH' and plan['training_launched'] and not plan['fixture']
assert plan['model_seeds']==r.protocol()['development']['model_seeds']
assert plan['train_blocks']==r.protocol()['development']['train_blocks']
assert set(plan['source_sha256'])==set(r.SOURCES)
for name,digest in plan['source_sha256'].items():
    assert r.base.old.sha(ROOT/name)==r.base.old.sha(O/'source_snapshot'/name)==digest,name
assert not (O/'FAILED.json').exists()
train=list(O.glob('*/fit_*/train_*.json.gz'))
verified=dict(state='launched_running_source_freeze_verified',checked_at=datetime.datetime.now().astimezone().isoformat(),
    worker_pid=status['worker_pid'],supervisor_pid=status['supervisor_pid'],saved_episodes=len(train),
    source_files_verified=len(plan['source_sha256']),protocol_sha256=plan['protocol_sha256'],final_launched=False,thesis_changed=False)
r.write(C/'LAUNCH_VERIFIED.json',verified)
tracker=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json';s=r.read(tracker)
s['current_workstream']='terminal_anchor_seventeen_arm_development_running'
s['next']='allow_frozen_run_to_complete_then_review_all_saved_results_and_audit_without_tuning'
s['terminal_anchor_development_launch']=dict(control=str(C.relative_to(ROOT)),output=str(O.relative_to(ROOT)),
    training_slots=680000,controllers=17,initializations=5,worker_pid=status['worker_pid'],
    supervisor_pid=status['supervisor_pid'],started_at=status['started_at'],research_launched=True,
    final_launched=False,thesis_changed=False,live_status=str((C/'LIVE_STATUS.txt').relative_to(ROOT)))
r.write(tracker,s)
print(json.dumps(verified))
