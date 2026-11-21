"""Record verified evaluation-only startup; leave original run and thesis unchanged."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run_anchor_posttraining_continuation_v2 as c
M=ROOT/'validation/anchor_evaluation_monitor_2026-11-11_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
status=c.read(M/'LIVE_STATUS.json');auth=c.read(M/'USER_AUTHORIZATION.json')
assert status['state']=='running' and status['worker_exit_code'] is None
out=Path(auth['output']);control=Path(auth['control']);receipt=c.read(control/'LAUNCH_AUTHORIZATION.json')
assert receipt['mode']=='development_evaluation_only' and receipt['explicit_cli_confirmation']
assert receipt['new_training_slots']==receipt['new_optimizer_updates']==0
assert receipt['readiness_decision_sha256']==auth['readiness_decision_sha256']
plan=c.read(out/'runtime_plan.json');c.validate_research_plan(plan)
assert c.read(out/'ALL_FINAL_CHECKPOINTS_FREEZE.json')==c.read(c.RESEARCH_SOURCE/'ALL_FINAL_CHECKPOINTS_FREEZE.json')
assert not (out/'CONTINUATION_FAILED.json').exists()
assert not (M/'worker_stderr.log').read_text(encoding='utf-8').strip()
c.write(M/'LAUNCH_VERIFIED.json',dict(state='running_evaluation_only_continuation',worker_pid=status['worker_pid'],supervisor_pid=status['supervisor_pid'],
    verified_source_checkpoints=85,source_plan_matches_frozen_development=True,new_training_slots=0,new_optimizer_updates=0,
    final_evaluation_launched=False,thesis_changed=False))
tracker=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json';s=c.read(tracker)
s['current_workstream']='anchor_development_evaluation_and_full_artifact_audit_running_no_retraining'
s['next']='allow_evaluation_and_full_audit_to_complete_then_review_all_fits_and_predeclared_screens'
s['anchor_development_evaluation_launch']=dict(monitor=str(M.relative_to(ROOT)),output=str(out.relative_to(ROOT)),control=str(control.relative_to(ROOT)),
    worker_pid=status['worker_pid'],supervisor_pid=status['supervisor_pid'],started_at=status['started_at'],controllers=17,initializations=5,
    controller_evaluation_episodes=4080,heuristic_episodes=144,new_training_slots=0,new_optimizer_updates=0,full_audit_scheduled=True,
    final_evaluation_launched=False,word_thesis_changed=False)
c.write(tracker,s)
paths=[M/'USER_AUTHORIZATION.json',M/'PROJECT_STATUS_BEFORE_LAUNCH.json',M/'LAUNCH_VERIFIED.json',M/'SUPERVISOR_STARTED.json',M/'WORKER_STARTED.json',
    control/'LAUNCH_AUTHORIZATION.json',ROOT/'supervise_anchor_evaluation.py',ROOT/'analysis/record_anchor_evaluation_launch_2026_11_11.py',Path(__file__)]
c.write(M/'LAUNCH_EVIDENCE_MANIFEST.json',dict(files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}))
print(json.dumps(dict(state=status['state'],worker_pid=status['worker_pid'],monitor=str(M/'LIVE_STATUS.txt'))))
