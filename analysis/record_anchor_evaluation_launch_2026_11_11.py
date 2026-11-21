"""Verify readiness and record explicit evaluation/audit launch authorization."""
import datetime,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import launch_anchor_evaluation as launch
import run_anchor_posttraining_continuation_v2 as c
M=ROOT/'validation/anchor_evaluation_monitor_2026-11-11_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not M.exists()
    decision=c.read(launch.DECISION);assert decision['state']=='ready_for_development_evaluation_only'
    for name,digest in c.read(ROOT/'validation/anchor_evaluation_readiness_completion_2026-11-11.json')['files'].items():assert sha(ROOT/name)==digest,name
    for name,digest in decision['source_sha256'].items():assert sha(ROOT/name)==digest,name
    checked=launch.preflight(decision['source'],decision['planned_output'])
    assert not Path(decision['planned_control']).exists()
    M.mkdir()
    shutil.copy2(ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json',M/'PROJECT_STATUS_BEFORE_LAUNCH.json')
    c.write(M/'USER_AUTHORIZATION.json',dict(user_instruction='Next, run the remaining development evaluation and full artifact audit. No retraining is needed.',
        explicit_evaluation_and_full_audit=True,recorded_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        source=decision['source'],output=decision['planned_output'],control=decision['planned_control'],
        readiness_decision_sha256=sha(launch.DECISION),preflight=checked,
        executable_sha256={**decision['source_sha256'],'supervise_anchor_evaluation.py':sha(ROOT/'supervise_anchor_evaluation.py')},
        new_training_slots=0,new_optimizer_updates=0,final_evaluation_authorized=False,thesis_changed=False))
    print(json.dumps(dict(monitor=str(M),preflight=checked['state'],checkpoints=85,controller_episodes=4080,heuristic_episodes=144)))
if __name__=='__main__':main()
