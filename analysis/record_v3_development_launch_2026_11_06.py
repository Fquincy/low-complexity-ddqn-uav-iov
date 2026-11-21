"""Record verified launch state and preserve the preceding tracker bytes."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'validation/v3_development_launch_2026-11-06_01'
OUTPUT=ROOT/'development_runs/controller_v3_development_2026-11-06_01'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2),encoding='utf-8')

def main():
    plan=read(OUTPUT/'runtime_plan.json');spec=read(ROOT/'LOW_COMPLEXITY_V3_RESEARCH_PROTOCOL_2026-11-05.json')
    assert plan['kind']=='RESEARCH' and plan['phase']=='development' and plan['training_launched'] and not plan['fixture']
    assert plan['model_seeds']==spec['development']['model_seeds'] and plan['train_blocks']==spec['development']['train_blocks']
    assert plan['evaluation_seeds']==list(range(420000000,420000048))
    assert plan['diagnostic_validation_seeds']==spec['development']['diagnostic_validation_seeds']
    assert plan['episodes']==200 and plan['cfg']['learning_rate']==.0001 and plan['cfg']['warmup']==1024
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==sha(OUTPUT/'source_snapshot'/name)==digest,name
    assert sha(OUTPUT/'PROJECT_STATUS_BEFORE.json')==sha(CONTROL/'PROJECT_STATUS_BEFORE_LAUNCH.json')
    saved=list(OUTPUT.glob('*/fit_*/train_*.json.gz'));assert saved
    assert not (OUTPUT/'FAILED.json').exists()
    current=read(CONTROL/'LIVE_STATUS.json');assert current['state']=='running' and current['worker_exit_code'] is None
    write(CONTROL/'LAUNCH_VERIFIED.json',dict(verified_at=datetime.datetime.now().astimezone().isoformat(),
        phase='development',training_launched=True,training_episodes_saved_at_check=len(saved),
        protocol_sha256=sha(ROOT/'LOW_COMPLEXITY_V3_RESEARCH_PROTOCOL_2026-11-05.json'),
        source_hashes_checked=len(plan['source_sha256']),runtime_plan_sha256=sha(OUTPUT/'runtime_plan.json'),
        supervisor_pid=current['supervisor_pid'],worker_pid=current['worker_pid'],final_phase_launched=False,thesis_changed=False))
    report=ROOT/'LOW_COMPLEXITY_V3_DEVELOPMENT_LAUNCH_NOTE_2026-11-06.txt'
    report.write_text('''VERSION 3 DEVELOPMENT COMPARISON LAUNCHED
Date: November 6, 2026

The explicitly authorized frozen development comparison is running locally in
a persistent background process. Initialization completed and training episode
files are being saved. The11 controller arms, five paired initializations,
200 episodes per fit and40 slots per episode give440,000 training slots.
Expected optimizer updates:383,735; target copies:1,870; final checkpoints:55.

Preflight verified the471-entry engineering completion manifest using the
archived pre-launch tracker, the frozen protocol hash, source hashes, saved
test evidence and seed-audit evidence. Missing development confirmation was
rejected; final execution was rejected; a confirmed check-only request passed
without creating research output. No research models, rewards, thresholds,
optimizer settings or seed schedules were modified to launch this experiment.

The new authorized entrypoint uses the tested Version3 revision. The original
engineering runner remains disabled when invoked directly. The launch wrapper
enables development execution only for this confirmed process; it offers no
final-phase execution. New entrypoint/supervisor sources and all training
sources are snapshotted and hashed inside the research output.

MONITORING
Open validation/v3_development_launch_2026-11-06_01/LIVE_STATUS.txt
or LIVE_STATUS.json. The supervisor refreshes them about every15 seconds.
They report fully saved training episodes, training slots, final checkpoints,
the latest runner event and process IDs. The current incomplete episode is
not included, so the slot count may lag by up to39 slots.

PowerShell live view from the Low Complexity folder:
Get-Content '.\validation\v3_development_launch_2026-11-06_01\LIVE_STATUS.txt' -Wait
For continuously refreshed contents, reopen the file or use a polling loop;
the supervisor replaces the file atomically, so -Wait can retain an older file
handle on some systems.

Logs:
validation/v3_development_launch_2026-11-06_01/worker_stdout.log
validation/v3_development_launch_2026-11-06_01/worker_stderr.log
validation/v3_development_launch_2026-11-06_01/supervisor_stderr.log

Research output:
development_runs/controller_v3_development_2026-11-06_01/

After training, the process checks exported frozen checkpoints, runs paired
held-out development evaluation, computes comparisons/value distributions and
audits saved artifacts. Training reaching100% is not completion of those
later stages. The monitor reports completed_and_audited only after success;
COMPLETION.json records the runner/auditor results. A protocol performance
screen can fail while the saved-artifact audit passes; all outcomes are kept.
On process failure the supervisor reports failed_requires_review. No automatic
restart, overwrite or checkpoint selection occurs.

The run uses local files and does not require an internet connection. Keep
the computer powered and prevent sleep/hibernation while it runs. The current
final-only checkpoint policy does not support automatic mid-fit recovery.

The prior tracker bytes are preserved in PROJECT_STATUS_BEFORE_LAUNCH.json
in the launch control folder so the preceding completion manifest remains
verifiable after the live project tracker advances.

Cloud and charging stations remain excluded. Security/trust remains deferred
to contribution two. The final experiment has not been authorized or launched.
The Word thesis is unchanged. This launch does not establish the contribution;
the saved results and every declared screen must still be reviewed.
''',encoding='utf-8')
    status=read(ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json')
    status['current_workstream']='contribution_one_v3_frozen_440000_slot_development_comparison_running'
    status['next']='monitor_development_then_audit_all_fits_and_predeclared_screens_before_final_decision'
    status['v3_development_launch_2026_11_06']=dict(state='running',report=report.name,
        control=str(CONTROL.relative_to(ROOT)).replace('\\','/'),output=str(OUTPUT.relative_to(ROOT)).replace('\\','/'),
        live_status=str((CONTROL/'LIVE_STATUS.txt').relative_to(ROOT)).replace('\\','/'),
        research_training_launched=True,total_training_slots=440000,final_phase_launched=False,
        word_thesis_changed=False,performance_claim_established=False,
        prior_tracker_archive=str((CONTROL/'PROJECT_STATUS_BEFORE_LAUNCH.json').relative_to(ROOT)).replace('\\','/'))
    write(ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json',status)
    static=[CONTROL/'LAUNCH_AUTHORIZATION.json',CONTROL/'PREFLIGHT_AND_ENTRYPOINT_TESTS.json',
        CONTROL/'PROJECT_STATUS_BEFORE_LAUNCH.json',CONTROL/'LAUNCH_VERIFIED.json',report,
        ROOT/'launch_v3_development.py',ROOT/'supervise_v3_development.py',Path(__file__),
        ROOT/'analysis/prepare_v3_authorized_launch_2026_11_06.py']
    write(CONTROL/'LAUNCH_EVIDENCE_MANIFEST.json',dict(files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in static},
        excluded_live_files=['LIVE_STATUS.txt','LIVE_STATUS.json','worker_stdout.log','worker_stderr.log','supervisor_stdout.log','supervisor_stderr.log']))
    print(json.dumps(dict(state='running_launch_verified',saved_training_episodes=len(saved),
        report=report.name,live_status=str(CONTROL/'LIVE_STATUS.txt')),indent=2))

if __name__=='__main__':main()
