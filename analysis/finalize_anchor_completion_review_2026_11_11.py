"""Preserve the failed post-training gate and record verified completion scope."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'validation/anchor_development_launch_2026-11-06_01'
O=ROOT/'development_runs/terminal_anchor_development_2026-11-06_01'
D=ROOT/'validation/anchor_development_completion_check_2026-11-11_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
x=read(D/'diagnosis.json');assert x['checkpoints_verified']==85 and not x['heldout_event_present']
for src in [O/'FAILED.json',O/'ALL_FINAL_CHECKPOINTS_FREEZE.json',O/'runtime_plan.json',O/'events.json',C/'LIVE_STATUS.txt',C/'LIVE_STATUS.json',C/'worker_stderr.log']:
    shutil.copy2(src,D/(('launch_' if src.parent==C else '')+src.name))
report=ROOT/'TERMINAL_ANCHOR_DEVELOPMENT_COMPLETION_REVIEW_2026-11-11.txt'
report.write_text('''TERMINAL-ANCHOR DEVELOPMENT COMPLETION REVIEW
Date: November 11, 2026

OUTCOME
Training completed; the full development comparison did not complete.
17,000 saved training episodes account for all 680,000 training slots.
All 85 final checkpoint file hashes match the pre-evaluation freeze.
Every checkpoint's online state matches its recorded parameter digest.
All 28 runtime source files match their originals and source snapshots.
These checks establish checkpoint/source integrity, not a full trace audit.

The worker stopped on November 7 at the frozen-checkpoint export equivalence
gate. LIVE_STATUS correctly says failed_requires_review, not completed.
There is no COMPLETION.json, development comparison summary or final saved-
artifact audit. Held-out development evaluation never started according to
the stage events. Final evaluation was not launched. The trained checkpoints
are usable for further engineering diagnosis; no training rerun is needed
for the observed export failure.

REPRODUCED SAVED-WEIGHT FAILURE
Replayed the two already exported fit-0 graphs over all 128 fixed engineering
observations. Primary price_both matched all 128 joint actions. Comparator
no_price_both disagreed on one branch of one observation: observation 110,
vehicle branch 18, destination 4 in PyTorch versus destination 0 in runtime.
No other fits/arms were export-tested in this diagnosis. Do not extrapolate
the primary's fit-0 match to full-run equivalence or cost-screen success.

PyTorch branch costs for destinations 0 and 4:
0.006834309082478285 and 0.006834303494542837.
Export/native costs for those destinations:
0.006834306754171848 and 0.006834306754171848.
PyTorch distinguishes these nearly tied choices; exported arithmetic rounds
them to a tie and the decoder selects destination 0. Neural core raw outputs
also differ at FP32 precision. Supplying the original PyTorch core output
to the same native decoder restores destination 4. This localizes the observed
action change to core numerical differences at a near tie; it does not show
that all potential failures share this cause.
Q at that observation is -2.839221477508545 in PyTorch versus
-2.839221954345703 in runtime. Small value error does not satisfy the separate
exact-action condition. The frozen acceptance condition was not relaxed.
No evidence yet supports declaring the controller successful or unsuccessful
on return, deadlines, calibration or complete deployment cost.

NEXT ENGINEERING STEP
Preserve this failed attempt and all checkpoints. Diagnose and predeclare a
numerically faithful runtime correction using fixed engineering observations
and these saved weights. Investigate backend arithmetic and action margins;
do not alter critic weights, rewards, task seeds, objective or thresholds.
Test the correction on all 85 frozen models and all 128 fixed observations,
including an explicit near-tie regression. If equivalence passes, measure
complete action latency/storage using the declared fixed procedure and every
fit. Retain all failures. Do not repeat timings to seek a favorable result.
If exact equivalence cannot be obtained, record that the original deployment
gate remains failed; a route change requires an explicit prospective amendment
and new honest cost evidence, not retrospective acceptance-rule changes.
Then implement and verify a post-training continuation that loads only these
frozen checkpoints, adds no optimizer steps, preserves original failure files,
and runs the remaining comparison and full artifact audit in separately
tracked outputs. Review that continuation before opening reserved held-out
development tasks. Keep the untouched final evaluation separate.
This review does not authorize retraining, new research runs or thesis claims.

PRESERVATION AND SCOPE
Original run, failure, exports and live-status files were not edited.
Copies and diagnosis are under validation/anchor_development_completion_check_2026-11-11_01.
No simulator calls, optimizer updates or new export/timing measurements were
performed in this diagnosis; existing graphs were replayed for equivalence.
Cloud/charging stations excluded; security/trust deferred. Word thesis unchanged.
''',encoding='utf-8')
tracker=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json';backup=D/'PROJECT_STATUS_BEFORE_REVIEW.json'
shutil.copy2(tracker,backup);status=read(tracker)
status['current_workstream']='terminal_anchor_training_complete_export_equivalence_failed_requires_runtime_engineering'
status['next']='predeclare_numerically_faithful_runtime_correction_on_saved_weights_before_posttraining_continuation'
status['terminal_anchor_development_completion_review']=dict(report=report.name,training_slots=680000,training_episodes=17000,
    checkpoints_hash_verified=85,source_files_verified=28,comparison_complete=False,artifact_audit_complete=False,
    heldout_development_started=False,observed_failure='no_price_both_fit0_observation110_branch18_near_tie_export_action_mismatch',
    new_training_slots=0,new_optimizer_updates=0,final_launched=False,word_thesis_changed=False)
status['terminal_anchor_development_launch']['current_state']='training_complete_posttraining_export_gate_failed'
write(tracker,status)
paths=[p for p in D.rglob('*') if p.is_file()]+[report,Path(__file__),ROOT/'analysis/diagnose_anchor_completion_2026_11_11.py']
files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}
# Seal the original training payload as it stands, without modifying the run.
for p in O.rglob('*'):
    if p.is_file() and (p.suffix=='.pt' or p.name in ['FAILED.json','ALL_FINAL_CHECKPOINTS_FREEZE.json','runtime_plan.json','events.json']):
        files[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
write(ROOT/'validation/anchor_development_completion_review_manifest_2026-11-11.json',dict(state='training_complete_comparison_incomplete_export_gate_failed',files=files))
print(json.dumps(dict(report=report.name,sealed_files=len(files),training_complete=True,comparison_complete=False)))
