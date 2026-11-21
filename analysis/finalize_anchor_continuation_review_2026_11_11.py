"""Seal the tested engineering continuation; never launch research evaluation."""
import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run_anchor_posttraining_continuation as c
OUT=ROOT/'validation/anchor_posttraining_continuation_fixture_2026-11-11_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    tests=c.read(ROOT/'validation/anchor_posttraining_continuation_tests_2026-11-11.json')
    audit=c.read(OUT/'saved_artifact_audit.json');verification=c.read(OUT/'verification.json')
    assert tests['passed'] and tests['tests']==5
    assert verification['new_training_slots']==verification['new_optimizer_updates']==0
    assert audit['checkpoints_checked']==34 and audit['episodes_checked']==244
    assert audit['optimizer_updates_checked']==2482
    for name,digest in tests['source_sha256'].items():assert sha(ROOT/name)==digest,name
    c.audit_continuation_inputs(OUT)
    report=ROOT/'TERMINAL_ANCHOR_POSTTRAINING_CONTINUATION_ENGINEERING_REVIEW_2026-11-11.txt'
    report.write_text('''TERMINAL-ANCHOR POST-TRAINING CONTINUATION ENGINEERING REVIEW
Date: November 11, 2026

OUTCOME
Five tests passed. The complete reduced continuation loads existing saved
weights, executes the amended runtime/export/cost gate, evaluates the isolated
synthetic task set, computes comparisons and value distributions, and completes
the full saved-training/evaluation artifact audit. No training is repeated.
Reserved development evaluation has not started. Final execution is disabled.
This result is implementation evidence, not research policy-quality evidence.

IMPLEMENTATION
run_anchor_posttraining_continuation.py reuses the unchanged seventeen-arm
comparison, value summaries, screen thresholds and batch-one greedy evaluation.
The input checkpoint set and original source snapshots remain unchanged.
A new output directory contains the continuation evidence, input hashes,
original event history, separate continuation-source snapshots and regenerated
post-training artifacts. Existing output directories are rejected.
Input trace/checkpoint payloads can share storage via hard links; the code
never writes input paths. Input hashes are rechecked against original files.
The original failed run, FAILED.json, frozen checkpoints and status are intact.

anchor_continuation_inference.py installs the prospectively declared original
PyTorch-core/native-decoder route uniformly for native-decoder families. The
four coupled full-policy ORT families retain their existing route. Sources and
runtime amendment are hash-covered before inference. Every checkpoint is
exported and checked before evaluation, including all comparators and fits.
Complete cost is measured once per fixture, using the fixed five-round schedule.

The new full auditor independently checks original training/validation traces,
optimizer histories and terminal replay accounting, checkpoint identities,
paired workload fingerprints/action streams, saved evaluation states/actions,
chosen/target values, weight invariance, all predeclared comparisons and value
summaries, exported runtime equivalence, and raw timing/order/cost calculations.
It checks both original-source and continuation-source bytes and all input
hashes. It reports historical training totals separately from zero new training.

REDUCED END-TO-END EVIDENCE
Source: validation/anchor_research_route_v2_2026-11-06_01.
Output: validation/anchor_posttraining_continuation_fixture_2026-11-11_01.
Seventeen controllers x two saved synthetic initializations:34 checkpoints.
244 episodes audited:68 preserved training,102 preserved validation,
68 regenerated evaluation and6 heuristic episodes. The74 regenerated episodes
use only the already declared synthetic engineering seeds (2,960 simulator
slots); none is a reserved research task. There were zero new training slots,
zero optimizer updates and zero target-network copies.
2,482 historical optimizer updates were audited, not rerun.
4,352 runtime action/value checks;21,760 raw timing rows recomputed.
Storage, median-latency and p95-latency screens passed in both fixture fits.
All paired workloads, evaluation invariance and saved-artifact checks passed.
Synthetic results do not establish return/deadline/calibration success.

TESTS
1. Complete continuation and full audit while optimizer construction, training
   agent construction and update calls are patched to fail on any invocation.
2. CLI and direct-API guards reject development/final execution, including
   attempts with the confirmation flag, before creating an output directory.
3. Reserved research input is rejected in fixture mode before output creation.
4. An existing output is rejected; no overwrite or automatic restart.
5. Deliberate summary corruption is rejected by the auditor without retiming;
   exact original bytes are restored and the saved manifest is reverified.
All five passed. The existing complete reduced fixture was not retrained.

REMAINING LAUNCH BOUNDARY
The research continuation path remains intentionally disabled in this gate.
Before reserved development evaluation, review this implementation and evidence
and prepare a separately versioned evaluation-only entry point. That entry
point must validate the original failed-run source, all85 frozen checkpoint
identities, the exact frozen five model/task blocks and held-out seed schedule,
and the runtime amendment. Keep final mode disabled and require explicit CLI
confirmation. Use a new continuation output; never edit the original run.
The reduced test exercises the shared post-checkpoint orchestration; it does
not claim a production evaluation run or complete audit of the real680,000-slot
training traces. Those comparisons and that audit still remain to be executed.
No additional training is necessary to perform the continuation.

SCOPE
Original rewards, architectures, objectives, optimizer settings, seeds and
thresholds unchanged. All fits/comparators retained; no best-fit selection.
Cloud/charging stations excluded. Security/trust deferred. Word thesis unchanged.
''',encoding='utf-8')
    tracker=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json';backup=ROOT/'validation/project_status_before_anchor_continuation_review_2026-11-11.json'
    assert not backup.exists();shutil.copy2(tracker,backup);status=c.read(tracker)
    status['current_workstream']='anchor_posttraining_continuation_synthetic_workflow_and_full_audit_passed_no_reserved_evaluation'
    status['next']='review_and_prepare_versioned_evaluation_only_continuation_launch_with_frozen_85_checkpoints'
    status['anchor_posttraining_continuation_engineering']=dict(report=report.name,tests=5,
        fixture=str(OUT.relative_to(ROOT)),checkpoints_audited=34,episodes_audited=244,historical_updates_audited=2482,
        new_training_slots=0,new_optimizer_updates=0,synthetic_evaluation_slots=2960,
        research_evaluation_started=False,final_launched=False,word_thesis_changed=False)
    c.write(tracker,status)
    paths=[p for p in OUT.rglob('*') if p.is_file()]
    paths += [ROOT/n for n in [*c.CONTINUATION_SOURCES,*c.original.SOURCES,'test_anchor_posttraining_continuation.py']]
    paths += [report,backup,Path(__file__),ROOT/'validation/anchor_posttraining_continuation_tests_2026-11-11.json']
    paths += [ROOT/'analysis'/n for n in ['prepare_anchor_continuation_2026_11_11.py','prepare_anchor_continuation_helpers_2026_11_11.py']]
    files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}
    c.write(ROOT/'validation/anchor_posttraining_continuation_completion_2026-11-11.json',dict(state='engineering_passed_research_continuation_disabled',files=files))
    for name,digest in files.items():assert sha(ROOT/name)==digest,name
    print(json.dumps(dict(report=report.name,tests=5,completion_files=len(files),research_evaluation_started=False)))
if __name__=='__main__':main()
