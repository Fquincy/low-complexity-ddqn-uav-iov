"""Record evaluation-only launch readiness; do not start reserved evaluation."""
import json,hashlib,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run_anchor_posttraining_continuation_v2 as c
OUT=ROOT/'validation/anchor_evaluation_readiness_fixture_2026-11-11_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    tests=c.read(ROOT/'validation/anchor_evaluation_readiness_tests_2026-11-11.json');assert tests['passed'] and tests['tests']==9
    for name,digest in tests['source_sha256'].items():assert sha(ROOT/name)==digest,name
    audit=c.read(OUT/'saved_artifact_audit.json');assert audit['episodes_checked']==244 and audit['checkpoints_checked']==34
    assert audit['optimizer_updates_checked']==2482 and audit['new_optimizer_updates']==0
    c.audit_continuation_inputs(OUT)
    source=c.RESEARCH_SOURCE
    target=ROOT/'development_runs/terminal_anchor_development_continuation_2026-11-11_01'
    control=ROOT/'validation/anchor_evaluation_launch_2026-11-11_01'
    assert not target.exists() and not control.exists()
    # Exercise the actual prepared CLI in check-only mode, including source checks.
    cmd=[sys.executable,str(ROOT/'launch_anchor_evaluation.py'),'--mode','development','--source',str(source),
         '--output',str(target),'--control',str(control),'--confirm-development-evaluation','--check-only']
    checked=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    assert checked.returncode==0,checked.stderr
    preflight=json.loads(checked.stdout);assert preflight['checkpoints']==85 and len(preflight['evaluation_seeds'])==48
    assert not target.exists() and not control.exists()
    c.write(ROOT/'validation/anchor_evaluation_check_only_2026-11-11.json',preflight)
    report=ROOT/'TERMINAL_ANCHOR_EVALUATION_LAUNCH_READINESS_REVIEW_2026-11-11.txt'
    report.write_text('''TERMINAL-ANCHOR EVALUATION-ONLY LAUNCH READINESS
Date: November 11, 2026

DECISION
Ready to resume the remaining development evaluation and full artifact audit
from the existing85 frozen checkpoints. Recommend proceeding with this
post-training continuation. This review has not launched reserved evaluation.
No retraining, optimizer updates or final evaluation is required or authorized
by the prepared development entry point. Word thesis remains unchanged.

PREPARED ENTRY POINT
launch_anchor_evaluation.py requires --mode development and an explicit
--confirm-development-evaluation flag. Final requests are rejected. --check-only
performs source validation without creating launch or research output folders.
Actual execution requires a passing readiness decision and exact reviewed
executable hashes. Generic continuation entry points remain disabled.
Separate new control/output directories prevent overwrite or automatic restart.
The executable seals an evaluation-only launch receipt linked to the original
680,000-slot launch authorization. The original failed run stays preserved.

RESEARCH PREFLIGHT
The v2 continuation accepts only the original failed development run as input.
It verifies the frozen protocol and original/runtime-repair/continuation
engineering evidence; checks all85 checkpoint file and online/target parameter
hashes; and checks each checkpoint's arm, model seed,8,000 historical slots,
6,977 historical updates,34 target copies and exact training configuration.
The original source bytes and snapshots must match. Training episode file
coverage is checked before entry; detailed trace/optimizer validity is checked
by the full post-evaluation auditor, not claimed established by preflight.
The five model seeds, disjoint200-episode training blocks,48 evaluation seeds,
diagnostic seeds, optimizer configuration, validation interval and development
phase must equal the frozen protocol. Source evaluation must not have started.
The prepared CLI passed check-only mode against the actual research source.
It created no continuation output and executed no reserved task episodes.

VERSIONED IMPLEMENTATION AND TESTS
Created run_anchor_posttraining_continuation_v2.py,
anchor_continuation_inference_v2.py and audit_anchor_posttraining_continuation_v2.py.
Predecessors remain unchanged. AST checks confirm shared post-checkpoint
orchestration and comparisons are unchanged except the explicit entry guard;
research-source preflight is added separately. Continuation source freezing
now explicitly includes the launch entry point and inherited dependencies.

Nine tests passed: reduced complete workflow/full audit with training blocked;
existing-output rejection; generic development/final CLI and API guards;
research-source rejection in fixture mode; corruption rejection; actual frozen
source preflight without simulator/optimizer access; rejection of changed seeds,
settings or phase; launcher confirmation/final guards; and unchanged shared
orchestration. The reduced new-version fixture used34 existing synthetic
checkpoints, audited244 episodes and2,482 historical optimizer updates, replayed
4,352 runtime actions and recomputed21,760 timing rows. It added zero training
slots and optimizer updates. Its74 regenerated synthetic evaluation/heuristic
episodes used2,960 simulator slots exclusively in the declared synthetic set.
It is excluded from research quality evidence. Its cost screens passed in both
synthetic fits. No favorable retiming or initialization selection occurred.

CONTINUATION BEHAVIOR
Use the already declared original-PyTorch-core/native-decoder amendment for all
native-decoder families. Coupled full-policy ORT families retain their route.
Export/check every saved checkpoint before reserved development evaluation,
and measure the fixed complete action-selection cost once. Preserve all fits,
comparators and failures. Compute the frozen per-initialization comparisons,
value distributions and all predeclared screens; finish the full artifact audit.
No changes to rewards, architecture, training, task seeds or acceptance rules.
The real training dataset's full audit remains outstanding until this executes.
Readiness establishes engineering readiness, not return/deadline/calibration
success, novelty or a supported thesis low-complexity claim.

PLANNED OUTPUTS (NOT CREATED BY THIS REVIEW)
Research: development_runs/terminal_anchor_development_continuation_2026-11-11_01
Control: validation/anchor_evaluation_launch_2026-11-11_01
Original: development_runs/terminal_anchor_development_2026-11-06_01

SCOPE
Development evaluation only. Untouched final evaluation remains separate and
disabled. Cloud/charging stations excluded; security/trust deferred.
Word thesis unchanged. Do not repeat the680,000-slot training run.
''',encoding='utf-8')
    decision=ROOT/'validation/anchor_evaluation_readiness_decision_2026-11-11.json'
    assert not decision.exists()
    c.write(decision,dict(state='ready_for_development_evaluation_only',report=report.name,tests=9,
        source_sha256=tests['source_sha256'],source=str(source),planned_output=str(target),planned_control=str(control),
        protocol_sha256=sha(c.PROTOCOL),checkpoint_freeze_sha256=sha(source/'ALL_FINAL_CHECKPOINTS_FREEZE.json'),
        runtime_amendment_sha256=sha(ROOT/'ANCHOR_TORCH_CORE_RUNTIME_AMENDMENT_2026-11-11.json'),
        research_evaluation_started=False,new_training_slots=0,new_optimizer_updates=0,final_enabled=False,word_thesis_changed=False))
    tracker=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json';backup=ROOT/'validation/project_status_before_anchor_evaluation_readiness_2026-11-11.json'
    assert not backup.exists();shutil.copy2(tracker,backup);status=c.read(tracker)
    status['current_workstream']='anchor_evaluation_only_continuation_ready_not_launched'
    status['next']='execute_reviewed_evaluation_only_continuation_from_85_frozen_checkpoints_then_review_full_audit'
    status['anchor_evaluation_launch_readiness']=dict(report=report.name,tests=9,decision=str(decision.relative_to(ROOT)),
        checkpoints=85,development_evaluation_seeds=48,research_evaluation_started=False,new_training_slots=0,
        new_optimizer_updates=0,final_enabled=False,word_thesis_changed=False)
    c.write(tracker,status)
    paths=[p for p in OUT.rglob('*') if p.is_file()]+[ROOT/n for n in tests['source_sha256']]
    paths += [report,decision,backup,Path(__file__),ROOT/'validation/anchor_evaluation_readiness_tests_2026-11-11.json',
        ROOT/'validation/anchor_evaluation_check_only_2026-11-11.json',ROOT/'analysis/prepare_anchor_evaluation_readiness_2026_11_11.py']
    files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}
    c.write(ROOT/'validation/anchor_evaluation_readiness_completion_2026-11-11.json',dict(state='ready_no_reserved_evaluation_launched',files=files))
    for name,digest in files.items():assert sha(ROOT/name)==digest,name
    print(json.dumps(dict(report=report.name,tests=9,checkpoint_preflight=85,completion_files=len(files),research_evaluation_started=False)))
if __name__=='__main__':main()
