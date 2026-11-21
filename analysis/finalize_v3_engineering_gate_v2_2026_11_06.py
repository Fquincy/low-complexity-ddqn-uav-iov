"""Archive engineering evidence and advance the tracker without research launch."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def main():
    tests_path=ROOT/'validation/v3_engineering_gate_v2_tests_2026-11-06.json'
    tests=read(tests_path);assert tests['passed'] and tests['tests']==7 and not tests['errors'] and not tests['failures']
    fixture=ROOT/'validation/v3_multifit_engineering_2026-11-06_01'
    result=read(fixture/'verification.json');audit=read(fixture/'saved_artifact_audit.json')
    assert audit['state']=='passed_v3_synthetic_shared_research_audit'
    assert result['training_slots']==1760 and result['optimizer_updates']==1606
    assert audit['episodes_checked']==160 and audit['checkpoints_checked']==22
    assert not result['research_training_launched'] and not audit['research_quality_evidence']
    for name,digest in tests['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in read(fixture/'run_manifest.json')['files'].items():assert sha(fixture/name)==digest,name
    previous=read(ROOT/'validation/v3_research_preparation_2026-11-05_COMPLETION_MANIFEST.json')
    for name,digest in previous['files'].items():assert sha(ROOT/name)==digest,name
    seed_path=ROOT/'validation/v3_seed_consumption_review_2026-11-06.json';seeds=read(seed_path)
    assert not seeds['collisions'] and not seeds['errors']
    archive=ROOT/'validation/v3_engineering_gate_v2_completion_2026-11-06'
    archive.mkdir(exist_ok=False)
    tracker=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json'
    shutil.copy2(tracker,archive/'PROJECT_STATUS_AT_PREPARATION.json')
    manifest_mapping={'LOW_COMPLEXITY_PROJECT_STATUS.json':str((archive/'PROJECT_STATUS_AT_PREPARATION.json').relative_to(ROOT)).replace('\\','/')}
    write(archive/'historical_manifest_mapping.json',dict(
        manifest='validation/v3_research_preparation_2026-11-05_COMPLETION_MANIFEST.json',
        preserved_file_mapping=manifest_mapping,
        reason='The status tracker advances; archived bytes preserve the prior completion manifest.'))
    report=ROOT/'LOW_COMPLEXITY_V3_ENGINEERING_CORRECTION_REVIEW_2026-11-06.txt'
    report.write_text('''VERSION 3 ENGINEERING CORRECTION GATE
Date: November 6, 2026

OUTCOME
The corrected runner and shared results auditor passed the reduced multi-fit
engineering workflow and targeted failure checks. Research training remains
disabled and has not been launched. The frozen research protocol, rewards,
model definitions, thresholds, optimizer settings and research seed schedules
are unchanged. The Word thesis is unchanged.

CORRECTIONS
Historical compatibility hashes now use the original archived status bytes;
the current project tracker is saved separately for each new run. Exported
graphs, extra decoder arrays, backend settings and complete arm order are
frozen before the first timed call and checked afterward. The auditor verifies
that ordering and payload integrity. It recomputes statistics from saved
timings without remeasuring latency.

Overall research success now requires all frozen quality AND cost screens.
The auditor reports empirical_protocol_screens_passed rather than asserting
novelty or thesis sufficiency. Duplicate, missing and unpaired comparison rows
are rejected. Per-fit comparison identities must be complete. The original
per-arm value-screen requirement is retained.

The same plan-driven runner and auditor cover research and the new synthetic
fixture. The auditor checks episode identities and counts, replay wrapping,
training/held-out workload pairing, actor/RNG pairing, final-only checkpoint
inventory, exact greedy evaluation decisions, replayed Q values, per-fit
comparisons, value distributions/time bins, checkpoint invariance, exports,
timing order and cost calculations. Synthetic artifacts cannot establish
research quality and carry explicit nonresearch labels.

VERIFICATION
Seven tests passed. The isolated temporary fixture was retained for inspection:
2 initializations x11 controllers x2 episodes x40 slots =1,760 synthetic
training slots;1,606 optimizer updates;198 target copies;22 final checkpoints.
The shared auditor checked160 episodes (44 training,66 diagnostic validation,
44 held-out controller evaluation and6 heuristic episodes). It replayed2,816
complete export decisions and recalculated14,080 original timing rows.
This fixture exercised replay-buffer wrapping and both initial and periodic
diagnostic validation. All synthetic checkpoint inference cost screens passed;
their latency measurements do not establish trained research performance.

Tests also confirmed that cost failure prevents overall success, incomplete
comparisons are rejected, and checkpoint inventory corruption, invalid timing
stage order, decoder payload changes and manifest tampering are detected.
Development and final CLI calls, with and without confirmation, all rejected
execution without creating output. Direct API research calls were also blocked.

The137 prior preparation manifest entries and114 prior compatibility entries
were preserved. The tracker snapshot mapping in the completion folder allows
the November5 preparation manifest to remain verifiable after status advances.
New versioned files preserve the original runner, auditor, tests and reports.

SEED CHECK
A refreshed bounded local metadata/episode-header audit found no reserved
development or final seed hits in checked saved runs. Its JSON records counts,
folders and exclusions. It covers selected runtime metadata and compressed
episode headers; it does not establish absence of seed use in tensor files,
external folders or unlogged executions. Those limitations remain explicit.

NEXT DECISION
The identified engineering blockers are resolved within this fixture's tested
scope. The next step is a separate launch-readiness decision for the frozen
440,000-slot development comparison (55 fits). Research CLI modes remain
disabled even with confirmation in this engineering revision; a later explicit
launch instruction must be implemented with confirmation and frozen-plan
checks before executing research. No final phase is authorized.

The first contribution still needs fresh return/deadline, mechanism, value and
cost evidence across all five research initializations. No initialization may
be discarded or chosen by outcome. Passing engineering tests does not prove
novelty or guarantee supervisor acceptance.

Scope: local CPU, UAV and RSU execution. Cloud and charging stations remain
excluded. Security/trust remains deferred to contribution two.

FILES
run_v3_research_gate_v2.py
audit_v3_research_gate_v2.py
test_v3_research_gate_v2.py
validation/v3_multifit_engineering_2026-11-06_01/
validation/v3_engineering_gate_v2_tests_2026-11-06.json
validation/v3_seed_consumption_review_2026-11-06.json
validation/v3_engineering_gate_v2_completion_2026-11-06/
''',encoding='utf-8')
    status=read(tracker)
    status['current_workstream']='contribution_one_v3_engineering_corrections_passed_synthetic_multifit_no_research_launch'
    status['next']='separate_launch_readiness_decision_for_frozen_440000_slot_development_comparison'
    status['v3_engineering_correction_gate_2026_11_06']=dict(
        state='passed_shared_multifit_engineering_audit',report=report.name,
        runner='run_v3_research_gate_v2.py',auditor='audit_v3_research_gate_v2.py',
        tests=tests['tests'],synthetic_fits=22,synthetic_training_slots=1760,
        synthetic_optimizer_updates=1606,synthetic_episodes_audited=160,
        complete_export_actions_audited=2816,original_timing_rows_recomputed=14080,
        prior_preparation_tracker_archive=manifest_mapping['LOW_COMPLEXITY_PROJECT_STATUS.json'],
        research_training_launched=False,final_training_launched=False,
        word_thesis_changed=False,low_complexity_contribution_established=False,
        seed_consumption_audit=seed_path.name,protocol_sha256=tests['protocol_sha256'])
    write(tracker,status)
    sources=[ROOT/name for name in tests['source_sha256']]
    files=list(fixture.rglob('*'))+list(archive.rglob('*'))+sources+[
        report,tracker,tests_path,seed_path,ROOT/'analysis/audit_v3_seed_consumption_2026_11_06.py',Path(__file__),
        ROOT/'analysis/build_v3_engineering_revision_2026_11_06.py']
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files if p.is_file()}
    write(archive/'COMPLETION_MANIFEST.json',dict(state='engineering_complete_no_research_launch',files=manifest))
    for name,digest in manifest.items():assert sha(ROOT/name)==digest,name
    print(json.dumps(dict(state='engineering_complete_no_research_launch',tests=tests['tests'],
        manifest_entries=len(manifest),seed_files_checked=seeds['files_checked'],report=report.name),indent=2))

if __name__=='__main__':main()
