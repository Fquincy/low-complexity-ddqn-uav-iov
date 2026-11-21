"""Verify saved engineering evidence and record a decision without training."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    completion=ROOT/'validation/v3_engineering_gate_v2_completion_2026-11-06/COMPLETION_MANIFEST.json'
    files=read(completion)['files']
    for name,digest in files.items():assert sha(ROOT/name)==digest,name
    protocol=ROOT/'LOW_COMPLEXITY_V3_RESEARCH_PROTOCOL_2026-11-05.json'
    p=read(protocol)
    assert sha(protocol)=='5feece24ac058b547140b0513f4991a3f8035536b59b4002893f0ae1fc8d96d9'
    tests=read(ROOT/'validation/v3_engineering_gate_v2_tests_2026-11-06.json')
    assert tests['passed'] and tests['tests']==7 and tests['errors']==tests['failures']==0
    for name,digest in tests['source_sha256'].items():assert sha(ROOT/name)==digest,name
    fixture=ROOT/'validation/v3_multifit_engineering_2026-11-06_01'
    audit=read(fixture/'saved_artifact_audit.json')
    assert audit['state']=='passed_v3_synthetic_shared_research_audit'
    assert audit['checkpoints_checked']==22 and audit['episodes_checked']==160
    assert not audit['research_quality_evidence']
    seeds=read(ROOT/'validation/v3_seed_consumption_review_2026-11-06.json')
    assert seeds['protocol_sha256']==sha(protocol) and not seeds['collisions'] and not seeds['errors']
    assert p['development']['training_slots']==11*5*200*40==440000
    assert p['development']['optimizer_updates']==383735
    assert p['development']['target_copies']==1870
    runner=(ROOT/'run_v3_research_gate_v2.py').read_text()
    assert 'RESEARCH_ENABLED=False' in runner and 'Research modes are disabled in this preparation gate' in runner
    assert not p['scope']['cloud'] and not p['scope']['charging_stations'] and not p['scope']['security_trust']
    evidence=dict(decision='recommend_frozen_development_launch_separate_explicit_instruction_required',
        completion_manifest_entries_verified=len(files),completion_manifest_sha256=sha(completion),
        protocol_sha256=sha(protocol),saved_tests_passed=7,tests_rerun_this_review=False,
        checkpoints_audited=22,synthetic_episodes_audited=160,
        development_training_slots=440000,development_optimizer_updates=383735,
        development_target_copies=1870,development_model_fits=55,
        bounded_seed_audit_files_checked=seeds['files_checked'],seed_audit_limitations=seeds['limitations'],
        research_execution_enabled=False,research_training_launched=False,thesis_changed=False,
        final_phase_authorized=False,low_complexity_claim_established=False)
    output=ROOT/'validation/v3_development_launch_decision_2026-11-06.json'
    output.write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    report=ROOT/'LOW_COMPLEXITY_V3_DEVELOPMENT_LAUNCH_DECISION_2026-11-06.txt'
    report.write_text('''VERSION 3 DEVELOPMENT LAUNCH DECISION
Date: November 6, 2026

DECISION
Recommend proceeding with the frozen eleven-controller development comparison.
The engineering corrections and reduced shared-workflow audit provide enough
implementation evidence to justify that research experiment. No additional
synthetic training is required for the identified blockers. Research training
has NOT been launched by this review.

CURRENT VERIFICATION
All471 engineering completion-manifest entries match their files. The frozen
protocol hash and all source hashes in the saved seven-test evidence match.
The shared synthetic auditor checked22 final checkpoints and160 episodes;
these artifacts remain intact. The tests were not rerun in this decision review.
The recorded seed audit found no reserved hits in12,968 checked files, subject
to its explicit limitations for binary files, external history and unlogged use.

EXPERIMENT TO LAUNCH
5 paired initializations x11 controllers x200 episodes x40 slots:
440,000 training slots;383,735 optimizer updates;1,870 target copies.
Preserve all55 final checkpoints. Freeze every checkpoint before held-out
evaluation. Use the declared48 common development evaluation workloads and
all predeclared per-fit quality, mechanism, value-stability and cost screens.
Retain failing fits and report every arm; no best-seed selection.

No rewards, model definitions, thresholds, optimizer settings or seed schedules
were changed in this review or selected from synthetic performance. Cloud and
charging stations remain excluded. Security/trust remains deferred to the
second contribution. The Word thesis is unchanged.

WHY RUN IT
The experiment will measure whether the proposed pricing controller preserves
return/deadline performance while reducing storage and action-selection cost,
and whether pricing improves on its matched no-price ablation. The smaller
plain24 reference remains a required comparison. The synthetic engineering
success cannot answer those policy-performance or contribution questions.

LAUNCH BOUNDARY
The current engineering runner deliberately blocks research modes even with
confirmation. A separate explicit instruction to launch development is still
required under the frozen protocol. On that instruction, enable the development
entrypoint with explicit confirmation and frozen-plan/source checks, prepare
persistent logs/status and execute the frozen schedule. Keep the final phase
disabled pending its own later review and explicit instruction. This review
does not enable execution or authorize final evaluation.

Exact next instruction, if proceeding:
Launch the frozen 440,000-slot Version 3 development comparison.

AFTER DEVELOPMENT
Audit the saved traces and artifacts, report every initialization and decide
whether all predeclared development screens pass. Only then consider the
separately reserved final experiment. A successful development run alone
does not establish novelty, thesis sufficiency or a proven lightweight claim.

EVIDENCE
validation/v3_development_launch_decision_2026-11-06.json
validation/v3_engineering_gate_v2_completion_2026-11-06/COMPLETION_MANIFEST.json
LOW_COMPLEXITY_V3_RESEARCH_PROTOCOL_2026-11-05.json
''',encoding='utf-8')
    print(json.dumps(dict(evidence=evidence,report=report.name),indent=2))

if __name__=='__main__':main()
