"""Record separate readiness decision without enabling research execution."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def main():
    evidence={}
    for name in ['anchor_research_preparation_completion_2026-11-06.json','anchor_runtime_overhead_completion_2026-11-06.json','anchor_runtime_route_v2_completion_2026-11-06.json']:
        path=ROOT/'validation'/name;m=read(path)
        for rel,digest in m['files'].items():assert sha(ROOT/rel)==digest,rel
        evidence[name]=dict(sha256=sha(path),files_verified=len(m['files']))
    tests=read(ROOT/'validation/anchor_launch_readiness_tests_2026-11-06.json')
    assert tests['passed'] and tests['tests']==8
    for name,digest in tests['source_sha256'].items():assert sha(ROOT/name)==digest,name
    protocol=ROOT/'TERMINAL_ANCHOR_RESEARCH_PROTOCOL_2026-11-06.json';p=read(protocol)
    amendment=ROOT/'ANCHOR_RUNTIME_ROUTE_AMENDMENT_2026-11-06.json'
    assert sha(protocol)==read(amendment)['base_protocol_sha256']
    seed=read(ROOT/'validation/anchor_research_seed_reservation_2026-11-06.json')
    assert seed['protocol_sha256']==sha(protocol) and not seed['collisions'] and not seed['errors']
    report=ROOT/'TERMINAL_ANCHOR_LAUNCH_READINESS_REVIEW_2026-11-06.txt'
    report.write_text('''TERMINAL-ANCHOR SEVENTEEN-ARM LAUNCH-READINESS REVIEW
Date: November 6, 2026

DECISION
Technically ready for a separate explicit development launch instruction.
Recommend proceeding with the frozen 680,000-slot comparison. No research
execution was enabled or launched in this review. Final execution remains
disabled. This is an engineering readiness decision, not a policy result.

SOURCE COVERAGE CORRECTION
Created run_anchor_research_route_v3.py, audit_anchor_research_route_v3.py
and test_anchor_research_route_v3.py, preserving every predecessor.
The prototype's transitive model factory, run_v3_research_gate_v2.py,
is now included directly in SOURCES. Its hash is checked against the
original terminal-anchor completion evidence before output creation.
The auditor requires an exact match between the recorded and declared
source sets and verifies each original and snapshot hash.
The frozen scientific protocol and runtime-route amendment are unchanged.

EVIDENCE REVIEWED
The previous complete reduced workflow passed six tests: seventeen arms,
two synthetic initializations, 2,720 slots, 2,482 updates, 306 target copies,
34 final checkpoints, 244 audited episodes, 4,352 exported action checks
and 21,760 recomputed timing rows. Storage/median/p95 screens passed in
both fits. Synthetic policy/value screens were not all passing and are
not evidence for research success or a basis for tuning.
Eight focused checks passed in this review, with zero simulator steps
and optimizer updates. They check replay and screen arithmetic, runtime
dispatch, protocol scope, CLI/direct-API guards, complete source snapshots
before the first model, and rejection of changed model-factory bytes
before an output folder exists. AST comparisons confirm all shared runner
functions are unchanged except the added provenance check in run_plan.
The full simulator fixture was not repeated for this provenance-only change.
All file hashes in the three preceding completion manifests were rechecked.

FROZEN COMPARISON
Seventeen arms x five initializations x 200 episodes x 40 slots:
680,000 training slots, 593,045 optimizer updates, 2,890 target copies,
85 final checkpoints. Keep all fits and all comparators, including failures.
Same paired training blocks and validation/evaluation schedules, unchanged
rewards, optimizer, thresholds, architecture and terminal-anchor objective.
Candidate-specific value screening is the already declared prospective
amendment; historical eleven-arm failed results retain their original rules.
Gate complete final-checkpoint action selection cost and exported actions
before held-out development evaluation. No best-fit/checkpoint selection.
Fresh development model seeds: 61001, 61011, 61017, 61023, 61039.
The saved bounded seed audit checked 28,436 files and found no reserved
seed collisions. Its metadata/header-only limitations remain; this review
does not claim a new exhaustive seed scan. No reserved seeds were executed
in this engineering review. Untouched final task episodes were not opened.

LAUNCH BOUNDARY
A separate explicit instruction to launch this 680,000-slot development
comparison is still required. The CLI remains disabled even with its
confirmation flag during engineering; after explicit authorization, prepare
a versioned development-only entry point with its confirmation guard,
record exact executable hashes, preserve the evidence chain, and launch
once. Keep final mode disabled. Research payload hashes must match this
review or a separately documented source-only launcher change.
Research outcomes must pass the predeclared return, deadline, value and
cost/mechanism screens before any claim of low-complexity improvement.
A separate amended final-evaluation protocol and authorization follow only
if development evidence supports proceeding. Novelty and thesis acceptance
are not established by engineering readiness.

SCOPE
Local/UAV/RSU offloading only; no cloud or charging stations.
Security/trust deferred to the second contribution. Word thesis unchanged.
''',encoding='utf-8')
    decision=ROOT/'validation/anchor_launch_readiness_decision_2026-11-06.json'
    write(decision,dict(state='technically_ready_pending_explicit_development_launch',report=report.name,
        protocol_sha256=sha(protocol),runtime_amendment_sha256=sha(amendment),
        prior_completion_manifests=evidence,focused_tests=8,new_simulator_slots=0,optimizer_updates=0,
        full_fixture_repeated=False,full_fixture_evidence='validation/anchor_research_route_v2_2026-11-06_01',
        source_sha256=tests['source_sha256'],development_training_slots=680000,
        research_enabled=False,research_launched=False,final_enabled=False,word_thesis_changed=False))
    tracker=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json'
    backup=ROOT/'validation/project_status_before_anchor_launch_readiness_2026-11-06.json'
    assert not backup.exists();shutil.copy2(tracker,backup)
    status=read(tracker);status['current_workstream']='terminal_anchor_technically_ready_pending_explicit_development_launch'
    status['next']='explicit_launch_instruction_required_for_frozen_680000_slot_development_comparison'
    status['anchor_launch_readiness']=dict(report=report.name,decision=str(decision.relative_to(ROOT)),focused_tests=8,
        transitive_factory_in_pre_run_freeze=True,development_slots=680000,research_launched=False,word_thesis_changed=False)
    write(tracker,status)
    paths=[ROOT/name for name in tests['source_sha256']]
    paths += [report,decision,backup,Path(__file__),ROOT/'analysis/prepare_anchor_launch_readiness_2026_11_06.py',ROOT/'analysis/test_anchor_launch_readiness_2026_11_06.py',ROOT/'validation/anchor_launch_readiness_tests_2026-11-06.json']
    files={str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for path in paths}
    write(ROOT/'validation/anchor_launch_readiness_completion_2026-11-06.json',dict(state='ready_no_launch',files=files))
    print(json.dumps(dict(report=report.name,focused_tests=8,prior_manifests_verified=evidence,completion_files=len(files),research_launched=False)))
if __name__=='__main__':main()
