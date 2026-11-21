"""Archive preparation evidence, without enabling or launching research."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/anchor_research_preparation_2026-11-06_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')

def main():
    tests=read(ROOT/'validation/anchor_research_preparation_tests_2026-11-06.json')
    assert tests['passed'] and tests['tests']==5
    for name,digest in tests['source_sha256'].items():assert sha(ROOT/name)==digest,name
    audit=read(OUT/'saved_artifact_audit.json');result=read(OUT/'verification.json')
    assert audit['state']=='passed_v3_synthetic_shared_research_audit'
    assert audit['episodes_checked']==244 and result['training_slots']==2720 and result['optimizer_updates']==2482
    assert result['checkpoint_count']==34 and result['target_copies']==306
    for name,digest in read(OUT/'run_manifest.json')['files'].items():assert sha(OUT/name)==digest,name
    seeds=read(ROOT/'validation/anchor_research_seed_reservation_2026-11-06.json')
    assert not seeds['collisions'] and not seeds['errors']
    assert seeds['protocol_sha256']==tests['protocol_sha256']
    inference=read(OUT/'FROZEN_CHECKPOINT_INFERENCE.json')
    report=ROOT/'TERMINAL_ANCHOR_RESEARCH_PREPARATION_REVIEW_2026-11-06.txt'
    report.write_text('''TERMINAL-ANCHOR SEVENTEEN-CONTROLLER RESEARCH PREPARATION
Date: November 6, 2026

DECISION
The prospective development protocol, separate runner, results auditor and
reduced paired workflow passed this engineering preparation gate. Research
execution remains disabled, including with confirmation. No680,000-slot
development experiment or final experiment was launched. A separate review
and explicit launch decision remain required. The Word thesis is unchanged.

FROZEN COMPARISON
Protocol: TERMINAL_ANCHOR_RESEARCH_PROTOCOL_2026-11-06.json.
Keep the original eleven controllers and add the six declared corrections:
priced horizon+anchor candidate, matched no-price horizon+anchor, priced
horizon-only, priced anchor-only, corrected plain24 and corrected compact64.
Keep all three heuristics. The primary is price_both. All original eleven
arms rerun on the fresh paired schedule; no old result is substituted for
a new fit, and no failed initialization is discarded.

Five model initializations x17 controllers x200 episodes x40 slots:
680,000 training slots,593,045 optimizer updates,2,890 target copies and85
final-only checkpoints. Evaluation uses48 common development workloads.
All checkpoints freeze before held-out evaluation. Diagnostic validation is
recorded initially and every50 episodes, without weight selection.
Adam1e-4, gamma1, Huber1, clip10, replay50,000, batch128, warmup1024 and
target interval200 retain the original research settings. Corrected anchored
arms add32 uniformly sampled terminal transitions from an independent RNG,
with terminal-loss coefficient1 and no extra optimizer steps. Replay arrays,
dtypes and original uniform sampling stream are retained for fair pairing;
only the declared horizon/objective changes apply to corrected arms.

COMPARISON AND VALUE RULES
Candidate must meet the original per-fit return>=-2 and deadline>=-0.01
differences against the five original required references AND corrected
plain24/compact64. Against each heuristic, both differences must be>=0 in
every fit. Against corrected no-price, both differences must be>=0 per fit,
and mean improvement must be>=2 return units and>=0.01 deadline fraction.
Both correction ablations must also have candidate-minus-ablation return
and deadline>=0 in every fit; all contrasts remain reported.

The explicitly prospective candidate value gate uses the unchanged numeric
thresholds: normalized RMSE<=.25, absolute bias<=2, near-zero/large-return
fraction<=.05 and terminal error p95<=2. All comparator values, time bins and
distributions remain reported, finite, and nonpositive where applicable.
This candidate-specific gating replaces the old all-arm value gate ONLY in
this new experiment. The old Version3 run remains failed under its own rule.
Comparator instability never excuses candidate failure. No final-phase
protocol is amended or authorized by this change.

COMPLETE INFERENCE AND COST RULES
Every final checkpoint is re-exported and checked against128 fixed existing
engineering observations. The four historical coupled arms retain complete
ORT graphs; other arms retain native core/decoder paths. Corrected paths add
full state/time checks and horizon scaling. Exact actions and Q atol1e-6/
rtol1e-5 are required. Graphs, decoder arrays, backend settings and timing
order freeze before timing. Use five rounds,32 warmups and128 calls per round,
single-thread CPU. Candidate online parameter bytes must be<=50% compact64,
median action latency<=wide128 and p95<=1ms in every fit. Quality AND costs
must pass the aggregate decision. Timing is audited from original samples,
without replacing samples with a favorable retiming run.
Peak process memory and training-time superiority are not established by
these parameter/latency criteria; do not make those claims.

FRESH DEVELOPMENT SEED RESERVATION
Model seeds:61001,61011,61017,61023,61039.
Training blocks:200 episodes each starting610000000,611000000,612000000,
613000000,614000000. Diagnostic validation621000000..621000004.
Held-out development620000000..620000047. Source protocol status says pending
collision audit because it was written before scanning; the separate seed
reservation evidence now records the bounded check. Do not edit the frozen
protocol hash merely to replace that descriptive status string.
The scan found no reserved development/final hits in the checked metadata and
compressed episode headers. It excludes binary tensor contents, external
history and unlogged executions; absence of all historical use is not proven.
Existing final510xx/520xx reservations remain untouched and final execution
is unavailable. A future amended final protocol requires separate review.

ENGINEERING TEST EVIDENCE
Five tests passed, covering protocol arithmetic/scope, original replay-index
and dtype equivalence, candidate-specific value screens and aggregate cost
failure, research CLI/API blocking, and full paired orchestration/audit.
Reduced fixture:2 synthetic initializations x17 arms x2 episodes x40 slots:
2,720 training simulator slots,2,482 updates,306 target copies,34 checkpoints.
The shared research auditor checked244 episodes:68 training,102 diagnostic
validation,68 held-out controller episodes and6 heuristic episodes.
It replayed4,352 complete exported decisions and recalculated21,760 timing
rows. It checks checkpoint freeze/order, exact inventory, paired workloads,
actor/RNG streams, evaluation Q/greedy actions, value summaries/comparisons,
anchor replay membership/eviction and loss arithmetic. A deliberate checkpoint
inventory corruption was rejected even with the outer manifest check bypassed.
No synthetic outcome was used to select weights, tune rules or claim quality.

LIMITATIONS AND NEXT DECISION
This gate verifies the reduced workflow; it cannot establish full-run learning
quality, novelty or supervisor acceptance. Candidate-only value screens and
additional correction-ablation rules are prospective scientific decisions and
are now explicit before new research. Review the new protocol, source snapshot,
seed check and fixture evidence before a separate explicit launch instruction.
Do not launch from an unconfirmed Next message. Final execution stays disabled.

Historical experiments, original models and the failed440,000-slot results
remain preserved. Cloud and charging stations remain excluded. Security/trust
remains deferred to contribution two. Word thesis unchanged.

FILES
run_anchor_research_preparation.py
audit_anchor_research_preparation.py
test_anchor_research_preparation.py
validation/anchor_research_preparation_2026-11-06_01/
validation/anchor_research_preparation_tests_2026-11-06.json
validation/anchor_research_seed_reservation_2026-11-06.json
''',encoding='utf-8')
    cost_lines=['','COST CAVEAT FROM THIS FIXTURE',
        'The workflow and auditor passed, but the synthetic cost screens did not',
        'all pass. Fit1 (second initialization) failed the median<=wide128 criterion.',
        'Storage and p95 passed in both fits. These are fixed-input engineering',
        'timings, not policy-quality outcomes. No timing rerun, threshold change or',
        'weight selection was performed to erase the failure.',
        'fit | primary median ms | wide128 median ms | all cost checks pass']
    for i in range(2):
        primary=inference['metrics']['price_both'][i];wide=inference['metrics']['additive128_unbounded'][i]
        cost_lines.append(f"{i} | {primary['median_ms']:.6f} | {wide['median_ms']:.6f} | {all(inference['screens_per_fit'][str(i)].values())}")
    cost_lines+=['Before any launch recommendation, review the wrapper timing path and',
        'this original sample. If implementation overhead is corrected, preserve',
        'these records and use a prospectively declared engineering check with',
        'unchanged model outputs and thresholds. Do not claim the cost gate passed.',
        'Research launch is not recommended by this preparation report.']
    with report.open('a',encoding='utf-8') as stream:stream.write('\n'.join(cost_lines)+'\n')
    tracker=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json';backup=ROOT/'validation/project_status_before_anchor_research_preparation_2026-11-06.json'
    assert not backup.exists();shutil.copy2(tracker,backup)
    s=read(tracker);s['current_workstream']='terminal_anchor_seventeen_arm_research_preparation_passed_no_launch'
    s['next']='review_synthetic_median_latency_failure_and_frozen_protocol_before_any_launch_decision'
    s['anchor_research_preparation_2026_11_06']=dict(report=report.name,tests=5,training_budget=680000,
        synthetic_fixture_slots=2720,synthetic_fixture_updates=2482,fixture_checkpoints=34,
        episodes_audited=244,research_launched=False,final_launched=False,word_thesis_changed=False,
        protocol_sha256=tests['protocol_sha256'],bounded_seed_files_checked=seeds['files_checked'],
        fixture_cost_screens_passed=inference['all_fit_cost_screens_pass'])
    write(tracker,s)
    files=[x for x in OUT.rglob('*') if x.is_file()]+[ROOT/name for name in tests['source_sha256']]+[
        report,backup,Path(__file__),ROOT/'validation/anchor_research_preparation_tests_2026-11-06.json',
        ROOT/'validation/anchor_research_seed_reservation_2026-11-06.json',
        ROOT/'analysis/prepare_anchor_research_2026_11_06.py',ROOT/'analysis/audit_anchor_research_seeds_2026_11_06.py']
    manifest={str(x.relative_to(ROOT)).replace('\\','/'):sha(x) for x in files}
    write(ROOT/'validation/anchor_research_preparation_completion_2026-11-06.json',dict(
        state='preparation_passed_no_research_launch',files=manifest))
    print(json.dumps(dict(report=report.name,tests=5,manifest_entries=len(manifest),
        seed_files_checked=seeds['files_checked'],research_launched=False),indent=2))

if __name__=='__main__':main()
