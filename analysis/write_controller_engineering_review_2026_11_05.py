"""Summarize the completed engineering gate; do not execute any learner."""
import ast,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
GATE=ROOT/'validation/controller_redesign_engineering_2026-11-05_01'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

tests=read(GATE/'test_verification.json')
cost=read(GATE/'cost_gate/verification.json')
audit=read(GATE/'cost_gate/saved_artifact_audit.json')
before=read(GATE/'cost_gate/PRE_TIMING_FREEZE.json')
assert tests['passed'] and tests['tests_run']==18
assert tests['fixture']['state']=='passed_synthetic_saved_artifact_audit'
assert cost['state']=='cost_gate_failed' and audit['state']=='passed_cost_artifact_audit'
for name,h in before['source_sha256'].items():assert sha(ROOT/name)==h,name
prior=read(ROOT/'analysis/controller_redesign_2026-11-05/design_stage_verification.json')
for name,h in prior['protected_file_sha256'].items():assert sha(ROOT/name)==h,name
old_inputs=read(ROOT/'analysis/controller_redesign_2026-11-05/saved_trace_diagnosis.json')['input_sha256']
for name,h in old_inputs.items():assert sha(ROOT/name)==h,name
for name,h in prior['outputs_sha256'].items():assert sha(ROOT/name)==h,name
for name in ['controller_redesign_v2.py','run_controller_redesign_engineering.py','audit_controller_redesign_engineering.py','test_controller_redesign_engineering.py']:
    ast.parse((ROOT/name).read_text(encoding='utf-8'))

metrics=cost['metrics'];primary=metrics['shared_congestion_cost'];small=metrics['additive64_nonpositive'];wide=metrics['additive128_unbounded']
ratio=primary['median_ms']/wide['median_ms']
lines='\n'.join(f"{name:27s} | {x['parameters']:7d} | {x['parameter_bytes']:7d} | {x['median_ms']:8.5f} | {x['p95_ms']:8.5f}" for name,x in metrics.items())
text=f'''
CONTROLLER REDESIGN VERSION 2 - ENGINEERING GATE REVIEW
Date: November 5, 2026

DECISION
Mechanics and reduced runner/auditor integration: PASSED.
Predeclared complete-action cost gate: FAILED.
Do NOT launch the 320,000-slot development experiment with this implementation.
No research training or final evaluation was started. The Word thesis is unchanged.
The candidate is implemented and mechanically checked; it is not a proven
low-complexity contribution, a trained successful policy or a security result.

1. CONTRIBUTION SCOPE AND THE USER'S CLOUD/CHARGING INSTRUCTION
Contribution one is the efficient DDQN-style offloading controller.
Contribution two will add security/trust after contribution one succeeds.
Security/trust is absent from this engineering experiment.
Cloud and charging stations are ABSENT from the ACTIVE simulator and controller.
Five execution choices only:local vehicle,UAV1,UAV2,RSU1,RSU2. UAV movement
choices remain hover/north/east/south/west. UAV batteries are not recharged.
The compliant active environment did not require deletion or reconstruction.
The source/configuration exclusion checks pass. The historical files and
completed experiments are preserved as requested; their older cloud/charging
implementations are not imported by the current simulator/controller path.
User-confirmed scope record: CONTRIBUTION_SCOPE_DECISION_2026-11-05.txt.

2. WHAT WAS IMPLEMENTED IN NEW FILES
controller_redesign_v2.py:
  shared per-vehicle encoder, pooled context, shared placement/mobility heads;
  termwise nonnegative cost, resource-load term, two price passes and exactly
  four single-move repairs with full-score acceptance. No exact joint argmax
  or convergence guarantee is asserted. No security/cloud/charging feature.
  No-congestion, untied-weight and outer-joint-bound ablations are implemented.
  The declared 64-64 BDQ adaptation is implemented with branch-mean target,
  per-branch Huber loss and shared-trunk gradient rescaling by1/33.

run_controller_redesign_engineering.py:
  all eight declared arms, frozen targets, terminal masking, uniform replay,
  paired exploration, final-only fixture checkpoints, freeze-before-evaluation,
  full task/reward/energy/motion/transition/optimizer records, own-trajectory
  online/target values and realized return-to-go, all fixture comparisons, and
  the same-backend CPU timing path. Research development/final CLI modes are
  explicitly disabled, even if confirmation is supplied. The confirmation flags
  are still checked first; absence of a flag also fails before creating output.
  This is an engineering harness, not a finished launchable five-fit research
  runner. Actual multi-fit development/final reporting remains a later gate,
  contingent on resolving the cost failure and freezing executable sources.

audit_controller_redesign_engineering.py:
  manifests, source snapshots, checkpoint/replay/counter checks, physical task
  accounting, paired workloads, optimizer traces, stored value/return arithmetic,
  timing quantiles, cost-screen recomputation, and active-system exclusions.
  Audits added zero simulator steps and zero optimizer updates.

test_controller_redesign_engineering.py:
  the model, target, decoder, gradient, scope, CLI and integration checks below.

3. MECHANICS AND INTEGRATION EVIDENCE
All18 tests passed in {tests['seconds']:.2f} seconds. The tests cover exact observation
index mapping and workload units, parameter/MAC arithmetic, legal actions,
nonpositive/finite scores, scored-action agreement, gradient flow and the
intentional zero-load ablation, correct decreasing-transform action selection,
online selection versus target evaluation, all-terminal bootstrap skipping,
BDQ trunk-gradient scaling, own-policy return-consistency arithmetic, cost-screen
failure detection and both research CLI guards.

The complete reduced fixture used model99001, training workload990000000 and
evaluation workload990000100. It was labeled SYNTHETIC_NOT_RESEARCH throughout.
Eight models x1 episode x40 slots =320 synthetic training slots;33 updates and
4 target copies each, totaling264 updates and32 copies. This is engineering
integration only. Its evaluation seed was intentionally reused for synthetic
validation; no fresh-research interpretation is permitted.
The auditor checked8 final checkpoints,35 saved physical episodes,264 optimizer
rows and104 manifest entries. It recomputed comparisons and value summaries,
checked replay against the saved transitions, and checked all source snapshots.
Corrupted and unmanifested artifact fixtures were both rejected as expected.
Temporary model/trace fixture outputs were removed after the tests. Their
audit summary is preserved in test_verification.json; they are not thesis data.

Verified parameter counts: shared4052; untied53845; compact additive48736;
wide additive105504; EMC69361; adapted BDQ48801. Verified shared Linear MACs
per state:51696. FP32 shared online parameter tensors occupy16208 bytes,
plus20 bytes of public configuration buffers. Serialized checkpoints, target,
optimizer, replay, activation and process memory are different quantities.

4. COST GATE - ONE PREDECLARED MEASUREMENT, NO FAVORABLE RETEST
CPU:AMD Ryzen7 4800H,8 physical cores/16 logical processors. Timing used ONE
PyTorch intra-op and ONE inter-op thread, eager CPU FP32 for EVERY arm; no GPU
or candidate-only compilation. Python3.11.9,PyTorch2.15.0+cpu,NumPy2.4.6.
Hardware/configuration/source hashes were recorded BEFORE any timing.

The declared8 timing-probe workloads422000000..422000007 generated320 slots
without learning. Their heuristic traces were used solely to form128 common
observations, not to assess the new policies. Each arm used the same synthetic
initialization seed99001. Every path received32 warmups/round and128 measured
calls/round over5 rounds in predeclared randomized order:640 calls/arm,
5120 total raw timing records. The measurement includes input parsing/conversion,
all neural modules, coordination/repairs/full scoring and final action conversion.
Environment stepping and file I/O are outside the timed region.

arm                         | params  | FP32 B  | median ms| p95 ms
{lines}

PREDECLARED SCREENS (UNCHANGED)
Storage <=50% of compact additive:PASS, actual ratio
{primary['parameter_bytes']/small['parameter_bytes']:.6f} ({100*(1-primary['parameter_bytes']/small['parameter_bytes']):.2f}% lower online weight storage).
Median <=100% of wide additive:FAIL, actual ratio {ratio:.3f}.
Complete-action p95 <=1.0ms:FAIL, actual {primary['p95_ms']:.5f}ms.
The raw timing artifact audit PASSED; the measured engineering gate FAILED.
All5120 readings, probe states and source hashes remain under cost_gate/.

5. INTERPRETATION AND LIMITS
The implementation achieves the predicted small online model but is much slower
in this eager CPU implementation. It cannot currently support a faster or
lower-computation claim. Even the shared no-congestion path is slower than the
flat references; the measured overhead is not necessarily only in the repairs.
The full path versus no-congestion timing difference suggests coordination
adds overhead, but a component profile is required before causal attribution.
These are untrained timing models and a synthetic runner fixture. They do not
show useful policy quality, improved deadlines, value calibration after research
training, empirical novelty, embedded deployment or controller energy savings.
No experimental settings, thresholds, rewards or model definitions were changed
after seeing this cost result. No best timing round or passing initialization
was selected. The failure is preserved.

6. SEED USAGE AND PRESERVATION
Used for engineering:declared timing probes422000000..422000007, synthetic
model99001, synthetic workload990000000/evaluation990000100, and component-test
RNG streams99003/99004. No development training/model/quality-gate seeds were
used, and all final-training/model/evaluation seeds remain untouched.
All66 original saved diagnosis-input hashes remain identical. The earlier
November4 protocol, its source/checkpoints/traces, and the November5 version2
design artifacts remain unchanged. New implementation and report files are
separate. No Word file was written.

7. NEXT ENGINEERING ACTION
Profile the complete inference path to separate parser/forward/nonlinear-cost/
price/repair/conversion overhead. Then predeclare an implementation optimization
that preserves the version2 mathematical model and complete actions, with
equivalence tests and fair backend treatment of ALL comparators. If changing
the architecture, decoder pass count or cost objective becomes necessary,
document a new version BEFORE training; retain this failure and its thresholds.
Do not silently relax the median/p95 gate or launch a long run to see whether
policy quality compensates. A storage-versus-latency tradeoff study would need
its own explicit protocol and a narrower claim.
Security/trust stays deferred until contribution one has adequate evidence.

8. ARTIFACT LOCATIONS AND REPRODUCTION
Engineering root:validation/controller_redesign_engineering_2026-11-05_01/
  unit_and_integration_tests.log
  test_verification.json
  hardware_cpu_before_timing.json
  cost_gate/PRE_TIMING_FREEZE.json
  cost_gate/raw_timings.json
  cost_gate/common_observations.npz
  cost_gate/verification.json
  cost_gate/saved_artifact_audit.json
  cost_gate/run_manifest.json
Interpreter:C:/Users/OMEN/AppData/Local/Temp/value_bound_smoke_env/Scripts/python.exe
Test command:-m unittest test_controller_redesign_engineering -v
Read-only cost audit:audit_controller_redesign_engineering.py <cost_gate_path> --cost
Do not rerun the cost measurement just to obtain a favorable result. Any new
implementation and its benchmark must have a recorded reason and frozen inputs.
'''
report=ROOT/'LOW_COMPLEXITY_CONTROLLER_REDESIGN_ENGINEERING_REVIEW_2026-11-05.txt'
if report.exists():raise FileExistsError(report)
report.write_text(text.strip()+'\n',encoding='utf-8')
assert report.read_text(encoding='utf-8').startswith('CONTROLLER REDESIGN VERSION 2')

status_path=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json'
raw=status_path.read_bytes();(GATE/'PROJECT_STATUS_BEFORE_ENGINEERING.json').write_bytes(raw)
status=json.loads(raw)
status['contribution_plan']=dict(first='efficient_DDQN_offloading_controller_not_yet_proven',
    second='security_trust_extension_deferred_until_first_succeeds',security_in_current_experiment=False,
    cloud_in_active_model=False,charging_stations_in_active_model=False,user_confirmed='2026-11-05')
status['controller_redesign_v2_engineering']=dict(implemented=True,mechanics_tests_passed=18,
    synthetic_runner_audit_passed=True,cost_artifact_audit_passed=True,cost_gate='failed',
    primary_online_parameter_bytes=16208,primary_median_ms=primary['median_ms'],primary_p95_ms=primary['p95_ms'],
    research_training_launched=False,low_complexity_claim_established=False,thesis_changed=False,
    review=str(report.relative_to(ROOT)),artifacts=str(GATE.relative_to(ROOT)),timing_probe_seeds_consumed=cost['probe_seeds_consumed'],
    development_training_and_quality_seeds_unconsumed=True,final_seeds_unconsumed=True)
status['current_workstream']='contribution_one_controller_redesign_v2_engineering_cost_gate_failed'
status['next']='profile_and_predeclare_semantics_preserving_inference_optimization_before_any_research_launch'
assert status_path.read_bytes()==raw,'status changed during review'
status_path.write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
assert read(status_path)['controller_redesign_v2_engineering']['cost_gate']=='failed'

result=dict(state='engineering_implementation_and_mechanics_passed_cost_gate_failed',tests=tests['tests_run'],
    synthetic_fixture_audit=tests['fixture'],cost_gate=cost['state'],cost_artifact_audit=audit['state'],
    primary_median_ratio_vs_wide=ratio,online_parameter_storage_ratio_vs_compact=primary['parameter_bytes']/small['parameter_bytes'],
    original_diagnosis_inputs_unchanged=len(old_inputs),original_design_output_hashes_unchanged=True,
    old_project_status_sha256=hashlib.sha256(raw).hexdigest(),new_project_status_sha256=sha(status_path),
    protected_source_hashes_match_pre_timing=True,research_training_launched=False,word_thesis_changed=False,
    scope=tests['fixture']['source_scope'],review=str(report),
    output_sha256={str(p.relative_to(ROOT)):sha(p) for p in [report,status_path,ROOT/'CONTRIBUTION_SCOPE_DECISION_2026-11-05.txt']})
(GATE/'ENGINEERING_GATE_SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
assert read(GATE/'ENGINEERING_GATE_SUMMARY.json')['tests']==18
print(json.dumps(result,indent=2))
