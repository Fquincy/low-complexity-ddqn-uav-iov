"""Record completed v3 engineering evidence, without modifying frozen inputs."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/controller_v3_engineering_2026-11-05_01'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')

v=read(OUT/'verification.json');audit=read(OUT/'saved_artifact_audit.json');tests=read(OUT/'mechanics_tests.json')
assert audit['state']=='passed_v3_engineering_artifact_audit' and tests['passed']
old_inputs=read(ROOT/'analysis/controller_redesign_2026-11-05/saved_trace_diagnosis.json')['input_sha256']
for name,h in old_inputs.items():
    path=Path(name)
    if not path.is_absolute():path=ROOT/path
    assert sha(path)==h,name
preserved=read(OUT/'prior_input_hashes.json')
for name,h in preserved.items():assert sha(ROOT/name)==h,name
preservation=dict(original_research_inputs_verified=len(old_inputs),prior_engineering_inputs_verified=len(preserved),
    all_unchanged=True,new_simulator_steps=0,optimizer_updates=0,word_thesis_changed=False)
write(OUT/'POST_AUDIT_PRESERVATION.json',preservation)

primary=v['metrics']['amortized_workload_price24'];wide=v['metrics']['additive128_unbounded'];compact=v['metrics']['additive64_nonpositive']
speedup=v['metrics']['shared_congestion_cost']['median_ms']/primary['median_ms']
ratio=primary['median_ms']/wide['median_ms'];saving=1-primary['parameter_bytes']/compact['parameter_bytes']
review=f'''VERSION 3 AMORTIZED WORKLOAD-PRICE DDQN: ENGINEERING REVIEW
Date: November 5, 2026

DECISION
The new design is implemented and mechanically checked, but its cost gate
FAILED. Do not start development/final training from this engineering result.
The low-complexity contribution and policy improvement remain unproven.

WHAT CHANGED
One small 533->24->170 network estimates individual action costs and four
workload prices, using a learned reference destination split to estimate other
vehicles' load once. The decoder then takes independent minima; no price-pass
loop, assignment repair or simulator look-ahead is used. This is exact greedy
selection for a NEW separable critic, not an equivalent faster version of the
old quadratic critic. It deliberately gives up action-dependent coordination.
The former critic and all earlier failed results are retained unchanged.

The network has 17,066 parameters, 68,264 FP32 online parameter bytes and 20
buffer bytes, with 16,872 Linear MACs. It is larger than the 4,052-parameter
shared critic, while using fewer neural MACs and much less decoder work.
Its online parameter storage is {saving:.2%} lower than compact64.
This says nothing about total process RAM, replay memory or electrical energy.

PREDECLARATION
Design: LOW_COMPLEXITY_CONTROLLER_V3_DESIGN_2026-11-05.txt
Protocol: LOW_COMPLEXITY_CONTROLLER_V3_ENGINEERING_PROTOCOL_2026-11-05.json
Both were saved before implementation and timing. This was one fixed design
and one declared timing attempt. No thresholds, old rewards, optimizer settings,
research/final seeds or historical model definitions were changed.
No timing-driven candidate revision or favorable rerun was performed.

WHAT PASSED
15 mechanics tests: exact parameter/MAC counts, observation offsets and units,
independent price-equation arithmetic, exhaustive reduced-model greedy actions,
deterministic ties, legal/nonpositive values, intended and masked head gradients,
DDQN online selection/separate target evaluation, terminal skipping, target
gradient isolation, checkpoint roundtrip, invalid-input/action/value rejection,
corrupt/unlisted artifact rejection, cost-screen logic, CLI launch blocking and
the absence of cloud, charging and security mechanisms.
Backward passes were used to test gradients; there were ZERO optimizer updates.

All 11 controllers were exported with the same ONNX Runtime 1.31.0 CPU FP32
backend, sequential execution and one intra/inter-op thread. Three synthetic
initializations per controller were checked against 128 saved observations and
64 numerical synthetic observations: 6,336 complete joint actions (202,752
individual choices) agreed exactly. Q values met rtol1e-5/atol1e-6. Every model
also passed deterministic save/reload checks and unchanged-weight checks.
This is finite-case equivalence, not a universal numerical proof or trained
policy evidence. The inherited expm1 lowering is Exp(x)-1 in FP32, and future
trained checkpoints need their own export/equivalence checks.

The independent auditor reconstructed all 33 models, replayed the 6,336 cases,
verified source/graph hashes and runtime settings, and recomputed all 7,040
timing records. It checked {audit['manifest_entries']} pre-audit manifest entries.
Audit state: {audit['state']}.

FULL ACTION-SELECTION TIMING (same run, milliseconds)
Controller                           Median       p95       FP32 parameter bytes
'''
for arm,item in v['metrics'].items():
    review+=f"{arm:<35} {item['median_ms']:>8.5f}  {item['p95_ms']:>8.5f}  {item['parameter_bytes']:>12}\n"
review+=f'''
The primary is {speedup:.2f}x faster than the old shared critic in THIS paired
timing run, but {ratio:.2f}x slower than the compiled wide reference. Do not
substitute timings from earlier sessions to calculate this comparison.
Primary p95 {primary['p95_ms']:.5f}ms passes the absolute1ms limit.
Unchanged screens: storage PASS; relative median speed FAIL; absolute p95 PASS.

Timing used five rounds, 32 warmups and 128 measured calls per arm/round,
640 readings per controller. Randomized interleaving order and every raw
nanosecond reading are saved. Input conversion, finite-input checks, all neural
and price calculations, full decoding/chosen-value computation and action copy
were included for every arm. Graph export and session startup were excluded
and saved separately. Export and audit processes both completed with exit0.
Parameter bytes and serialized graph sizes are reported separately; neither
is measured peak RAM. Untrained Q distributions are audit diagnostics only.

IMPORTANT LIMITATION THAT THE TESTS EXPOSED
In a symmetric two-vehicle algebra example, both vehicles still choose server1.
Their cost under the OLD coupled quadratic objective is0.5, versus0.25 for a
split assignment. Version3 minimizes its own separable approximation exactly;
it does not solve the original coupled minimization. Actual deadline/return
effects have not been measured. The reference fractions are latent learned
quantities, with no calibrated assignment-prediction guarantee.
The no-price ablation masks the price term to zero while retaining all
parameters and the declared computational path. Its runtime is reported as
implemented; the difference is not an isolated measure of pricing overhead.
No novel algorithm or successful deployment is established by this gate.

PRESERVATION AND SCOPE
Verified {len(old_inputs)} original saved research-input hashes and {len(preserved)} prior source,
design and engineering-artifact hashes unchanged. The pre-update tracker is
saved as PROJECT_STATUS_BEFORE.json. All old results remain available.
Zero new simulator slots; zero optimizer updates; no research/final seed use;
no long training run. The Word thesis remains unchanged.
Active execution locations are local vehicle, UAV1,UAV2,RSU1,RSU2 only.
Cloud and charging stations are absent; security/trust is deferred contribution2.

WHAT SHOULD HAPPEN NEXT
Stop before research training. First profile this frozen compiled version3
graph to locate the remaining cost, including price construction and model
output handling. The plain small model is also slower than the wide reference
in this run, so smaller widths alone do not establish lower runtime.
Any subsequent implementation simplification needs a new predeclared amendment,
the same complete cost accounting and equally optimized references. Do not
relax the cost screen or infer policy quality from untrained fixtures.
If that gate eventually passes, prepare and review a complete eleven-arm
research protocol/runner/auditor before a separate launch decision. Preserve
the prior return/deadline and value-stability screens, add the plain24 and
no-price24 comparisons, and freeze disjoint development/final tasks and fits.
The historical eight-arm budget is not a valid eleven-arm budget.
Literature comparison and supervisor discussion remain necessary before
claiming novelty or deciding the thesis contribution is sufficient.

FILES
controller_v3_amortized.py
run_controller_v3_engineering.py
audit_controller_v3_engineering.py
test_controller_v3_engineering.py
validation/controller_v3_engineering_2026-11-05_01/
  PRE_EXPORT_FREEZE.json,PRE_TIMING_FREEZE.json,mechanics_tests.json
  equivalence.json,metadata.json,raw_timings.json,verification.json
  run_manifest.json,saved_artifact_audit.json,audited_vectors.npz
  POST_AUDIT_PRESERVATION.json,exports/,source_snapshot/
'''
report=ROOT/'LOW_COMPLEXITY_CONTROLLER_V3_ENGINEERING_REVIEW_2026-11-05.txt'
report.write_text(review,encoding='utf-8')
tracker_path=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json';tracker=read(tracker_path)
assert tracker==read(OUT/'PROJECT_STATUS_BEFORE.json'),'tracker changed since freeze'
tracker['controller_v3_amortized_engineering']=dict(state=v['state'],mechanics_tests=tests['tests'],
    action_equivalence_cases=6336,independent_audit_passed=True,screens=v['screens'],
    primary_parameters=primary['parameters'],primary_parameter_bytes=primary['parameter_bytes'],
    primary_median_ms=primary['median_ms'],primary_p95_ms=primary['p95_ms'],reference_median_ms=wide['median_ms'],
    same_run_speedup_vs_previous_shared=speedup,relative_median_vs_wide=ratio,
    new_simulator_steps=0,optimizer_updates=0,research_training_launched=False,
    policy_quality_established=False,low_complexity_contribution_established=False,word_thesis_changed=False,
    report=report.name,artifacts=str(OUT.relative_to(ROOT)).replace('\\','/'))
tracker['current_workstream']='contribution_one_v3_amortized_critic_relative_speed_gate_failed'
tracker['next']='profile_frozen_v3_compiled_graph_before_any_new_predeclared_optimization_no_training'
write(tracker_path,tracker)
outer=ROOT/'validation/controller_v3_engineering_2026-11-05_01_COMPLETION_MANIFEST.json'
files=[p for p in OUT.rglob('*') if p.is_file()]+[report,tracker_path,Path(__file__)]
write(outer,dict(state='complete_failed_speed_gate_no_training',files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files}))
print(json.dumps(dict(report=report.name,gate=v['state'],audit=audit['state'],tests=tests['tests'],
    preserved=preservation,completion_manifest_files=len(files),speedup=speedup,relative_median=ratio),indent=2))
