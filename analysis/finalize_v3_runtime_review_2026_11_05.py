"""Save the stopped gate's review and advance only the mutable project tracker."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/controller_v3_runtime_optimization_2026-11-05_01'
GATE=OUT/'gate_01'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

failure=read(GATE/'FAILED.json');audit=read(GATE/'failure_record_audit.json');diag=read(GATE/'discrepancy_diagnosis.json')
profile=read(OUT/'profile/profile.json');freeze=read(GATE/'PRE_EXPORT_FREEZE.json')
assert audit['state']=='passed_failure_record_audit_not_equivalence_or_cost_pass'
assert not failure['timing_started'] and not (GATE/'raw_timings.json').exists()
for name,h in read(GATE/'FAILURE_ARTIFACT_MANIFEST.json')['files'].items():assert sha(GATE/name)==h
tracker=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json'
assert sha(tracker)==freeze['original_hashes'][tracker.name]
shutil.copy2(tracker,OUT/'PROJECT_STATUS_BEFORE_RUNTIME_OPTIMIZATION.json')
metadata={}
dependency=Path(freeze['dependencies']['dependency_path'])
for folder in dependency.glob('*.dist-info'):
    for name in ['METADATA','WHEEL','RECORD']:
        path=folder/name
        if path.exists():
            dest=OUT/'dependency_metadata'/folder.name/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
            metadata[str(dest.relative_to(OUT))]=sha(dest)
write(OUT/'dependency_metadata_manifest.json',metadata)

review='''VERSION 3 RUNTIME PROFILE AND NATIVE-FUSION REVIEW
Date: November 5, 2026

CURRENT DECISION
The runtime diagnosis and one predeclared optimization attempt are complete.
The optimization FAILED exact-action equivalence before timing. It is not
approved, and there is no new speed result or research-training launch.
The previous measured Version3 speed-gate failure remains the current evidence.
The Word thesis remains unchanged.

1. WHAT THE PROFILE FOUND
The saved full-graph implementations were profiled on the same 128 saved
observations, with 32 warmups and three diagnostic passes. ONNX Runtime CPU
FP32, sequential execution and one intra/inter-op thread were retained.
Optimized graph-node counts:
  Version3 amortized_workload_price24: 52
  Plain small DDQN plain24_nonpositive: 23
  Wide DDQN additive128_unbounded: 9

Version3 has many separate multiplication, shape, slice, softplus, sum and
selection operations around its small dense network. Under instrumentation,
dense Gemm/FusedGemm kernels account for 5.37% of its recorded kernel time,
versus 11.72% for plain24 and 42.39% for the wide model. Profiling adds substantial
per-node overhead: these percentages are diagnostic, NOT estimates of the
uninstrumented runtime split, a new speed comparison or a contribution claim.
The node counts and recorded operations support testing native fusion of
non-neural cost/decoder operations, rather than another width reduction alone.
Raw ORT event traces, segment timings and optimized graphs are preserved in
the profile/ subfolder. The optimized graphs may be hardware-specific.

2. ONE PREDECLARED IMPLEMENTATION CHANGE
Protocol: LOW_COMPLEXITY_V3_FUSED_RUNTIME_PROTOCOL_2026-11-05.json
Keep each mathematical critic, architecture, parameter values, action choices,
tie ordering, old repair count/threshold, training code, rewards and research
seeds unchanged. Export the neural cores to the existing ORT CPU FP32 backend;
fuse cost calculation and selection into Numba nopython native routines for
all eleven arms. Fast-math and parallel execution are disabled; bounds checks
are enabled. Compilation/first dispatch and session startup are recorded
separately and excluded from any prospective steady-state timing.

This is an inference implementation hypothesis, not a new learned controller.
The same native treatment was implemented for the compact/wide additive models,
plain24, BDQ, EMC and the four prior shared critics. There was no single-arm
fallback after failure, no timing-driven tuning and no favorable retest.
The predeclared cost thresholds remain unchanged.

Numba0.65.1 and llvmlite0.47.0 were installed with --no-deps into an isolated
temporary target. Existing NumPy2.4.6, Torch2.15.0+cpu and ORT1.31.0 were not
upgraded. Native-binary hashes, LLVM version and package metadata are saved.
This temporary dependency location is an engineering environment, not a
demonstrated portable deployment package. Official documentation retrieval was
attempted: ORT timed out; the Numba page returned HTTP403. API behavior and
versions were checked locally. No literature or novelty review is claimed.

3. WHAT PASSED BEFORE THE STOP
Eight focused mechanics tests passed: stable softplus/finite bounds, fixed
ties and branch selection, synthetic Version3 cost-table agreement, old
symmetry/repair-threshold fixtures, marginal identities, invalid-input and
corrupt-artifact rejection, nopython/no-fastmath/no-parallel settings, and
research CLI blocking both with and without explicit confirmation flags.
No simulator rollout, backward pass or optimizer update occurred in this gate.

Fourteen model/initialization combinations completed full checks on all 192
saved/numerical inputs each: 2,688 complete joint decisions agreed exactly,
with Q tolerances met, unchanged weights and deterministic graph reload.
These include the primary Version3 model on seeds99021 and99023 (384 cases).
The third planned synthetic initialization was not reached. Do not describe
this partial result as passing all 33 combinations or all 6,336 planned cases.

4. WHY THE GATE STOPPED
The next combination, older shared_congestion_cost with synthetic seed99023,
has one differing action component among its 192x32 tested components:
  Saved observation187 (a pre-existing numerical synthetic case), vehicle index2.
  Original eager and prior full-ORT policy: destination2 (UAV2).
  Split-core/native implementation: destination4 (RSU2).
All indices above are zero-based. No workload or actual trained trajectory
was changed. Reported chosen Q is exactly -1.8347185850143433 in all three
implementations; equal reported Q does not satisfy exact-action equivalence.

Read-only cross-checks localize this case:
  Original Torch decoder on split-ORT tables chooses the native action.
  Native decoder on original eager tables chooses the original action.
The split core perturbs u by up to3.7253e-9, workload queue L by1.1921e-7,
beta by5.9605e-8 and motion by2.9802e-8. This exposes sensitivity to FP32
table/score rounding near a candidate tie, rather than establishing a generic
wrong-action bug in the native decoder. No universal cause is asserted from
this one case.

The original eager decoder assigns all three initial candidates the same
FP32 cost and retains the earliest. In the split-core native trace, candidate0
costs1.8347187042236328 and candidate1 costs1.8347185850143433, so candidate1
wins. No repair is accepted in either trace. On the same original tables,
higher-precision arithmetic puts the two final assignment costs about3.0e-9
apart. These are numerical diagnostics; no offloading-performance impact was
measured. We did not discard this case, introduce a tie tolerance, relax exact
agreement, change the comparator or silently switch a backend.

5. WHAT WAS NOT RUN
The full equivalence stage did not pass. PRE_TIMING_FREEZE.json and raw timing
records do not exist for this attempt. ZERO benchmark readings were collected.
There is no evidence that the native-fusion candidate meets the speed gate.
No development/final training, research-seed consumption or model selection.
The prior 0.0833ms Version3 and0.0419ms wide-reference medians remain historical
results from the earlier same-run benchmark; they are not native-fusion timings.

6. AUDIT AND PRESERVATION
An independent failure-record auditor replayed all 15 reached combinations
(2,880 cases), reproduced the fourteen fully equivalent combinations and the
single failed-model mismatch, and confirmed all checked Q values meet tolerance.
It verified the preserved source snapshots, dependency binaries, all69 previous
completion artifacts before the tracker update and66 original research-input
hashes. The audit PASSES AS A FAILURE RECORD; equivalence and cost remain failed
or untested, respectively. The input fixture, full traces and replay vectors
are saved. The failed runner and native source have not been patched or retried.

The mutable project tracker is archived byte-for-byte before this review's
status update. The other68 previous completion artifacts remain unchanged.
Earlier run manifests describe their historical tracker snapshots; a later
live tracker update is not a modification of an earlier research result.
The new outer completion manifest covers this review and the updated tracker.
No Word document has been opened for editing or changed.
Cloud and charging stations remain absent from the active physical model.
Security/trust remains deferred contribution two.

7. NEXT ENGINEERING STEP
Predeclare a numerical-compatibility amendment that preserves the older
comparison controller's action-selection behavior across the export boundary.
Add this exact observation/seed/table discrepancy as a regression fixture.
Review the export boundary and candidate-score reductions before choosing the
implementation fix. Keep the current failed attempt and exact-agreement rule.
Repeat the full declared equivalence gate under the amended, frozen source;
only then run the equal-treatment cost comparison. No inference or speed claim
can be based on partial checks or instrumented profile timings.

If the existing cost gate subsequently passes, prepare the complete research
runner/protocol/auditor with disjoint seeds and separate launch review. Policy
return, deadlines, value consistency and the mechanism ablation still need
fresh evidence. An efficient implementation alone does not establish novelty.

FILES TO OPEN
LOW_COMPLEXITY_V3_FUSED_RUNTIME_PROTOCOL_2026-11-05.json
validation/controller_v3_runtime_optimization_2026-11-05_01/
  profile/profile.json and raw ORT profiles
  native_mechanics_tests.json
  PROJECT_STATUS_BEFORE_RUNTIME_OPTIMIZATION.json
  gate_01/FAILED.json
  gate_01/discrepancy_diagnosis.json
  gate_01/discrepancy_fixture.npz
  gate_01/failure_record_audit.json
  gate_01/failure_audit_vectors.npz
  gate_01/FAILURE_ARTIFACT_MANIFEST.json
  gate_01/PRE_EXPORT_FREEZE.json, exports/ and source_snapshot/
'''
report=ROOT/'LOW_COMPLEXITY_V3_RUNTIME_OPTIMIZATION_REVIEW_2026-11-05.txt'
report.write_text(review,encoding='utf-8')
state=read(tracker)
state['controller_v3_runtime_optimization']=dict(state='equivalence_failed_before_timing',
    profile_nodes=dict(primary=52,plain24=23,wide=9),mechanics_tests_passed=8,
    completed_equivalent_cases=2688,completed_equivalent_model_initializations=14,
    primary_checked_cases=384,failed_comparator='shared_congestion_cost',failed_synthetic_seed=99023,
    mismatch=dict(observation_index=187,vehicle_index=2,eager_destination=2,native_destination=4),
    failure_record_audit_passed=True,benchmark_readings=0,new_simulator_steps=0,optimizer_updates=0,
    research_training_launched=False,word_thesis_changed=False,policy_quality_established=False,
    report=report.name,artifacts=str(OUT.relative_to(ROOT)).replace('\\','/'))
state['current_workstream']='contribution_one_v3_native_fusion_exact_action_equivalence_failed_before_timing'
state['next']='predeclare_numerical_compatibility_amendment_for_split_reference_core_add_regression_before_any_timing_or_training'
write(tracker,state)
for name,h in freeze['original_hashes'].items():
    preserved=OUT/'PROJECT_STATUS_BEFORE_RUNTIME_OPTIMIZATION.json' if name==tracker.name else ROOT/name
    assert sha(preserved)==h,name
files=[p for p in OUT.rglob('*') if p.is_file()]+[report,tracker,
    ROOT/'controller_v3_fused_runtime.py',ROOT/'run_controller_v3_fused_gate.py',ROOT/'audit_controller_v3_fused_gate.py',
    ROOT/'test_controller_v3_fused_runtime.py',ROOT/'profile_controller_v3_compiled.py',
    ROOT/'LOW_COMPLEXITY_V3_FUSED_RUNTIME_PROTOCOL_2026-11-05.json',Path(__file__),
    ROOT/'analysis/diagnose_v3_native_discrepancy_2026_11_05.py',ROOT/'analysis/audit_v3_native_failure_2026_11_05.py']
write(ROOT/'validation/controller_v3_runtime_optimization_2026-11-05_01_COMPLETION_MANIFEST.json',
    dict(state='completed_stopped_equivalence_failure_no_benchmark_or_training',files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files}))
print(json.dumps(dict(report=report.name,gate='equivalence_failed_before_timing',failure_audit_passed=True,
    tests=8,equivalent_cases=2688,benchmark_readings=0,optimizer_updates=0,new_simulator_steps=0,
    archived_original_tracker=True,immutable_previous_completion_inputs_verified=68,completion_files=len(files)),indent=2))
