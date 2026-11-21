"""Record audited compatibility/cost pass, preserving all earlier experiments."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/controller_v3_runtime_compatibility_2026-11-05_01'
COMPLETION=ROOT/'validation/controller_v3_runtime_compatibility_2026-11-05_01_completion'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
v=read(OUT/'verification.json');a=read(OUT/'saved_artifact_audit.json');pre=read(OUT/'PRE_EXPORT_FREEZE.json')
assert v['state']=='compatibility_cost_passed' and all(v['screens'].values())
assert a['state']=='passed_compatibility_artifact_audit'
for name,h in read(OUT/'run_manifest.json')['files'].items():assert sha(OUT/name)==h
for name,h in pre['prior_artifacts'].items():
    path=OUT/'PROJECT_STATUS_BEFORE.json' if name=='LOW_COMPLEXITY_PROJECT_STATUS.json' else ROOT/name
    assert sha(path)==h,name
old=read(ROOT/'analysis/controller_redesign_2026-11-05/saved_trace_diagnosis.json')['input_sha256']
for name,h in old.items():
    path=Path(name)
    if not path.is_absolute():path=ROOT/path
    assert sha(path)==h,name
COMPLETION.mkdir(exist_ok=False)
write(COMPLETION/'preservation.json',dict(original_research_inputs_verified=len(old),
    prior_completion_artifacts_preserved=len(pre['prior_artifacts']),mutable_tracker_archived_in='PROJECT_STATUS_BEFORE.json',
    all_old_models_protocols_results_unchanged=True,new_simulator_steps=0,optimizer_updates=0,word_thesis_changed=False))
actual=dict(backend='ORT CPU neural-core/native FP32 decoder for7 arms; retained whole-policy ORT for historical4',
    routes=pre['routes'],dependencies=pre['dependencies'],
    annotation='Inherited hardware.backend text is a legacy eager label; protocol/routes and audited actual runtime settings are authoritative')
write(COMPLETION/'actual_runtime_annotation.json',actual)
primary=v['metrics']['compatibility']['amortized_workload_price24'];wide=v['metrics']['compatibility']['additive128_unbounded']
small=v['metrics']['compatibility']['additive64_nonpositive'];plain=v['metrics']['compatibility']['plain24_nonpositive']
legacy=v['metrics']['legacy']['amortized_workload_price24']
ratio=primary['median_ms']/wide['median_ms'];saving=1-primary['parameter_bytes']/small['parameter_bytes'];speed=legacy['median_ms']/primary['median_ms']
report=f'''VERSION 3 NUMERICAL COMPATIBILITY AND COST GATE REVIEW
Date: November 5, 2026

DECISION
PASS: the numerical regression, full finite-case equivalence, declared storage
and runtime screens, and independent saved-artifact audit all passed.
The implementation is ready for the next research-engineering preparation gate.
No research training was launched. Return/deadline performance, trained value
consistency and a sufficient thesis contribution remain unproven.
The Word thesis remains unchanged.

1. THE COMPATIBILITY AMENDMENT
Protocol: LOW_COMPLEXITY_V3_RUNTIME_COMPATIBILITY_PROTOCOL_v2_2026-11-05.json
Saved before the new implementation and before any timing under this amendment.
The first native-fusion failure is preserved with its full diagnosis and hashes.
The offending older controller is sensitive to small FP32 changes when its
tables are exported separately. We removed that new export boundary for the
ENTIRE four-model historical shared-critic family: every input uses its original,
validated complete ONNX policy graph. All 12 graphs (four arms, three seeds)
were copied byte-for-byte and independently verified. This does not fix native
table-split arithmetic universally; it avoids that numerical boundary for these
historical controls. Their parameters and action-selection policy did not change.

Seven arms use the same ORT neural-core plus native-decoder treatment: primary
Version3, its no-price ablation, plain24, compact64, wide128, EMC and adapted BDQ.
The route is fixed by model family before measurement; no per-input, mismatch-
triggered or timing-selected fallback is permitted. This differs from the first
attempt's all-native plan and is explicitly declared, not hidden.
Do not attribute cross-route speed differences versus the historical four arms
to architectural superiority. The primary cost gate uses equally native-treated
compact64 and wide128 references. All routes and all timing results are reported.

The mathematical critics, weights, old two-pass/four-repair decoder and1e-6
repair threshold, rewards, training settings, research seeds and acceptance
thresholds remain unchanged. No tie tolerance was introduced.

2. WHAT PASSED
Fourteen tests passed: the previous eight native mechanics/rejection tests plus
the saved mismatch regression, fixed historical-family routes, equal native
treatment of principal references, wrong-weight/graph rejection, unchanged
relative-speed screen and both research CLI guards. Development/final commands
remain disabled both with and without explicit confirmation flags.

All 33 model/initialization combinations were checked on the same192 inputs:
128 saved engineering observations plus64 previously saved synthetic numerical
observations. Synthetic model seeds:99021,99023,99029.
ALL 6,336 complete joint actions (202,752 individual choices) matched exactly.
Chosen Q values met rtol1e-5/atol1e-6; values were finite and nonpositive where
required. Graph save/reload and repeat were deterministic, and model parameter
hashes remained unchanged.
The former seed99023, observation187, vehicle-index2 regression now chooses
destination2 (UAV2), agreeing with the original complete action and Q.
These are finite engineering checks, not a universal numerical guarantee or
performance evaluation of trained offloading policies. Every final trained
checkpoint will need its own export/equivalence and runtime verification.

3. FAIR SAME-RUN TIMING
CPU inference, FP32, ORT1.31.0 CPUExecutionProvider, sequential execution and
one intra/inter-op thread; native Numba0.65.1/llvmlite0.47.0, nopython,
fast-math OFF, parallel OFF, bounds checks ON, cache OFF. No new dependency
installation or framework upgrade was required in this gate.
22 paths:11 original complete-ORT policies plus11 declared compatibility routes.
Five randomized rounds,32 warmups and128 measured calls/path/round:640 readings
per path and14,080 raw timing records in total. Timing-order seed99048.
Input conversion and finite-input checks, all neural computation, all cost and
decoder work, chosen value, and final action copy are included.
Export, session startup and first dispatch/JIT compilation are logged separately
and excluded. No simulator stepping or file I/O is included. One frozen timing
attempt; no favorable rerun or performance-driven model/threshold adjustment.

Arm                                  Original ORT  Declared route   Route p95
                                         median ms      median ms          ms
'''
for arm in v['metrics']['compatibility']:
    oldrow=v['metrics']['legacy'][arm];row=v['metrics']['compatibility'][arm]
    report+=f"{arm:<36} {oldrow['median_ms']:>10.5f} {row['median_ms']:>14.5f} {row['p95_ms']:>11.5f}\n"
report+=f'''
The historical shared four use retained complete policies; the other seven
use native cost/decoding. Read their route labels in verification.json.
For the primary, median complete selection is {primary['median_ms']:.5f}ms,
versus {wide['median_ms']:.5f}ms for the equally native-treated wide reference:
ratio {ratio:.6f}, or {(1-ratio)*100:.2f}% lower measured median latency.
Its p95 is {primary['p95_ms']:.5f}ms, below the unchanged1ms limit.
Compared with its own same-run original full-graph implementation, its median
is {speed:.2f}x faster. This is implementation improvement for the same critic.

Online FP32 parameter bytes: primary68,264 versus compact64 194,944,
or {saving*100:.2f}% fewer. Primary parameters:17,066; public buffer:20 bytes.
Each target copy would add another68,264 parameter bytes plus20 buffer bytes.
Serialized ONNX sizes and startup costs are in metadata.json; none of these
quantities measures total process RAM, replay memory or controller energy.
All three unchanged engineering screens PASS: <=50% compact parameter bytes;
median <=wide-reference median; p95 <=1ms.

IMPORTANT MATCHED-SMALL COMPARISON
Plain24 is smaller and faster than the proposed primary in this run:
67,364 parameter bytes and {plain['median_ms']:.5f}ms median, versus68,264 and
{primary['median_ms']:.5f}ms. The proposed price mechanism therefore has an
additional resource cost. It must earn that cost through fresh return/deadline
and ablation evidence. Do not claim that the primary is the fastest/smallest
controller, or that faster runtime alone proves a useful novel contribution.
The learned reference fractions still do not guarantee balanced assignments.

4. INDEPENDENT AUDIT AND PRESERVATION
Audit state:{a['state']}.
The auditor reconstructed and replayed all6,336 cases, independently recomputed
all14,080 timing records, reproduced the regression, verified routes/provider/
thread settings, preserved graph hashes, native modes, principal-reference
fairness and{a['manifest_entries']} pre-audit manifest entries.
It verified92 prior completion artifacts against their frozen versions.
This review also verified all66 original saved research-input hashes unchanged.
The mutable tracker's prior bytes are archived in PROJECT_STATUS_BEFORE.json;
the other prior completion artifacts remain unchanged. Previous manifests
continue to describe their historical tracker versions. Old failed gates are
retained, and this successful engineering amendment does not rewrite them.

The dependency hardware object includes an inherited legacy eager-backend text
label. The protocol, per-arm route labels and audited ORT/native settings state
the actual measured runtime; an explicit annotation is saved in the completion
folder. These timings are on this desktop CPU, not a UAV embedded deployment.
No trained-policy quality, generalization, novelty or publication outcome is
established. Zero new simulator slots and zero optimizer updates in this gate.
Cloud and charging stations are absent. Security/trust is deferred contribution2.
No Word thesis edit or automated thesis text replacement occurred.

5. NEXT GATE: RESEARCH RUNNER PREPARATION, THEN SEPARATE LAUNCH DECISION
Freeze a complete eleven-arm prospective research protocol, implement its
development-only runner and independent saved-artifact auditor, and exercise
the same orchestration in a small isolated synthetic fixture. Cover training,
terminal DDQN updates, paired workload/exploration schedules, final-only
checkpoints, checkpoint freeze before held-out evaluation, every comparison per
initialization, value-stability/own-trajectory diagnostics and all saved files.
Keep research launch guarded and disabled until the implementation review.

Retain the predeclared return/deadline margins and own-policy value screens.
Add the plain24 reference and same-size no-price ablation to the full report.
The mechanism needs its own useful-effect evidence; no best-seed/checkpoint
selection, dropped failed initialization or post-hoc threshold tuning.
Perform a seed-collision check and freeze disjoint development and untouched
final fits/workloads, including new research timing inputs. These reused saved
engineering observations are not an untouched research evaluation set.

At the proposed inherited schedule (200 episodes x40 slots/model, five fits,
11 arms), EACH research phase would require440,000 physical training slots,
383,735 optimizer updates and1,870 target copies. A later equally sized final
phase would be separate. These are prospective accounting numbers, not a launch
authorization. Review the full protocol and implementation evidence before any
explicit research-launch decision. A failed quality gate must remain a failure.
Literature comparison and supervisor discussion are still needed before making
novelty or thesis-sufficiency claims.

FILES
LOW_COMPLEXITY_V3_RUNTIME_COMPATIBILITY_PROTOCOL_v2_2026-11-05.json
controller_v3_runtime_compatibility.py
run_controller_v3_compatibility_gate.py
test_controller_v3_compatibility.py
audit_controller_v3_compatibility_gate.py
validation/controller_v3_runtime_compatibility_2026-11-05_01/
  mechanics_tests.json,regression.json,equivalence.json,metadata.json
  PRE_EXPORT_FREEZE.json,PRE_TIMING_FREEZE.json,raw_timings.json
  verification.json,run_manifest.json,saved_artifact_audit.json
  audited_vectors.npz,exports/,source_snapshot/,PROJECT_STATUS_BEFORE.json
validation/controller_v3_runtime_compatibility_2026-11-05_01_completion/
  preservation.json,actual_runtime_annotation.json
'''
report_path=ROOT/'LOW_COMPLEXITY_V3_COMPATIBILITY_AND_COST_REVIEW_2026-11-05.txt'
report_path.write_text(report,encoding='utf-8')
tracker_path=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json'
assert sha(tracker_path)==pre['mutable_tracker_snapshot_sha256']
tracker=read(tracker_path)
tracker['controller_v3_runtime_compatibility']=dict(state=v['state'],audit=a['state'],mechanics_tests_passed=14,
    complete_actions_equal=6336,timing_records=14080,regression_fixed=True,screens=v['screens'],
    primary_median_ms=primary['median_ms'],primary_p95_ms=primary['p95_ms'],wide_native_median_ms=wide['median_ms'],
    plain24_native_median_ms=plain['median_ms'],primary_parameter_bytes=primary['parameter_bytes'],
    same_run_median_reduction_vs_wide_fraction=1-ratio,storage_reduction_vs_compact_fraction=saving,
    native_arms=read(ROOT/'LOW_COMPLEXITY_V3_RUNTIME_COMPATIBILITY_PROTOCOL_v2_2026-11-05.json')['routes']['native_core_and_decoder'],
    retained_full_policy_arms=read(ROOT/'LOW_COMPLEXITY_V3_RUNTIME_COMPATIBILITY_PROTOCOL_v2_2026-11-05.json')['routes']['validated_complete_ORT_policy'],
    research_training_launched=False,policy_quality_established=False,novelty_established=False,
    low_complexity_contribution_established=False,new_simulator_steps=0,optimizer_updates=0,word_thesis_changed=False,
    report=report_path.name,artifacts=str(OUT.relative_to(ROOT)).replace('\\','/'))
tracker['current_workstream']='contribution_one_v3_compatibility_and_engineering_cost_passed_research_runner_preparation_next'
tracker['next']='freeze_eleven_arm_research_protocol_implement_runner_auditor_and_reduced_fixture_before_separate_launch_decision'
write(tracker_path,tracker)
all_files=[p for p in OUT.rglob('*') if p.is_file()]+[p for p in COMPLETION.rglob('*') if p.is_file()]
all_files += [report_path,tracker_path,Path(__file__)]
write(ROOT/'validation/controller_v3_runtime_compatibility_2026-11-05_01_COMPLETION_MANIFEST.json',
    dict(state='engineering_passed_no_research_launch',files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in all_files}))
print(json.dumps(dict(report=report_path.name,gate=v['state'],audit=a['state'],tests=14,
    primary_median_ms=primary['median_ms'],wide_native_median_ms=wide['median_ms'],
    primary_p95_ms=primary['p95_ms'],latency_reduction_fraction=1-ratio,storage_reduction_fraction=saving,
    original_research_inputs_verified=len(old),completion_files=len(all_files),research_training_launched=False),indent=2))
