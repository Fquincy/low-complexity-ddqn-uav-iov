"""Record the measured optimization result without modifying frozen experiments."""
import ast,hashlib,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/controller_inference_optimization_2026-11-05_01'
GATE=OUT/'compiled_gate_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

r=read(GATE/'verification.json');a=read(GATE/'saved_artifact_audit.json');e=read(GATE/'equivalence.json')
profile=read(OUT/'profile/profile.json');meta=read(GATE/'export_metadata.json')
assert a['state']=='passed_compiled_artifact_audit' and r['state']=='compiled_cost_gate_failed'
assert e['total_complete_actions']==4608 and r['total_timing_records']==10240
freeze=read(GATE/'PRE_EXPORT_FREEZE.json')
for name,h in freeze['source_sha256'].items():assert sha(ROOT/name)==h,name
original=read(ROOT/'analysis/controller_redesign_2026-11-05/saved_trace_diagnosis.json')
for name,h in original['input_sha256'].items():assert sha(ROOT/name)==h,name
previous=ROOT/'validation/controller_redesign_engineering_2026-11-05_01'
for name,h in read(previous/'ENGINEERING_ARTIFACT_MANIFEST.json')['files'].items():assert sha(previous/name)==h,name
for name in ['compiled_controller_inference.py','run_compiled_inference_gate.py','audit_compiled_inference_gate.py','test_compiled_inference_checks.py']:
    ast.parse((ROOT/name).read_text(encoding='utf-8'))
eager=r['metrics']['eager'];native=r['metrics']['compiled'];p=native['shared_congestion_cost'];wide=native['additive128_unbounded'];small=native['additive64_nonpositive']
speed=eager['shared_congestion_cost']['median_ms']/p['median_ms'];ratio=p['median_ms']/wide['median_ms']
lines='\n'.join(f"{arm:27s} | {eager[arm]['median_ms']:8.5f} | {x['median_ms']:8.5f} | {x['p95_ms']:8.5f} | {x['onnx_serialized_bytes']:8d}" for arm,x in native.items())
text=f'''
CONTROLLER INFERENCE OPTIMIZATION REVIEW
Date: November 5, 2026

CURRENT DECISION
Profiling, compiled inference adapter, equivalence checks and independent audit
completed. The candidate is {speed:.2f} times faster than its same-run eager
implementation, and now meets the absolute1ms p95 latency screen.
The overall cost gate STILL FAILS because the compiled wide reference is
{ratio:.2f} times faster at median complete action selection.
Do not launch research training under the existing version2 protocol.
Contribution one's policy quality and low-complexity claim remain unproven.
Cloud and charging stations remain excluded. Security/trust is deferred to
contribution two. The Word thesis remains unchanged.

1. WHAT WAS PROFILED
The unchanged eager candidate was instrumented on the128 saved engineering
observations, with3 profile rounds and CPU operator counts on8 observations.
No new simulator rollout or optimizer update was performed.
The profile attributes {profile['segment_mean_percent']['decoder_two_prices_four_repairs']:.2f}% of mean instrumented segment time
to the two price passes/four repair rounds, and
{profile['segment_mean_percent']['tables_including_parser_network_and_cost_terms']:.2f}% to observation parsing, neural modules and cost tables.
Frequent copies, indexing, gathering, allocation and small tensor arithmetic
support a dispatch/coordination-overhead explanation. These instrumented
segment means are diagnostic and must NOT be added to or substituted for the
previous cost-gate medians, which were measured in a different execution context.
Profile artifacts:profile/profile.json and profile/raw_profile.json.

2. PREDECLARED OPTIMIZATION AND THE EXPORT AMENDMENT
The chosen single backend was ONNX Runtime CPU for ALL eight models, with
FP32, sequential execution, one intra-op and one inter-op thread and the same
graph-optimization setting. No alternate backend or favorable timing result
was selected. No neural architecture, weight, reward, training seed, decoder
pass count, repair threshold or existing cost-screen threshold changed.
Both the original eager and compiled paths include complete action selection,
including all required cost calculations and final action conversion.
Unused diagnostic outputs are not returned by the compiled policy interface;
the decisions and chosen-Q output are preserved on the tested cases.

Protocol files:
  LOW_COMPLEXITY_INFERENCE_OPTIMIZATION_PROTOCOL_2026-11-05.json (v1, retained)
  LOW_COMPLEXITY_INFERENCE_OPTIMIZATION_PROTOCOL_v1_1_2026-11-05.json (current)
The first export failed before any compiled timing because the legacy exporter
lacks aten::expm1 support. This is retained in
export_feasibility/FAILED_FIRST_EXPORT.json. Before timing, amendment1.1 declared
the mathematical lowering expm1(x)=exp(x)-1 using ONNX primitives. This is not
bitwise identical in FP32 near zero; explicit numerical/action tests were required.
The maximum absolute difference in the small expm1 fixture was
{e['fixtures']['expm1_max_absolute_error']:.9g}, within the declared1e-6 absolute tolerance.
No candidate or reference silently fell back to another execution backend.

The existing PyTorch2.15.0+cpu,NumPy2.4.6 and ONNX Runtime1.31.0 were preserved.
ONNX1.23.2 and ml_dtypes0.5.4 were installed in the existing temporary test
environment with --no-deps; no framework upgrade or project environment rewrite.
The official ONNX Runtime Python API documentation was checked for the session
options. Model export/session creation costs are logged separately from inference.

3. SEMANTIC AND ARTIFACT CHECKS
Eight models x3 synthetic initializations (99001,99006,99007), each checked on
128 saved observations plus64 valid numerical synthetic observations from
RNG99005:4,608 complete joint actions, or147,456 individual action components.
ALL complete actions matched exactly. Chosen-Q values met rtol1e-5/atol1e-6;
the largest observed absolute difference was
{max(x['max_Q_absolute_error'] for x in e['models']):.9g}.
Nonpositive values were checked for bounded models. Weight hashes were unchanged.
Serialization/reload and deterministic-repeat checks passed for every model.
Wrong input shapes are rejected before calling the native runtime.
Additional decoder cases cover symmetry, all ties and both sides of the1e-6
repair-acceptance threshold. The fixed batch-one shape is [1,533],N=30.
This establishes equivalence on these finite cases and weights; it is not a
universal proof of bitwise equivalence, unseen-state policy identity or trained
policy quality. Future trained checkpoints require their own export/equivalence gate.

The read-only independent auditor reconstructed all24 model instances and
replayed all4,608 cases, recomputed all10,240 timing records and verified49
pre-audit manifest entries, graph hashes, source snapshots, the input cases,
provider/thread settings and unchanged screens. Replayed actions and Q vectors
are retained in compiled_gate_01/audited_equivalence_vectors.npz.
Audit state:{a['state']}.

Six additional focused tests passed:strict action identity despite equal Q;
nonfinite-Q rejection; enforced Q tolerances; shape rejection before runtime;
corrupted-artifact rejection; and refusing to override a relative-speed failure
with an absolute-latency pass. The previous18 model/runner tests and their
engineering evidence remain intact; unchanged old sources were hash-verified.

4. FAIR SAME-RUN TIMING RESULTS
CPU:AMD Ryzen7 4800H. All16 paths (eight eager,eight compiled) were interleaved
in predeclared randomized order, over5 rounds with32 warmups/path/round and
128 timed observations/path/round. There are640 readings/path and10,240 total.
Compilation/session creation, simulator stepping and file I/O are excluded.
Both backends include input handling, the complete decoder, required scoring
and copying out the32-element action. Neither backend's median is selected
from a favorable timing round; all readings are retained.

arm                         | eager med| native med|native p95| graph B
{lines}
All times are milliseconds. "Graph B" is serialized ONNX file size, not RAM.

For the primary candidate:
  same-run eager median:{eager['shared_congestion_cost']['median_ms']:.5f}ms
  compiled median:{p['median_ms']:.5f}ms
  compiled p95:{p['p95_ms']:.5f}ms
  same-run median speedup:{speed:.3f}x
  FP32 online parameter tensors:16,208 bytes, plus20 bytes of config buffers
  serialized ONNX graph:{p['onnx_serialized_bytes']:,} bytes
  export:{meta['shared_congestion_cost_99001']['export_seconds']:.3f}s;
  session initialization:{meta['shared_congestion_cost_99001']['session_startup_seconds']:.3f}s.
The compiled compact reference graph is{small['onnx_serialized_bytes']:,} bytes,
so the primary graph file is {100*(1-p['onnx_serialized_bytes']/small['onnx_serialized_bytes']):.2f}% smaller.
Peak process memory, session/activation RAM and electrical energy were not measured.
Online parameter savings must not be presented as the same percentage of total RAM.

5. UNCHANGED SCREENS AND CONCLUSION
Online parameter bytes <=50% of compact reference:PASS.
Compiled primary p95<=1ms:PASS, actual{p['p95_ms']:.5f}ms.
Compiled primary median<=compiled wide reference:FAIL,
  {p['median_ms']:.5f}ms versus{wide['median_ms']:.5f}ms; ratio{ratio:.3f}.
Compare compiled candidate to COMPILED references, never to their slower eager
paths to manufacture a pass. Audit success and action agreement do not turn
this failed performance gate into a successful contribution.
The original eager failure is preserved, and the compiled result is a new,
separately declared engineering measurement. No thresholds were relaxed.

6. LIMITATIONS AND NEXT ACTION
Compilation substantially reduces implementation overhead, but a simple flat
reference still has much less complete decision work at this tested scale.
Further progress must address the remaining coordination/decoder cost before
research training under the existing objective. Another mathematical controller
or decoder budget needs a newly documented design version; preserve this one.
Any further implementation optimization should preserve actions/equations and
be predeclared and applied with fair treatment of all comparators. No backend
or hyperparameter search was performed in this step.
A different storage/latency tradeoff objective would require an explicit new
protocol and narrower claim; this review does not silently adopt that objective.

These graphs have frozen weights. They cannot follow an optimizer update
automatically. Keep training and target evaluation on the unchanged PyTorch
path; export from a frozen checkpoint and repeat equivalence before using its
compiled deployment policy. This is an inference improvement, not evidence of
faster training, a better policy, new research novelty or embedded deployment.
Contribution two's security/trust work stays deferred.

7. PRESERVATION, SEEDS AND EXECUTION NOTES
New simulator steps:0. Optimizer updates:0. No research training/model/evaluation
seeds were consumed. Previously consumed timing observations were reused; all
new RNG values in this step are declared synthetic engineering-only streams.
All66 older saved diagnosis-input hashes still match, and the complete prior
engineering artifact manifest and its source/review snapshots are unchanged.
The user-facing project tracker is updated separately with its prior bytes saved.
No Word document or original research source file was written.

PowerShell marked the benchmark shell invocation as a NativeCommandError when
legacy-export/tracing warnings were written to stderr. The log contains the
completed verification and no Python traceback/FAILED.json. The generated
artifacts, equivalence vectors and timing summaries were independently replayed
and audited successfully. No benchmark was repeated to obtain a cleaner exit
status or more favorable result. Future command wrappers should preserve the
Python process return code and capture warning streams without PowerShell's
native-stderr classification. The warnings and original log are retained.

8. FILES TO USE
New implementation:compiled_controller_inference.py
Engineering workflow:run_compiled_inference_gate.py
Independent audit:audit_compiled_inference_gate.py
Focused tests:test_compiled_inference_checks.py
Artifact root:validation/controller_inference_optimization_2026-11-05_01/
  profile/profile.json
  export_feasibility/FAILED_FIRST_EXPORT.json
  compiled_gate_01/equivalence.json
  compiled_gate_01/export_metadata.json
  compiled_gate_01/raw_timings.json
  compiled_gate_01/verification.json
  compiled_gate_01/saved_artifact_audit.json
  compiled_gate_01/exports/
  compiled_gate_01.log
Read-only audit command (do not rerun timing):
  python audit_compiled_inference_gate.py <compiled_gate_01_path>
Focused test command:
  python -m unittest test_compiled_inference_checks -v
Interpreter:C:/Users/OMEN/AppData/Local/Temp/value_bound_smoke_env/Scripts/python.exe
'''
report=ROOT/'LOW_COMPLEXITY_INFERENCE_OPTIMIZATION_REVIEW_2026-11-05.txt'
if report.exists():raise FileExistsError(report)
report.write_text(text.strip()+'\n',encoding='utf-8')
assert report.read_text(encoding='utf-8').startswith('CONTROLLER INFERENCE OPTIMIZATION REVIEW')

test_info=dict(tests_passed=6,command='python -m unittest test_compiled_inference_checks -v',observed_process_exit_code=0,
    source_sha256=sha(ROOT/'test_compiled_inference_checks.py'),new_benchmark_run=False)
(OUT/'focused_test_verification.json').write_text(json.dumps(test_info,indent=2),encoding='utf-8')
status_path=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json';raw=status_path.read_bytes()
(OUT/'PROJECT_STATUS_BEFORE_INFERENCE_OPTIMIZATION.json').write_bytes(raw)
status=json.loads(raw)
status['controller_inference_optimization']=dict(protocol_id='controller-v2-onnx-cpu-inference-v1.1',
    original_model_and_scope_unchanged=True,action_equivalence_complete_actions=4608,independent_artifact_audit='passed',
    focused_tests_passed=6,compiled_gate='failed_relative_speed_screen',primary_compiled_median_ms=p['median_ms'],
    primary_compiled_p95_ms=p['p95_ms'],primary_same_run_median_speedup=speed,
    primary_relative_median_vs_compiled_wide=ratio,optimizer_updates=0,new_simulator_steps=0,research_training_launched=False,
    policy_quality_established=False,low_complexity_contribution_established=False,word_thesis_changed=False,
    report=report.name,artifacts=str(OUT.relative_to(ROOT)))
status['current_workstream']='contribution_one_compiled_inference_relative_speed_gate_failed'
status['next']='predeclare_next_engineering_design_to_address_relative_decoder_latency_no_research_launch'
assert status_path.read_bytes()==raw
status_path.write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
assert not read(status_path)['contribution_plan']['cloud_in_active_model']
assert not read(status_path)['contribution_plan']['charging_stations_in_active_model']
assert not read(status_path)['contribution_plan']['security_in_current_experiment']
snapshot=OUT/'audit_and_review_snapshot';snapshot.mkdir()
for name in ['audit_compiled_inference_gate.py','test_compiled_inference_checks.py',report.name]:shutil.copy2(ROOT/name,snapshot/name)
summary=dict(state='inference_optimization_equivalence_and_audit_passed_cost_gate_failed',
    improvement_speedup=speed,primary_relative_compiled_median=ratio,all_screens_unchanged=True,
    original_saved_inputs_unchanged=66,previous_engineering_artifacts_unchanged=True,
    no_research_training=True,word_thesis_changed=False,review_sha256=sha(report),status_sha256=sha(status_path),
    focused_tests=test_info,audit=a)
(OUT/'OPTIMIZATION_GATE_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
manifest={str(x.relative_to(OUT)).replace('\\','/'):sha(x) for x in OUT.rglob('*') if x.is_file() and x.name!='OPTIMIZATION_ARTIFACT_MANIFEST.json'}
(OUT/'OPTIMIZATION_ARTIFACT_MANIFEST.json').write_text(json.dumps(dict(files=manifest),indent=2),encoding='utf-8')
for name,h in read(OUT/'OPTIMIZATION_ARTIFACT_MANIFEST.json')['files'].items():assert sha(OUT/name)==h,name
print(json.dumps(summary,indent=2));print('Verified',len(manifest),'saved artifact hashes; review read back.')
