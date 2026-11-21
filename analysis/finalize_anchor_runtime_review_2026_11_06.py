"""Record the fixed saved-weight engineering benchmark; do not launch research."""
import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import check_anchor_runtime as check
OUT=ROOT/'validation/anchor_runtime_single_validation_2026-11-06_01'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def main():
    audit=check.audit(OUT);assert audit['all_cost_screens_pass']
    # Prove statistics auditing rejects deliberate corruption; restore exact bytes.
    timing_path=OUT/'timings.json';original=timing_path.read_bytes();bad=read(timing_path)
    first_key=next(iter(bad['metrics']))
    bad['metrics'][first_key]['median_ms']+=1.;write(timing_path,bad)
    try:
        try:check.audit(OUT)
        except AssertionError:pass
        else:raise AssertionError('tampered timing accepted')
    finally:timing_path.write_bytes(original)
    check.audit(OUT)
    metrics=read(timing_path)['metrics'];equivalence=read(OUT/'equivalence.json')
    assert equivalence['invalid_inputs_rejected']==36 and equivalence['all_old_new_values_exact']
    prior=ROOT/'validation/anchor_research_preparation_2026-11-06_01'
    old_protocol=ROOT/'TERMINAL_ANCHOR_RESEARCH_PROTOCOL_2026-11-06.json'
    amendment=ROOT/'ANCHOR_RUNTIME_ROUTE_AMENDMENT_2026-11-06.json'
    write(amendment,dict(id='anchor-runtime-route-amendment-v1',base_protocol_sha256=sha(old_protocol),
        correction='Replace corrected-arm duplicate-validation wrapper with anchor_runtime_single_validation.values; unchanged model, decoder and FP32 outputs.',
        historical_failed_cost_result='validation/anchor_research_preparation_2026-11-06_01/FROZEN_CHECKPOINT_INFERENCE.json',
        old_runtime_sha256=sha(ROOT/'run_anchor_research_preparation.py'),new_adapter_sha256=sha(ROOT/'anchor_runtime_single_validation.py'),
        evidence=str(OUT.relative_to(ROOT)).replace('\\','/'),research_enabled=False,
        thresholds_changed=False,weights_changed=False,reward_changed=False,
        next='Integrate declared route into separately versioned runner/auditor and verify reduced shared orchestration before launch review.'))
    lines=['TERMINAL-ANCHOR RUNTIME OVERHEAD REVIEW','Date: November 6, 2026','','OUTCOME',
        'The fixed saved-weight engineering benchmark passed all declared storage,',
        'median-latency and p95-latency criteria in both synthetic initializations.',
        'No training, simulator calls, model changes or research launch occurred.',
        'The original median-latency failure remains preserved. This result verifies',
        'a runtime correction on existing engineering weights, not policy quality.',
        '', 'CORRECTION',
        'The old corrected wrapper called np.asarray/isfinite, then the underlying',
        'native.values repeated conversion/finite validation. The new independent',
        'adapter validates once, invokes the unchanged ORT core and native decoder,',
        'then applies the original separate FP32 rounding/scaling operations.',
        'Shape, complete finite checks and nonterminal time-grid checks remain.',
        'No acceptance thresholds, rewards, seed schedules or neural weights changed.',
        'The old runner/adapters and their hash-covered evidence were not edited.',
        '', 'PROSPECTIVE FIXED CHECK',
        'ANCHOR_RUNTIME_CHECK_PROTOCOL_2026-11-06.json was written before measurement.',
        'Use all17 saved controller arms and both synthetic initializations, the',
        'same128 fixed engineering observations, five randomized timing rounds,',
        '32 warmups and128 calls per path/round. Time corrected six with both old',
        'and new routes, and the other eleven with their existing routes:46 paths',
        'and29,440 measurements. Include final action copying uniformly. The fixed',
        'single-thread backend, graphs, extra arrays, sources and order freeze before',
        'timing. No favorable retiming or fit/weight selection was performed.',
        '', 'EQUIVALENCE AND AUDIT',
        '4,352 saved-model eager/export action checks passed. For the corrected six,',
        '1,536 old/new actions and FP32 Q values matched exactly. Eager/export Q',
        'matched atol1e-6/rtol1e-5. All36 invalid-state/time cases were rejected.',
        'The artifact auditor reverified payload/source hashes and recomputed all',
        '29,440 raw timing rows and thresholds without new measurements. Deliberate',
        'summary-statistic corruption was rejected; original bytes were restored.',
        'Recomputing median/p95 alone cannot detect every single raw-sample change;',
        'the completion manifest below separately seals all final timing bytes.',
        '', 'PRIMARY RESULTS',
        'fit | old wrapper median ms | new wrapper median ms | wide128 median ms | new p95 ms']
    for fit in range(2):
        old=metrics[f'price_both/{fit}/old'];new=metrics[f'price_both/{fit}/new'];wide=metrics[f'additive128_unbounded/{fit}/original']
        lines.append(f"{fit} | {old['median_ms']:.6f} | {new['median_ms']:.6f} | {wide['median_ms']:.6f} | {new['p95_ms']:.6f}")
    lines+=['Primary online FP32 parameter bytes remain68,264; compact64 bytes194,944.',
        'Storage<=50% compact64, median<=wide128 and p95<=1ms pass in both fits.',
        'Small absolute timing differences remain machine/load/input dependent.',
        'Do not extrapolate this engineering benchmark to unseen trained policies',
        'or infer return/deadline/calibration success from runtime measurements.',
        '', 'NEXT ENGINEERING STEP',
        'The runtime amendment is recorded separately and is not yet installed',
        'in the frozen seventeen-arm research runner. Integrate it into a new',
        'versioned runner and auditor, preserving all previous artifacts. Verify',
        'the reduced shared workflow and amended source freezes, then make a',
        'separate launch-readiness decision. Research modes remain disabled.',
        'Do not rerun engineering timing simply to seek a better result.',
        '', 'FILES','anchor_runtime_single_validation.py','check_anchor_runtime.py',
        'ANCHOR_RUNTIME_CHECK_PROTOCOL_2026-11-06.json',amendment.name,str(OUT.relative_to(ROOT)),
        '', 'Historical failed research results, final seeds and Word thesis unchanged.',
        'Cloud/charging stations excluded; security/trust remains deferred.']
    report=ROOT/'TERMINAL_ANCHOR_RUNTIME_OVERHEAD_REVIEW_2026-11-06.txt'
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    tracker=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json';backup=ROOT/'validation/project_status_before_anchor_runtime_review_2026-11-06.json'
    assert not backup.exists();shutil.copy2(tracker,backup)
    status=read(tracker);status['current_workstream']='anchor_single_validation_runtime_check_passed_no_research'
    status['next']='integrate_declared_runtime_route_into_versioned_runner_auditor_and_verify_before_launch_review'
    status['anchor_runtime_overhead_review']=dict(report=report.name,engineering_cost_screens_passed=True,
        source_amendment=amendment.name,installed_in_research_runner=False,research_launched=False,word_thesis_changed=False)
    write(tracker,status)
    files=[x for x in OUT.rglob('*') if x.is_file()]+[ROOT/name for name in read(OUT/'PRE_TIMING_FREEZE.json')['source_hashes']]+[report,amendment,backup,Path(__file__)]
    manifest={str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for path in files}
    write(ROOT/'validation/anchor_runtime_overhead_completion_2026-11-06.json',dict(state='runtime_engineering_passed_no_research',files=manifest))
    print(json.dumps(dict(report=report.name,audit=audit,primary={str(f):dict(old=metrics[f'price_both/{f}/old'],new=metrics[f'price_both/{f}/new'],wide=metrics[f'additive128_unbounded/{f}/original']) for f in range(2)}),indent=2))

if __name__=='__main__':main()
