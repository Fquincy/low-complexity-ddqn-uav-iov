"""Seal runtime repair and independent audits; leave research output untouched."""
import json,hashlib,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import audit_anchor_torch_core_runtime as auditor
import run_anchor_development_v4 as r
A=ROOT/'validation/anchor_numerical_correction_2026-11-11_01'
B=ROOT/'validation/anchor_torch_core_runtime_2026-11-11_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    first=r.read(A/'verification.json');second=r.read(B/'verification.json');audit=r.read(B/'saved_artifact_audit.json');cost=r.read(B/'cost.json')
    assert not first['all_candidate_actions_and_values_pass'] and first['candidate_failed_models']==1
    assert second['all_candidate_actions_and_values_pass'] and cost['all_fit_cost_screens_pass']
    assert audit['observations_replayed']==10880 and audit['cost_audit']['raw_rows_recomputed']==54400
    for folder in [A,B]:
        for name,digest in r.read(folder/'COMPLETION_MANIFEST.json')['files'].items():assert sha(folder/name)==digest,name
    # Regression: corruption of a reported cost must be rejected, not re-timed.
    path=B/'cost.json';original=path.read_bytes();bad=r.read(path)
    bad['metrics']['price_both'][0]['median_ms']+=1.;r.write(path,bad)
    try:
        try:auditor.audit(B)
        except AssertionError as error:assert str(error)=='cost.json',str(error)
        else:raise AssertionError('corrupted cost accepted')
    finally:path.write_bytes(original)
    assert sha(path)==r.read(B/'COMPLETION_MANIFEST.json')['files']['cost.json']
    r.write(B/'tamper_regression.json',dict(state='passed',cost_summary_corruption_rejected=True,
        original_bytes_restored=True,new_timing_measurements=0))
    report=ROOT/'TERMINAL_ANCHOR_RUNTIME_NUMERICAL_REPAIR_REVIEW_2026-11-11.txt'
    lines=['TERMINAL-ANCHOR RUNTIME NUMERICAL REPAIR REVIEW','Date: November 11, 2026','','OUTCOME',
        'The replacement runtime passed exact actions and Q tolerance on all85',
        'frozen models x128 fixed engineering observations (10,880 checks).',
        'Storage, median and p95 gates passed in all five initializations.',
        'The saved-artifact auditor replayed all85 models and recomputed all',
        '54,400 timing rows without new measurements. This is a same-weight',
        'runtime engineering result, not a return/deadline/value-stability result.',
        '', 'PRESERVED FAILED CANDIDATE',
        'First prospectively tested disabling ORT graph optimization, preserving',
        'the original graphs/decoders. It failed the same no_price_both fit0',
        'observation110 branch18 near tie. An all85 model sweep found exactly',
        'one differing joint decision out of10,880 for the original and disabled',
        'optimization routes; every Q comparison passed its original tolerance.',
        'An independent audit confirmed the failed candidate. It was not timed.',
        'Its protocol, sources, predictions and failed result remain retained.',
        '', 'DECLARED REPLACEMENT',
        'Before the next candidate test, wrote',
        'ANCHOR_TORCH_CORE_RUNTIME_AMENDMENT_2026-11-11.json.',
        'Use the original single-thread PyTorch FP32 neural-core operations and',
        'frozen weights, followed by the unchanged native action decoder. Apply',
        'this replacement to ALL native-decoder families, including compact64,',
        'wide128, EMC and BDQ references; do not selectively repair one arm.',
        'The original four coupled full-policy ORT families retain their route.',
        'The six corrected arms retain their input checks and horizon scaling.',
        'No fallback, relaxed tie rule, adjusted tolerance, changed weights or',
        'altered reference policy was used. The old ONNX route remains failed.',
        'The new candidate is a separately declared route, not retrospective',
        'acceptance of the failed graph. Original rewards, training schedules,',
        'seeds, thresholds, optimizer settings and model definitions are intact.',
        '', 'COMPLETE ACTION-SELECTION MEASUREMENT',
        'One benchmark: all17 controllers x5fits x5randomized rounds x128calls;',
        '32warmups per path/round. Includes input validation, core execution,',
        'native decoding, horizon handling where applicable and action copying.',
        'Graphs, source hashes and checkpoint hashes frozen before the check;',
        'timing payload hashes and randomized orders written before timing.',
        'No favorable re-timing or fit selection. Measures this host and the',
        'fixed engineering input set; absolute times depend on machine/load.',
        'fit | primary median ms | wide128 median ms | primary p95 ms']
    for fit in range(5):
        p=cost['metrics']['price_both'][fit];w=cost['metrics']['additive128_unbounded'][fit]
        lines.append(f"{fit} | {p['median_ms']:.6f} | {w['median_ms']:.6f} | {p['p95_ms']:.6f}")
    lines+=['Online FP32 parameter bytes: primary68,264; compact64194,944;',
        'wide128422,016. This satisfies the unchanged primary<=50% compact64',
        'parameter-payload criterion. This quantity excludes framework process',
        'memory, serialized checkpoint/optimizer size and Python/PyTorch runtime.',
        'Required auxiliary buffers remain unchanged; parameter bytes are not',
        'a claim about total executable deployment memory or hardware suitability.',
        '', 'REGRESSIONS AND AUDIT',
        'Known near tie reproduces on the old route and is corrected by the new',
        'route. Exact actions and Q tolerance passed for every retained fit,',
        'including every comparator. Auditor independently rechecked PyTorch',
        'reference predictions, graph/parameter/source hashes and all metrics.',
        'Deliberate cost-summary corruption was rejected by the manifest check;',
        'the original bytes were restored. No timings were repeated for this.',
        'Evidence applies to fixed engineering observations, not every possible',
        'state. Do not infer deadline performance, calibration, policy improvement',
        'or research novelty from export-equivalence/cost success.',
        '', 'NEXT ENGINEERING STEP',
        'Implement and test a post-training continuation that loads the existing',
        '85 frozen checkpoints with this runtime amendment, uses no optimizer',
        'steps, and leaves the original failed run and source snapshots intact.',
        'Test the resume plan, workload identity, evaluation invariance and full',
        'saved-training/artifact audit on isolated reduced fixtures. Review the',
        'continuation before opening reserved held-out development episodes.',
        'The original comparison remains incomplete until its policy comparisons',
        'and full audit finish. Do not retrain or change the frozen acceptance',
        'rules. Final-evaluation execution remains disabled.',
        '', 'PRESERVATION',
        'No simulator calls, optimizer updates or held-out research episodes were',
        'executed during this gate. Original FAILED.json, status and85 frozen',
        'checkpoints unchanged. Current project tracker updated separately.',
        'Cloud and charging stations excluded; security/trust deferred.',
        'Word thesis unchanged.',
        '', 'ARTIFACTS',str(A.relative_to(ROOT)),str(B.relative_to(ROOT)),
        'anchor_original_torch_core_runtime.py','check_anchor_torch_core_runtime.py',
        'audit_anchor_torch_core_runtime.py']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    tracker=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json';backup=ROOT/'validation/project_status_before_anchor_runtime_repair_2026-11-11.json'
    assert not backup.exists();shutil.copy2(tracker,backup);status=r.read(tracker)
    status['current_workstream']='anchor_saved_weight_runtime_repaired_equivalence_and_cost_passed_comparison_still_incomplete'
    status['next']='implement_and_test_posttraining_continuation_and_full_audit_before_reserved_development_evaluation'
    status['anchor_runtime_numerical_repair']=dict(report=report.name,failed_candidate=str(A.relative_to(ROOT)),
        passing_candidate=str(B.relative_to(ROOT)),models_verified=85,exact_joint_action_checks=10880,
        timing_rows=54400,all_five_cost_screens_passed=True,heldout_development_started=False,
        research_relaunched=False,new_training_slots=0,word_thesis_changed=False)
    r.write(tracker,status)
    paths=[p for folder in [A,B] for p in folder.rglob('*') if p.is_file()]
    names=['check_anchor_numerical_correction.py','audit_anchor_numerical_correction.py',
        'anchor_original_torch_core_runtime.py','check_anchor_torch_core_runtime.py','audit_anchor_torch_core_runtime.py',
        'ANCHOR_RUNTIME_NUMERICAL_CORRECTION_PROTOCOL_2026-11-11.json','ANCHOR_TORCH_CORE_RUNTIME_AMENDMENT_2026-11-11.json']
    paths += [ROOT/n for n in names]+[report,backup,Path(__file__)]
    files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}
    manifest=ROOT/'validation/anchor_runtime_numerical_repair_completion_2026-11-11.json'
    r.write(manifest,dict(state='same_weight_runtime_equivalence_and_cost_passed_no_heldout_evaluation',files=files))
    for name,digest in files.items():assert sha(ROOT/name)==digest,name
    print(json.dumps(dict(report=report.name,sealed_files=len(files),all_equivalence_passed=True,all_cost_screens_passed=True,heldout_evaluation_started=False)))
if __name__=='__main__':main()
