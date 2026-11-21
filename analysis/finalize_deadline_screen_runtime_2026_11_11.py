"""Record verified faster screen runtime; preserve prior failure and research results."""
import hashlib,json,shutil,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import audit_deadline_screen_fast_runtime as auditor
O=R/'validation/deadline_screen_fast_runtime_2026-11-11_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def main():
    reportdata=read(O/'verification.json');audit=read(O/'saved_artifact_audit.json');cost=read(O/'cost.json')
    assert reportdata['checks']==audit['actions_replayed']==1792
    assert all(all(row[k] for k in ['storage','median','p95']) for row in cost['screens'])
    for name,digest in read(R/'validation/deadline_screen_engineering_completion_2026-11-11.json')['files'].items():assert sha(R/name)==digest,name
    # Deliberate report tampering must be caught without rebenchmarking.
    path=O/'cost.json';original=path.read_bytes();bad=read(path);bad['metrics']['price_both_fit_00']['median_ms']+=1.;write(path,bad)
    try:
        try:auditor.audit()
        except AssertionError as error:assert str(error)=='cost.json',str(error)
        else:raise AssertionError('altered cost accepted')
    finally:path.write_bytes(original)
    assert sha(path)==read(O/'COMPLETION_MANIFEST.json')['files']['cost.json']
    write(O/'tamper_test.json',dict(cost_summary_corruption_rejected=True,original_bytes_restored=True,no_retiming=True))
    report=R/'DEADLINE_SCREEN_RUNTIME_OPTIMIZATION_REVIEW_2026-11-11.txt'
    lines=['CONSERVATIVE DEADLINE SCREEN RUNTIME OPTIMIZATION REVIEW','Date: November11,2026','','OUTCOME',
        'The optimized runtime passed all declared equivalence, interval-boundary',
        'and synthetic storage/latency checks. Both retained initializations pass',
        'median<=screened wide128 andp95<=1ms. No research run was launched.',
        'This verifies implementation behavior/cost on fixed synthetic weights,',
        'not deadline or return improvement of a trained screened controller.',
        '', 'CHANGE',
        'deadline_screen_fast_runtime.py combines complete finite/task/time input',
        'checks and conservative interval predicates into one Numba CPU pass.',
        'Neural Linear/ReLU arithmetic remains original PyTorch FP32. Subsequent',
        'price/flat/branch selection uses existing native decoders and first-index',
        'tie behavior. Only local scores certified infeasible are excluded.',
        'Both initializations of every price/no-price/plain/compact/wide/BDQ family',
        'use the same route change, not a selective advantage for the primary.',
        'The optimized route computes core outputs directly and eliminates full',
        'PyTorch task-score tables plus repeated validation/time processing in',
        'the prior prototype. No learned weights, training updates, masks or',
        'thresholds changed. Old runtime and failed timing evidence remain intact.',
        '', 'PRESERVED SEMANTICS',
        'Outward adjacent FP32 task bounds, physical scaling and outward FP64',
        'rounding are retained, including TIME_EPS1e-10. Ambiguous local choices',
        'remain allowed; no remote deadline is certified. All four remotes and',
        'movement choices remain available. Terminal action selection is rejected.',
        'The time check uses the same FP32 multiply-by40 and grid tolerance as',
        'the prototype. No hidden simulator variables or outcome labels are used.',
        'This is an inference-only adapter. Training, objective and target code',
        'remain the separately tested prototype. No research runner is integrated.',
        '', 'VERIFICATION',
        'Same14 saved random-weight models, same128 synthetic observations, same',
        'seven model families and two initializations as the failed benchmark.',
        'All1,792 eager/prototype/optimized action comparisons matched exactly;',
        'chosen Q matched original rtol1e-5/atol1e-6. Q is not claimed bitwise equal.',
        '100,020 randomized task mask checks exactly matched the original screen;',
        'every excluded task was truly locally infeasible by its exact synthetic',
        'cycles/deadline.3,000 equality/near-boundary perturbations retained local',
        'and matched original masks. Invalid finite/range/time cases were rejected.',
        'A zero-weight tie fixture retained first-index remote choice when local',
        'was excluded. No near-tie rule or acceptance tolerance was loosened.',
        'Independent auditor reloaded all models, replayed decisions/masks/values,',
        'reproduced boundary tests and recomputed every timing statistic and gate.',
        'Cost-summary tampering was rejected; exact original bytes were restored.',
        '', 'FIXED COMPLETE-ACTION BENCHMARK',
        'Protocol declared before implementation measurement. Existing model/input',
        'payloads, source snapshots and randomized orders sealed before testing;',
        'all payloads rechecked and frozen before timing. Five rounds x128calls',
        'with32warmups per path/round:8,960 total timed calls. Only one benchmark.',
        'Complete action selection includes validation/screen, original core,',
        'native decoder, horizon handling and final action copy. All fits retained.',
        'No simulator episodes, parameter fitting, retiming or seed selection.',
        'fit | primary median ms | screened wide128 median ms | primary p95 ms']
    for fit in range(2):
        p=cost['metrics'][f'price_both_fit_{fit:02d}'];w=cost['metrics'][f'additive128_unbounded_fit_{fit:02d}']
        lines.append(f"{fit} | {p['median_ms']:.6f} | {w['median_ms']:.6f} | {p['p95_ms']:.6f}")
    lines+=['Primary online FP32 parameter payload68,264bytes, existing buffer20bytes;',
        'compact64194,944parameter bytes. Zero additional trainable parameters.',
        'Temporary arrays, Python/PyTorch/Numba binaries, process memory and peak',
        'deployment memory are not counted in online parameter bytes.',
        'Prior prototype primary median0.7089/0.74635ms andp951.477395/1.85237ms',
        'remain recorded as failed under its original measurement. The new same-',
        'weight benchmark passes; it does not retroactively convert that result.',
        'Timing sessions have different machine/load conditions, so historical',
        'ratios are descriptive. Current reference gates use contemporaneous',
        'measurements with the same screening and randomized protocol.',
        '', 'NEXT ENGINEERING STEP',
        'Integrate the screened training/exploration and mask-aware DDQN targets',
        'with this verified inference route into a separately versioned paired',
        'simulator fixture. Use isolated synthetic seeds, retain final checkpoints,',
        'verify masks on exact generated task records, paired workload/RNG identity,',
        'all equal-information controls, no terminal decoding and artifact hashes.',
        'Re-export/reverify all trained fixture models and complete-action cost,',
        'then review evidence before preparing a fresh-seed research comparison.',
        'Do not launch research from these random-weight engineering results.',
        'The old scientific failures, especially whole-trajectory value calibration,',
        'are unresolved. Screening must still demonstrate return/deadline benefit',
        'against equally screened controls; implementation success is not novelty.',
        'Broader UAV-IoV related-work check and supervisor scope discussion remain.',
        '', 'SCOPE',
        'No simulator calls or optimizer updates, no reserved task/model seeds, no',
        'research checkpoint loading, no final evaluation and no Word thesis edits.',
        'Cloud/charging stations excluded; security/trust remains deferred.']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    tracker=R/'LOW_COMPLEXITY_PROJECT_STATUS.json';backup=O/'PROJECT_STATUS_BEFORE_REVIEW.json'
    assert not backup.exists();shutil.copy2(tracker,backup);s=read(tracker)
    s['current_workstream']='deadline_screen_optimized_runtime_equivalence_and_synthetic_cost_passed_no_research'
    s['next']='integrate_screened_training_exploration_ddqn_and_fast_inference_into_paired_synthetic_simulator_fixture'
    s['deadline_screen_runtime_optimization']=dict(report=report.name,models=14,action_checks=1792,random_task_mask_checks=100020,
        boundary_checks=3000,timing_rows=8960,all_synthetic_cost_screens_passed=True,
        primary_median_ms=[cost['metrics'][f'price_both_fit_{i:02d}']['median_ms'] for i in range(2)],
        new_simulator_slots=0,new_optimizer_updates=0,research_launched=False,word_thesis_changed=False)
    write(tracker,s)
    paths=[p for p in O.rglob('*') if p.is_file()]+[R/n for n in ['deadline_screen_fast_runtime.py','check_deadline_screen_fast_runtime.py','audit_deadline_screen_fast_runtime.py','DEADLINE_SCREEN_RUNTIME_OPTIMIZATION_PROTOCOL_2026-11-11.json']]+[report,Path(__file__)]
    files={str(p.relative_to(R)).replace('\\','/'):sha(p) for p in paths}
    write(R/'validation/deadline_screen_runtime_optimization_completion_2026-11-11.json',dict(state='engineering_equivalence_and_cost_passed_no_research',files=files))
    for name,digest in files.items():assert sha(R/name)==digest,name
    print(json.dumps(dict(report=report.name,audit_passed=True,all_cost_screens_passed=True,research_launched=False)))
if __name__=='__main__':main()
