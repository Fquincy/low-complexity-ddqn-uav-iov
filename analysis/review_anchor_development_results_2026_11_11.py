"""Review audited development results without new simulation, fitting or selection."""
import gzip,hashlib,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1]
O=R/'development_runs/terminal_anchor_development_continuation_2026-11-11_01'
C=R/'validation/anchor_evaluation_launch_2026-11-11_01'
D=R/'validation/anchor_development_results_review_2026-11-11_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def main():
    assert not D.exists();D.mkdir()
    completion=read(C/'COMPLETION.json');audit=read(O/'saved_artifact_audit.json');v=read(O/'verification.json')
    assert completion['audit']==audit and completion['result']==v
    assert audit['state']=='passed_posttraining_continuation_full_audit'
    manifest=read(O/'run_manifest.json')['files']
    for name,digest in manifest.items():assert sha(O/name)==digest,name
    x=read(O/'comparisons_and_value_stability.json');cost=read(O/'FROZEN_CHECKPOINT_INFERENCE.json');plan=read(O/'runtime_plan.json')
    assert not v['empirical_protocol_screens_passed'] and cost['all_fit_cost_screens_pass']
    assert v['new_training_slots']==v['new_optimizer_updates']==0
    comparisons={name:dict(mean_return_difference=row['mean_return_difference'],mean_deadline_difference_percentage_points=100*row['mean_deadline_difference'],per_fit=row['per_fit']) for name,row in x['comparisons'].items()}
    # Extract descriptive means from the saved evaluation episodes, no environment execution.
    primary=[]
    for fit,seed in enumerate(plan['model_seeds']):
        rows=[]
        for task in plan['evaluation_seeds']:
            with gzip.open(O/'price_both'/f'fit_{fit:02d}'/f'evaluation_{task}.json.gz','rt',encoding='utf-8') as stream:rows.append(json.load(stream))
        primary.append(dict(model_seed=seed,mean_return=sum(y['return_sum'] for y in rows)/len(rows),
            mean_deadline_success=sum(y['deadline_success'] for y in rows)/len(rows)))
    values=[dict(model_seed=seed,**{k:row[k] for k in ['normalized_rmse','absolute_bias','near_zero_large_return_fraction','terminal_abs_error_p95']}) for seed,row in zip(plan['model_seeds'],x['value_stability']['price_both'])]
    summary=dict(state='audited_development_failed_predeclared_quality_screens',audit=audit,quality_screens=x['quality_screens'],comparisons=comparisons,
        primary_per_fit=primary,primary_value_stability=values,primary_cost=cost['metrics']['price_both'],cost_screens=cost['screens_per_fit'],
        full_payload_hashes_verified=len(manifest),new_simulator_slots=0,new_optimizer_updates=0,final_launched=False,thesis_changed=False)
    write(D/'results_summary.json',summary)
    lines=['TERMINAL-ANCHOR DEVELOPMENT RESULTS REVIEW','Date: November 11, 2026','','DECISION',
        'The continuation completed and its full artifact audit passed. The',
        'controller did NOT pass the frozen development acceptance criteria.',
        'Do not launch final evaluation or claim a proven low-complexity policy',
        'improvement from this run. Preserve all fits and failed comparisons.',
        '', 'VERIFIED COMPLETION',
        'Completed November11 at04:12:59 +08:00; elapsed approximately1.50hours.',
        '4,080 controller evaluation episodes and144 heuristic episodes saved.',
        '85 checkpoints,23,349 total saved episodes and593,045 historical optimizer',
        'updates audited;10,880 runtime equivalence checks and54,400 timing rows.',
        'The continuation added zero training slots and optimizer updates.',
        f"This review rehashed all{len(manifest):,} files in the completed run manifest",
        'and matched the completion receipt to its saved result and audit. No',
        'second simulation, training or timing measurement was performed.',
        '', 'COST EVIDENCE',
        'Storage and measured complete action-selection median/p95 gates passed',
        'in all five initializations under the declared runtime amendment.',
        'Primary online FP32 parameter payload68,264bytes versus compact64194,944',
        'bytes: approximately65% fewer bytes. Payload excludes framework/process',
        'memory and does not prove lower total deployment memory.',
        'Primary median0.0742-0.0747ms; p950.0790-0.08382ms on this CPU/input set.',
        '', 'POLICY COMPARISONS',
        'Differences below are primary minus comparator, averaged over all five',
        'initializations; higher is better. Acceptance still applies per fit,',
        'so a favorable overall mean cannot offset a failed initialization.',
        'comparator | return difference | deadline difference percentage points']
    for name in ['additive64_nonpositive','additive128_unbounded','no_price_both','all_local','nearest_rsu','uav_selective']:
        row=comparisons[name];lines.append(f"{name} | {row['mean_return_difference']:.4f} | {row['mean_deadline_difference_percentage_points']:.4f}")
    lines +=['The primary has higher mean return than the compact/wide references,',
        'but lower mean deadline success by10.9854 and8.7236 percentage points.',
        'Reference noninferiority requires return difference>=-2 and deadline',
        'difference>=-1percentage point in EVERY fit; the combined screen failed.',
        'Nearest-RSU and UAV-selective heuristics outperform the primary in both',
        'return and deadline success in every fit. The all-local baseline is',
        'outperformed, but this does not satisfy the stronger heuristic screen.',
        '', 'MECHANISM AND ABLATIONS',
        'Against corrected no-price, primary mean return is1.8640 lower and',
        'deadline success2.4295 percentage points lower. The required positive',
        'mean mechanism improvement and per-fit nondegradation are unsupported.',
        'The primary does not dominate both correction ablations in every fit.',
        'These results do not establish a useful workload-price mechanism.',
        '', 'VALUE STABILITY',
        'seed | normalized RMSE | absolute bias | nearzero fraction | terminal error p95']
    for row in values:lines.append(f"{row['model_seed']} | {row['normalized_rmse']:.6f} | {row['absolute_bias']:.6f} | {row['near_zero_large_return_fraction']:.6f} | {row['terminal_abs_error_p95']:.6f}")
    lines +=['Terminal error p95 passes<=2 and nearzero/large-return fraction passes',
        '<=.05 in all fits. However all five fits fail normalized RMSE<=.25 and',
        'absolute bias<=2. Finite-horizon/terminal handling has not established',
        'adequate whole-trajectory value calibration. Saved residual means show',
        'overoptimistic Q relative to realized greedy return-to-go. This is an',
        'empirical policy-return diagnostic, not a direct test of optimal Q.',
        '', 'PREDECLARED SCREEN OUTCOMES']
    lines += [f'{key}: {value}' for key,value in x['quality_screens'].items()]
    lines +=['All six combined quality screens failed; the separate cost gate passed.',
        '', 'NEXT WORK',
        'Use the saved evaluation/training traces to diagnose deadline misses by',
        'destination, queue/load, task deadline and time stage; compare primary',
        'against corrected no-price, compact64, wide128 and the strong heuristics.',
        'Check whether long-horizon optimistic values, local/offload allocation',
        'or workload-price behavior explain the deficits. These are hypotheses;',
        'this results review does not establish their cause. Do not start another',
        'training run simply by increasing budget or tuning against these seeds.',
        'After diagnosis, prospectively declare one justified correction and its',
        'comparison on fresh development seeds. The48 development evaluation',
        'seeds are now consumed; they are not an untouched final test set.',
        'Keep final seeds unused, retain every fit, and do not retroactively alter',
        'thresholds, drop comparators or choose a winning initialization.',
        '', 'THESIS IMPLICATION',
        'We now have audited evidence of smaller parameter payload and faster',
        'measured action selection under the declared routes, but not equal or',
        'better deadline performance or the proposed mechanism benefit. The',
        'intended combined low-complexity contribution is not yet demonstrated.',
        'Do not revise the thesis to claim that combined improvement on this basis.',
        'No security/trust claim is tested; that contribution remains deferred.',
        'Cloud/charging stations excluded. Word thesis and final evaluation unchanged.']
    report=R/'TERMINAL_ANCHOR_DEVELOPMENT_RESULTS_REVIEW_2026-11-11.txt';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    tracker=R/'LOW_COMPLEXITY_PROJECT_STATUS.json';shutil.copy2(tracker,D/'PROJECT_STATUS_BEFORE_RESULTS_REVIEW.json');status=read(tracker)
    status['current_workstream']='anchor_development_completed_full_audit_passed_cost_passed_quality_screens_failed'
    status['next']='saved_trace_deadline_and_value_diagnosis_before_any_fresh_seed_design_or_final_evaluation'
    status['anchor_development_results_review']=dict(report=report.name,full_audit_passed=True,all_cost_screens_passed=True,
        quality_screens=x['quality_screens'],development_evaluation_complete=True,development_evaluation_seeds_consumed=48,
        final_launched=False,new_training_slots=0,word_thesis_changed=False)
    write(tracker,status)
    paths=[report,D/'results_summary.json',D/'PROJECT_STATUS_BEFORE_RESULTS_REVIEW.json',Path(__file__),C/'COMPLETION.json',O/'run_manifest.json',O/'saved_artifact_audit.json',O/'verification.json',O/'comparisons_and_value_stability.json',O/'FROZEN_CHECKPOINT_INFERENCE.json']
    write(D/'REVIEW_MANIFEST.json',dict(state=summary['state'],files={str(p.relative_to(R)).replace('\\','/'):sha(p) for p in paths}))
    print(json.dumps(dict(report=report.name,files_rehashed=len(manifest),audit_passed=True,cost_passed=True,quality_screens=x['quality_screens'],primary_per_fit=primary)))
if __name__=='__main__':main()
