"""Review frozen development outputs; no training or model changes."""
import gzip
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'development_runs/controller_v3_development_2026-11-06_01'
CONTROL=ROOT/'validation/v3_development_launch_2026-11-06_01'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def main():
    status=read(CONTROL/'LIVE_STATUS.json');assert status['state']=='completed_and_audited'
    result=read(OUT/'verification.json');audit=read(OUT/'saved_artifact_audit.json')
    analysis=read(OUT/'comparisons_and_value_stability.json');cost=read(OUT/'FROZEN_CHECKPOINT_INFERENCE.json')
    plan=read(OUT/'runtime_plan.json');completion=read(CONTROL/'COMPLETION.json')
    assert completion['runner']==result and completion['audit']==audit
    assert audit['state']=='passed_v3_research_saved_artifact_audit'
    manifest=read(OUT/'run_manifest.json')['files']
    for name,digest in manifest.items():assert sha(OUT/name)==digest,name
    assert result['training_slots']==440000 and result['optimizer_updates']==383735
    assert result['checkpoint_count']==55 and not audit['empirical_protocol_screens_passed']
    assert cost['all_fit_cost_screens_pass']
    comparisons=analysis['comparisons'];seeds=plan['model_seeds']
    primary_values=analysis['value_stability']['amortized_workload_price24']
    primary=[]
    for i,seed in enumerate(seeds):
        rows=[]
        for task in plan['evaluation_seeds']:
            with gzip.open(OUT/'amortized_workload_price24'/f'fit_{i:02d}'/f'evaluation_{task}.json.gz','rt',encoding='utf-8') as stream:rows.append(json.load(stream))
        primary.append(dict(seed=seed,mean_return=sum(r['return_sum'] for r in rows)/len(rows),
            deadline_success=sum(r['deadline_success'] for r in rows)/len(rows)))
    refs=['additive64_nonpositive','additive128_unbounded','emc_q_nonpositive','bdq64_uniform_adapted','plain24_nonpositive']
    failures={name:[r['seed'] for r in comparisons[name]['per_fit'] if r['return_difference']< -2 or r['deadline_difference']< -.01] for name in refs}
    metrics=cost['metrics'];storage=metrics['amortized_workload_price24'][0]['parameter_bytes']
    compact=metrics['additive64_nonpositive'][0]['parameter_bytes'];wide=metrics['additive128_unbounded'][0]['parameter_bytes']
    evidence=dict(state='completed_audited_development_performance_gate_failed',
        manifest_entries_verified=len(manifest),training_slots=440000,optimizer_updates=383735,
        artifact_audit=audit,quality_screens=analysis['quality_screens'],cost_screens_passed=True,
        primary_absolute_per_fit=primary,comparison_failures=failures,comparisons=comparisons,
        primary_value_stability=primary_values,all_arm_value_stability=analysis['value_stability'],
        primary_parameter_bytes=storage,compact_parameter_bytes=compact,wide_parameter_bytes=wide,
        storage_reduction_vs_compact=1-storage/compact,storage_reduction_vs_wide=1-storage/wide,
        final_phase_launched=False,thesis_changed=False,
        next='diagnose_saved_terminal_value_learning_and_deadline_errors_before_any_new_design_or_run')
    evidence_path=ROOT/'validation/v3_development_results_review_2026-11-06.json';write(evidence_path,evidence)
    lines=['VERSION 3 DEVELOPMENT RESULTS REVIEW','Date: November 6, 2026','','DECISION',
        'The experiment completed and passed the saved-artifact audit. The frozen',
        'development performance gate FAILED. Do not launch the reserved final phase',
        'or present this controller as preserving performance under the agreed rules.',
        'This is valid negative development evidence, retained in full. It does not',
        'prove that a successful low-complexity controller is impossible.','',
        'COMPLETION AND INTEGRITY',
        f"440,000 training slots;383,735 optimizer updates;55 final checkpoints.",
        f"Auditor checked{audit['episodes_checked']:,} episodes,7,040 exported decisions and35,200 timing rows.",
        f"This review rehashed all{len(manifest):,} run-manifest entries successfully.",
        'All five initializations and all eleven arms were retained. No additional',
        'training, threshold changes or thesis edits were performed in this review.','',
        'WHAT PASSED',
        f'Primary online FP32 parameters: {storage:,} bytes; compact64: {compact:,}; wide128: {wide:,}.',
        f'Storage reduction: {100*(1-storage/compact):.2f}% versus compact64; {100*(1-storage/wide):.2f}% versus wide128.',
        'All five fits passed the declared storage, median-latency and p95-latency',
        'screens on the saved128 engineering observations with the fixed runtime.',
        'These are CPU benchmark results for the declared inputs and backend.',
        'The smaller plain24 reference remains smaller/faster than the primary.','',
        'PRIMARY COST PER FIT (complete action selection)',
        'seed | median ms | p95 ms']
    for seed,row in zip(seeds,metrics['amortized_workload_price24']):lines.append(f"{seed} | {row['median_ms']:.5f} | {row['p95_ms']:.5f}")
    lines+=['','PRIMARY ABSOLUTE HELD-OUT PERFORMANCE','Each fit uses the same48 development workloads.','seed | mean episode return | deadline success %']
    for row in primary:lines.append(f"{row['seed']} | {row['mean_return']:.4f} | {100*row['deadline_success']:.4f}")
    lines+=['','PREDECLARED SCREEN RESULTS']
    for name,passed in analysis['quality_screens'].items():lines.append(f"{name}: {'PASS' if passed else 'FAIL'}")
    lines+=['','ALL PAIRED CONTRASTS',
        'Differences are primary minus comparator; higher is better. Deadline',
        'differences below are percentage points, not relative percentage changes.',
        'Required neural-reference margins: return>=-2 and deadline>=-1 point',
        'in EVERY fit. Heuristics and no-price ablation require both>=0 per fit.',
        'No-price across-fit means also require return>=2 and deadline>=1 point.']
    for name,comparison in comparisons.items():
        lines+=['',name,'seed | return difference | deadline difference (points)']
        for row in comparison['per_fit']:lines.append(f"{row['seed']} | {row['return_difference']:+.4f} | {100*row['deadline_difference']:+.4f}")
        lines.append(f"All-fit mean: {comparison['mean_return_difference']:+.4f} return; {100*comparison['mean_deadline_difference']:+.4f} deadline points.")
    lines+=['','INTERPRETATION',
        'The primary fails the deadline margin against compact64 in all five fits.',
        'Its average deadline deficit is12.2181 points against compact64 and6.7358',
        'points against wide128. It also trails the nearest-RSU and UAV-selective',
        'heuristics on both metrics in every fit. The primary meets the declared',
        'joint margins versus plain24 and adapted BDQ in all five fits, but success',
        'against these references does not offset failure against stronger ones.',
        'Pricing beats the matched no-price arm on average by27.0825 return units',
        'and7.0601 deadline points. However seed41023 loses6.3177 deadline points',
        'against that ablation, so the required per-fit mechanism screen fails.',
        'These comparisons are development observations, not final superiority tests.','',
        'VALUE ESTIMATES',
        'Primary values are finite and nonpositive, but every primary fit fails',
        'the normalized RMSE, absolute-bias and terminal-error thresholds.',
        'Nonpositive values alone do not ensure calibration or stable magnitude.',
        'The diagnostic compares learned Q to realized return on its own greedy',
        'trajectories. It is not a proof of Q-star accuracy or a causal diagnosis.',
        'seed | normalized RMSE (limit .25) | abs bias (limit2) | terminal p95 (limit2)']
    for seed,row in zip(seeds,primary_values):lines.append(f"{seed} | {row['normalized_rmse']:.4f} | {row['absolute_bias']:.4f} | {row['terminal_abs_error_p95']:.4f}")
    lines+=['','NEXT WORK',
        '1. Preserve this completed run, including failed fits and all comparator',
        '   results. Keep the final evaluation seed set untouched.',
        '2. Diagnose saved terminal/nonterminal predictions, Bellman targets,',
        '   gradient histories and action/destination decisions. Compare the',
        '   primary with compact64, wide128, no-price and the strong heuristics.',
        '   Check why finite-horizon values remain very negative near termination',
        '   and which decisions account for deadline misses. Investigate using',
        '   saved data first; the present evidence does not establish the cause.',
        '3. Only after that diagnosis, write one prospective correction/design',
        '   brief and comparison amendment. Any later change must be declared',
        '   before a new run, with fresh development seeds. Do not change the',
        '   frozen rules retrospectively or select the best initialization.',
        '4. Judge the amended controller on fresh evidence before considering',
        '   the separately authorized final phase or thesis claim revision.','',
        'Scope remains local/UAV/RSU offloading, with no cloud or charging station.',
        'Security/trust stays deferred to contribution two. Word thesis unchanged.',
        'No new training or final experiment was launched by this review.','',
        'DETAILS',str(evidence_path.relative_to(ROOT)),str(OUT.relative_to(ROOT))]
    report=ROOT/'LOW_COMPLEXITY_V3_DEVELOPMENT_RESULTS_REVIEW_2026-11-06.txt'
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive=CONTROL/'PROJECT_STATUS_AT_RESULTS_REVIEW.json'
    tracker=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json'
    if not archive.exists():shutil.copy2(tracker,archive)
    tracker_data=read(tracker)
    tracker_data['current_workstream']='contribution_one_v3_completed_audited_development_performance_gate_failed'
    tracker_data['next']='diagnose_saved_value_and_deadline_failures_before_prospective_redesign_no_final_launch'
    tracker_data['v3_development_launch_2026_11_06'].update(state='completed_and_audited_performance_screens_failed',performance_claim_established=False)
    tracker_data['v3_development_results_review_2026_11_06']=dict(report=report.name,
        artifact_audit_passed=True,performance_screens_passed=False,cost_screens_passed=True,
        all_fits_retained=True,final_phase_launched=False,word_thesis_changed=False,
        evidence=str(evidence_path.relative_to(ROOT)).replace('\\','/'))
    write(tracker,tracker_data)
    write(CONTROL/'RESULTS_REVIEW_MANIFEST.json',dict(files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p)
        for p in [evidence_path,report,archive,Path(__file__)]},prior_tracker_archive=str(archive)))
    print(json.dumps(dict(report=report.name,state=evidence['state'],manifest_entries_verified=len(manifest),
        primary=primary,primary_value_failures=[dict(seed=seed,nrmse=row['normalized_rmse'],bias=row['absolute_bias'],terminal_p95=row['terminal_abs_error_p95']) for seed,row in zip(seeds,primary_values)]),indent=2))

if __name__=='__main__':main()
