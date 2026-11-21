"""Read-only completion verification and frozen-criteria review. No training imports."""
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'development_runs/horizon_loss_development_2026-11-14_01'
SAVED=RUN/'comparison/run'
OUT=ROOT/'validation/horizon_loss_development_results_2026-11-14_01'
REPORT='HORIZON_LOSS_DEVELOPMENT_RESULTS_REVIEW_2026-11-14.txt'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    manifests=[(RUN,RUN/'completion_manifest.json'),(RUN/'comparison',RUN/'comparison/workflow_manifest.json'),
               (SAVED,SAVED/'run_manifest.json'),(RUN/'comparison/deployment_storage',RUN/'comparison/deployment_storage/manifest.json')]
    checks=[]
    for base,path in manifests:
        m=read(path)['files'];print(f'Verifying {len(m)} entries: {path.name}',flush=True)
        for i,(name,h) in enumerate(m.items(),1):
            target=(base/name).resolve();assert target.is_relative_to(base.resolve())
            assert sha(target)==h,name
            if i%10000==0:print(f'  {i} hashes checked',flush=True)
        checks.append(dict(manifest=str(path.relative_to(ROOT)),entries=len(m),sha256=sha(path)))
    workflow=read(RUN/'comparison/workflow_verification.json');assert workflow['artifact_audit_passed'] and workflow['deployment_storage_audit']['passed']
    startup=read(RUN/'startup_verification.json');assert startup['passed'] and startup['models']==70
    completion=read(RUN/'completion_manifest.json')
    assert completion['comparison_workflow_manifest_sha256']==sha(RUN/'comparison/workflow_manifest.json')
    for name,h in completion['sources'].items():assert sha(ROOT/name)==h,name
    pre=read(SAVED/'PRE_RUN_FREEZE.json')
    for name,h in pre['source_sha256'].items():assert sha(ROOT/name)==sha(SAVED/'source_snapshot'/name)==h,name
    p=read(ROOT/'HORIZON_LOSS_RESEARCH_PROTOCOL_2026-11-14.json')
    assert pre['protocol_sha256']==sha(ROOT/'HORIZON_LOSS_RESEARCH_PROTOCOL_2026-11-14.json')
    a=read(SAVED/'analysis.json');v=read(SAVED/'verification.json');audit=read(SAVED/'saved_artifact_audit.json')
    assert [v[k] for k in ('training_slots','checkpoints','optimizer_updates','target_copies','evaluation_slots','raw_timing_rows')]==[560000,70,488390,2380,140160,44800]
    assert (audit['episodes'],audit['task_records'])==(17504,21004800)
    assert a['screens_applied'] and a['quality_evidence'] and not a['all_scientific_screens_passed']
    primary=p['primary'];absolute={}
    for arm in p['learned_arms']:
        fits=[]
        for fit in range(5):
            rewards=[];deadlines=[]
            for seed in range(p['development']['evaluation_seeds']['start'],p['development']['evaluation_seeds']['start']+48):
                with gzip.open(SAVED/arm['id']/f'fit_{fit:02d}'/f'evaluation_{seed}.json.gz','rt') as stream:row=json.load(stream)
                rewards.append(row['return_sum']);deadlines.append(row['deadline_success'])
            fits.append(dict(fit=fit,seed=p['development']['model_seeds'][fit],return_mean=float(np.mean(rewards)),deadline_fraction=float(np.mean(deadlines))))
        absolute[arm['id']]=fits
    # Independent gate arithmetic from saved paired effects, without thresholds changed.
    failed_refs={}
    for name in p['comparisons']['required_reference_arms']:
        failed_refs[name]=[i for i,x in enumerate(a['comparisons'][name]['per_fit']) if x['return_difference'] < -2 or x['deadline_difference'] < -.01]
    mechanisms={}
    for m in p['comparisons']['mechanisms']:
        c=a['comparisons'][m['right']]
        passed=(all(x['return_difference']>=m['each_fit_return_difference_min'] and x['deadline_difference']>=m['each_fit_deadline_fraction_difference_min'] for x in c['per_fit'])
                and c['mean_return_difference']>=m['equal_fit_mean_return_difference_min']
                and c['mean_deadline_difference']>=m['equal_fit_mean_deadline_fraction_difference_min'])
        assert passed==a['computed_screens'][m['id']+'_mechanism']
        mechanisms[m['id']]=dict(passed=passed,comparison=m['right'],**c)
    remote=read(SAVED/'remote_decision_diagnostics.json')['policies']
    remote_primary={}
    for fit in range(5):
        d=remote[f'{primary}/{fit}'];stages={}
        for key,count in d['failure_stages_by_time_destination'].items():
            category=key.split('/')[-1];stages[category]=stages.get(category,0)+count
        counts=np.asarray(d['task_counts_by_time_destination']);misses=np.asarray(d['misses_by_time_destination'])
        remote_primary[str(fit)]=dict(destination_counts=counts.sum(0).tolist(),misses_by_destination=misses.sum(0).tolist(),failure_stages=stages)
    costs=read(SAVED/'costs.json')[primary]
    summary=dict(outcome='completed_audited_combined_scientific_screen_failed',manifest_checks=checks,
        verification={k:value for k,value in v.items() if k!='counts'},
        audit={k:value for k,value in audit.items() if k!='optimizer_distributions'},
        computed_screens=a['computed_screens'],absolute_all_learners=absolute,
        comparisons=a['comparisons'],mechanisms=mechanisms,failed_reference_fits=failed_refs,
        primary_value=[{k:x[k] for k in ('normalized_rmse','absolute_bias','terminal_abs_error_p95','near_zero_large_return_fraction','positive_Q_fraction')} for x in a['value_stability'][primary]],
        primary_remote_failures=remote_primary,primary_cost=costs,
        new_training_slots=0,new_evaluation_slots=0,new_optimizer_updates=0,retiming=False,
        final_phase_launched=False,word_thesis_changed=False)
    write(OUT/'results_summary.json',summary)
    lines=['HORIZON-LOSS DEVELOPMENT: AUDITED RESULTS REVIEW','2026-11-14','',
        'DECISION','The full development workflow completed. Artifact/storage/startup audits passed.',
        'The combined predeclared scientific decision FAILED. Do not launch final',
        'confirmation or revise the Word thesis to claim success. Preserve all fits.',
        'This is a completed negative test of the prospective loss correction, not',
        'a training crash and not proof that efficient DDQN is impossible.','',
        'COMPLETION AND INTEGRITY',
        '560,000 training slots;70 final checkpoints;488,390 optimizer updates;',
        '2,380 target copies;140,160 evaluation slots;44,800 timing readings.',
        'Independent audit:17,504 episodes;21,004,800 task records;8,960 runtime',
        'actions. Diagnostic reports:70 training fits and73 policies. Deployment',
        'storage and separate startup checks passed for all70 checkpoints.',
        'This review rehashed all entries in the completion, paired-workflow, run',
        'and deployment-storage manifests and verified the saved/current source',
        'and protocol hashes. It did not repeat the auditor, training or timings.','',
        'FROZEN SCIENTIFIC CRITERIA']
    lines += [f'{k}: {"PASS" if value else "FAIL"}' for k,value in a['computed_screens'].items()]
    lines += ['', 'PRIMARY ABSOLUTE PERFORMANCE','fit | model seed | mean return | deadline success %']
    lines += [f"{x['fit']} | {x['seed']} | {x['return_mean']:.6f} | {100*x['deadline_fraction']:.6f}" for x in absolute[primary]]
    lines += ['', 'DIRECT LOSS-CORRECTION TEST: PRIMARY MINUS ORIGINAL-LOSS PRIMARY',
        'fit | return difference | deadline difference (percentage points)']
    correction=a['comparisons']['screened_price24']
    lines += [f"{i} | {x['return_difference']:+.6f} | {100*x['deadline_difference']:+.6f}" for i,x in enumerate(correction['per_fit'])]
    lines += [f"Equal-fit mean: {correction['mean_return_difference']:+.6f} return; {100*correction['mean_deadline_difference']:+.6f} deadline points.",
        'Both metrics deteriorated in every fit. The proposed normalization is not',
        'supported as an improvement under this frozen budget/setup. This comparison',
        'uses fresh paired original-loss controls, not results from older seeds.','',
        'ALL PRIMARY COMPARISONS: FIVE-FIT MEAN',
        'comparator | return difference | deadline difference (percentage points)']
    lines += [f"{name} | {x['mean_return_difference']:+.6f} | {100*x['mean_deadline_difference']:+.6f}" for name,x in a['comparisons'].items()]
    lines += ['', 'INTERPRETING THE TWO POSITIVE MECHANISM SCREENS',
        'Price and same-width architecture pass against their NORMALIZED-loss',
        'comparators. Those arms performed much worse; the exact loss-correction',
        'effects within all three architectures are listed below. These positive',
        'relative contrasts do not establish correction benefit, overall superiority',
        'or independent final novelty. Do not substitute a successful comparator',
        'for the predeclared primary or select only favorable initializations.']
    for x in a['auxiliary_contrasts'][-3:]:
        lines.append(f"{x['left']} minus {x['right']}: mean return {np.mean([v['return_difference'] for v in x['per_fit']]):+.6f}; deadline {100*np.mean([v['deadline_difference'] for v in x['per_fit']]):+.6f} points.")
    lines += ['', 'VALUE CONSISTENCY: ORIGINAL UNITS','fit | normalized RMSE (max.25) | absolute bias (max2) | terminal p95 (max2)']
    lines += [f"{i} | {x['normalized_rmse']:.6f} | {x['absolute_bias']:.6f} | {x['terminal_abs_error_p95']:.6f}" for i,x in enumerate(summary['primary_value'])]
    lines += ['All five primary fits fail global RMSE and bias; terminal p95, Q sign',
        'and near-zero checks pass. Lower training loss is not a calibration result.',
        'Realized own-policy return consistency is not exact optimal-Q ground truth.','',
        'COST','Online primary parameters68,264bytes +20buffer bytes versus194,944',
        'parameter bytes for unmasked compact64:64.98% fewer parameter bytes.',
        f"Median action latency range {min(x['median_ms'] for x in costs):.6f}..{max(x['median_ms'] for x in costs):.6f}ms; p95 range {min(x['p95_ms'] for x in costs):.6f}..{max(x['p95_ms'] for x in costs):.6f}ms.",
        'All five frozen cost screens pass. Same host/procedure/backend; not a claim',
        'about total device memory or electrical energy. Startup is separately',
        'reported and includes full training-checkpoint loading and existing caches.','',
        'SAVED PRIMARY DEADLINE FAILURE COUNTS',
        'Destinations ordered local,UAV1,UAV2,RSU1,RSU2; lifecycle labels descriptive.']
    for fit,x in remote_primary.items():lines.append(f"fit{fit}: misses {x['misses_by_destination']}; stages {x['failure_stages']}")
    lines += ['', 'NEXT WORK',
        'Do not automatically start another design or training run. The correction',
        'hypothesis failed its direct policy test. Preserve normalized and original',
        'arms, failed fits, thresholds and reserved final seeds.',
        'Use the saved residual-by-horizon, gradient/anchor histories and decision',
        'reports to compare the three normalized/original pairs. Establish where',
        'policy deterioration and value error occur and whether the original-loss',
        'screened price24 remains a plausible development candidate. Any such',
        'assessment is exploratory; it does not promote that control to primary.',
        'Then make an explicit claim/design decision with the supervisor rather',
        'than assume normalization needs another automatic tuning run. Literature',
        'novelty and independently justified acceptance remain separate questions.',
        'A later change requires prospective declaration and fresh development',
        'evidence. Final confirmation remains disabled. Word thesis unchanged.',
        'Cloud/charging excluded; security/trust deferred.','',
        f'Evidence: {OUT.relative_to(ROOT)}',f'Run: {RUN.relative_to(ROOT)}']
    (ROOT/REPORT).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copy2(ROOT/REPORT,OUT/REPORT);shutil.copy2(__file__,OUT/Path(__file__).name)
    status_path=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json';shutil.copy2(status_path,OUT/'PROJECT_STATUS_BEFORE_RESULTS_REVIEW.json')
    status=read(status_path);status['current_workstream']='horizon_loss_development_complete_audited_scientific_screen_failed'
    status['next']='Saved-data diagnosis of normalized/original pairs, then explicit claim/design decision; no automatic retraining or final launch'
    status['next_engineering_step']=status['next']
    status['horizon_loss_development_results']=dict(review=REPORT,evidence=str(OUT.relative_to(ROOT)),
        training_complete=True,evaluation_complete=True,artifact_and_storage_audits_passed=True,
        startup_complete=True,scientific_screens=a['computed_screens'],all_scientific_screens_passed=False,
        mean_correction_return_difference=correction['mean_return_difference'],
        mean_correction_deadline_difference_points=100*correction['mean_deadline_difference'],
        final_phase_launched=False,word_thesis_changed=False)
    write(status_path,status)
    write(OUT/'verification.json',dict(passed=True,manifest_checks=checks,
        protocol_and_source_hashes_verified=True,mechanism_decisions_recomputed=True,
        absolute_means_recomputed_all_learners=True,combined_scientific_screen_passed=False,
        new_training_slots=0,new_evaluation_slots=0,retiming=False,word_thesis_changed=False))
    write(OUT/'manifest.json',{x.relative_to(OUT).as_posix():sha(x) for x in OUT.rglob('*') if x.is_file()})
    print(json.dumps(dict(review=REPORT,integrity_verified=True,scientific_pass=False,
        correction_return=correction['mean_return_difference'],correction_deadline_points=100*correction['mean_deadline_difference'])),flush=True)

if __name__=='__main__':main()
