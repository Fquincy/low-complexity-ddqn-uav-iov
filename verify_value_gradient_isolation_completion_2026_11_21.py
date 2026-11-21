"""Independent completion/archive rehash and saved-episode outcome recomputation."""
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation/value_gradient_isolation_comparison_completion_2026-11-21_01'
BINDING=ROOT/'VALUE_GRADIENT_ISOLATION_COMPARISON_LAUNCH_BINDING_2026-11-21.json'
SPEC=ROOT/'VALUE_GRADIENT_ISOLATION_COMPARISON_PROTOCOL_2026-11-20.json'
SPEC_SHA='2a3a4cd58f3f04eabf1d54958e058af7294fbb73a2f41000c3242999bcc4f683'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def close(a,b):return np.allclose(a,b,rtol=1e-7,atol=1e-8)
def means(rows):
    return dict(return_value=float(np.mean([v['return_value'] for v in rows])),
                deadline=float(np.mean([v['summary']['deadline_success_rate'] for v in rows])))


def main():
    if OUT.exists():raise RuntimeError('Preserve previous completion verification')
    binding=read(BINDING);out=ROOT/binding['output'];run=out/'run';archive=Path(binding['archive'])
    spec=read(SPEC);assert sha(SPEC)==SPEC_SHA
    for name,digest in binding['sha256'].items():assert sha(ROOT/name)==digest,name
    auth=read(out/'launch_authorization.json')
    assert auth['authorized'] and auth['binding']==BINDING.name and auth['binding_sha256']==sha(BINDING)
    assert sha(out/'launch_binding.json')==sha(BINDING)
    complete=read(out/'completion.json');audit=read(out/'saved_artifact_audit.json')
    assert complete['status']=='DEVELOPMENT_AUDITED_COMPLETE' and complete['audit_passed'] and complete['research_launch_authorized'] and audit['passed']
    for key,want in dict(simulation_slots_recorded=67120,historical_bc_updates=4800,
        historical_ddqn_updates=44928,retention_batches=22464,ridge_solves=15,final_checkpoints=27,
        deployment_weight_files=27,audit_gradient_evaluations=18,measured_action_calls=10752,warmup_action_calls=2688).items():assert audit[key]==want,key
    assert len(audit['first_update_reconstructions'])==9
    child=read(ROOT/binding['monitoring_output']/'child_result.json');assert child['child_exit_code']==0 and child['child_exited_successfully']
    plan=read(run/'runtime_plan.json');assert plan['mode']=='development' and plan['seeds']==spec['seeds'] and plan['fits']==3
    freeze=read(run/'PRE_RUN_FREEZE.json');assert freeze['protocol_sha256']==SPEC_SHA
    for name,digest in freeze['source_sha256'].items():
        assert sha(ROOT/name)==sha(run/'source_snapshot'/name)==digest,name
    manifest=read(run/'manifest.json');assert sha(run/'manifest.json')==complete['manifest_sha256']
    actual={p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file() and p!=run/'manifest.json'}
    assert actual==set(manifest)
    for name,digest in manifest.items():assert sha(run/name)==digest,name
    ar=read(out/'archive_verification.json');assert ar['passed'] and Path(ar['archive']).resolve()==archive.resolve()
    inventory=ar['sha256'];actual={p.relative_to(archive).as_posix() for p in archive.rglob('*') if p.is_file()}
    expected={'run/'+name for name in manifest}|{'run/manifest.json','completion.json','saved_artifact_audit.json',
        'launch_authorization.json','launch_binding.json','audit_phase_timing.json','phase_timings.json'}
    assert set(inventory)==expected and actual==expected|{'archive_verification.json'}
    for name,digest in inventory.items():
        assert sha(archive/name)==digest,name
        if name.startswith('run/') and name!='run/manifest.json':assert manifest[name[4:]]==digest,name
        else:assert sha(out/name)==digest,name
    assert sha(out/'archive_verification.json')==sha(archive/'archive_verification.json')
    assert not (out/'failure.txt').exists()
    assert len(list(run.glob('*/fit_*/final.pt')))==27
    assert len(list(run.glob('*/fit_*/deployment_weights.pt')))==27
    assert len(list(run.rglob('*.pt')))==54
    print('Source, research artifact and independent archive hashes verified.',flush=True)
    analysis=read(run/'analysis.json');arms=[a['id'] for a in spec['arms']];P,R,A,B,C,D,E,F,G=arms
    teacher_rows=[read(run/'evaluation/teacher'/f'episode_{i:03d}.json') for i in range(16)]
    teacher=means(teacher_rows);policies={};values={};task_outcomes={};handoffs={};costs={}
    for fit in range(3):
        for arm in arms:
            key=f'{arm}/{fit}';base=run/'evaluation'/key
            rows=[read(base/f'episode_{i:03d}.json') for i in range(16)]
            for i,row in enumerate(rows):
                assert row['seed']==spec['seeds']['evaluation_seed_block']['start']+i
                tasks=row['records'];assert len(tasks)==1200
                assert sum(bool(t['on_time']) for t in tasks)==row['summary']['tasks_on_time']
                assert close(row['summary']['deadline_success_rate'],sum(bool(t['on_time']) for t in tasks)/1200)
            policies[key]=means(rows);errors=[];gs=[];qs=[]
            for i,row in enumerate(rows):
                with np.load(base/f'episode_{i:03d}.npz') as saved:
                    assert close(float(saved['rewards'].sum(dtype=np.float64)),row['return_value'])
                    # This re-evaluates metrics from saved Q; completed full audit
                    # already reproduced Q/actions from final checkpoints.
                    g=np.cumsum(saved['rewards'][::-1],dtype=np.float64)[::-1]
                    errors.extend(saved['q']-g);gs.extend(g);qs.extend(saved['q'])
            values[key]=dict(bias=float(np.mean(errors)),nrmse=float(np.sqrt(np.mean(np.square(errors)))/max(np.sqrt(np.mean(np.square(gs))),1e-12)))
            assert close(list(values[key].values()),list(analysis['value_metrics'][key].values()))
            tasks=[t for row in rows for t in row['records']]
            task_outcomes[key]=dict(total=19200,misses=sum(not t['on_time'] for t in tasks),
                unfinished=sum(t['unfinished_at_horizon'] for t in tasks),
                destinations=np.bincount([t['destination'] for t in tasks],minlength=5).tolist(),
                misses_by_destination=[sum(t['destination']==d and not t['on_time'] for t in tasks) for d in range(5)],
                positive_Q_fraction=float(np.mean(np.array(qs)>0)),nearzero_Q_fraction=float(np.mean(np.abs(qs)<1e-6)))
            folder=run/arm/f'fit_{fit}';counts=read(folder/'counts.json')
            costs[key]=dict(parameter_bytes=counts['parameter_bytes'],buffer_bytes=counts['buffer_bytes'],
                checkpoint_bytes=(folder/'final.pt').stat().st_size,deployment_file_bytes=(folder/'deployment_weights.pt').stat().st_size,
                bc_wall_ns=counts['bc_wall_ns'],calibration_wall_ns=counts['calibration_wall_ns'],online_wall_ns=counts['online_wall_ns'],
                median_action_ms=audit['timing'][key]['median_ns']/1e6,p95_action_ms=audit['timing'][key]['p95_ns']/1e6)
            if counts['ridge_solves']:
                cal=read(folder/'calibration.json')
                residuals={part:{stage:entry['rmse'] for stage,entry in data.items()} for part,data in cal['residuals'].items()}
                assert cal['withheld_fit_worsened_vs_after_bc']==(residuals['withheld_returns']['after_value_fit']>residuals['withheld_returns']['after_bc'])
                handoffs[key]=dict(alpha=counts['alpha'],residual_rmse=residuals,
                    withheld_fit_worsened_vs_after_bc=cal['withheld_fit_worsened_vs_after_bc'],ridge_condition=cal['ridge']['regularized_condition'])
        print(f'Recomputed all nine saved-policy outcomes and value metrics for fit {fit}.',flush=True)
    per_fit={};failures=[]
    for fit in range(3):
        absolute={a:policies[f'{a}/{fit}'] for a in arms};primary=absolute[P]
        contrasts={arm:{m:primary[m]-v[m] for m in primary} for arm,v in absolute.items() if arm!=P}
        per_fit[str(fit)]=dict(absolute=absolute,teacher=teacher,contrasts=contrasts,
            minus_teacher={arm:{m:v[m]-teacher[m] for m in v} for arm,v in absolute.items()})
        assert per_fit[str(fit)]==analysis['per_fit'][str(fit)]
        for arm in [E,F,A,R]:
            for metric in ['return_value','deadline']:
                if contrasts[arm][metric]<0:failures.append(dict(fit=fit,comparator=arm,metric=metric,difference=contrasts[arm][metric]))
    mean_results={arm:dict(mean_return=statistics.mean(policies[f'{arm}/{fit}']['return_value'] for fit in range(3)),
        mean_deadline_pct=100*statistics.mean(policies[f'{arm}/{fit}']['deadline'] for fit in range(3))) for arm in arms}
    mean_results['teacher']=dict(mean_return=teacher['return_value'],mean_deadline_pct=100*teacher['deadline'])
    mean_differences={arm:{metric:statistics.mean(row['contrasts'][arm][metric] for row in per_fit.values()) for metric in ['return_value','deadline']} for arm in arms if arm!=P}
    policy=not failures and all(mean_differences[arm][metric]>0 for arm in [E,F,A,R] for metric in ['return_value','deadline'])
    value_checks={str(i):dict(nrmse_smaller=values[f'{P}/{i}']['nrmse']<values[f'{D}/{i}']['nrmse'],
        abs_bias_smaller=abs(values[f'{P}/{i}']['bias'])<abs(values[f'{D}/{i}']['bias'])) for i in range(3)}
    value=all(all(check.values()) for check in value_checks.values())
    assert (policy,value,policy and value)==(analysis['policy_screen'],analysis['value_screen'],analysis['promising'])
    assert audit['recorded_promising']==(policy and value)
    for fit,row in per_fit.items():
        for metric in ['return_value','deadline','nrmse','abs_bias']:
            def cell(arm):
                if metric in ['return_value','deadline']:return row['absolute'][arm][metric]
                return values[f'{arm}/{fit}']['nrmse'] if metric=='nrmse' else abs(values[f'{arm}/{fit}']['bias'])
            assert close(analysis['factorial_interaction'][fit][metric],cell(A)-cell(B)-cell(C)+cell(D))
    mean_failures=[dict(comparator=arm,metric=metric,difference=mean_differences[arm][metric]) for arm in [E,F,A,R] for metric in ['return_value','deadline'] if mean_differences[arm][metric]<=0]
    for arm in arms:
        for metric in ['return_value','deadline']:
            assert close(analysis['equal_fit_means'][arm][metric],np.mean([per_fit[str(i)]['absolute'][arm][metric] for i in range(3)]))
            if arm!=P:assert close(analysis['equal_fit_mean_contrasts'][arm][metric],mean_differences[arm][metric])
    for fit,row in per_fit.items():
        for label,left,right in [('P-A',P,A),('P-R',P,R),('R-A',R,A)]:
            want={m:row['absolute'][left][m]-row['absolute'][right][m] for m in ['return_value','deadline']}
            want.update(nrmse=values[f'{left}/{fit}']['nrmse']-values[f'{right}/{fit}']['nrmse'],
                abs_bias=abs(values[f'{left}/{fit}']['bias'])-abs(values[f'{right}/{fit}']['bias']))
            assert all(close(v,analysis['mechanism_contrasts'][fit][label][m]) for m,v in want.items())
        for arm in arms:
            if arm==P:continue
            assert close(analysis['value_contrasts'][fit][arm]['nrmse'],values[f'{P}/{fit}']['nrmse']-values[f'{arm}/{fit}']['nrmse'])
            assert close(analysis['value_contrasts'][fit][arm]['abs_bias'],abs(values[f'{P}/{fit}']['bias'])-abs(values[f'{arm}/{fit}']['bias']))
    ckfreeze=read(run/'CHECKPOINT_FREEZE.json')
    assert len(ckfreeze['checkpoints'])==len(ckfreeze['deployments'])==27
    for key,digest in ckfreeze['checkpoints'].items():
        arm,fit=key.split('/');assert sha(run/arm/f'fit_{fit}/final.pt')==digest
    for key,digest in ckfreeze['deployments'].items():
        arm,fit=key.split('/');assert sha(run/arm/f'fit_{fit}/deployment_weights.pt')==digest
    rules={P:'isolate_value',R:'rescale_shared_td'}
    log_counts=dict(updates=0,retention_batches=0,BC_updates=0,ridge_solves=0,initial_target_copies=0,scheduled_target_copies=0,online_slots=0)
    gradient_by_fit={};warnings={}
    for arm in arms:
        for fit in range(3):
            key=f'{arm}/{fit}';folder=run/arm/f'fit_{fit}';count=read(folder/'counts.json');logs=read(folder/'optimizer.json')
            assert all(v['gradient_rule']==rules.get(arm,'ordinary') for v in logs)
            assert len(logs)==count['updates']
            assert sum(v['retention_indices'] is not None for v in logs)==count['retention_batches']
            for field,cfield in [('updates','updates'),('retention_batches','retention_batches'),('BC_updates','bc_updates'),('ridge_solves','ridge_solves'),('initial_target_copies','initial_target_copy'),('scheduled_target_copies','target_copies'),('online_slots','slots')]:log_counts[field]+=count[cfield]
            gradient_by_fit[key]=dict(clip_fraction=float(np.mean([v['clipped'] for v in logs])) if logs else None,
                preclip_norm_quantiles=np.quantile([v['clip_norm'] for v in logs],[0,.5,.95,1]).tolist() if logs else None)
    assert log_counts==dict(updates=44928,retention_batches=22464,BC_updates=4800,ridge_solves=15,initial_target_copies=24,scheduled_target_copies=216,online_slots=48000)
    for fit in range(3):
        for label in ['bc_end_hashes','handoff_hashes']:
            eligible=[a['id'] for a in spec['arms'] if a['bc'] if label=='bc_end_hashes' or a['calibration']]
            assert len({ckfreeze[label][f'{arm}/{fit}'] for arm in eligible})==1
    stages=[v['stage'] for v in read(run/'events.json')]
    assert stages.count('checkpoints_frozen')==1 and stages.index('checkpoints_frozen')<stages.index('evaluation')
    assert len(read(run/'timing.json'))==10752 and len(read(run/'timing_warmups.json'))==2688
    record=dict(status='AUDIT_AND_ARCHIVE_REVERIFIED_POLICY_FAILED_VALUE_PASSED',
        protocol_sha256=SPEC_SHA,manifest_files_rehashed=len(manifest),archive_files_rehashed=len(inventory)+1,
        archive=str(archive),slots=67120,student_online_slots=48000,bc_updates=4800,td_updates=44928,
        retention_batches=22464,ridge_solves=15,final_checkpoints=27,deployment_weight_files=27,source_and_authorization_bindings_verified=True,
        policy_screen=policy,value_screen=value,promising=policy and value,per_fit_policy_failures=failures,
        equal_fit_policy_failures=mean_failures,value_checks=value_checks,mean_results=mean_results,mean_primary_minus_control=mean_differences,
        per_fit=per_fit,value_metrics=values,factorial_interaction=analysis['factorial_interaction'],
        tasks=task_outcomes,handoffs=handoffs,costs=costs,gradient_by_fit=gradient_by_fit,actual_log_counts=log_counts,
        mechanism_contrasts=analysis['mechanism_contrasts'],first_update_contexts_verified=len(audit['first_update_reconstructions']),
        audit_gradient_evaluations=audit['audit_gradient_evaluations'],process_result=child,teacher_timing=audit['timing']['teacher'],
        phase_timings=read(out/'phase_timings.json'),
        evidence_sha256={name:sha(out/name) for name in ['completion.json','saved_artifact_audit.json','archive_verification.json','run/analysis.json']},
        script_sha256=sha(Path(__file__)),new_simulator_steps=0,new_optimizer_updates=0,new_teacher_calls=0,
        limitation='All original and archived hashes verified; saved-episode outcome/metric/screens recomputed. Uses completed auditor for final policy and training arithmetic verification; no new gradient or checkpoint evaluation replay.',
        new_research_launch_authorized=False,final_phase_launched=False,word_thesis_changed=False)
    OUT.mkdir(parents=True)
    (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    with (OUT/'verification.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:record[k] for k in ['status','manifest_files_rehashed','archive_files_rehashed','policy_screen','value_screen','promising','mean_results','mean_primary_minus_control']},indent=2))


if __name__=='__main__':main()

