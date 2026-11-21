"""Read-only paired diagnosis from independently audited saved summaries.

No simulator, model inference, optimizer, new timing or repeated full-file audit.
"""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'development_runs/horizon_loss_development_2026-11-14_01/comparison/run'
OUT=ROOT/'validation/horizon_loss_saved_pair_diagnosis_2026-11-15_01'
REPORT='HORIZON_LOSS_SAVED_PAIR_DIAGNOSIS_2026-11-15.txt'
PAIRS=[('normalized_screened_price24','screened_price24'),
       ('normalized_screened_no_price24','screened_no_price24'),
       ('normalized_screened_plain24_anchored','screened_plain24_anchored')]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    prior=ROOT/'validation/horizon_loss_development_results_2026-11-14_01'
    assert read(prior/'verification.json')['passed']
    prior_manifest=read(prior/'manifest.json')
    for n,h in prior_manifest.items():assert sha(prior/n)==h,n
    names=['analysis.json','saved_artifact_audit.json','remote_decision_diagnostics.json',
           'training_residuals_by_horizon.json','costs.json','verification.json']
    manifest=read(RUN/'run_manifest.json')['files']
    outer=read(RUN.parent/'workflow_manifest.json')['files']
    inputs={}
    for n in names:
        h=sha(RUN/n)
        expected=manifest[n] if n in manifest else outer['run/'+n]
        assert h==expected,n
        inputs[n]=h
    a=read(RUN/'analysis.json');audit=read(RUN/'saved_artifact_audit.json')
    remote=read(RUN/'remote_decision_diagnostics.json')['policies']
    td=read(RUN/'training_residuals_by_horizon.json')['fits']
    absolute=read(prior/'results_summary.json')['absolute_all_learners']
    p=read(ROOT/'HORIZON_LOSS_RESEARCH_PROTOCOL_2026-11-14.json')
    arms={}
    for name in [arm for pair in PAIRS for arm in pair]:
        fits=[]
        for i in range(5):
            d=remote[f'{name}/{i}'];cnt=np.array(d['task_counts_by_time_destination']);miss=np.array(d['misses_by_time_destination'])
            assert cnt.shape==miss.shape==(40,5) and int(cnt.sum())==57600
            stages={};bytime={}
            for key,value in d['failure_stages_by_time_destination'].items():
                t,dest,stage=key.split('/')
                stages[stage]=stages.get(stage,0)+value
                bucket=f'{int(t)//10}/{dest}/{stage}'
                bytime[bucket]=bytime.get(bucket,0)+value
            assert sum(stages.values())==int(miss.sum())
            assert np.isclose(1-miss.sum()/cnt.sum(),absolute[name][i]['deadline_fraction'])
            opt=next(x for x in audit['optimizer_distributions'][name] if x['fit']==i)
            horizons=td[f'{name}/{i}']
            assert sum(x['original']['count'] for x in horizons.values())==6977*128
            weighted=sum(x['original']['mean']*x['original']['count'] for x in horizons.values())/(6977*128)
            assert np.isclose(weighted,opt['TD_residuals']['mean'],rtol=1e-12,atol=1e-12)
            bins=[]
            for lo,hi in [(1,1),(2,10),(11,20),(21,30),(31,40)]:
                selected=[horizons[str(h)] for h in range(lo,hi+1)]
                count=sum(x['original']['count'] for x in selected)
                bins.append(dict(current_h_min=lo,current_h_max=hi,count=count,
                    mean_original_TD=sum(x['original']['mean']*x['original']['count'] for x in selected)/count,
                    mean_normalized_TD=sum(x['normalized']['mean']*x['normalized']['count'] for x in selected)/count))
            value=a['value_stability'][name][i]
            fits.append(dict(fit=i,**{k:v for k,v in absolute[name][i].items() if k!='fit'},
                destination_counts=cnt.sum(0).tolist(),misses_by_destination=miss.sum(0).tolist(),
                failures_by_lifecycle=stages,failure_stage_by_time_destination=bytime,
                misses_by_ten_slot_bin=[int(miss[t:t+10].sum()) for t in range(0,40,10)],
                assignment_fraction=(cnt.sum(0)/cnt.sum()).tolist(),
                remote_miss_delays=d['remote_miss_delays'],observed_cpu=d['observed_log_cpu_workload_remote'],
                observed_remote_queues=d['observed_log_remote_queue_length'],movement_counts=d['movement_counts'],
                movement_distance=d['movement_distance_m'],optimizer=opt,
                anchor_to_uniform_mean_loss_ratio=opt['anchor_loss']['mean']/opt['uniform_loss']['mean'],
                TD_horizon_bin_means=bins,
                value={k:value[k] for k in ('normalized_rmse','absolute_bias','terminal_abs_error_p95','positive_Q_fraction')},
                signed_Q_minus_G=value['all_slots']['Q_minus_G']['mean'],
                value_error_by_time=[dict(start=x['start'],end=x['end'],**x['distributions']['Q_minus_G']) for x in value['time_bins']]))
        arms[name]=fits
    effects=[]
    for new,old in PAIRS:
        contrast=next(x for x in a['auxiliary_contrasts'] if x['left']==new and x['right']==old)
        for i,x in enumerate(contrast['per_fit']):
            assert np.isclose(arms[new][i]['return_mean']-arms[old][i]['return_mean'],x['return_difference'])
            assert np.isclose(arms[new][i]['deadline_fraction']-arms[old][i]['deadline_fraction'],x['deadline_difference'])
        effects.append(dict(normalized=new,original=old,per_fit=contrast['per_fit'],
            mean_return_difference=float(np.mean([x['return_difference'] for x in contrast['per_fit']])),
            mean_deadline_points=float(100*np.mean([x['deadline_difference'] for x in contrast['per_fit']]))))
    # Exploratory characterization only: original controller was not the primary.
    original='screened_price24';delta=a['comparisons'][original]['per_fit']
    references={}
    for name in p['comparisons']['required_reference_arms']+p['comparisons']['strong_heuristics']:
        if name==original:continue
        rows=[]
        for i,x in enumerate(a['comparisons'][name]['per_fit']):
            rd=x['return_difference']-delta[i]['return_difference'];dd=x['deadline_difference']-delta[i]['deadline_difference']
            margin=(0,0) if name in p['comparisons']['strong_heuristics'] else (-2,-.01)
            rows.append(dict(return_difference=rd,deadline_points=100*dd,passes_same_policy_margin=rd>=margin[0] and dd>=margin[1]))
        references[name]=rows
    exploratory=dict(status='descriptive_control_assessment_not_primary_replacement_or_new_acceptance',
        original_controller=original,policy_references=references,
        all_policy_margins_same_as_primary=all(x['passes_same_policy_margin'] for rows in references.values() for x in rows),
        primary_value_limits_pass=[x['value']['normalized_rmse']<=.25 and x['value']['absolute_bias']<=2 and x['value']['terminal_abs_error_p95']<=2 for x in arms[original]],
        unchanged_price_contrasts=[dict(fit=i,return_difference=arms[original][i]['return_mean']-arms['screened_no_price24'][i]['return_mean'],deadline_points=100*(arms[original][i]['deadline_fraction']-arms['screened_no_price24'][i]['deadline_fraction'])) for i in range(5)])
    write(OUT/'diagnosis.json',dict(input_sha256=inputs,paired_effects=effects,arms=arms,
        original_control_exploratory_assessment=exploratory,
        causal_training_mechanism_proven=False,new_simulator_steps=0,new_optimizer_updates=0,
        fresh_model_inference=False,retiming=False,word_thesis_changed=False))
    lines=['HORIZON-LOSS: SAVED-PAIR DIAGNOSIS','2026-11-15','',
        'FINDING','The normalization correction failed across all three architecture pairs.',
        'It greatly reduced gradient clipping but did not produce better policies.',
        'The price primary mostly overloaded a single RSU. Normalized no-price and',
        'plain variants used UAVs heavily and suffered unfinished/late uploads.',
        'These are saved outcome associations, not proof of one optimizer cause.','',
        'EVIDENCE AND LIMITS','Read-only diagnosis of the completed fresh 560,000-slot comparison.',
        'All five fits for each of six arms retained; no fitting/retiming/inference.',
        'Uses independently audited optimizer distributions, horizon residual and',
        'remote-decision reports, and previously recomputed absolute evaluation means.',
        'Input hashes match the saved run/workflow manifests; prior completion review',
        'and its evidence manifest verified. The full multi-gigabyte raw logs and',
        'full run audit were not reread/repeated. Optimizer distributions aggregate',
        'all6977 updates: they do not establish the temporal onset of deterioration.',
        'TD summaries pool logged training predictions, not final-network residuals.',
        'Means by horizon are sample-count weighted; medians/quantiles are not averaged.','',
        'PAIRED CORRECTION EFFECTS: NORMALIZED MINUS ORIGINAL',
        'architecture | mean return difference | mean deadline difference (points)']
    for x in effects:lines.append(f"{x['normalized']} | {x['mean_return_difference']:+.6f} | {x['mean_deadline_points']:+.6f}")
    lines+=['All three pairs lose both metrics in every initialization; exact per-fit',
        'differences and checks are saved in diagnosis.json.','',
        'LEARNING AND VALUE SUMMARY','arm | fit | deadline% | median gradient | clip fraction | mean anchor/uniform loss | Q-G bias | RMSE']
    for name,rows in arms.items():
        for x in rows:lines.append(f"{name} | {x['fit']} | {100*x['deadline_fraction']:.4f} | {x['optimizer']['grad_norm_before_clip']['median']:.4f} | {x['optimizer']['clipping_fraction']:.4f} | {x['anchor_to_uniform_mean_loss_ratio']:.4f} | {x['signed_Q_minus_G']:.4f} | {x['value']['normalized_rmse']:.4f}")
    lines+=['', 'INTERPRETATION','Original arms clipped approximately every update. Normalized arms clipped',
        'less frequently, yet all three paired policies deteriorated. Therefore the',
        'observed reduction in clipping is not evidence of learning success, and',
        'near-universal clipping in the originals was not sufficient to predict worse',
        'policy quality. This does not prove clipping is harmless or causal.',
        'Loss scales differ by design. The anchor/uniform scalar-loss ratio is not',
        'a gradient-contribution ratio; the saved summaries do not isolate anchoring,',
        'Huber-regime changes and horizon weighting as separate causes. Raw normalized',
        'and original losses cannot be compared as equal-unit objectives.',
        'All five original-price fits still fail RMSE<=.25, so better original policy',
        'performance does not establish the declared value-consistency contribution.',
        'Prediction accuracy and policy action ranking remain distinct questions.','',
        'DESTINATION AND STAGE ACCOUNTING',
        'Destination order:local,UAV1,UAV2,RSU1,RSU2. Counts cover48 episodes/57600tasks.']
    for name,rows in arms.items():
        for x in rows:
            top=sorted(x['failures_by_lifecycle'].items(),key=lambda kv:-kv[1])[:2]
            lines.append(f"{name} fit{x['fit']}: assigned{x['destination_counts']}; missed{x['misses_by_destination']}; dominant{top}")
    lines+=['', 'PRIMARY POLICY DETERIORATION',
        'Normalized price fit0 assigned41784/57600 tasks toRSU2 and missed17420',
        'there. Its paired original assigned23858 toRSU2 and missed2018 there.',
        'Fit1 also concentrated atRSU2; fits2/3 atRSU1; fit4 atRSU2. Across the',
        'normalized primary the dominant failure label is CPU started after deadline.',
        'The lifecycle timing supports late CPU admission; counts alone do not prove',
        'which learned estimate, queue history or motion choice caused concentration.',
        'Original and normalized policies have different subsequent queue histories,',
        'so task comparisons are not isolated counterfactual single-action tests.',
        'A few local misses remain in some fits. Do not claim the conservative screen',
        'guarantees zero local or remote misses. This diagnosis does not introduce a',
        'new feasibility mask or access to private queue/future-arrival information.','',
        'NORMALIZED NO-PRICE AND PLAIN CONTROLS',
        'High UAV allocations coincide with almost all UAV-assigned tasks missing',
        'deadlines in several fits. Unfinished upload and upload finished after',
        'deadline dominate. The primary therefore passes relative price/architecture',
        'contrasts partly in the context of very poor normalized comparators. That',
        'is valid relative development evidence, not proof of overall claim success.','',
        'EXPLORATORY ORIGINAL-LOSS CONTROL ASSESSMENT',
        f"All retained primary-style policy reference margins pass for the original control: {exploratory['all_policy_margins_same_as_primary']}.",
        'This is exploratory re-expression of saved paired differences. The control',
        'is not promoted to primary, the frozen experiment remains failed, and',
        'none of these consumed seeds become independent confirmation data.']
    for name,rows in references.items():
        failed=[i for i,x in enumerate(rows) if not x['passes_same_policy_margin']]
        lines.append(f'Original vs{name}: fits failing same policy margin{failed}.')
    lines+=['Original primary value limits pass by fit: '+str(exploratory['primary_value_limits_pass']),
        'Original vs original no-price, per-fit return/deadline points:']
    for x in exploratory['unchanged_price_contrasts']:lines.append(str(x))
    lines+=['', 'DECISION AND NEXT BOUNDARY',
        'Close this normalization correction as an unsuccessful development design',
        'under its frozen exposure and settings. Preserve its useful negative evidence.',
        'Do not automatically add another controller or adjust loss weights/rewards.',
        'The original-loss screened controller is the more plausible policy candidate',
        'in this paired run, but still lacks the complete value/mechanism evidence.',
        'The next decision is a contribution/claim brief for supervisor discussion:',
        'separate established size/latency and screening evidence from unproven price',
        'superiority, calibration and novelty. Decide whether a narrowly specified',
        'controller improvement remains justified before any new prospective trial.',
        'A different scientific claim needs its own prospective criteria and fresh',
        'development/final confirmation, not retrospective relaxation of this run.',
        'No thesis revision, new training or final launch is authorized by this report.',
        'Final seeds remain protected; security/trust deferred; cloud/charging excluded.',
        '',f'Evidence:{OUT.relative_to(ROOT)}']
    (ROOT/REPORT).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copy2(ROOT/REPORT,OUT/REPORT);shutil.copy2(__file__,OUT/Path(__file__).name)
    write(OUT/'verification.json',dict(passed=True,input_hashes_verified=True,
        six_arms_five_fits_retained=True,task_and_failure_counts_reconciled=True,
        horizon_residual_counts_and_means_reconciled=True,paired_policy_differences_recomputed=True,
        temporal_optimizer_onset_established=False,causal_optimizer_mechanism_proven=False,
        new_simulator_steps=0,new_optimizer_updates=0,retiming=False,word_thesis_changed=False))
    status_path=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json';shutil.copy2(status_path,OUT/'PROJECT_STATUS_BEFORE_DIAGNOSIS.json')
    status=read(status_path);status['current_workstream']='horizon_loss_saved_pair_diagnosis_complete_correction_unsuccessful'
    status['next']='Prepare contribution/claim decision brief for supervisor; separate established efficiency/screening from unproven mechanism/calibration. No automatic redesign, training or final launch.'
    status['next_engineering_step']=status['next']
    status['horizon_loss_saved_pair_diagnosis']=dict(report=REPORT,evidence=str(OUT.relative_to(ROOT)),
        arms=6,fits=30,verification_passed=True,correction_helped_policy=False,
        causal_optimizer_mechanism_proven=False,new_simulator_steps=0,new_optimizer_updates=0,
        thesis_changed=False,final_phase_launched=False)
    write(status_path,status)
    write(OUT/'manifest.json',{x.relative_to(OUT).as_posix():sha(x) for x in OUT.rglob('*') if x.is_file()})
    print(json.dumps(dict(report=REPORT,verified=True,effects=effects,
        original_policy_margins_all_pass=exploratory['all_policy_margins_same_as_primary']),indent=2))

if __name__=='__main__':main()
