"""Read-only diagnosis of consumed development evidence; no simulator/training."""
import gzip,json,hashlib,sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'development_runs/deadline_screen_development_2026-11-14_01/run'
OUT=ROOT/'validation/deadline_screen_saved_data_diagnosis_2026-11-14_01'
ARMS=['screened_price24','screened_no_price24','screened_plain24_anchored']

def read(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else path.open()) as f:return json.load(f)
def write(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stats(x):
    x=np.asarray(x,np.float64)
    return dict(count=int(x.size),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),min=float(x.min()),max=float(x.max())) if x.size else dict(count=0)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    manifest=read(RUN/'run_manifest.json')['files'];summary=read(RUN/'analysis.json');audit=read(ROOT/'validation/deadline_screen_audit_correction_2026-11-14_01/development_saved_artifact_audit.json')
    outputs={};comparisons={};checked={};seeds=list(range(720000000,720000048))
    for fit in range(5):
        accum={arm:dict(destinations=Counter(),misses=Counter(),causes=Counter(),status=Counter(),moves=Counter(),delays={d:[] for d in range(5)},
            return_sum=[],deadline=[],reward_components=Counter(),Q=[],T=[],G=[],R=[],local_impossible=0,local_uncertain_boundary=0,
            failures_by_time=Counter(),failures_by_time_destination=Counter(),remote_queue=[],remote_uplink=[],remote_compute=[]) for arm in ARMS}
        paired={arm:Counter() for arm in ARMS[1:]}
        paired_by_dest={arm:Counter() for arm in ARMS[1:]}
        for seed in seeds:
            rows={}
            for arm in ARMS:
                path=RUN/arm/f'fit_{fit:02d}'/f'evaluation_{seed}.json.gz'
                for file in (path,path.with_suffix('').with_suffix('.npz')):
                    rel=file.relative_to(RUN).as_posix();digest=sha(file)
                    assert manifest[rel]==digest;checked[rel]=digest
                row=read(path);rows[arm]=row;a=accum[arm]
                assert row['seed']==seed and len(row['tasks'])==1200
                a['return_sum'].append(row['return_sum']);a['deadline'].append(row['deadline_success'])
                a['Q'].append(row['chosen_q']);a['T'].append(row['target_q']);a['G'].append(row['realized_return_to_go']);a['R'].append(row['slot_rewards'])
                assert sum(t['on_time'] for t in row['tasks'])==row['tasks_on_time']
                for slot in row['slots']:
                    for i,move in enumerate(slot['action'][30:]):a['moves'][(i,move)]+=1
                    for k,v in slot['reward_components'].items():a['reward_components'][k]+=v
                for task in row['tasks']:
                    d=task['destination'];failed=not task['on_time'];time=int(task['task_id']//30)
                    a['destinations'][d]+=1;a['misses'][d]+=int(failed)
                    if d==0:
                        impossible=task['required_cycles']/1.5e9>task['deadline_s']+1e-10
                        a['local_impossible']+=int(impossible)
                    if task['delay_s'] is not None:a['delays'][d].append(task['delay_s'])
                    if not failed:continue
                    a['status'][task['status']]+=1;a['failures_by_time'][time//10]+=1;a['failures_by_time_destination'][(time//10,d)]+=1
                    if d==0:
                        a['causes']['local_empty_queue_impossible' if impossible else 'local_not_empty_queue_impossible']+=1
                    else:
                        a['remote_queue'].append(task['queue_delay_s']) if task['queue_delay_s'] is not None else None
                        a['remote_uplink'].append(task['uplink_wait_s']) if task['uplink_wait_s'] is not None else None
                        a['remote_compute'].append(task['computation_delay_s']) if task['computation_delay_s'] is not None else None
                        up=task['upload_finish_s'];start=task['compute_start_s'];deadline=task['arrival_s']+task['deadline_s']
                        cause='unfinished_upload' if up is None else 'upload_finished_after_deadline' if up>deadline+1e-10 else 'cpu_not_started' if start is None else 'cpu_started_after_deadline' if start>deadline+1e-10 else 'cpu_completed_late'
                        a['causes'][cause]+=1
            primary=rows[ARMS[0]]
            for arm in ARMS[1:]:
                other=rows[arm];assert primary['workload_sha256']==other['workload_sha256']
                assert [t['task_id'] for t in primary['tasks']]==[t['task_id'] for t in other['tasks']]
                for x,y in zip(primary['tasks'],other['tasks']):
                    for field in ('input_bits','required_cycles','deadline_s','vehicle_id','arrival_s'):assert x[field]==y[field]
                    key='both_success' if x['on_time'] and y['on_time'] else 'primary_only_success' if x['on_time'] else 'comparator_only_success' if y['on_time'] else 'both_fail'
                    paired[arm][key]+=1
                    if key in ('primary_only_success','comparator_only_success'):paired_by_dest[arm][(key,x['destination'],y['destination'])]+=1
        for arm,a in accum.items():
            q,t,g,rewards=[np.array(a[k],np.float64) for k in ('Q','T','G','R')]
            next_online=np.concatenate((q[:,1:],np.zeros((48,1))),axis=1)
            next_target=np.concatenate((t[:,1:],np.zeros((48,1))),axis=1)
            residual=q-g;self_residual=q-rewards-next_online;target_residual=q-rewards-next_target
            assert abs(float(abs(residual.mean()))-summary['value_stability'][arm][fit]['absolute_bias'])<1e-12
            outputs[f'{arm}/{fit}']=dict(mean_return=float(np.mean(a['return_sum'])),deadline_fraction=float(np.mean(a['deadline'])),
                destination_counts=dict(a['destinations']),deadline_misses_by_destination=dict(a['misses']),
                destination_success_fraction={d:1-a['misses'][d]/a['destinations'][d] for d in a['destinations']},
                failed_tasks=sum(a['misses'].values()),failure_stage_counts=dict(a['causes']),failure_final_status=dict(a['status']),
                local_provably_impossible_actions=a['local_impossible'],movement_counts={str(k):v for k,v in a['moves'].items()},
                failures_by_time_bin=dict(a['failures_by_time']),failures_by_time_destination={str(k):v for k,v in a['failures_by_time_destination'].items()},
                completed_delays={d:stats(x) for d,x in a['delays'].items()},remote_failure_queue_delay=stats(a['remote_queue']),
                remote_failure_uplink_wait=stats(a['remote_uplink']),remote_failure_compute=stats(a['remote_compute']),
                mean_reward_components_per_slot={k:v/(48*40) for k,v in a['reward_components'].items()},
                value_by_time=[dict(start=i,end=i+9,Q=stats(q[:,i:i+10]),target_Q=stats(t[:,i:i+10]),G=stats(g[:,i:i+10]),
                    Q_minus_G=stats(residual[:,i:i+10]),own_trajectory_online_Bellman_residual=stats(self_residual[:,i:i+10]),
                    own_trajectory_target_Bellman_residual=stats(target_residual[:,i:i+10]),Q_per_remaining_step=stats(q[:,i:i+10]/np.arange(40-i,30-i,-1))) for i in range(0,40,10)],
                online_minus_target=stats(q-t),own_policy_residual_telescoping_max_error=float(abs(residual-np.cumsum(self_residual[:,::-1],axis=1)[:,::-1]).max()),
                optimizer_diagnostics=next(x for x in audit['optimizer_distributions'][arm] if x['fit']==fit))
        for arm in ARMS[1:]:
            counts=paired[arm];assert sum(counts.values())==48*1200
            difference=(counts['primary_only_success']-counts['comparator_only_success'])/(48*1200)
            expected=summary['comparisons'][arm]['per_fit'][fit]['deadline_difference'];assert abs(difference-expected)<1e-12
            comparisons[f'{arm}/{fit}']=dict(outcomes=dict(counts),deadline_difference=difference,
                discordant_destinations={str(k):v for k,v in paired_by_dest[arm].items()},causal_interpretation=False)
        print('Completed saved trace fit',fit,flush=True)
    write(OUT/'trace_diagnosis.json',dict(arms=ARMS,fits=5,evaluation_episodes=720,unique_tasks_across_policy_traces=864000,
        results=outputs,paired_task_outcomes=comparisons,new_training_slots=0,new_evaluation_slots=0,new_optimizer_updates=0,retiming=False,
        warning='Policies have different queues and UAV trajectories. Matched-task outcome differences are descriptive policy comparisons, not counterfactual destination effects. Own-policy Bellman residuals are trajectory identities, not sampled replay training losses.'))
    write(OUT/'trace_inputs_manifest.json',checked)
    print('Trace diagnosis complete.',flush=True)

if __name__=='__main__':main()
