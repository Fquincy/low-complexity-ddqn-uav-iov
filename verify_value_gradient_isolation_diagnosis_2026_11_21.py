"""Independent saved diagnosis export/arithmetic checks; no model imports."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation/value_gradient_isolation_saved_trace_diagnosis_2026-11-21_01'
RUN=ROOT/'development_runs/value_gradient_isolation_comparison_2026-11-20_01/run'
P='value_gradient_isolated_ddqn';R='dueling_shared_td_rescale_ddqn';A='calibrated_retained_ddqn'
E='calibrated_bc_only';F='old_softplus_bc_ddqn'
CORE=[P,R,A,E,F]


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def check_stats(data,record):
    x=np.asarray(data,dtype=np.float64).ravel();assert len(x)==record['n']
    if len(x):
        assert np.isfinite(x).all()
        assert np.isclose(x.mean(),record['mean'],rtol=1e-7,atol=1e-8)
        assert np.allclose(np.quantile(x,[0,.05,.5,.95,1]),record['quantiles'],rtol=1e-7,atol=1e-8)


def main():
    dest=OUT/'independent_verification.json'
    if dest.exists():raise RuntimeError('Preserve prior independent diagnosis verification')
    receipt=read(OUT/'verification.json');v=read(OUT/'diagnosis.json')
    for name,digest in receipt['artifacts_sha256'].items():assert sha(OUT/name)==digest,name
    assert sha(ROOT/'diagnose_value_gradient_isolation_saved_traces_2026_11_21.py')==receipt['script_sha256']
    for name,digest in v['consumed_sha256'].items():assert sha(ROOT/name)==digest,name
    population_checks=0;pair_checks=0;queue_tasks=0
    with np.load(OUT/'fixed_state_predictions.npz',allow_pickle=False) as raw:
        for pop,entries in v['fixed_state'].items():
            population,fit=pop.rsplit('/',1);reference=f'fit{fit}_{population}_{E}'
            ea=raw[reference+'_actions'];eq=raw[reference+'_q'];ev=raw[reference+'_value']
            for arm,entry in entries.items():
                prefix=f'fit{fit}_{population}_{arm}';actions=raw[prefix+'_actions'];q=raw[prefix+'_q'];value=raw[prefix+'_value']
                qe=raw[prefix+'_q_on_E'];change=actions!=ea
                assert float(change[:,:30].mean())==entry['placement_change_vs_E']
                assert float(change[:,30:].mean())==entry['movement_change_vs_E']
                assert float(change.any(1).mean())==entry['joint_change_vs_E']
                transition=np.zeros((5,5),dtype=np.int64)
                np.add.at(transition,(ea[:,:30].ravel(),actions[:,:30].ravel()),1)
                assert transition.tolist()==entry['E_to_model_destinations']
                for name,data in [('q',q),('value',value),('fixed_E_action_Q_difference',qe-eq),('value_difference',value-ev),
                    ('greedy_gain_over_E_action',q-qe),('placement_margin',raw[prefix+'_margins'][:,:30]),
                    ('symmetric_value_readout_contribution',raw[prefix+'_readout']),('symmetric_value_feature_contribution',raw[prefix+'_feature'])]:
                    check_stats(data,entry[name])
                if arm!=F:
                    branch=(qe-value)-(eq-ev);check_stats(branch,entry['centered_branch_difference'])
                    assert np.allclose(qe-eq,(value-ev)+branch,rtol=3e-5,atol=3e-5)
                assert np.allclose(raw[prefix+'_readout']+raw[prefix+'_feature'],raw[prefix+'_explicit_value_delta'],rtol=1e-9,atol=1e-9)
                check_stats(raw[prefix+'_explicit_value_delta']-(value-ev),entry['explicit64_minus_FP32_value_difference'])
                population_checks+=1
        spec=read(ROOT/'VALUE_GRADIENT_ISOLATION_COMPARISON_PROTOCOL_2026-11-20.json');cfg=spec['simulator_defaults']
        weights=dict(normalized_delay_accrual=-cfg['delay_reward_weight'],normalized_uav_energy=-cfg['uav_energy_reward_weight'],
            new_deadline_misses_per_vehicle=-cfg['deadline_miss_penalty'])
        cross_population={}
        for fit in range(3):
            eps={a:[read(RUN/'evaluation'/a/str(fit)/f'episode_{ep:03d}.json') for ep in range(16)] for a in CORE}
            states={}
            for arm in CORE:
                arrays=[]
                for ep in range(16):
                    with np.load(RUN/'evaluation'/arm/str(fit)/f'episode_{ep:03d}.npz',allow_pickle=False) as data:arrays.append(data['states'])
                states[arm]=np.concatenate(arrays)
            cross_population[str(fit)]={}
            for control in [E,A,R,F]:
                ps=states[P];cs=states[control]
                assert np.array_equal(ps[:,:240],cs[:,:240]) and np.array_equal(ps[:,-1],cs[:,-1])
                cross_population[str(fit)][control]=dict(saved_states_different_fraction=float(np.any(ps!=cs,axis=1).mean()),
                    policy_changes_across_observed_populations={})
                for arm in CORE:
                    pp=raw[f'fit{fit}_{P}_{arm}_actions'];cc=raw[f'fit{fit}_{control}_{arm}_actions']
                    cross_population[str(fit)][control]['policy_changes_across_observed_populations'][arm]=dict(
                        placement=float((pp[:,:30]!=cc[:,:30]).mean()),movement=float((pp[:,30:]!=cc[:,30:]).mean()))
                counts=np.zeros(4,np.int64);transitions=np.zeros((5,5),np.int64);new_transitions=np.zeros((5,5),np.int64)
                rsu_deltas={3:[],4:[]};new_ids=[];avoided_ids=[];reward={k:0. for k in weights}
                for ep in range(16):
                    primary=eps[P][ep];other=eps[control][ep]
                    for p,c in zip(primary['records'],other['records']):
                        assert all(p[k]==c[k] for k in ['task_id','vehicle_id','arrival_s','required_cycles','input_bits','deadline_s'])
                        transitions[c['destination'],p['destination']]+=1
                        if not p['on_time'] and c['on_time']:
                            counts[0]+=1;new_ids.append([ep,p['task_id']]);new_transitions[c['destination'],p['destination']]+=1
                            if p['destination']==c['destination']:
                                counts[3]+=1;d=p['destination']
                                if d in [3,4]:
                                    queue_delta=p['queue_delay_s']-c['queue_delay_s'];delay_delta=p['delay_s']-c['delay_s']
                                    assert abs(queue_delta-delay_delta)<1e-8
                                    assert all(abs(p[k]-c[k])<1e-8 for k in ['uplink_wait_s','transmission_delay_s','computation_delay_s'])
                                    rsu_deltas[d].append(queue_delta);queue_tasks+=1
                        elif p['on_time'] and not c['on_time']:counts[1]+=1;avoided_ids.append([ep,p['task_id']])
                        elif not p['on_time'] and not c['on_time']:counts[2]+=1
                    for name,w in weights.items():
                        reward[name]+=w*(sum(t['reward_components'][name] for t in primary['step_info'])-sum(t['reward_components'][name] for t in other['step_info']))/16
                saved=v['paired_tasks'][str(fit)][control]
                assert counts.tolist()==[saved[k] for k in ['primary_only_misses','control_only_misses','both_miss','primary_only_same_destination']]
                assert new_ids==saved['new_miss_task_ids'] and avoided_ids==saved['avoided_miss_task_ids']
                assert transitions.tolist()==saved['all_control_to_primary_destinations'] and new_transitions.tolist()==saved['new_miss_control_to_primary_destinations']
                for d,data in rsu_deltas.items():check_stats(data,saved['same_RSU_queue_accounting'][str(d)]['extra_CPU_wait_s'])
                for name,value in reward.items():assert np.isclose(value,v['reward_components'][f'contrasts/{fit}'][control][name],rtol=1e-9,atol=1e-9)
                pair_checks+=1
            for arm in CORE:
                record=v['observed_states'][f'{arm}/{fit}'];qindices=list(range(240,514))+[514,515,517,518,520,521,523,524]
                assert float(np.abs(states[arm][:,qindices]).max())==record['max_abs_observed_queue_feature']
                for destination in [3,4]:
                    counts=[];service=[];misses=[]
                    for ep in eps[arm]:
                        for slot in range(40):
                            part=[t for t in ep['records'] if t['arrival_s']==slot and t['destination']==destination]
                            counts.append(len(part));service.append(sum(t['required_cycles'] for t in part)/cfg['rsu_cpu_hz'][destination-3]);misses.append(sum(not t['on_time'] for t in part))
                    load=v['server_loads'][f'{arm}/{fit}'][str(destination)]
                    check_stats(counts,load['tasks_per_slot']);check_stats(service,load['CPU_service_s_per_slot'])
                    assert sum(misses)==load['misses'] and sum(x>0 for x in misses)==load['slots_with_misses']
            for arm in [P,R,A,F]:
                record=v['training'][f'{arm}/{fit}'];windows=record['windows']
                assert windows[0]['first_update']==1 and windows[-1]['last_update']==1872
                assert all(b['first_update']==a['last_update']+1 for a,b in zip(windows,windows[1:]))
                assert record['full']['all']['prediction']['n']==1872*64
                terminal=record['full']['terminal'];assert terminal['target_q']['mean']==0
    assert v['policy_screen'] is False and v['value_screen'] is True and v['promising'] is False
    assert v['new_optimizer_updates']==v['new_simulator_steps']==v['new_teacher_calls']==v['new_autograd_evaluations']==0
    result=dict(status='SAVED_DIAGNOSIS_INPUT_HASHES_EXPORTS_TASK_QUEUE_AND_REWARD_ARITHMETIC_VERIFIED',
        consumed_files_rehashed=len(v['consumed_sha256']),fixed_state_population_model_entries_verified=population_checks,
        paired_task_comparisons_verified=pair_checks,same_RSU_new_miss_queue_identities_verified=queue_tasks,
        cross_population=cross_population,diagnosis_receipt_sha256=sha(OUT/'verification.json'),script_sha256=sha(Path(__file__)),
        new_models_created=0,new_simulator_steps=0,new_optimizer_updates=0,new_teacher_calls=0,new_autograd_evaluations=0,
        policy_screen=False,value_screen=True,promising=False,new_research_launch_authorized=False,word_thesis_changed=False)
    with dest.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cross_population'},indent=2))


if __name__=='__main__':main()
