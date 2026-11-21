"""Read-only saved-artifact audit for the frozen five-arm development run."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np,torch
import run_value_bound_tradeoff as r
from chapter3_iov_env_v2 import Chapter3Config

def close(a,b):assert abs(float(a)-float(b))<=1e-7*max(1.,abs(float(a)),abs(float(b))),(a,b)

def check_episode(row,physical):
 slots=row['slots'];tasks=row['tasks'];summary=row['summary']
 assert len(slots)==40 and len(row['slot_rewards'])==40
 assert [s['done'] for s in slots]==[False]*39+[True] and [s['slot'] for s in slots]==list(range(40))
 close(row['return_sum'],sum(row['slot_rewards']))
 assert row['workload_sha256']==r.digest([s['tasks_arrived'] for s in slots])
 assert len(tasks)==row['tasks_generated']==summary['tasks_generated']==1200
 assert sum(t['on_time'] for t in tasks)==row['tasks_on_time']==summary['tasks_on_time']
 close(row['deadline_success'],row['tasks_on_time']/row['tasks_generated']);close(row['deadline_success'],summary['deadline_success_rate'])
 assert sum(t['status']=='completed' for t in tasks)==summary['tasks_completed']
 assert sum(t['unfinished_at_horizon'] for t in tasks)==summary['tasks_unfinished']
 assert sum(t['deadline_missed'] for t in tasks)==summary['deadline_misses_observed']
 by_id={t['task_id']:t for t in tasks}
 for i,slot in enumerate(slots):
  action=np.asarray(slot['action']);assert action.shape==(32,) and np.all((0<=action)&(action<5))
  assert len(slot['tasks_arrived'])==30;close(slot['reward'],row['slot_rewards'][i])
  c=slot['reward_components'];close(slot['reward'],-physical.delay_reward_weight*c['normalized_delay_accrual']-
   physical.uav_energy_reward_weight*c['normalized_uav_energy']-physical.deadline_miss_penalty*c['new_deadline_misses_per_vehicle'])
  for task in slot['tasks_arrived']:
   saved=by_id[task['task_id']]
   for name,value in task.items():assert saved[name]==value
   assert saved['destination']==slot['action'][task['vehicle_id']]
  np.testing.assert_allclose(slot['actual_distance_m'],np.linalg.norm(np.asarray(slot['uav_after'])-slot['uav_before'],axis=1),rtol=0,atol=1e-9)
 for component in slots[0]['energy_j']:close(sum(s['energy_j'][component] for s in slots),summary[component+'_energy_j'])

def read(path):
 opener=gzip.open if str(path).endswith('.gz') else open
 with opener(path,'rt',encoding='utf-8') as f:return json.load(f)

def check_summary_structure(spec,settings,verification,analysis):
 keys={f'{arm}_{rep}' for arm in r.ARMS for rep in settings['replicates']}
 assert set(verification['checkpoint_sha256'])==keys
 assert set(verification['counters'])==keys
 assert set(verification['fresh'])==keys and all(len(v)==len(settings['fresh']) for v in verification['fresh'].values())
 assert set(analysis['replicates'])=={str(rep) for rep in settings['replicates']}
 assert analysis['state'] in ('all_three_primary_initializations_pass','one_or_more_predeclared_screens_failed')
 required={'versus_additive128_unbounded','versus_additive64_nonpositive','versus_emc_q_unbounded',
  'versus_nearest_rsu','versus_uav_selective','fresh_terminal_calibration','all_bounded_probe_signs',
  'all_bounded_fresh_terminal_signs','finite_learning_values','online_storage_ratio','median_latency_ratio','p95_latency_ceiling'}
 for result in analysis['replicates'].values():
  assert set(result['screens'])==required
  assert set(result['value_diagnostics'])==set(r.ARMS)
  assert set(result['costs'])==set(r.ARMS)
 return keys

def assert_paired_fingerprints(by_arm):
 if not by_arm:raise AssertionError('no arm fingerprints')
 first=next(iter(by_arm.values()))
 assert all(value==first for value in by_arm.values()),'paired workload fingerprints differ'

def main(folder):
 torch.set_num_threads(1);spec=read(folder/'runtime_protocol.json');settings=r.settings('development',spec)
 synthetic=spec.get('run_kind')=='synthetic_fixture_not_research_evidence'
 assert read(folder/'verification.json')['run_kind']==('synthetic_fixture_not_research_evidence' if synthetic else 'research_protocol_run')
 manifest=read(folder/'run_manifest.json')['files']
 for rel,h in manifest.items():assert r.sha(folder/rel)==h,rel
 frozen=read(folder/'FRESH_GATE_FREEZE.json');verification=read(folder/'verification.json')
 assert verification['mode']=='development' and verification['quality_screen_applied']
 assert verification['all_checkpoints_frozen_before_fresh'] and verification['checkpoint_sha256']==frozen
 analysis=read(folder/'development_analysis.json');keys=check_summary_structure(spec,settings,verification,analysis)
 assert len(read(folder/'timings.json'))==5*128*len(keys)
 common=np.load(folder/'common_probe_states.npz')['observations'];assert common.shape==(128,533) and np.isfinite(common).all()
 invariance=read(folder/'evaluation_invariance.json');assert len(invariance)==len(keys)*(len(settings['diagnostics'])+3)
 episode_counts=dict(training=0,validation=0,fresh_model=0,fresh_heuristic=0,probe=0);updates_checked=0
 source_hashes={}
 for name in spec['source_sha256']:
  source_hashes[name]=r.sha(r.ROOT/name);assert source_hashes[name]==spec['source_sha256'][name]
 for rel,h in manifest.items():
  if rel.startswith('source_snapshot/'):
   source_hashes[rel.removeprefix('source_snapshot/')]=r.sha(folder/rel)
 workload_groups={};fresh_workloads={};actor_histories={};timing=read(folder/'timings.json');physical=Chapter3Config()
 model_module=__import__('train_chapter3_ddqn')
 for rep in settings['replicates']:
  paired_training={}
  for arm in r.ARMS:
   key=f'{arm}_{rep}';d=folder/key;ck=torch.load(d/'final.pt',map_location='cpu',weights_only=False)
   assert r.sha(d/'final.pt')==frozen[key]==verification['checkpoint_sha256'][key]
   assert (ck['arm'],ck['replicate'],ck['model_seed'])==(arm,rep,settings['model_seeds'][rep])
   assert ck['spec_sha256']==r.sha(folder/'runtime_protocol.json') and ck['training_config']==settings['cfg']
   metrics=verification['counters'][key]
   for source_name,h in ck['source_sha256'].items():
    snapshot=folder/'source_snapshot'/Path(source_name).name
    assert snapshot.exists() and r.sha(snapshot)==h
    if source_name in spec['source_sha256'] or source_name==r.SPEC.name:assert h==r.sha(r.ROOT/source_name)
   counter=read(d/'counters.json');episodes=settings['cfg']['episodes'];expected_slots=episodes*40
   expected_updates=expected_slots-settings['cfg']['warmup_physical_records']+1
   expected_copies=expected_updates//settings['cfg']['target_copy_every_updates']
   expected_replay_size=min(expected_slots,settings['cfg']['replay_capacity_physical_records'])
   assert (counter['slots'],counter['updates'],counter['target_copies'],ck['replay_size'])==(expected_slots,expected_updates,expected_copies,expected_replay_size)
   assert ck['replay_position']==expected_slots%settings['cfg']['replay_capacity_physical_records']
   assert counter['value_stability']['updates']==expected_updates
   assert counter['value_stability']['aggregation_scope']=='mean_of_optimizer_minibatch_statistics'
   first_update_episode=(settings['cfg']['warmup_physical_records']-1)//spec['episode_slots']+1
   expected_trend_episodes=settings['cfg']['episodes']-first_update_episode+1
   assert len(counter['value_stability']['by_training_episode'])==expected_trend_episodes
   assert r.summarize_optimizer_log(d/'optimizer.jsonl')==counter['value_stability']
   for arr in ck['replay'].values():assert np.isfinite(arr).all()
   actor_histories[key]=r.digest(ck['actor_history'])
   assert len(ck['actor_history'])==expected_slots
   summaries=read(d/'training_summary.json');assert len(summaries)==episodes
   fingerprints=[]
   for ep in range(1,episodes+1):
    row=read(d/f'train_ep{ep}.json.gz');seed=settings['train_starts'][rep]+ep-1
    assert (row['seed'],row['episode'])==(seed,ep);check_episode(row,physical)
    with np.load(d/f'train_ep{ep}.npz') as arrays:
     assert arrays['states'].shape==arrays['next_states'].shape==(40,533)
     assert arrays['actions'].shape==(40,32) and arrays['rewards'].shape==(40,) and arrays['dones'].shape==(40,)
     assert np.isfinite(arrays['states']).all() and np.isfinite(arrays['next_states']).all()
     np.testing.assert_array_equal(arrays['actions'],[slot['action'] for slot in row['slots']])
     np.testing.assert_allclose(arrays['rewards'],row['slot_rewards'],rtol=0,atol=1e-6)
     np.testing.assert_array_equal(arrays['dones'],[False]*39+[True])
     np.testing.assert_array_equal(arrays['next_states'][:-1],arrays['states'][1:])
    assert summaries[ep-1]['seed']==seed and summaries[ep-1]['workload_sha256']==row['workload_sha256']
    fingerprints.append(row['workload_sha256']);workload_groups.setdefault(('train',rep,ep),{})[arm]=row['workload_sha256']
    episode_counts['training']+=1
   paired_training[arm]=fingerprints
   logs=[json.loads(x) for x in (d/'optimizer.jsonl').read_text(encoding='utf-8').splitlines()]
   assert len(logs)==expected_updates and [x['step'] for x in logs]==list(range(1,expected_updates+1))
   for i,item in enumerate(logs):
    assert item['physical_slots']==i+settings['cfg']['warmup_physical_records'] and item['target_copy']==((i+1)%settings['cfg']['target_copy_every_updates']==0)
    assert len(item['replay_indices'])==settings['cfg']['batch_size'] and min(item['replay_indices'])>=0 and max(item['replay_indices'])<min(i+settings['cfg']['warmup_physical_records'],settings['cfg']['replay_capacity_physical_records'])
    assert np.isfinite([item['loss'],item['grad_norm_before_clip']]).all()
    for field in ('chosen_q','bellman_target','td_residual'):
     values=item[field];assert values['count']==settings['cfg']['batch_size'] and np.isfinite(list(values.values())).all()
     assert 0.0<=values['positive_fraction']<=1.0
    if r.bounded(arm):
     assert item['chosen_q']['maximum']<=1e-7 and item['bellman_target']['maximum']<=1e-7
    assert item['gradient_clipped']==(item['grad_norm_before_clip']>settings['cfg']['gradient_clip'])
   assert sum(x['target_copy'] for x in logs)==expected_copies
   updates_checked+=len(logs)
   validation_index=read(d/'validation.json');assert set(validation_index)=={str(ep) for ep in settings['diagnostics']}
   for ep in settings['diagnostics']:
    rows=read(d/f'validation_ep{ep}.json.gz');assert len(rows)==len(settings['validation']) and [x['seed'] for x in rows]==settings['validation']
    for row in rows:
     assert len(row['slots'])==40 and len(row['tasks'])==1200
     if r.bounded(arm):assert row['terminal_value']<=0
     workload_groups.setdefault(('validation',rep,ep,row['seed']),{})[arm]=row['workload_sha256'];episode_counts['validation']+=1
   fresh=read(d/'fresh.json.gz');assert len(fresh)==len(settings['fresh']) and [x['seed'] for x in fresh]==settings['fresh']
   for row in fresh:
    check_episode(row,physical)
    if r.bounded(arm):assert row['terminal_value']<=0
    workload_groups.setdefault(('fresh',rep,row['seed']),{})[arm]=row['workload_sha256']
    fresh_workloads.setdefault(row['seed'],{})[key]=row['workload_sha256'];episode_counts['fresh_model']+=1
   widths=(128,128) if arm=='additive128_unbounded' else (64,64)
   model=r.EMCQ() if arm in r.EMC_ARMS else model_module.FactorizedQNetwork(533,32,5,widths)
   model.load_state_dict(ck['online']);assert r.digest(model.state_dict())==r.digest(ck['online'])
   target=r.build(arm,settings['model_seeds'][rep],settings['cfg'])['target'];target.load_state_dict(ck['target'])
   with torch.inference_mode():
    states=torch.tensor(common);actions,online_values,_=r.greedy(model,states,arm)
    target_values,_=r.chosen_value(target,states,actions,arm)
   assert torch.isfinite(online_values).all() and torch.isfinite(target_values).all()
   if r.bounded(arm):assert torch.all(online_values<=0) and torch.all(target_values<=0)
   assert metrics['online_parameter_bytes']==sum(p.numel()*p.element_size() for p in model.parameters())
   assert metrics['target_parameter_bytes']==sum(p.numel()*p.element_size() for p in target.parameters())
   expected_replay_bytes=settings['cfg']['replay_capacity_physical_records']*(2*533*4+32*1+4+4)
   assert metrics['replay_array_bytes']==expected_replay_bytes
   assert metrics['checkpoint_replay_bytes']==sum(x.nbytes for x in ck['replay'].values())
   opt=torch.optim.Adam(model.parameters(),lr=settings['cfg']['learning_rate']);opt.load_state_dict(ck['optimizer'])
   optimizer_bytes=sum(x.numel()*x.element_size() for state in opt.state.values() for x in state.values() if isinstance(x,torch.Tensor))
   assert metrics['optimizer_tensor_bytes']==optimizer_bytes
   assert metrics['learning_values_finite'] and metrics['value_stability']['updates']==len(logs)
   if r.bounded(arm):assert metrics['fixed_probe_positive_greedy_fraction']==0 and metrics['fresh_terminal_positive_fraction']==0
   times=np.asarray([x['nanoseconds'] for x in timing if x['run']==key]);assert len(times)==640 and np.all(times>0)
   assert abs(np.median(times)/1e6-metrics['median_action_ms'])<1e-10
   assert abs(np.quantile(times,.95)/1e6-metrics['p95_action_ms'])<1e-10
   assert {x['round'] for x in timing if x['run']==key}==set(range(5))
   assert metrics['fixed_probe_online_q_distribution']['count']==128
   assert metrics['fixed_probe_target_q_distribution']['count']==128
   assert metrics['fresh_terminal_q_distribution']['count']==len(settings['fresh'])
   assert metrics['fresh_terminal_abs_error_distribution']['count']==len(settings['fresh'])
  assert_paired_fingerprints(paired_training)
  assert len({actor_histories[f'{arm}_{rep}'] for arm in r.ARMS})==1
 heuristic_summaries=read(folder/'fresh_heuristics.json');assert set(heuristic_summaries)==set(spec['heuristics'])
 for name in spec['heuristics']:
  rows=read(folder/f'fresh_heuristic_{name}.json.gz');assert len(rows)==len(settings['fresh']) and [x['seed'] for x in rows]==settings['fresh']
  assert [r.compact_episode(row) for row in rows]==heuristic_summaries[name]
  for row in rows:
   check_episode(row,physical)
   workload_groups.setdefault(('heuristic',row['seed']),{})[name]=row['workload_sha256'];episode_counts['fresh_heuristic']+=1
 for seed in settings['probes']:
  row=read(folder/f'fixed_probe_{seed}.json.gz');assert row['seed']==seed;check_episode(row,physical)
  episode_counts['probe']+=1
 for group,by_name in workload_groups.items():
  if group[0] in ('train','validation','fresh'):
   assert_paired_fingerprints(by_name)
  elif group[0]=='heuristic':
   assert len(by_name)==3 and len(set(by_name.values()))==1
 for seed in settings['fresh']:
  model_group=fresh_workloads[seed]
  expected={name:row['workload_sha256'] for name,rows in heuristic_summaries.items() for row in rows if row['seed']==seed}
  assert len(set(model_group.values())|set(expected.values()))==1
 recomputed=r.analyze_development(spec,{},verification['fresh'],heuristic_summaries,verification['counters'])
 assert recomputed==analysis
 for key in keys:
  events=[x for x in invariance if x['run']==key]
  assert len(events)==len(settings['diagnostics'])+3 and all(x['before']==x['after'] for x in events)
  assert sorted(x['episode'] for x in events if x['phase']=='scheduled_validation')==settings['diagnostics']
  expected_counts=dict(training=len(keys)*settings['cfg']['episodes'],
   validation=len(keys)*len(settings['diagnostics'])*len(settings['validation']),
   fresh_model=len(keys)*len(settings['fresh']),fresh_heuristic=len(spec['heuristics'])*len(settings['fresh']),probe=len(settings['probes']))
  assert episode_counts==expected_counts
  assert updates_checked==sum(x['updates'] for x in verification['counters'].values())
 result=dict(state='passed_development_saved_artifact_audit',simulator_steps_added=0,optimizer_updates_added=0,
   run_kind='synthetic_fixture_not_research_evidence' if synthetic else 'research_protocol_run',
   episodes_checked=episode_counts,training_slots=sum(x['slots'] for x in verification['counters'].values()),optimizer_rows_checked=updates_checked,
  checkpoint_hashes_checked=len(frozen),manifest_entries_checked=len(manifest),paired_training_validation_and_fresh_workloads=True,
  source_checkpoint_replay_and_value_outputs_checked=True,quality_screens_recomputed=True,quality_screen=analysis['state'],
  bootstrap_rng_seed=spec['uncertainty']['rng_seed'],bootstrap_draws=spec['uncertainty']['bootstrap_draws'],
  interpretation='development engineering evidence only; no formal noninferiority or novelty claim',auditor_sha256=r.sha(Path(__file__)))
 with (folder/'saved_artifact_audit.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps(result,indent=2));return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);main(p.parse_args().folder)
