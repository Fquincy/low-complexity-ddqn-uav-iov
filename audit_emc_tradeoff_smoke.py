"""Read-only saved-artifact audit. No simulator or optimizer steps."""
import argparse,gzip,json,math
from pathlib import Path
import numpy as np,torch
import run_emc_tradeoff as r
from chapter3_iov_env_v2 import Chapter3Config

def close(a,b):
 assert math.isclose(float(a),float(b),rel_tol=1e-7,abs_tol=1e-7),(a,b)

def finite(x):
 if isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,list):
  for v in x:finite(v)
 elif isinstance(x,float):assert math.isfinite(x)

def read(path):
 if str(path).endswith('.gz'):
  with gzip.open(path,'rt',encoding='utf-8') as f:value=json.load(f)
 else:value=json.loads(path.read_text(encoding='utf-8'))
 finite(value);return value

def check_episode(row,physical):
 slots=row['slots'];tasks=row['tasks'];summary=row['summary']
 assert len(slots)==40 and len(row['slot_rewards'])==40
 assert [s['done'] for s in slots]==[False]*39+[True]
 assert [s['slot'] for s in slots]==list(range(40))
 close(row['return_sum'],sum(row['slot_rewards']))
 assert row['workload_sha256']==r.digest([s['tasks_arrived'] for s in slots])
 assert len(tasks)==row['tasks_generated']==summary['tasks_generated']==1200
 assert len({t['task_id'] for t in tasks})==1200
 assert sum(t['on_time'] for t in tasks)==row['tasks_on_time']==summary['tasks_on_time']
 close(row['deadline_success'],row['tasks_on_time']/row['tasks_generated'])
 close(row['deadline_success'],summary['deadline_success_rate'])
 assert sum(t['status']=='completed' for t in tasks)==summary['tasks_completed']
 assert sum(t['unfinished_at_horizon'] for t in tasks)==summary['tasks_unfinished']
 assert sum(t['deadline_missed'] for t in tasks)==summary['deadline_misses_observed']
 by_id={t['task_id']:t for t in tasks}
 for i,s in enumerate(slots):
  action=np.array(s['action']);assert action.shape==(32,) and np.all((0<=action)&(action<5))
  assert len(s['tasks_arrived'])==30
  close(s['reward'],row['slot_rewards'][i])
  c=s['reward_components']
  close(s['reward'],-physical.delay_reward_weight*c['normalized_delay_accrual']-physical.uav_energy_reward_weight*c['normalized_uav_energy']-physical.deadline_miss_penalty*c['new_deadline_misses_per_vehicle'])
  for task in s['tasks_arrived']:
   saved=by_id[task['task_id']]
   for k,v in task.items():assert saved[k]==v
   assert saved['destination']==s['action'][task['vehicle_id']]
  np.testing.assert_allclose(s['actual_distance_m'],np.linalg.norm(np.array(s['uav_after'])-s['uav_before'],axis=1),rtol=0,atol=1e-9)
 for component in slots[0]['energy_j']:
  close(sum(s['energy_j'][component] for s in slots),summary[component+'_energy_j'])
 return 40

def main(folder):
 torch.set_num_threads(1)
 spec=read(r.SPEC);cfg=r.settings('smoke',spec)['cfg'];v=read(folder/'verification.json')
 assert v['state']=='passed_smoke' and v['mode']=='smoke'
 freeze=read(folder/'FRESH_GATE_FREEZE.json');assert len(freeze)==3
 tests=read(r.ROOT/'validation/emc_tradeoff_runner_tests_2026-11-04_09/verification.json')
 assert tests['state']=='passed' and tests['tests_run']==9 and tests['runner_sha256']==r.sha(r.ROOT/'run_emc_tradeoff.py')
 physical=Chapter3Config();all_rows=[];models={};training_fingerprints=[];actor_histories=[];sample_rows=0
 timing=read(folder/'timings.json');assert len(timing)==1920
 common=np.load(folder/'common_probe_states.npz')['observations'];assert common.shape==(128,533) and np.isfinite(common).all()
 for arm in r.ARMS:
  key=arm+'_0';d=folder/key;ck=torch.load(d/'final.pt',map_location='cpu',weights_only=False)
  assert ck['spec_sha256']==r.sha(r.SPEC)
  assert r.sha(d/'final.pt')==freeze[key]==v['checkpoint_sha256'][key]
  for name,h in ck['source_sha256'].items():
   assert h==r.sha(folder/'source_snapshot'/Path(name).name)
   if name in spec['source_sha256'] or name==r.SPEC.name:assert h==r.sha(r.ROOT/name)
  for name,h in spec['source_sha256'].items():assert ck['source_sha256'][name]==h
  assert ck['model_seed']==2399 and ck['arm']==arm and ck['training_config']==cfg
  assert (ck['slots'],ck['updates'],ck['copies'],ck['replay_size'],ck['replay_position'])==(80,73,9,64,16)
  agent=r.build(arm,2399,cfg);agent['online'].load_state_dict(ck['online']);agent['target'].load_state_dict(ck['target']);agent['opt'].load_state_dict(ck['optimizer'])
  assert r.digest(agent['online'].state_dict())==r.digest(ck['online'])
  assert r.digest(agent['target'].state_dict())==r.digest(ck['target'])
  assert r.digest(agent['opt'].state_dict())==r.digest(ck['optimizer'])
  replay=r.Replay(64);rows=[];physical_arrays=[]
  for ep,seed in enumerate(spec['smoke']['train_seeds'],1):
   row=read(d/f'train_ep{ep}.json.gz');assert row['seed']==seed and row['episode']==ep
   rows.append(row);all_rows.append(('training',row))
   with np.load(d/f'train_ep{ep}.npz') as data:
    assert data['states'].shape==data['next_states'].shape==(40,533)
    assert data['actions'].shape==(40,32)
    for name in data.files:assert np.isfinite(data[name]).all()
    np.testing.assert_array_equal(data['actions'],[s['action'] for s in row['slots']])
    np.testing.assert_array_equal(data['dones'],[False]*39+[True])
    np.testing.assert_array_equal(data['rewards'],row['slot_rewards'])
    np.testing.assert_array_equal(data['next_states'][:-1],data['states'][1:])
    physical_arrays.extend([(data['states'][i].copy(),data['actions'][i].copy(),data['rewards'][i],data['next_states'][i].copy(),data['dones'][i]) for i in range(40)])
  logs=[json.loads(line) for line in (d/'optimizer.jsonl').read_text().splitlines()];assert len(logs)==73
  action_rng=np.random.default_rng(102399);replay_rng=np.random.default_rng(202399);history=[]
  for i,record in enumerate(physical_arrays):
   replay.add(*record);epsilon=1. if i<8 else 1-.95*min(i/80,1.)
   explore=bool(i<8 or action_rng.random()<epsilon)
   if explore:np.testing.assert_array_equal(record[1],action_rng.integers(0,5,32,dtype=np.int64))
   history.append(dict(epsilon=epsilon,warmup=i<8,explore=explore))
   if i>=7:
    log=logs[i-7];finite(log);ix=replay_rng.integers(0,replay.size,4)
    assert log['step']==i-6 and log['physical_slots']==i+1 and log['reward_samples']==4
    assert log['target_copy']==((i-6)%8==0)
    np.testing.assert_array_equal(ix,log['replay_indices'])
    assert log['terminal_samples']==int(replay.dones[ix].sum())
  assert ck['actor_history']==history
  assert r.digest(ck['action_rng'])==r.digest(action_rng.bit_generator.state)
  assert r.digest(ck['replay_rng'])==r.digest(replay_rng.bit_generator.state)
  for name,saved in ck['replay'].items():np.testing.assert_array_equal(saved,getattr(replay,name))
  actor_histories.append(r.digest(history));training_fingerprints.append([x['workload_sha256'] for x in rows]);sample_rows+=len(logs)
  validation=read(d/'validation.json');assert set(validation)=={'0','1','2'}
  for phase,phase_rows in validation.items():
   assert [x['seed'] for x in phase_rows]==spec['smoke']['validation_seeds']
   all_rows.extend(('validation',x) for x in phase_rows)
  fresh=read(d/'fresh.json.gz');assert [x['seed'] for x in fresh]==spec['smoke']['fresh_seeds']
  all_rows.extend(('fresh_model',x) for x in fresh)
  metrics=v['counters'][key]
  close(metrics['fresh_terminal_abs_error_p95'],np.quantile([abs(x['terminal_value']-x['slot_rewards'][-1]) for x in fresh],.95))
  with torch.inference_mode():
   tensor=torch.tensor(common);q=agent['online'].greedy(tensor)[1] if arm==r.ARMS[2] else agent['online'](tensor).max(2).values.sum(1)
  assert torch.isfinite(q).all();close(metrics['fixed_probe_positive_greedy_fraction'],float((q>0).float().mean()))
  ns=np.array([t['nanoseconds'] for t in timing if t['run']==key]);assert len(ns)==640 and np.all(ns>0)
  assert {(t['round'],t['observation']) for t in timing if t['run']==key}=={(i,j) for i in range(5) for j in range(128)}
  close(metrics['median_action_ms'],np.median(ns)/1e6);close(metrics['p95_action_ms'],np.quantile(ns,.95)/1e6)
  close(metrics['online_parameter_bytes'],sum(p.numel()*p.element_size() for p in agent['online'].parameters()))
  close(metrics['target_parameter_bytes'],metrics['online_parameter_bytes']);close(metrics['replay_array_bytes'],replay.storage_bytes)
  close(metrics['optimizer_tensor_bytes'],sum(z.numel()*z.element_size() for st in agent['opt'].state.values() for z in st.values() if isinstance(z,torch.Tensor)))
  models[key]=dict(training_slots=80,optimizer_updates=len(logs),target_copies=9,median_action_ms=metrics['median_action_ms'],p95_action_ms=metrics['p95_action_ms'],online_parameter_bytes=metrics['online_parameter_bytes'])
 assert len(set(actor_histories))==1 and all(f==training_fingerprints[0] for f in training_fingerprints)
 for seed in spec['smoke']['probe_seeds']:all_rows.append(('probe',read(folder/f'fixed_probe_{seed}.json.gz')))
 heuristics=read(folder/'fresh_heuristics.json');assert set(heuristics)==set(spec['heuristics'])
 for p,rows in heuristics.items():
  assert [x['seed'] for x in rows]==spec['smoke']['fresh_seeds'];all_rows.extend(('fresh_heuristic',x) for x in rows)
 counts={};fingerprints={}
 for kind,row in all_rows:
  check_episode(row,physical);counts[kind]=counts.get(kind,0)+1
  fingerprints.setdefault(row['seed'],set()).add(row['workload_sha256'])
 assert all(len(values)==1 for values in fingerprints.values())
 assert counts==dict(training=6,validation=18,fresh_model=6,probe=2,fresh_heuristic=6)
 invariance=read(folder/'evaluation_invariance.json');assert len(invariance)==18
 for key in models:
  inv=[x for x in invariance if x['run']==key]
  assert len(inv)==6 and sorted(x['phase'] for x in inv)==sorted(['scheduled_validation']*3+['fresh_evaluation','timing','fixed_probe'])
  assert all(x['before']==x['after'] for x in inv)
 result=dict(state='passed_saved_artifact_audit',new_simulator_steps=0,new_optimizer_updates=0,episodes_checked=counts,
  training_slots=240,evaluation_slots=1280,optimizer_rows_checked=sample_rows,models=models,
  source_and_checkpoint_hashes_checked=True,replay_and_rng_reconstructed=True,task_reward_energy_accounting_checked=True,
  timing_quantiles_recomputed=True,evaluation_digest_pairs_checked=len(invariance),quality_claim=False,
  auditor_sha256=r.sha(Path(__file__)),runner_sha256=r.sha(r.ROOT/'run_emc_tradeoff.py'))
 output=folder/'saved_artifact_audit.json'
 with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps(result,indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);main(parser.parse_args().folder)
