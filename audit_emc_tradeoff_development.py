"""Read-only structural and saved-evidence audit for completed development runs."""
import argparse,gzip,json
from pathlib import Path
import numpy as np,torch
import run_emc_tradeoff as r
from audit_emc_tradeoff_smoke import check_episode,read
from chapter3_iov_env_v2 import Chapter3Config

def episode_file(path):return read(path)

def main(folder):
 torch.set_num_threads(1);spec=read(r.SPEC);settings=r.settings('development',spec);physical=Chapter3Config()
 manifest=read(folder/'run_manifest.json')['files']
 for rel,h in manifest.items():assert r.sha(folder/rel)==h,rel
 freeze=read(folder/'FRESH_GATE_FREEZE.json');expected_keys={f'{arm}_{rep}' for arm in r.ARMS for rep in range(3)}
 assert set(freeze)==expected_keys
 ver=read(folder/'verification.json');assert ver['mode']=='development' and ver['quality_screen_applied']
 assert ver['checkpoint_sha256']==freeze and ver['all_checkpoints_frozen_before_fresh']
 analysis=read(folder/'development_analysis.json');assert analysis['state'] in ('all_three_emc_initializations_pass','one_or_more_predeclared_screens_failed')
 timing=read(folder/'timings.json');assert len(timing)==5*128*9
 common=np.load(folder/'common_probe_states.npz')['observations'];assert common.shape==(128,533) and np.isfinite(common).all()
 counters={};histories={};fingerprints={};episode_count={'training':0,'validation':0,'fresh_model':0,'fresh_heuristic':0,'probe':0};training_slots=update_rows=0
 for rep in range(3):
  paired={}
  for arm in r.ARMS:
   key=f'{arm}_{rep}';d=folder/key;ck=torch.load(d/'final.pt',map_location='cpu',weights_only=False)
   assert r.sha(d/'final.pt')==freeze[key]==ver['checkpoint_sha256'][key]
   assert ck['replicate']==rep and ck['arm']==arm and ck['model_seed']==settings['model_seeds'][rep]
   assert ck['spec_sha256']==r.sha(r.SPEC)
   for name,h in ck['source_sha256'].items():
    assert h==r.sha(folder/'source_snapshot'/Path(name).name)
    if name in spec['source_sha256'] or name==r.SPEC.name:assert h==r.sha(r.ROOT/name)
   counter=read(d/'counters.json');assert (counter['slots'],counter['updates'],counter['target_copies'],ck['replay_size'],ck['replay_position'])==(8000,6977,34,8000,8000)
   assert ck['training_config']==settings['cfg']
   for values in ck['replay'].values():assert np.isfinite(values).all()
   counters[key]=counter;histories[key]=r.digest(ck['actor_history'])
   episode_fingerprints=[];summary=read(d/'training_summary.json');assert len(summary)==200
   for ep in range(1,201):
    row=episode_file(d/f'train_ep{ep}.json.gz');seed=settings['train_starts'][rep]+ep-1
    assert (row['seed'],row['episode'])==(seed,ep);check_episode(row,physical);episode_fingerprints.append(row['workload_sha256']);episode_count['training']+=1;training_slots+=40
    with np.load(d/f'train_ep{ep}.npz') as arrays:
     assert arrays['states'].shape==arrays['next_states'].shape==(40,533)
     assert arrays['actions'].shape==(40,32) and arrays['rewards'].shape==(40,) and arrays['dones'].shape==(40,)
     assert np.isfinite(arrays['states']).all() and np.isfinite(arrays['next_states']).all()
     np.testing.assert_array_equal(arrays['actions'],[s['action'] for s in row['slots']])
     np.testing.assert_allclose(arrays['rewards'],row['slot_rewards'],rtol=0,atol=1e-6)
     np.testing.assert_array_equal(arrays['dones'],[False]*39+[True])
     np.testing.assert_array_equal(arrays['next_states'][:-1],arrays['states'][1:])
   assert summary[ep-1]['seed']==seed and summary[ep-1]['workload_sha256']==row['workload_sha256']
   paired[arm]=episode_fingerprints
   for ep,fp in enumerate(episode_fingerprints,1):fingerprints.setdefault(('train',rep,ep),set()).add(fp)
   opt=[json.loads(x) for x in (d/'optimizer.jsonl').read_text(encoding='utf-8').splitlines()]
   assert len(opt)==6977 and [x['step'] for x in opt]==list(range(1,6978))
   assert opt[0]['physical_slots']==1024 and opt[-1]['physical_slots']==8000
   for i,x in enumerate(opt):
    assert x['physical_slots']==i+1024 and x['target_copy']==((i+1)%200==0)
    assert len(x['replay_indices'])==128 and min(x['replay_indices'])>=0 and max(x['replay_indices'])<min(i+1024,50000)
    assert np.isfinite([x['loss'],x['grad_norm_before_clip']]).all()
   assert sum(x['target_copy'] for x in opt)==34
   update_rows+=len(opt)
   val=read(d/'validation.json');assert set(val)=={'0','50','100','150','200'}
   for ep in val:
    rows=read(d/f'validation_ep{ep}.json.gz');assert len(rows)==5
    assert [x['seed'] for x in rows]==settings['validation']
    for row in rows:
     check_episode(row,physical);episode_count['validation']+=1
     fingerprints.setdefault(('validation',rep,int(ep),row['seed']),set()).add(row['workload_sha256'])
   fresh=read(d/'fresh.json.gz');assert len(fresh)==48 and [x['seed'] for x in fresh]==settings['fresh']
   for row in fresh:
    check_episode(row,physical);episode_count['fresh_model']+=1
    fingerprints.setdefault(('fresh',rep,row['seed']),set()).add(row['workload_sha256'])
   widths=(128,128) if arm==r.ARMS[0] else (64,64) if arm==r.ARMS[1] else None
   model=r.EMCQ() if arm==r.ARMS[2] else __import__('train_chapter3_ddqn').FactorizedQNetwork(533,32,5,widths)
   model.load_state_dict(ck['online']);assert r.digest(model.state_dict())==r.digest(ck['online'])
   with torch.inference_mode():
    values=model.greedy(torch.tensor(common))[1] if arm==r.ARMS[2] else model(torch.tensor(common)).max(2).values.sum(1)
   diag=ver['counters'][key];assert abs(float((values>0).float().mean())-diag['fixed_probe_positive_greedy_fraction'])<1e-9
   lengths=np.array([x['nanoseconds'] for x in timing if x['run']==key]);assert len(lengths)==640 and (lengths>0).all()
   assert abs(float(np.median(lengths)/1e6)-diag['median_action_ms'])<1e-10
   assert abs(float(np.quantile(lengths,.95)/1e6)-diag['p95_action_ms'])<1e-10
  for ep in range(200):
   assert len({paired[a][ep] for a in r.ARMS})==1
   assert len({histories[f'{a}_{rep}'] for a in r.ARMS})==1
  actor_rngs=[torch.load(folder/f'{a}_{rep}'/'final.pt',map_location='cpu',weights_only=False)['action_rng'] for a in r.ARMS]
  assert len({r.digest(value) for value in actor_rngs})==1
 heuristic_summaries=read(folder/'fresh_heuristics.json')
 assert set(heuristic_summaries)==set(spec['heuristics'])
 for name in spec['heuristics']:
  rows=read(folder/f'fresh_heuristic_{name}.json.gz')
  assert len(rows)==48 and [x['seed'] for x in rows]==settings['fresh']
  assert [r.compact_episode(row) for row in rows]==heuristic_summaries[name]
  for row in rows:
   check_episode(row,physical);episode_count['fresh_heuristic']+=1
   fingerprints.setdefault(('fresh_heuristic',row['seed']),set()).add(row['workload_sha256'])
 for seed in settings['probes']:
  row=episode_file(folder/f'fixed_probe_{seed}.json.gz');check_episode(row,physical);episode_count['probe']+=1
 for k,values in fingerprints.items():assert len(values)==1,k
 recomputed=r.analyze_development(spec,{},ver['fresh'],heuristic_summaries,ver['counters'])
 assert recomputed==analysis
 for rep in range(3):
  for arm in r.ARMS:
   key=f'{arm}_{rep}';events=[x for x in read(folder/'evaluation_invariance.json') if x['run']==key]
   assert len(events)==8 and all(x['before']==x['after'] for x in events)
   assert sorted(x['episode'] for x in events if x['phase']=='scheduled_validation')==[0,50,100,150,200]
 assert episode_count==dict(training=1800,validation=225,fresh_model=432,fresh_heuristic=144,probe=8)
 assert training_slots==72000 and update_rows==62793
 out=dict(state='passed_development_saved_artifact_audit',simulator_steps_added=0,optimizer_updates_added=0,
  episodes_checked=episode_count,training_slots=training_slots,optimizer_rows_checked=update_rows,
  checkpoint_hashes_checked=len(freeze),manifest_entries_checked=len(manifest),paired_training_and_fresh_workloads=True,
  source_and_model_outputs_checked=True,reward_task_energy_accounting_checked=True,quality_screens_recomputed=True,quality_screen=analysis['state'],
  bootstrap_rng_seed=spec['uncertainty']['rng_seed'],bootstrap_draws=spec['uncertainty']['bootstrap_draws'],
  interpretation='development engineering evidence only; no formal noninferiority or novelty claim',auditor_sha256=r.sha(Path(__file__)))
 with (folder/'saved_artifact_audit.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False)
 print(json.dumps(out,indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);main(parser.parse_args().folder)
