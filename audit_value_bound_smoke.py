"""Read-only integrity audit of the five-arm value-bound smoke artifacts."""
import argparse,gzip,hashlib,json,math
from pathlib import Path
import numpy as np,torch
import run_value_bound_tradeoff as r

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
 opener=gzip.open if str(path).endswith('.gz') else open
 with opener(path,'rt',encoding='utf-8') as f:return json.load(f)

def main(folder):
 torch.set_num_threads(1);spec=load(r.SPEC);cfg=r.settings('smoke',spec);v=load(folder/'verification.json')
 runtime_protocol=folder/'runtime_protocol.json'
 protocol_hash=sha(runtime_protocol) if runtime_protocol.exists() else sha(r.SPEC)
 assert v['mode']=='smoke' and v['state']=='passed_smoke'
 assert sum(x['slots'] for x in v['counters'].values())==400
 assert sum(x['updates'] for x in v['counters'].values())==365 and len(v['checkpoint_sha256'])==5
 manifest=load(folder/'run_manifest.json')['files']
 for name,expected in manifest.items():assert sha(folder/name)==expected,name
 frozen=load(folder/'FRESH_GATE_FREEZE.json');assert set(frozen)=={f'{a}_0' for a in r.ARMS}
 assert frozen==v['checkpoint_sha256']
 common=np.load(folder/'common_probe_states.npz')['observations'];assert common.shape==(128,533) and np.isfinite(common).all()
 timing=load(folder/'timings.json');assert len(timing)==5*5*128
 paired_train=[];paired_fresh=[];invariance=load(folder/'evaluation_invariance.json')
 assert len(invariance)==5*(3+1+1+1)
 for arm in r.ARMS:
  key=f'{arm}_0';d=folder/key;ck=torch.load(d/'final.pt',map_location='cpu',weights_only=False)
  assert sha(d/'final.pt')==frozen[key]
  assert ck['arm']==arm and ck['model_seed']==3091 and ck['replicate']==0
  assert ck['spec_sha256']==protocol_hash and ck['training_config']==cfg['cfg']
  assert (ck['slots'],ck['updates'],ck['copies'],ck['replay_size'],ck['replay_position'])==(80,73,9,64,16)
  for name,h in ck['source_sha256'].items():
   assert h==sha(folder/'source_snapshot'/Path(name).name)
   if name in spec['source_sha256'] or name==r.SPEC.name:assert h==sha(r.ROOT/name)
  model=r.build(arm,3091,cfg['cfg']);model['online'].load_state_dict(ck['online']);model['target'].load_state_dict(ck['target']);model['opt'].load_state_dict(ck['optimizer'])
  assert r.digest(model['online'].state_dict())==r.digest(ck['online'])
  assert r.digest(model['target'].state_dict())==r.digest(ck['target'])
  logs=[json.loads(x) for x in (d/'optimizer.jsonl').read_text(encoding='utf-8').splitlines()]
  assert len(logs)==73 and all(np.isfinite([x['loss'],x['grad_norm_before_clip']]).all() for x in logs)
  for i,row in enumerate(logs):assert row['step']==i+1 and row['target_copy']==((i+1)%8==0)
  trains=[load(d/f'train_ep{i}.json.gz') for i in (1,2)]
  assert [x['seed'] for x in trains]==spec['smoke']['train_seeds']
  for ep,row in enumerate(trains,1):
   arrays=np.load(d/f'train_ep{ep}.npz');assert arrays['states'].shape==arrays['next_states'].shape==(40,533)
   assert arrays['actions'].shape==(40,32) and np.isfinite(arrays['states']).all()
   assert row['return_sum']==sum(row['slot_rewards']) and len(row['tasks'])==1200
   assert row['workload_sha256']==r.digest([slot['tasks_arrived'] for slot in row['slots']])
   assert row['tasks_on_time']==sum(t['on_time'] for t in row['tasks'])
   np.testing.assert_array_equal(arrays['actions'],[x['action'] for x in row['slots']])
   np.testing.assert_array_equal(arrays['dones'],[False]*39+[True])
  paired_train.append([x['workload_sha256'] for x in trains])
  val=load(d/'validation.json');assert set(val)=={'0','1','2'}
  for phase,rows in val.items():
   assert [x['seed'] for x in rows]==spec['smoke']['validation_seeds']
   if r.bounded(arm):assert all(x['terminal_value']<=0 for x in rows)
  fresh=load(d/'fresh.json.gz');assert [x['seed'] for x in fresh]==spec['smoke']['fresh_seeds']
  if r.bounded(arm):assert all(x['terminal_value']<=0 for x in fresh)
  paired_fresh.append([x['workload_sha256'] for x in fresh])
  metrics=v['counters'][key]
  with torch.inference_mode():_,values,_=r.greedy(model['online'],torch.tensor(common),arm)
  assert torch.isfinite(values).all()
  if r.bounded(arm):assert torch.all(values<=0) and metrics['fixed_probe_positive_greedy_fraction']==0
  per_run=[x for x in timing if x['run']==key]
  assert len(per_run)==640 and all(x['nanoseconds']>0 for x in per_run)
  assert metrics['online_parameter_bytes']==sum(p.numel()*p.element_size() for p in model['online'].parameters())
  inv=[x for x in invariance if x['run']==key];assert len(inv)==6 and all(x['before']==x['after'] for x in inv)
 assert all(p==paired_train[0] for p in paired_train) and all(p==paired_fresh[0] for p in paired_fresh)
 probe_fingerprints=[load(folder/f'fixed_probe_{seed}.json.gz')['workload_sha256'] for seed in spec['smoke']['probe_seeds']]
 heuristic=load(folder/'fresh_heuristics.json')
 for rows in heuristic.values():assert [x['seed'] for x in rows]==spec['smoke']['fresh_seeds']
 assert len(probe_fingerprints)==2
 result=dict(state='passed_saved_artifact_audit',simulator_steps=0,optimizer_updates=0,training_slots=400,
  optimizer_rows_checked=365,arms_checked=len(r.ARMS),paired_training_workloads=True,
  paired_fresh_workloads=True,bounded_greedy_values_nonpositive=True,
  evaluation_state_invariance_checks=len(invariance),checkpoint_and_manifest_hashes_checked=True,
  thesis_modified=False,quality_claim=False)
 out=folder/'saved_artifact_audit.json'
 with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps(result,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);main(p.parse_args().folder)
