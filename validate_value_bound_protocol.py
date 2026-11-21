"""Validate frozen seed separation, arithmetic and value-transform mechanics only."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from emc_q import decode_tables

ROOT=Path(__file__).resolve().parent
SPEC=ROOT/'LOW_COMPLEXITY_VALUE_BOUND_PROTOCOL_2026-11-04.json'
OUT=ROOT/'validation/value_bound_protocol_2026-11-04_01'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def seeds(spec):
 groups={f"train_{b['replicate']}":set(range(b['start'],b['start']+b['count'])) for b in spec['seeds']['train_blocks']}
 groups.update({k:set(v) for k,v in spec['seeds'].items() if k!='train_blocks'})
 smoke=spec['smoke']
 groups.update({f"smoke_{k}":set(smoke[k]) for k in ('train_seeds','validation_seeds','probe_seeds','fresh_seeds')})
 for name in ('model_seeds',):groups[name]=set(spec[name])
 for left,a in groups.items():
  for right,b in groups.items():
   if left!=right:assert not a&b,(left,right,a&b)
 proposed=set.union(*groups.values());scanned=[]
 other=ROOT.parent/'October 31/Working/Project';paths=set()
 for folder in (ROOT,ROOT/'Source',other):
  for pattern in ('*SPEC*.json','*PLAN*.json'):paths.update(folder.glob(pattern))
 def collect(obj,context=False):
  found=set()
  if isinstance(obj,dict):
   if context and isinstance(obj.get('start'),int) and isinstance(obj.get('count'),int):
    start,count=obj['start'],obj['count']
    if 1_000_000<=start<100_000_000 and 0<count<=100_000:found.update(range(start,start+count))
   for k,value in obj.items():found.update(collect(value,context or 'seed' in k.lower() or 'task_blocks' in k.lower()))
  elif isinstance(obj,list):
   for value in obj:found.update(collect(value,context))
  elif context and isinstance(obj,int) and 1_000_000<=obj<100_000_000:found.add(obj)
  return found
 for path in sorted(paths):
  if path==SPEC:continue
  values=collect(json.loads(path.read_text(encoding='utf-8-sig')))
  assert not proposed&values,(str(path),proposed&values)
  scanned.append(str(path))
 return groups,scanned

def transform_checks():
 raw=torch.tensor([-100.,-10.,-1.,0.,1.,10.,100.],dtype=torch.float64)
 q=-torch.nn.functional.softplus(raw)
 assert torch.isfinite(q).all() and torch.all(q<=0)
 # The negative-softplus transform is decreasing; its argmax is raw argmin.
 assert int(q.argmax())==int(raw.argmin())
 rng=np.random.default_rng(4109);b,v,d,rank=2,3,3,2
 unary=torch.tensor(rng.normal(size=(b,v,d)),dtype=torch.float64)
 motion=torch.tensor(rng.normal(size=(b,2,2)),dtype=torch.float64)
 factors=torch.tensor(rng.normal(size=(b,v,d,rank)),dtype=torch.float64)
 gates=torch.tensor(rng.normal(size=(b,rank)),dtype=torch.float64)
 embeddings=torch.tensor(rng.normal(size=(2,2,rank)),dtype=torch.float64)
 positive_actions,positive_values=decode_tables(unary,motion,factors,gates,embeddings)
 negative_actions,negative_raw=decode_tables(-unary,-motion,-factors,gates,embeddings)
 # Exhaustive action-space check verifies the transformed decoder is exact.
 for batch in range(b):
  values=[]
  for a0 in range(2):
   for a1 in range(2):
    for d0 in range(d):
     for d1 in range(d):
      for d2 in range(d):
       actions=torch.tensor([[d0,d1,d2,a0,a1]],dtype=torch.long)
       value=motion[batch,a0,a1]
       context=(embeddings[0,a0]+embeddings[1,a1])*gates[batch]
       for vehicle,dest in enumerate((d0,d1,d2)):
        value=value+unary[batch,vehicle,dest]+(factors[batch,vehicle,dest]*context).sum()
       values.append((float(value),actions[0].tolist()))
  min_raw,expected=min(values,key=lambda x:x[0])
  assert negative_actions[batch].tolist()==expected
  assert abs(float(negative_raw[batch])-(-min_raw))<1e-10
 # The original exact decoder is still max-raw; only the bounded variant must min-raw.
 assert torch.isfinite(positive_values).all()
 return dict(nonpositive_finite_grid=True,negative_transform_action_is_raw_argmin=True,
  exact_negated_table_decoder_matches_exhaustive=True,enumerated_joint_actions_per_state=2*2*d**v)

def main():
 spec=json.loads(SPEC.read_text());groups,scanned=seeds(spec);t=spec['training'];n=len(spec['arms']);sm=spec['smoke']
 updates=spec['training']['episodes_per_model']*spec['episode_slots']-t['warmup_physical_records']+1
 assert n==5 and spec['replicates']==3
 assert updates==6977 and updates//t['target_copy_every_updates']==34
 assert n*3*8000==120000 and n*3*updates==104655
 assert n*sm['expected_slots_per_arm']==sm['total_training_slots']==400
 assert n*sm['expected_updates_per_arm']==sm['total_optimizer_updates']==365
 assert sm['expected_updates_per_arm']//sm['target_interval']==sm['expected_target_copies_per_arm']==9
 assert spec['quality_screen']['primary_arm']=='emc_q_nonpositive'
 assert spec['quality_screen']['all3_primary_initializations_must_pass']
 assert spec['value_bound']['additional_parameters']==0
 for name,h in spec['source_sha256'].items():assert sha(ROOT/name)==h
 mechanics=transform_checks();OUT.mkdir(parents=True,exist_ok=False)
 result=dict(state='passed_predeclaration_checks_only',protocol_sha256=sha(SPEC),validator_sha256=sha(Path(__file__)),
  arms=n,replicates=3,group_sizes={k:len(v) for k,v in groups.items()},registered_protocols_scanned=scanned,
  training_slots=120000,optimizer_updates=104655,target_copies=510,smoke_slots=400,smoke_updates=365,
  transform_mechanics=mechanics,source_hashes_checked=True,optimizer_steps=0,simulator_steps=0,
  runner_implemented=False,development_launch_authorized=False)
 (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()
