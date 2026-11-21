"""Predeclared five-arm nonpositive-Q runner; smoke is the only launch mode."""
import argparse,copy,csv,gzip,hashlib,json,shutil,time,traceback
import tempfile
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
import sys
sys.path.insert(0,str(Path(__file__).parent/'Source'))
from chapter3_iov_env_v2 import Chapter3IoVEnv
from train_chapter3_ddqn import FactorizedQNetwork,to_action_dict
from validate_chapter3_model import select_destinations
from emc_q import EMCQ,decode_tables

ROOT=Path(__file__).resolve().parent
SPEC=ROOT/'LOW_COMPLEXITY_VALUE_BOUND_PROTOCOL_2026-11-04.json'
ARMS=('additive128_unbounded','additive64_unbounded','emc_q_unbounded','additive64_nonpositive','emc_q_nonpositive')
EMC_ARMS={'emc_q_unbounded','emc_q_nonpositive'}
BOUNDED_ARMS={'additive64_nonpositive','emc_q_nonpositive'}

def bounded(arm):return arm in BOUNDED_ARMS

def q_transform(raw,arm):return -F.softplus(raw) if bounded(arm) else raw

def greedy(model,states,arm):
 """Return exact greedy actions and their transformed joint value."""
 if arm in EMC_ARMS:
  if bounded(arm):
   unary,motion,factors,gates=model.tables(states)
   # Negated action tables let the existing exact max decoder minimize raw Q.
   actions,_=decode_tables(-unary,-motion,-factors,gates,model.movement_embeddings)
   raw=model(states,actions)
  else:actions,raw=model.greedy(states)
 else:
  heads=model(states);actions=heads.argmin(2) if bounded(arm) else heads.argmax(2)
  raw=heads.gather(2,actions[:,:,None]).squeeze(2).sum(1)
 return actions,q_transform(raw,arm),raw

def chosen_value(model,states,actions,arm):
 raw=model(states,actions) if arm in EMC_ARMS else model(states).gather(2,actions[:,:,None]).squeeze(2).sum(1)
 return q_transform(raw,arm),raw

def distribution(values):
 arr=values.detach().cpu().numpy().astype(np.float64,copy=False).reshape(-1)
 if not np.isfinite(arr).all():raise FloatingPointError('nonfinite diagnostic values')
 return dict(count=int(arr.size),mean=float(arr.mean()),std=float(arr.std()),minimum=float(arr.min()),
  p05=float(np.quantile(arr,.05)),median=float(np.quantile(arr,.5)),p95=float(np.quantile(arr,.95)),
  maximum=float(arr.max()),positive_fraction=float((arr>0).mean()))

def summarize_optimizer_log(path):
 rows=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
 if not rows:raise ValueError('optimizer trace is empty')
 keys=('chosen_q','bellman_target','td_residual')
 result={}
 for key in keys:
  names=('mean','std','minimum','p05','median','p95','maximum','positive_fraction')
  result[key]={name:float(np.mean([row[key][name] for row in rows])) for name in names}
 by_episode={}
 for row in rows:
  episode=(row['physical_slots']-1)//40+1;by_episode.setdefault(episode,[]).append(row)
 trend=[]
 for episode,items in sorted(by_episode.items()):
  trend.append(dict(episode=episode,updates=len(items),loss_mean=float(np.mean([x['loss'] for x in items])),
   chosen_q_mean=float(np.mean([x['chosen_q']['mean'] for x in items])),bellman_target_mean=float(np.mean([x['bellman_target']['mean'] for x in items])),
   td_residual_mean=float(np.mean([x['td_residual']['mean'] for x in items])),
   clipped_update_fraction=float(np.mean([x['gradient_clipped'] for x in items])),
   gradient_norm_preclip_p95=float(np.quantile([x['grad_norm_before_clip'] for x in items],.95))))
 result.update(aggregation_scope='mean_of_optimizer_minibatch_statistics',updates=len(rows),by_training_episode=trend,clipped_update_fraction=float(np.mean([row['gradient_clipped'] for row in rows])),
  gradient_norm_preclip_mean=float(np.mean([row['grad_norm_before_clip'] for row in rows])),
  gradient_norm_preclip_p95=float(np.quantile([row['grad_norm_before_clip'] for row in rows],.95)),
  maximum_gradient_norm_preclip=float(max(row['grad_norm_before_clip'] for row in rows)))
 return result

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):
 h=hashlib.sha256()
 def walk(v):
  if isinstance(v,torch.Tensor):v=v.detach().cpu().numpy()
  if isinstance(v,np.ndarray):h.update(str((v.dtype.str,v.shape)).encode());h.update(v.tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=str):walk(k);walk(v[k])
  elif isinstance(v,(list,tuple)):
   for z in v:walk(z)
  else:h.update(repr(v).encode())
  h.update(b'\0')
 walk(x);return h.hexdigest()

class Replay:
 def __init__(self,capacity):
  self.capacity=capacity;self.size=0;self.position=0
  self.states=np.empty((capacity,533),np.float32);self.next_states=np.empty_like(self.states)
  self.actions=np.empty((capacity,32),np.int8);self.rewards=np.empty(capacity,np.float32);self.dones=np.empty(capacity,np.float32)
 def add(self,s,a,r,ns,d):
  i=self.position;self.states[i]=s;self.next_states[i]=ns;self.actions[i]=a;self.rewards[i]=r;self.dones[i]=d
  self.position=(i+1)%self.capacity;self.size=min(self.size+1,self.capacity)
 def sample(self,n,rng):
  ix=rng.integers(0,self.size,n)
  return self.states[ix],self.actions[ix].astype(np.int64),self.rewards[ix],self.next_states[ix],self.dones[ix],ix
 @property
 def storage_bytes(self):return sum(x.nbytes for x in (self.states,self.next_states,self.actions,self.rewards,self.dones))

def build(arm,model_seed,training):
 if arm not in ARMS:raise ValueError('unknown model arm')
 torch.manual_seed(model_seed)
 model=EMCQ() if arm in EMC_ARMS else FactorizedQNetwork(533,32,5,(128,128) if arm=='additive128_unbounded' else (64,64))
 target=copy.deepcopy(model).eval();target.requires_grad_(False)
 return dict(arm=arm,seed=model_seed,online=model,target=target,opt=torch.optim.Adam(model.parameters(),lr=training['learning_rate']),
  replay=Replay(training['replay_capacity_physical_records']),action_rng=np.random.default_rng(model_seed+training['action_rng_seed_offset']),
  replay_rng=np.random.default_rng(model_seed+training['replay_rng_seed_offset']),slots=0,updates=0,copies=0,actor_history=[])

def act(agent,state,epsilon,warmup):
 explore=bool(warmup or agent['action_rng'].random()<epsilon)
 if explore:action=agent['action_rng'].integers(0,5,32,dtype=np.int64)
 else:
  with torch.inference_mode():action=greedy(agent['online'],torch.tensor(state)[None],agent['arm'])[0][0].numpy().copy()
 agent['actor_history'].append(dict(epsilon=float(epsilon),warmup=bool(warmup),explore=explore))
 return action,explore

@torch.no_grad()
def targets(agent,rewards,next_states,dones):
 if not torch.all((dones==0)|(dones==1)):raise ValueError('binary done mask required')
 y=rewards.detach().clone();active=~dones.bool()
 if active.any():
  following=next_states[active]
  actions=greedy(agent['online'],following,agent['arm'])[0]
  value=chosen_value(agent['target'],following,actions,agent['arm'])[0]
  y[active]+=value
 return y

def train_update(agent,cfg):
 if agent['slots']<cfg['warmup_physical_records']:return None
 s,a,r,ns,d,ix=agent['replay'].sample(cfg['batch_size'],agent['replay_rng'])
 s=torch.tensor(s);a=torch.tensor(a);r=torch.tensor(r);ns=torch.tensor(ns);d=torch.tensor(d)
 q=chosen_value(agent['online'],s,a,agent['arm'])[0]
 y=targets(agent,r,ns,d)
 if y.requires_grad or not torch.isfinite(q).all() or not torch.isfinite(y).all():raise FloatingPointError('invalid online Q or detached target')
 residual=q.detach()-y
 loss=F.smooth_l1_loss(q,y,beta=1.)
 agent['opt'].zero_grad(set_to_none=True);loss.backward()
 grad=torch.nn.utils.clip_grad_norm_(agent['online'].parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
 if agent['target'].training or any(p.requires_grad or p.grad is not None for p in agent['target'].parameters()):raise AssertionError('target gradient/grad state')
 agent['opt'].step();agent['updates']+=1
 if any(not torch.isfinite(p).all() for p in agent['online'].parameters()):raise FloatingPointError('nonfinite parameters')
 copied=agent['updates']%cfg['target_copy_every_updates']==0
 if copied:agent['target'].load_state_dict(agent['online'].state_dict());agent['copies']+=1
 return dict(step=agent['updates'],physical_slots=agent['slots'],loss=float(loss.detach()),grad_norm_before_clip=float(grad),
  gradient_clipped=bool(float(grad)>cfg['gradient_clip']),chosen_q=distribution(q),bellman_target=distribution(y),td_residual=distribution(residual),
  replay_indices=ix.tolist(),target_copy=copied,reward_samples=len(ix),terminal_samples=int(d.sum()))

def workload(env):
 return [{k:t[k] for k in ('task_id','vehicle_id','arrival_s','input_bits','required_cycles','deadline_s')}
         for t in sorted(env.current_tasks,key=lambda x:x['vehicle_id'])]

def episode(agent,seed,training=False,policy=None,episode_id=0,trace=None,collect_states=False):
 env=Chapter3IoVEnv(seed=seed);slot_rewards=[];all_work=[];slot_rows=[];all_states=[];next_states=[];terminal_value=None;uav_before=[];uav_after=[]
 for slot in range(40):
  state=env._observation();w=workload(env);before=env.uav_positions.copy()
  if collect_states or training:all_states.append(state.copy())
  if policy:
   action=np.concatenate((select_destinations(env,policy),np.zeros(2,dtype=np.int64)))
  elif training:
   cfg=agent['cfg'];eps=1. if agent['slots']<cfg['warmup_physical_records'] else 1-.95*min(agent['slots']/cfg['epsilon_decay_slots'],1.)
   action,_=act(agent,state,eps,agent['slots']<cfg['warmup_physical_records'])
  else:
   with torch.inference_mode():
    tensor=torch.tensor(state)[None];action,value,_=greedy(agent['online'],tensor,agent['arm']);action=action[0].numpy().copy()
    if slot==39:terminal_value=float(value[0])
  ns,reward,done,info=env.step(to_action_dict(action,30));after=env.uav_positions.copy()
  if not np.isfinite(reward) or not np.isfinite(ns).all() or bool(done)!=(slot==39):raise AssertionError('physical transition invalid')
  if terminal_value is not None and not np.isfinite(terminal_value):raise FloatingPointError('nonfinite evaluation terminal Q')
  if training:next_states.append(ns.copy())
  if training:
   agent['replay'].add(state,action,reward,ns,done);agent['slots']+=1;u=train_update(agent,agent['cfg'])
   if trace is not None and u is not None:trace.write(json.dumps(u)+'\n');trace.flush()
  slot_rewards.append(float(reward));all_work.append(w);uav_before.append(before);uav_after.append(after)
  slot_rows.append(dict(slot=slot,action=action.tolist(),reward=float(reward),done=bool(done),
    reward_components=info['reward_components'],energy_j=info['energy_j'],tasks_arrived=w,
    uav_before=before.tolist(),uav_after=after.tolist(),actual_distance_m=np.linalg.norm(after-before,axis=1).tolist()))
 assert done and slot==39
 summary=env.episode_summary();records=env.export_task_records()
 return dict(seed=seed,episode=episode_id,return_sum=float(sum(slot_rewards)),deadline_success=summary['deadline_success_rate'],
  tasks_generated=summary['tasks_generated'],tasks_on_time=summary['tasks_on_time'],slot_rewards=slot_rewards,
  terminal_value=terminal_value,workload_sha256=digest(all_work),summary=summary,slots=slot_rows,tasks=records,
  observations=np.stack(all_states) if collect_states or training else None,
  next_observations=np.stack(next_states) if training else None)

def learner_snapshot(a):
 return digest((a['online'].state_dict(),a['target'].state_dict(),a['opt'].state_dict(),a['action_rng'].bit_generator.state,
  torch.get_rng_state(),a['online'].training,a['target'].training,
  a['replay_rng'].bit_generator.state,a['slots'],a['updates'],a['copies'],a['actor_history'],
  a['replay'].states[:a['replay'].size],a['replay'].next_states[:a['replay'].size],a['replay'].actions[:a['replay'].size],
  a['replay'].rewards[:a['replay'].size],a['replay'].dones[:a['replay'].size],a['replay'].size,a['replay'].position))

def save_episode(path,row):
 with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(row,f,allow_nan=False)

def compact_episode(row):
 return {k:row[k] for k in ('seed','episode','return_sum','deadline_success','tasks_generated','tasks_on_time','workload_sha256','terminal_value','slot_rewards')}

def evaluate_to_file(agent,seeds,path,episode_id,policy=None):
 """Stream full records and retain small per-episode analysis inputs only."""
 summaries=[]
 with gzip.open(path,'wt',encoding='utf-8') as f:
  f.write('[')
  for i,seed in enumerate(seeds):
   row=episode(agent,seed,False,policy=policy,episode_id=episode_id)
   if i:f.write(',')
   json.dump(row,f,allow_nan=False);summaries.append(compact_episode(row))
  f.write(']')
 return summaries

def checkpoint(a,names,spec_sha256=None):
 return dict(arm=a['arm'],model_seed=a['seed'],online=a['online'].state_dict(),target=a['target'].state_dict(),optimizer=a['opt'].state_dict(),
  action_rng=a['action_rng'].bit_generator.state,replay_rng=a['replay_rng'].bit_generator.state,slots=a['slots'],updates=a['updates'],copies=a['copies'],
  replay={n:getattr(a['replay'],n)[:a['replay'].size].copy() for n in ('states','next_states','actions','rewards','dones')},replay_size=a['replay'].size,replay_position=a['replay'].position,
  actor_history=a['actor_history'],torch_rng=torch.get_rng_state(),source_sha256={n:sha(ROOT/n) for n in names},spec_sha256=spec_sha256 or sha(SPEC),training_config=a['cfg'])

def settings(mode,s):
 if mode=='smoke':
  x=s['smoke'];t=copy.deepcopy(s['training']);t.update(episodes=x['episodes_per_arm'],warmup_physical_records=x['warmup'],batch_size=x['batch'],replay_capacity_physical_records=x['capacity'],target_copy_every_updates=x['target_interval'],epsilon_decay_slots=x['epsilon_decay_slots'])
  return dict(cfg=t,replicates=[0],model_seeds=[x['model_seed']],train_starts=[x['train_seeds'][0]],validation=x['validation_seeds'],probes=x['probe_seeds'],fresh=x['fresh_seeds'],diagnostics=x['diagnostic_episodes'],order_seed=s['run_order_rng_seed'])
 if mode=='development':
  blocks=sorted(s['seeds']['train_blocks'],key=lambda x:x['replicate']);t=copy.deepcopy(s['training']);t['episodes']=t['episodes_per_model']
  return dict(cfg=t,replicates=list(range(s['replicates'])),model_seeds=s['model_seeds'],
   train_starts=[block['start'] for block in blocks],validation=s['seeds']['scheduled_validation'],
   probes=s['seeds']['timing_probe'],fresh=s['seeds']['fresh_development_gate'],
   diagnostics=s['diagnostic_episodes'],order_seed=s['run_order_rng_seed'])
 raise ValueError(f'unknown run mode: {mode}')

def paired_bootstrap(left,right,draws,seed=None,rng=None):
 if len(left)!=len(right) or not left:raise ValueError('paired outcome arrays must have equal nonzero lengths')
 if rng is None:rng=np.random.default_rng(seed)
 ix=rng.integers(0,len(left),size=(draws,len(left)))
 def estimate(a,b):
  delta=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)
  sampled=delta[ix].mean(axis=1)
  return dict(mean_difference=float(delta.mean()),percentile_95=[float(x) for x in np.quantile(sampled,[.025,.975])])
 return dict(episodes=len(left),unit='paired_episode_within_fitted_model',draws=draws,
  return_sum=estimate([x['return_sum'] for x in left],[x['return_sum'] for x in right]),
  deadline_fraction=estimate([x['deadline_success'] for x in left],[x['deadline_success'] for x in right]))

def analyze_development(spec,agents,fresh,heuristics,diagnostics):
 """Apply every frozen point-estimate screen per fitted initialization."""
 q=spec['quality_screen'];cost=spec['cost_screen'];unc=spec['uncertainty'];by_rep={};all_gates=True
 bootstrap_rng=np.random.default_rng(unc['rng_seed'])
 for rep in range(spec['replicates']):
  primary=fresh[f'emc_q_nonpositive_{rep}'];m=diagnostics[f'emc_q_nonpositive_{rep}']
  comparators={
   'additive128_unbounded':fresh[f'additive128_unbounded_{rep}'],
   'additive64_nonpositive':fresh[f'additive64_nonpositive_{rep}'],
   'emc_q_unbounded':fresh[f'emc_q_unbounded_{rep}'],
   'nearest_rsu':heuristics['nearest_rsu'],'uav_selective':heuristics['uav_selective']}
  comparisons={name:paired_bootstrap(primary,rows,unc['bootstrap_draws'],rng=bootstrap_rng) for name,rows in comparators.items()}
  screens={}
  def screen(name,threshold):
   delta=comparisons[name];return (delta['return_sum']['mean_difference']>=threshold['mean_episode_return_difference_at_least'] and
    delta['deadline_fraction']['mean_difference']>=threshold['deadline_fraction_difference_at_least'])
  screens['versus_additive128_unbounded']=screen('additive128_unbounded',q['versus_additive128_unbounded'])
  screens['versus_additive64_nonpositive']=screen('additive64_nonpositive',q['versus_additive64_nonpositive'])
  screens['versus_emc_q_unbounded']=screen('emc_q_unbounded',q['versus_emc_q_unbounded'])
  for name in ('nearest_rsu','uav_selective'):
   screens[f'versus_{name}']=screen(name,q['versus_each_heuristic'])
  screens['fresh_terminal_calibration']=m['fresh_terminal_abs_error_p95']<=q['calibration']['fresh_final_slot_abs_error_p95_at_most']
  screens['all_bounded_probe_signs']=all(diagnostics[f'{name}_{rep}']['fixed_probe_positive_greedy_fraction']<=q['calibration']['primary_greedy_q_positive_fraction'] for name in BOUNDED_ARMS)
  screens['all_bounded_fresh_terminal_signs']=all(diagnostics[f'{name}_{rep}']['fresh_terminal_positive_fraction']==0.0 for name in BOUNDED_ARMS)
  screens['finite_learning_values']=m['learning_values_finite']
  base=diagnostics[f'additive128_unbounded_{rep}']
  screens['online_storage_ratio']=m['online_parameter_bytes']<=cost['online_parameter_bytes_at_most_ratio_additive128']*base['online_parameter_bytes']
  screens['median_latency_ratio']=m['median_action_ms']<=cost['complete_action_median_ratio_vs_additive128_at_most']*base['median_action_ms']
  screens['p95_latency_ceiling']=m['p95_action_ms']<=cost['complete_action_p95_ms_at_most']
  passed=all(screens.values());all_gates &= passed
  by_rep[str(rep)]=dict(model_seed=spec['model_seeds'][rep],episodes=len(primary),comparisons=comparisons,screens=screens,
   value_diagnostics={name:diagnostics[f'{name}_{rep}']['value_stability'] for name in ARMS},
   costs={name:{k:diagnostics[f'{name}_{rep}'][k] for k in ('online_parameter_bytes','target_parameter_bytes','replay_array_bytes','optimizer_tensor_bytes','median_action_ms','p95_action_ms')} for name in ARMS},all_screens_passed=passed)
 return dict(state='all_three_primary_initializations_pass' if all_gates else 'one_or_more_predeclared_screens_failed',
  interpretation='point_estimate_engineering_screen_not_formal_noninferiority_or_significance',
  bootstrap_rng_seed=unc['rng_seed'],bootstrap_draws=unc['bootstrap_draws'],
  bootstrap_unit='episode_within_each_fitted_model_pair; no pooling across model seeds',
  training_seed_uncertainty_claim=False,replicates=by_rep)

def run(mode,out,confirm_development=False):
 if mode=='development' and not confirm_development:
  raise ValueError('Development launch requires --confirm-development-run after reviewing the development protocol and readiness report')
 if mode not in ('smoke','development'):raise ValueError(f'unknown run mode: {mode}')
 return execute_run(mode,out)

def execute_run(mode,out,spec_override=None,synthetic_fixture=False):
 base_spec=json.loads(SPEC.read_text());spec=copy.deepcopy(spec_override) if spec_override is not None else base_spec
 if synthetic_fixture:
  temp=Path(tempfile.gettempdir()).resolve();out=Path(out).resolve()
  if not out.is_relative_to(temp) or spec_override is None or spec.get('run_kind')!='synthetic_fixture_not_research_evidence':
   raise ValueError('Synthetic fixture runs require a marked reduced spec and output under the system temporary directory')
 else:
  if spec_override is not None:raise ValueError('Protocol overrides are allowed only in the guarded synthetic fixture path')
  if mode=='development' and tuple(x['id'] for x in spec['arms'])!=ARMS:raise ValueError('development protocol arm mismatch')
 active_spec_path=Path(out)/'runtime_protocol.json'
 valid=json.loads((ROOT/'validation/value_bound_protocol_2026-11-04_01/verification.json').read_text())
 assert valid['state']=='passed_predeclaration_checks_only' and sha(SPEC)==valid['protocol_sha256']
 for p,h in base_spec['source_sha256'].items():assert sha(ROOT/p)==h
 assert tuple(x['id'] for x in spec['arms'])==ARMS
 if mode=='smoke':assert spec['launch_authorized'] is False
 if not synthetic_fixture:
  for p,h in spec['source_sha256'].items():assert sha(ROOT/p)==h
 cfg=settings(mode,spec);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();agents={};diagnostics={};invariance=[];order_rng=np.random.default_rng(cfg['order_seed'])
 active_spec_path.write_text(json.dumps(spec,indent=2),encoding='utf-8')
 active_spec_hash=sha(active_spec_path)
 def status(state,**kw):
  d=dict(state=state,mode=mode,updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),**kw)
  (out/'status.json').write_text(json.dumps(d,indent=2),encoding='utf-8');(out/'LIVE_STATUS.txt').write_text(json.dumps(d,indent=2),encoding='utf-8')
 try:
  source=out/'source_snapshot';source.mkdir()
  names=['run_value_bound_tradeoff.py','test_value_bound_tradeoff.py','audit_value_bound_smoke.py','audit_value_bound_development.py','emc_q.py','Source/chapter3_iov_env.py','Source/chapter3_iov_env_v2.py','Source/train_chapter3_ddqn.py','Source/validate_chapter3_model.py',SPEC.name]
  for n in names:shutil.copy2(ROOT/n,source/Path(n).name)
  probe_states=[];probe_generation_seconds=0.
  for i,seed in enumerate(cfg['probes']):
   phase=time.perf_counter()
   row=episode(None,seed,policy=spec['heuristics'][i%3],collect_states=True)
   probe_generation_seconds+=time.perf_counter()-phase
   probe_states.extend(row.pop('observations'));save_episode(out/f'fixed_probe_{seed}.json.gz',row)
  common=np.stack(probe_states)[np.linspace(0,len(probe_states)-1,128,dtype=int)]
  np.savez_compressed(out/'common_probe_states.npz',observations=common)
  order=[]
  for rep in cfg['replicates']:
   seq=list(ARMS);order_rng.shuffle(seq);order.append(seq)
   for arm in seq:
    model_seed=cfg['model_seeds'][rep];key=f'{arm}_{rep}';a=build(arm,model_seed,cfg['cfg']);a['cfg']=cfg['cfg'];agents[key]=a;d=out/key;d.mkdir();history=[];train_seconds=0.;eval_seconds=0.
    validation_rows={}
    with (d/'optimizer.jsonl').open('w') as trace:
     for ep in range(cfg['cfg']['episodes']+1):
      if ep in cfg['diagnostics']:
       before=learner_snapshot(a);status('scheduled_validation',run=key,episode=ep,slots=a['slots'],updates=a['updates'])
       phase=time.perf_counter()
       if mode=='development':evalrows=evaluate_to_file(a,cfg['validation'],d/f'validation_ep{ep}.json.gz',ep)
       else:evalrows=[episode(a,seed,False,episode_id=ep) for seed in cfg['validation']]
       eval_seconds+=time.perf_counter()-phase
       after=learner_snapshot(a);assert before==after
       invariance.append(dict(run=key,phase='scheduled_validation',episode=ep,before=before,after=after))
       validation_rows[str(ep)]=evalrows
       (d/'validation.json').write_text(json.dumps(validation_rows,indent=2,allow_nan=False),encoding='utf-8')
      if ep==cfg['cfg']['episodes']:break
      status('training',run=key,episode=ep+1,slots=a['slots'],updates=a['updates'])
      phase=time.perf_counter();row=episode(a,cfg['train_starts'][rep]+ep,True,episode_id=ep+1,trace=trace);train_seconds+=time.perf_counter()-phase
      np.savez_compressed(d/f'train_ep{ep+1}.npz',states=row.pop('observations'),next_states=row.pop('next_observations'),actions=np.array([x['action'] for x in row['slots']]),rewards=np.array(row['slot_rewards']),dones=np.array([x['done'] for x in row['slots']]))
      save_episode(d/f'train_ep{ep+1}.json.gz',row)
      history.append(compact_episode(row) if mode=='development' else row)
      (d/'training_summary.json').write_text(json.dumps([{k:r[k] for k in ('seed','episode','return_sum','deadline_success','tasks_generated','tasks_on_time','workload_sha256')} for r in history],indent=2),encoding='utf-8')
      status('training_episode_complete',run=key,episode=ep+1,slots=a['slots'],updates=a['updates'])
    expected=cfg['cfg']['episodes']*40-cfg['cfg']['warmup_physical_records']+1
    assert a['slots']==cfg['cfg']['episodes']*40 and a['updates']==expected
    assert a['copies']==expected//cfg['cfg']['target_copy_every_updates']
    ck=checkpoint(a,names,active_spec_hash);ck['replicate']=rep
    torch.save(ck,d/'final.pt');agents[key]['history_summary']=history
    replay_storage_bytes=a['replay'].storage_bytes
    checkpoint_replay_bytes=sum(value.nbytes for value in ck['replay'].values())
    diagnostics[key]=dict(slots=a['slots'],updates=a['updates'],target_copies=a['copies'],training_and_validation_wall_seconds=train_seconds+eval_seconds,
      action_rng_final=a['action_rng'].bit_generator.state,replay_array_bytes=replay_storage_bytes,checkpoint_replay_bytes=checkpoint_replay_bytes)
    if mode=='development':
     diagnostics[key]['value_stability']=summarize_optimizer_log(d/'optimizer.jsonl')
     # Preserve replay in the frozen checkpoint without retaining its unused
     # 50,000-record allocation for every fitted model during evaluation.
     a['replay']=Replay(0)
    (d/'counters.json').write_text(json.dumps(diagnostics[key],indent=2),encoding='utf-8')
  for rep in cfg['replicates']:
   triplet=[agents[f'{arm}_{rep}'] for arm in ARMS]
   assert len({digest(a['actor_history']) for a in triplet})==1
   assert len({digest(a['action_rng'].bit_generator.state) for a in triplet})==1
   assert len({tuple(r['workload_sha256'] for r in a['history_summary']) for a in triplet})==1
  frozen={k:sha(out/k/'final.pt') for k in agents};(out/'FRESH_GATE_FREEZE.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
  assert set(frozen)=={f'{arm}_{rep}' for arm in ARMS for rep in cfg['replicates']}
  heuristic={};shared_eval_seconds=0.
  for p in spec['heuristics']:
   phase=time.perf_counter()
   if mode=='development':heuristic[p]=evaluate_to_file(None,cfg['fresh'],out/f'fresh_heuristic_{p}.json.gz','fresh',p)
   else:heuristic[p]=[episode(None,seed,False,policy=p) for seed in cfg['fresh']]
   shared_eval_seconds+=time.perf_counter()-phase
  (out/'fresh_heuristics.json').write_text(json.dumps(heuristic,allow_nan=False),encoding='utf-8')
  fresh={}
  for key,a in agents.items():
   before=learner_snapshot(a);phase=time.perf_counter()
   if mode=='development':rows=evaluate_to_file(a,cfg['fresh'],out/key/'fresh.json.gz','fresh')
   else:rows=[episode(a,seed,False,episode_id='fresh') for seed in cfg['fresh']]
   diagnostics[key]['fresh_evaluation_wall_seconds']=time.perf_counter()-phase
   after=learner_snapshot(a);assert before==after and sha(out/key/'final.pt')==frozen[key]
   invariance.append(dict(run=key,phase='fresh_evaluation',before=before,after=after))
   fresh[key]=rows
   if mode=='smoke':
    with gzip.open(out/key/'fresh.json.gz','wt',encoding='utf-8') as f:json.dump(rows,f,allow_nan=False)
   for rep in cfg['replicates']:
    fingerprints={key:[r['workload_sha256'] for r in rows] for key,rows in fresh.items() if key.endswith(f'_{rep}')}
    fingerprints.update({p:[r['workload_sha256'] for r in rows] for p,rows in heuristic.items()})
    assert len({tuple(x) for x in fingerprints.values()})==1
  timing_before={key:learner_snapshot(a) for key,a in agents.items()};timing_seconds_start=time.perf_counter()
  timing=[];timing_rng=np.random.default_rng(spec['cost_measurement']['path_order_rng_seed'])
  for round_id in range(5):
   seq=list(agents);timing_rng.shuffle(seq)
   for key in seq:
    a=agents[key]
    def selection(state):
     with torch.inference_mode():
      tensor=torch.tensor(state)[None]
      vector=greedy(a['online'],tensor,a['arm'])[0][0].numpy().copy()
     return to_action_dict(vector,30)
    for i in range(32):selection(common[i])
    for i,state in enumerate(common):
     t=time.perf_counter_ns();selection(state);ns=time.perf_counter_ns()-t
     timing.append(dict(round=round_id,run=key,observation=i,nanoseconds=ns))
  (out/'timings.json').write_text(json.dumps(timing),encoding='utf-8')
  timing_seconds=time.perf_counter()-timing_seconds_start
  for key,a in agents.items():
   after=learner_snapshot(a);assert timing_before[key]==after
   invariance.append(dict(run=key,phase='timing',before=timing_before[key],after=after))
   assert sha(out/key/'final.pt')==frozen[key]
   before=learner_snapshot(a)
   with torch.inference_mode():
    tensor=torch.tensor(common)
    probe_actions,values,_=greedy(a['online'],tensor,a['arm'])
    target_probe_values,_=chosen_value(a['target'],tensor,probe_actions,a['arm'])
   after=learner_snapshot(a);assert before==after and torch.isfinite(values).all()
   invariance.append(dict(run=key,phase='fixed_probe',before=before,after=after))
   ns=np.array([r['nanoseconds'] for r in timing if r['run']==key])/1e6
   terminal_values=np.asarray([r['terminal_value'] for r in fresh[key]],dtype=np.float64)
   terminal_errors=np.abs(terminal_values-np.asarray([r['slot_rewards'][-1] for r in fresh[key]],dtype=np.float64))
   diagnostics[key].update(fresh_terminal_abs_error_p95=float(np.quantile(terminal_errors,.95)),
    fresh_terminal_positive_fraction=float((terminal_values>0).mean()),fixed_probe_positive_greedy_fraction=float((values>0).float().mean()),
    fixed_probe_online_q_distribution=distribution(values),fixed_probe_target_q_distribution=distribution(target_probe_values),
    fresh_terminal_q_distribution=distribution(torch.tensor(terminal_values)),fresh_terminal_abs_error_distribution=distribution(torch.tensor(terminal_errors)),
    median_action_ms=float(np.median(ns)),p95_action_ms=float(np.quantile(ns,.95)),
    online_parameter_bytes=sum(p.numel()*p.element_size() for p in a['online'].parameters()),target_parameter_bytes=sum(p.numel()*p.element_size() for p in a['target'].parameters()),
    replay_array_bytes=diagnostics[key]['replay_array_bytes'],checkpoint_replay_bytes=diagnostics[key]['checkpoint_replay_bytes'],
    optimizer_tensor_bytes=sum(v.numel()*v.element_size() for st in a['opt'].state.values() for v in st.values() if isinstance(v,torch.Tensor)))
   if mode=='development':
    for p in a['online'].parameters():
     if not torch.isfinite(p).all():raise FloatingPointError(f'nonfinite online parameter in {key}')
    for p in a['target'].parameters():
     if not torch.isfinite(p).all():raise FloatingPointError(f'nonfinite target parameter in {key}')
    diagnostics[key]['learning_values_finite']=bool(torch.isfinite(values).all() and torch.isfinite(target_probe_values).all() and np.isfinite(terminal_values).all())
    if not diagnostics[key]['learning_values_finite']:raise FloatingPointError(f'nonfinite evaluated values in {key}')
  for key in diagnostics:
   (out/key/'counters.json').write_text(json.dumps(diagnostics[key],indent=2,allow_nan=False),encoding='utf-8')
  (out/'evaluation_invariance.json').write_text(json.dumps(invariance,indent=2),encoding='utf-8')
  frozen_after={k:sha(out/k/'final.pt') for k in agents};assert frozen_after==frozen
  for key in frozen:assert sha(out/key/'final.pt')==frozen[key]
  screens=None
  if mode=='development':
   screens=analyze_development(spec,agents,fresh,heuristic,diagnostics)
   (out/'development_analysis.json').write_text(json.dumps(screens,indent=2,allow_nan=False),encoding='utf-8')
  result=dict(state=('passed_smoke' if mode=='smoke' else 'development_screen_passed' if screens['state']=='all_three_primary_initializations_pass' else 'development_screen_failed'),mode=mode,order=order,
   counters=diagnostics,checkpoint_sha256=frozen,
   fresh={k:[{n:r[n] for n in ('seed','return_sum','deadline_success','tasks_generated','tasks_on_time','workload_sha256')} for r in v] for k,v in fresh.items()},
   paired_workloads=True,evaluation_state_invariance=True,all_checkpoints_frozen_before_fresh=True,
   quality_screen_applied=mode=='development',new_full_quality_claim=False,run_kind='synthetic_fixture_not_research_evidence' if synthetic_fixture else 'research_protocol_run',
   runtime_protocol_sha256=active_spec_hash,wall_seconds=time.perf_counter()-start,
   shared_wall_seconds=dict(timing=timing_seconds,probe_generation=probe_generation_seconds,heuristic_evaluation=shared_eval_seconds))
  (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');status('complete',updates=sum(a['updates'] for a in agents.values()),slots=sum(a['slots'] for a in agents.values()))
  manifest={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('run_manifest.json','LIVE_STATUS.txt','status.json')}
  (out/'run_manifest.json').write_text(json.dumps(dict(files=manifest),indent=2),encoding='utf-8')
  print(json.dumps({'state':result['state'],'slots':sum(a['slots'] for a in agents.values()),'updates':sum(a['updates'] for a in agents.values()),'fresh':result['fresh']},indent=2))
  return result
 except BaseException as e:
  (out/'failure_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
  for key,a in agents.items():
   try:torch.save(checkpoint(a,names),out/key/'partial_failure.pt')
   except BaseException as save_error:(out/f'{key}_partial_save_error.txt').write_text(repr(save_error),encoding='utf-8')
  status('failed_no_retry',error=repr(e));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','development'],required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--confirm-development-run',action='store_true');args=p.parse_args();run(args.mode,args.output.resolve(),args.confirm_development_run)
