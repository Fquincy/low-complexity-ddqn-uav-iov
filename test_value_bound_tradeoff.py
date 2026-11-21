"""Focused checks for the frozen nonpositive-Q transform and paired runner."""
import copy,io,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np,torch
import run_value_bound_tradeoff as r
from emc_q import decode_tables
import audit_value_bound_development as auditor


class TinyEMC(torch.nn.Module):
 def __init__(self):
  super().__init__();self.unary=torch.nn.Parameter(torch.tensor([[[-2.,1.],[-1.,3.]]]))
  self.motion=torch.nn.Parameter(torch.tensor([[[0.,1.],[2.,-1.]]]))
  self.factors=torch.nn.Parameter(torch.tensor([[[[.2],[.5]],[[.7],[-.3]]]]))
  self.gates=torch.nn.Parameter(torch.tensor([[.8]]));self.movement_embeddings=torch.nn.Parameter(torch.tensor([[[.1],[.4]],[[.2],[-.5]]]))
 def tables(self,states):return self.unary.expand(len(states),-1,-1),self.motion.expand(len(states),-1,-1),self.factors.expand(len(states),-1,-1,-1),self.gates.expand(len(states),-1)
 def forward(self,states,actions):
  unary,motion,factors,gates=self.tables(states);b=torch.arange(len(states));v=torch.arange(2)
  d=actions[:,:2];i,j=actions[:,2],actions[:,3];ctx=(self.movement_embeddings[0,i]+self.movement_embeddings[1,j])*gates
  return motion[b,i,j]+(unary[b[:,None],v,d]+(factors[b[:,None],v,d]*ctx[:,None]).sum(-1)).sum(1)


class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);cls.spec=json.loads(r.SPEC.read_text());cls.cfg=r.settings('smoke',cls.spec)['cfg']
 def agent(self,arm):
  a=r.build(arm,3091,self.cfg);a['cfg']=self.cfg;return a
 def test_transform_bound_finiteness_and_monotonicity(self):
  raw=torch.tensor([-100.,-10.,-1.,0.,1.,10.,100.]);v=r.q_transform(raw,'emc_q_nonpositive')
  self.assertTrue(torch.isfinite(v).all());self.assertTrue(torch.all(v<=0));self.assertEqual(int(v.argmax()),int(raw.argmin()))
  torch.testing.assert_close(r.q_transform(raw,'emc_q_unbounded'),raw)
 def test_additive_bounded_policy_is_raw_argmin_per_head(self):
  class Heads(torch.nn.Module):
   def forward(self,x):return torch.tensor([[[4.,-2.,1.],[0.,5.,-1.]]]).expand(len(x),-1,-1)
  actions,value,raw=r.greedy(Heads(),torch.zeros(1,1),'additive64_nonpositive')
  self.assertEqual(actions.tolist(),[[1,2]]);self.assertEqual(float(raw),-3.);self.assertLessEqual(float(value),0.)
 def test_negated_emc_decoder_equals_exhaustive_raw_minimum(self):
  model=TinyEMC();state=torch.zeros(1,1);actions,value,raw=r.greedy(model,state,'emc_q_nonpositive')
  candidates=[]
  for d0 in range(2):
   for d1 in range(2):
    for i in range(2):
      for j in range(2):candidates.append((float(model(state,torch.tensor([[d0,d1,i,j]])).detach()),[d0,d1,i,j]))
  expected=min(candidates,key=lambda x:x[0]);self.assertEqual(actions[0].tolist(),expected[1]);self.assertAlmostEqual(float(raw.detach()),expected[0]);self.assertLessEqual(float(value.detach()),0.)
  u,m,f,g=model.tables(state);neg_action,_=decode_tables(-u,-m,-f,g,model.movement_embeddings)
  torch.testing.assert_close(neg_action,actions)
 def test_double_q_uses_raw_min_target_transform_and_skips_terminal(self):
  class Fixed(torch.nn.Module):
   def __init__(self,heads):super().__init__();self.heads=heads
   def forward(self,x):return self.heads.expand(len(x),-1,-1)
  online=Fixed(torch.tensor([[[2.,-3.],[4.,-5.]]]))
  target=Fixed(torch.tensor([[[1.,2.],[3.,4.]]]))
  agent=dict(arm='additive64_nonpositive',online=online,target=target)
  next_states=torch.zeros(2,533);next_states[1]=float('nan')
  out=r.targets(agent,torch.tensor([-2.,-5.]),next_states,torch.tensor([0.,1.]))
  self.assertFalse(out.requires_grad);self.assertTrue(torch.isfinite(out).all())
  self.assertAlmostEqual(float(out[0]),-2.-float(torch.nn.functional.softplus(torch.tensor(2.+4.))))
  self.assertEqual(float(out[1]),-5.)
 def test_five_arm_builds_have_frozen_sizes_and_checkpoint_roundtrip(self):
  expected=tuple(x['id'] for x in self.spec['arms']);self.assertEqual(expected,r.ARMS)
  for arm in r.ARMS:
   a=self.agent(arm);ck=r.checkpoint(a,['run_value_bound_tradeoff.py']);buf=io.BytesIO();torch.save(ck,buf);buf.seek(0)
   saved=torch.load(buf,weights_only=False);b=self.agent(arm)
   b['online'].load_state_dict(saved['online']);b['target'].load_state_dict(saved['target']);b['opt'].load_state_dict(saved['optimizer'])
   self.assertEqual(r.digest(a['online'].state_dict()),r.digest(b['online'].state_dict()))
   self.assertEqual(saved['training_config'],self.cfg)
   self.assertEqual(r.bounded(arm),arm in ('additive64_nonpositive','emc_q_nonpositive'))
 def test_training_update_counters_and_paired_exploration(self):
  agents=[self.agent(a) for a in r.ARMS];rng=np.random.default_rng(1903)
  for i in range(80):
   state=rng.normal(size=533).astype(np.float32);ns=rng.normal(size=533).astype(np.float32)
   outputs=[r.act(a,state,1.-.95*i/80,i<8) for a in agents]
   self.assertEqual(len({bool(x[1]) for x in outputs}),1)
   if outputs[0][1]:
    for x in outputs[1:]:np.testing.assert_array_equal(x[0],outputs[0][0])
   for a,(action,_) in zip(agents,outputs):
    a['replay'].add(state,action,-1.,ns,i%40==39);a['slots']+=1;r.train_update(a,self.cfg)
  self.assertEqual({a['updates'] for a in agents},{73});self.assertEqual({a['copies'] for a in agents},{9})
  self.assertEqual(len({r.digest(a['actor_history']) for a in agents}),1)
  for a in agents:self.assertTrue(all(p.grad is None and not p.requires_grad for p in a['target'].parameters()))
 def test_paired_environment_episode_and_evaluation_invariance(self):
  rows=[]
  for arm in r.ARMS:
   a=self.agent(arm);before=r.learner_snapshot(a);row=r.episode(a,24990000,episode_id='smoke')
   self.assertEqual(before,r.learner_snapshot(a));self.assertEqual(len(row['slots']),40)
   self.assertAlmostEqual(sum(row['slot_rewards']),row['return_sum']);rows.append(row)
   if r.bounded(arm):self.assertLessEqual(row['terminal_value'],0.)
  self.assertEqual(len({row['workload_sha256'] for row in rows}),1)
 def test_development_is_not_launchable_in_this_gate(self):
  out=Path('validation/development_must_not_launch')
  with self.assertRaisesRegex(ValueError,'--confirm-development-run'):
   r.run('development',out,False)
  self.assertFalse(Path('validation/development_must_not_launch').exists())
  with tempfile.TemporaryDirectory() as td:
   cli_out=Path(td)/'must_not_launch'
   result=subprocess.run([sys.executable,str(r.ROOT/'run_value_bound_tradeoff.py'),'--mode','development','--output',str(cli_out)],
    capture_output=True,text=True,timeout=20)
   self.assertNotEqual(result.returncode,0);self.assertIn('--confirm-development-run',result.stderr)
   self.assertFalse(cli_out.exists())
 def test_development_settings_match_frozen_budget_without_launch(self):
  full=r.settings('development',self.spec)
  self.assertEqual(full['replicates'],[0,1,2]);self.assertEqual(full['model_seeds'],[3109,3119,3121])
  self.assertEqual(full['train_starts'],[24100000,24200000,24300000])
  self.assertEqual(full['cfg']['episodes']*40*len(r.ARMS)*3,120000)
  self.assertEqual((full['cfg']['episodes']*40-full['cfg']['warmup_physical_records']+1)*len(r.ARMS)*3,104655)
 def test_optimizer_diagnostics_are_summarized_by_value_and_clipping(self):
  import tempfile
  rows=[]
  for i in range(3):
   base=dict(count=4,mean=float(i),std=.5,minimum=-1.,p05=-.8,median=float(i),p95=float(i+1),maximum=2.,positive_fraction=.5)
   rows.append(dict(chosen_q=base,bellman_target=base,td_residual=base,gradient_clipped=i!=0,grad_norm_before_clip=float(i+1),physical_slots=1024+i,loss=float(i+1)))
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'optimizer.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
   result=r.summarize_optimizer_log(p)
  self.assertEqual(result['updates'],3);self.assertAlmostEqual(result['clipped_update_fraction'],2/3)
  self.assertAlmostEqual(result['chosen_q']['mean'],1.)
 def test_all_predeclared_screens_are_evaluated_per_replicate(self):
  fresh={};heuristics={name:[] for name in ('all_local','nearest_rsu','uav_selective')};diagnostics={}
  for rep in range(3):
   for arm in r.ARMS:
    key=f'{arm}_{rep}';fresh[key]=[]
    diagnostics[key]=dict(fresh_terminal_abs_error_p95=1.,fresh_terminal_positive_fraction=0.,fixed_probe_positive_greedy_fraction=0.,
     learning_values_finite=True,online_parameter_bytes=277444 if arm.startswith('emc') else 194944 if '64' in arm else 422016,
     target_parameter_bytes=277444 if arm.startswith('emc') else 194944 if '64' in arm else 422016,
     replay_array_bytes=1234,optimizer_tensor_bytes=4321,median_action_ms=.1,p95_action_ms=.2,
     value_stability={'chosen_q':{},'bellman_target':{},'td_residual':{}})
   for i in range(48):
    row=dict(return_sum=-100.+i,deadline_success=.7,seed=24890000+i,workload_sha256=f'{i}')
    for arm in r.ARMS:fresh[f'{arm}_{rep}'].append(dict(row))
  for i in range(48):
   row=dict(return_sum=-100.+i,deadline_success=.7,seed=24890000+i,workload_sha256=f'{i}')
   for name in heuristics:heuristics[name].append(dict(row))
  analysis=r.analyze_development(self.spec,{},fresh,heuristics,diagnostics)
  self.assertEqual(set(analysis['replicates']),{'0','1','2'})
  self.assertIn('versus_emc_q_unbounded',analysis['replicates']['0']['screens'])
  self.assertIn('value_diagnostics',analysis['replicates']['0'])
 def test_development_auditor_helpers_accept_and_reject_synthetic_fixtures(self):
  settings=r.settings('development',self.spec);keys={f'{arm}_{rep}' for arm in r.ARMS for rep in range(3)}
  screens={'versus_additive128_unbounded':True,'versus_additive64_nonpositive':True,'versus_emc_q_unbounded':True,
   'versus_nearest_rsu':True,'versus_uav_selective':True,'fresh_terminal_calibration':True,'all_bounded_probe_signs':True,
   'all_bounded_fresh_terminal_signs':True,'finite_learning_values':True,'online_storage_ratio':True,'median_latency_ratio':True,'p95_latency_ceiling':True}
  analysis=dict(state='one_or_more_predeclared_screens_failed',replicates={str(rep):dict(screens=screens,
   value_diagnostics={arm:{} for arm in r.ARMS},costs={arm:{} for arm in r.ARMS}) for rep in range(3)})
  verification=dict(checkpoint_sha256={key:'hash' for key in keys},counters={key:dict(slots=8000,updates=6977) for key in keys},
   fresh={key:[{} for _ in range(48)] for key in keys})
  self.assertEqual(auditor.check_summary_structure(self.spec,settings,verification,analysis),keys)
  good={arm:['a','b','c'] for arm in r.ARMS};auditor.assert_paired_fingerprints(good)
  bad=dict(good);bad[r.ARMS[-1]]=['x','b','c']
  with self.assertRaises(AssertionError):auditor.assert_paired_fingerprints(bad)
 def test_end_to_end_synthetic_development_orchestration_and_audit(self):
  fixture=copy.deepcopy(self.spec);fixture['run_kind']='synthetic_fixture_not_research_evidence'
  fixture['replicates']=1;fixture['model_seeds']=[39071]
  fixture['training']['episodes_per_model']=1;fixture['training']['slots_per_model']=40
  fixture['training']['optimizer_updates_per_model']=37;fixture['training']['warmup_physical_records']=4
  fixture['training']['batch_size']=4;fixture['training']['replay_capacity_physical_records']=64
  fixture['training']['target_copy_every_updates']=2;fixture['training']['target_copies_per_model']=18
  fixture['training']['epsilon_decay_slots']=40
  fixture['seeds']['train_blocks']=[dict(replicate=0,start=97000001,count=1)]
  fixture['seeds']['scheduled_validation']=[97010001]
  fixture['seeds']['timing_probe']=[97020001]
  fixture['seeds']['fresh_development_gate']=[97030001,97030002]
  fixture['diagnostic_episodes']=[0,1]
  fixture['uncertainty']['bootstrap_draws']=20;fixture['uncertainty']['rng_seed']=97040001
  fixture['run_order_rng_seed']=97040002;fixture['cost_measurement']['path_order_rng_seed']=97040003
  reserved=set(self.spec['model_seeds'])|set(self.spec['seeds']['scheduled_validation'])|set(self.spec['seeds']['timing_probe'])|set(self.spec['seeds']['fresh_development_gate'])
  for block in self.spec['seeds']['train_blocks']:reserved.update(range(block['start'],block['start']+block['count']))
  for key in ('train_seeds','validation_seeds','probe_seeds','fresh_seeds'):reserved.update(self.spec['smoke'][key])
  fixture_seeds={39071,97000001,97010001,97020001,97030001,97030002,97040001,97040002,97040003}
  self.assertFalse(fixture_seeds&reserved)
  with tempfile.TemporaryDirectory(prefix='value_bound_fixture_') as td:
   output=Path(td)/'synthetic_development'
   result=r.execute_run('development',output,fixture,synthetic_fixture=True)
   self.assertEqual(result['run_kind'],'synthetic_fixture_not_research_evidence')
   self.assertEqual((sum(x['slots'] for x in result['counters'].values()),sum(x['updates'] for x in result['counters'].values())),(200,185))
   audited=auditor.main(output)
   self.assertEqual(audited['run_kind'],'synthetic_fixture_not_research_evidence')
   self.assertEqual((audited['training_slots'],audited['optimizer_rows_checked']),(200,185))
   self.assertEqual(audited['episodes_checked'],dict(training=5,validation=10,fresh_model=10,fresh_heuristic=6,probe=1))
   self.assertTrue((output/'FRESH_GATE_FREEZE.json').is_file())
   self.assertTrue((output/'development_analysis.json').is_file())
   self.assertTrue((output/'saved_artifact_audit.json').is_file())


if __name__=='__main__':
 unittest.main(verbosity=2)
