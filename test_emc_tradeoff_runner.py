"""Synthetic mechanics and one paired physical episode; no pilot launch."""
import argparse,copy,io,json,unittest
from pathlib import Path
import numpy as np,torch
import run_emc_tradeoff as r


class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);cls.spec=json.loads(r.SPEC.read_text());cls.cfg=r.settings('smoke',cls.spec)['cfg']
 def agent(self,arm):
  a=r.build(arm,2399,self.cfg);a['cfg']=self.cfg;return a
 def test_smoke_build_config_and_checkpoint_roundtrip(self):
  for arm in r.ARMS:
   a=self.agent(arm);self.assertEqual(a['replay'].capacity,64)
   ck=r.checkpoint(a,['run_emc_tradeoff.py']);buffer=io.BytesIO();torch.save(ck,buffer);buffer.seek(0)
   saved=torch.load(buffer,weights_only=False);b=self.agent(arm)
   b['online'].load_state_dict(saved['online']);b['target'].load_state_dict(saved['target']);b['opt'].load_state_dict(saved['optimizer'])
   self.assertEqual(r.digest(a['online'].state_dict()),r.digest(b['online'].state_dict()))
   self.assertEqual(saved['training_config'],self.cfg)
 def test_all_arm_update_counters_and_no_target_grad(self):
  for arm in r.ARMS:
   a=self.agent(arm);rng=np.random.default_rng(50)
   for i in range(80):
    s=rng.normal(size=533).astype(np.float32);ns=rng.normal(size=533).astype(np.float32)
    a['replay'].add(s,rng.integers(0,5,32),-1.,ns,i%40==39);a['slots']+=1;u=r.train_update(a,self.cfg)
    self.assertEqual(a['updates'],max(0,i+1-7))
   self.assertEqual(a['copies'],9)
   self.assertTrue(all(p.grad is None and not p.requires_grad for p in a['target'].parameters()))
 def test_paired_exploration_and_separate_replay(self):
  agents=[self.agent(arm) for arm in r.ARMS]
  for i in range(80):
   values=[r.act(a,np.zeros(533,dtype=np.float32),1-.95*i/80,i<8) for a in agents]
   self.assertEqual(len({v[1] for v in values}),1)
   if values[0][1]:
    for v in values[1:]:np.testing.assert_array_equal(v[0],values[0][0])
   for j,a in enumerate(agents):a['replay_rng'].integers(0,64,4*(j+1))
  self.assertEqual(len({r.digest(a['action_rng'].bit_generator.state) for a in agents}),1)
 def test_target_terminal_skip_online_selection_and_detachment(self):
  class Raw(torch.nn.Module):
   def __init__(self,target):super().__init__();self.target=target
   def forward(self,x):
    assert not torch.isnan(x).any()
    return torch.tensor([9.,3.,2.,1.,0.] if self.target else [0.,7.,3.,2.,1.]).repeat(len(x),32,1)
  a=dict(arm=r.ARMS[0],online=Raw(False),target=Raw(True))
  ns=torch.zeros(2,533,requires_grad=True)
  with torch.no_grad():ns[1]=float('nan')
  y=r.targets(a,torch.tensor([-2.,-5.]),ns,torch.tensor([0.,1.]))
  np.testing.assert_array_equal(y.numpy(),[94.,-5.]);self.assertFalse(y.requires_grad)
 def test_eval_invariance_and_task_pairing(self):
  rows=[]
  for arm in r.ARMS:
   a=self.agent(arm);before=r.learner_snapshot(a)
   row=r.episode(a,17910000,episode_id='unit')
   self.assertEqual(before,r.learner_snapshot(a))
   self.assertEqual(sum(row['slot_rewards']),row['return_sum'])
   self.assertEqual(row['deadline_success'],row['tasks_on_time']/row['tasks_generated'])
   rows.append(row)
  self.assertEqual(len({x['workload_sha256'] for x in rows}),1)
 def test_replay_ring_storage(self):
  replay=r.Replay(3)
  for i in range(5):replay.add(np.full(533,i),np.full(32,i%5),-i,np.full(533,i+1),False)
  self.assertEqual(replay.size,3);self.assertEqual(replay.position,2)
  np.testing.assert_array_equal(replay.states[:,0],[3,4,2])
 def test_development_budget_and_paired_bootstrap(self):
  full=r.settings('development',self.spec)
  self.assertEqual(full['replicates'],[0,1,2]);self.assertEqual(len(full['model_seeds']),3)
  self.assertEqual(full['cfg']['episodes']*40*9,72000)
  left=[dict(return_sum=float(i),deadline_success=i/10) for i in range(4)]
  right=[dict(return_sum=float(i-2),deadline_success=i/10-.1) for i in range(4)]
  a=r.paired_bootstrap(left,right,2000,77);b=r.paired_bootstrap(left,right,2000,77)
  self.assertEqual(a,b);self.assertEqual(a['return_sum']['mean_difference'],2.)
  self.assertEqual(a['deadline_fraction']['mean_difference'],.1)
 def test_development_launch_is_gated(self):
  with self.assertRaisesRegex(ValueError,'--confirm-development-run'):
   r.run('development',Path('validation/development_must_not_launch'))
  self.assertFalse(Path('validation/development_must_not_launch').exists())
 def test_development_screen_uses_each_seed_and_each_comparator(self):
  fresh={};heuristics={'nearest_rsu':[],'uav_selective':[]};diagnostics={}
  for rep in range(3):
   for arm in r.ARMS:
    fresh[f'{arm}_{rep}']=[];diagnostics[f'{arm}_{rep}']=dict(fresh_terminal_abs_error_p95=1.,fixed_probe_positive_greedy_fraction=0.,
      median_action_ms=.1 if arm==r.ARMS[0] else .2,online_parameter_bytes=422016 if arm==r.ARMS[0] else 277444,p95_action_ms=.5)
   emc=fresh[f'{r.ARMS[2]}_{rep}'];a128=fresh[f'{r.ARMS[0]}_{rep}'];a64=fresh[f'{r.ARMS[1]}_{rep}']
   for i in range(48):
    base=100.+i
    a128.append(dict(return_sum=base,deadline_success=.70))
    a64.append(dict(return_sum=base-3,deadline_success=.72))
    emc.append(dict(return_sum=base,deadline_success=.74))
  for i in range(48):
   heuristics['nearest_rsu'].append(dict(return_sum=100.+i,deadline_success=.71))
   heuristics['uav_selective'].append(dict(return_sum=100.+i,deadline_success=.70))
  good=r.analyze_development(self.spec,{},fresh,heuristics,diagnostics)
  self.assertEqual(good['state'],'all_three_emc_initializations_pass')
  fresh[f'{r.ARMS[2]}_1'][-1]['return_sum']-=200
  seed_failure=r.analyze_development(self.spec,{},fresh,heuristics,diagnostics)
  self.assertFalse(seed_failure['replicates']['1']['all_screens_passed'])
  fresh,heuristics,diagnostics=self._synthetic_data()
  for row in fresh[f'{r.ARMS[2]}_2']:row['deadline_success']=.70
  failed=r.analyze_development(self.spec,{},fresh,heuristics,diagnostics)
  self.assertEqual(failed['state'],'one_or_more_predeclared_screens_failed')
  self.assertFalse(failed['replicates']['2']['screens']['versus_additive64'])

 def _synthetic_data(self):
  fresh={};heuristics={name:[] for name in ('nearest_rsu','uav_selective')};diagnostics={}
  for rep in range(3):
   for arm in r.ARMS:
    fresh[f'{arm}_{rep}']=[];diagnostics[f'{arm}_{rep}']=dict(fresh_terminal_abs_error_p95=1.,fixed_probe_positive_greedy_fraction=0.,median_action_ms=.1 if arm==r.ARMS[0] else .2,online_parameter_bytes=422016 if arm==r.ARMS[0] else 277444,p95_action_ms=.5)
   for i in range(48):
    base=100.+i;fresh[f'{r.ARMS[0]}_{rep}'].append(dict(return_sum=base,deadline_success=.70));fresh[f'{r.ARMS[1]}_{rep}'].append(dict(return_sum=base-3,deadline_success=.72));fresh[f'{r.ARMS[2]}_{rep}'].append(dict(return_sum=base,deadline_success=.74))
  for i in range(48):
   heuristics['nearest_rsu'].append(dict(return_sum=100.+i,deadline_success=.71));heuristics['uav_selective'].append(dict(return_sum=100.+i,deadline_success=.70))
  return fresh,heuristics,diagnostics


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 out=args.output;out.mkdir(parents=True,exist_ok=False)
 stream=io.StringIO();res=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 (out/'tests.txt').write_text(stream.getvalue(),encoding='utf-8')
 (out/'verification.json').write_text(json.dumps(dict(state='passed' if res.wasSuccessful() else 'failed',tests_run=res.testsRun,runner_sha256=r.sha(Path(r.__file__)),synthetic_optimizer_updates=219,physical_evaluation_steps=120,pilot_launched=False),indent=2),encoding='utf-8')
 print(stream.getvalue())
 if not res.wasSuccessful():raise SystemExit(1)
