"""Focused update-oracle and decision-boundary tests; no simulator episodes."""
import copy,json,unittest
import numpy as np
import torch
import run_deadline_screen_research as r


class Decisions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):r.configure();cls.p=r.protocol()

    def test_actual_update_matches_independent_ddqn_and_loss_oracle(self):
        state=np.load(r.ROOT/self.p['action_cost']['common_inputs'])['observations'][0].copy()
        state[-1]=0.;state[6:240:8]=1.;state[7:240:8]=np.float32(.25/.3)
        terminal=state.copy();terminal[-1]=1.
        # This controlled batch includes two terminal transitions. No simulator seeds consumed.
        for arm in ('screened_price24','unmasked_price24','screened_bdq64'):
            definition=next(x for x in self.p['learned_arms'] if x['id']==arm)
            cfg=r.plan(self.p,'integration-fixture')['cfg'];a=r.build(definition,99921,cfg);a['slots']=8
            current=np.stack([state]*4);current[[1,3],-1]=np.float32(39/40)
            following=np.stack([state,terminal,state,terminal]);actions=np.ones((4,32),np.int64)
            rewards=np.array([-1.,-2.,-3.,-4.],np.float32);dones=np.array([0.,1.,0.,1.],np.float32)
            for i in range(4):a['replay'].add(current[i],actions[i],rewards[i],following[i],dones[i])
            # Fixed sampler gives an exact oracle batch, independent of random duplicate draws.
            a['replay'].sample=lambda n,rng:(current,actions,rewards,following,dones,np.arange(4))
            # Target parameters intentionally differ from online parameters.
            with torch.no_grad():
                for i,x in enumerate(a['target'].parameters()):x.add_(.01*(i+1))
            def forbidden(*args,**kwargs):raise AssertionError('target greedy must never select DDQN action')
            a['target'].greedy=forbidden
            x=torch.from_numpy(current);acts=torch.from_numpy(actions);ns=torch.from_numpy(following)
            with torch.no_grad():
                next_act=a['online'].greedy(ns[[0,2]])[0]
                expected=rewards.copy();expected[[0,2]]+=a['target'](ns[[0,2]],next_act).numpy()
                q=a['online'](x,acts)
                if arm=='screened_bdq64':branch=a['online'].model(x).gather(2,acts[:,:,None]).squeeze(2);uniform=torch.nn.functional.smooth_l1_loss(branch,torch.from_numpy(expected)[:,None].expand_as(branch))
                else:uniform=torch.nn.functional.smooth_l1_loss(q,torch.from_numpy(expected))
                # Reproduce only the independent anchor draw to build expected loss.
                arng=np.random.default_rng(99921+300000);ix=arng.choice([1,3],32,replace=True) if a['online'].anchored else []
                anchor=torch.nn.functional.smooth_l1_loss(a['online'](x[ix],acts[ix]),torch.from_numpy(rewards[ix])) if len(ix) else torch.tensor(0.)
                expected_loss=float(uniform+anchor)
            entry=r.update(a)
            np.testing.assert_allclose(entry['bellman_targets'],expected,rtol=1e-6,atol=1e-6)
            np.testing.assert_array_equal(np.array(entry['next_actions'])[[0,2]],next_act.numpy())
            self.assertTrue(np.all(np.array(entry['next_actions'])[[1,3]]==-1))
            self.assertAlmostEqual(entry['loss'],expected_loss,places=5)

    def fixtures(self):
        ev={}
        def row(ret,deadline,seed):
            rewards=[ret/40]*40;g=np.cumsum(rewards[::-1])[::-1].tolist()
            return dict(seed=seed,workload_sha256=str(seed),return_sum=ret,deadline_success=deadline,slot_rewards=rewards,
                chosen_q=g,target_q=g,realized_return_to_go=g,tasks=[dict(destination=0,on_time=True)],slots=[dict(excluded_local=[False]*30) for _ in range(40)])
        for d in self.p['learned_arms']:
            primary=d['id']==self.p['primary']
            ev[d['id']]=[[row(-10. if primary else -13.,.82 if primary else .80,seed) for seed in (1,2)] for _ in range(5)]
        heur={x['id']:[row(-13.,.80,seed) for seed in (1,2)] for x in self.p['heuristics']}
        cost={d['id']:[dict(parameter_bytes=d['expected_fp32_parameter_bytes'],median_ms=.1 if d['id']==self.p['primary'] else .2,p95_ms=.3) for _ in range(5)] for d in self.p['learned_arms']}
        return ev,heur,cost

    def test_one_failed_initialization_cannot_hide_in_average(self):
        ev,h,c=self.fixtures();baseline=r.analysis.summarize(self.p,ev,h,c)
        self.assertTrue(baseline['all_scientific_screens_passed'])
        for x in ev[self.p['primary']][4]:x['return_sum']=-16.
        bad=r.analysis.summarize(self.p,ev,h,c)
        self.assertGreater(bad['comparisons']['unmasked_wide128']['mean_return_difference'],-2.)
        self.assertFalse(bad['computed_screens']['references_every_fit']);self.assertFalse(bad['all_scientific_screens_passed'])

    def test_cost_ceiling_and_fixture_quality_label(self):
        ev,h,c=self.fixtures();c[self.p['primary']][4]['p95_ms']=1.000001
        bad=r.analysis.summarize(self.p,ev,h,c)
        self.assertFalse(bad['computed_screens']['primary_cost_every_fit']);self.assertFalse(bad['all_scientific_screens_passed'])
        fixture=r.analysis.summarize(self.p,ev,h,c,True)
        self.assertFalse(fixture['screens_applied']);self.assertFalse(fixture['quality_evidence']);self.assertIsNone(fixture['all_scientific_screens_passed'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Decisions))
    path=r.ROOT/'validation/deadline_screen_research_engineering_2026-11-14_01/decision_tests.json'
    if path.exists():raise SystemExit('Refusing to overwrite test evidence')
    r.write(path,dict(passed=result.wasSuccessful(),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),new_simulator_steps=0,new_optimizer_updates=3))
    raise SystemExit(0 if result.wasSuccessful() else 1)
