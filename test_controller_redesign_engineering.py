"""Mechanics and marked synthetic integration tests; no research seed training."""
import copy,itertools,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import controller_redesign_v2 as m
import run_controller_redesign_engineering as r
import audit_controller_redesign_engineering as audit

FIXTURE_EVIDENCE={}

class Mechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):r.configure()
    def states(self,n=3):
        return torch.tensor(np.random.default_rng(99003).uniform(.01,.9,(n,533)),dtype=torch.float32)

    def test_parameter_counts_and_linear_macs(self):
        expected=dict(shared_congestion_cost=4052,shared_no_congestion=4052,untied_congestion_cost=53845,
            shared_joint_bound=4052,additive64_nonpositive=48736,additive128_unbounded=105504,emc_q_nonpositive=69361,bdq64_uniform_adapted=48801)
        for arm in r.ARMS:
            agent=r.build(arm,99001,r.fixture_cfg());self.assertEqual(sum(p.numel() for p in agent['online'].parameters()),expected[arm])
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in agent['target'].parameters()))
        model=m.SharedCostNetwork();macs=[]
        def count(module,inputs,outputs):macs.append(inputs[0].numel()*module.out_features)
        handles=[layer.register_forward_hook(count) for layer in model.modules() if isinstance(layer,torch.nn.Linear)]
        model.tables(self.states(1))
        for handle in handles:handle.remove()
        self.assertEqual(sum(macs),51696)

    def test_observation_parser_exact_indices(self):
        x=torch.arange(533,dtype=torch.float32)[None]
        local,g=m.split_observation(x)
        self.assertEqual(local.shape,(1,30,17));self.assertEqual(g.shape,(1,23))
        for i in range(30):
            torch.testing.assert_close(local[0,i],torch.cat((x[0,i*8:(i+1)*8],x[0,240+i:241+i],x[0,274+i*8:282+i*8])))
        torch.testing.assert_close(g[0],torch.cat((x[0,270:274],x[0,514:])))
        with self.assertRaises(ValueError):m.split_observation(torch.zeros(1,532))

    def test_workload_units_reconstructed_from_public_config(self):
        model=m.SharedCostNetwork();x=torch.zeros(1,533)
        x[0,6]=.8;x[0,270]=np.log1p(2.)
        u,w,L,beta,motion,base=model.tables(x)
        self.assertAlmostEqual(float(L[0,1]),2.,places=6)
        self.assertAlmostEqual(float(w[0,0,1]),400e6/10e9/.3,places=6)
        self.assertAlmostEqual(float(w[0,0,3]),400e6/20e9/.3,places=6)
        self.assertEqual(float(w[0,0,0]),0.)

    def test_shared_scores_actions_finite_and_bound(self):
        states=self.states()
        for arm in m.SHARED_ARMS:
            model=m.SharedCostNetwork(arm);actions,q,details=model.greedy(states)
            self.assertEqual(actions.shape,(3,32));self.assertTrue(torch.all((actions>=0)&(actions<5)))
            self.assertTrue(torch.isfinite(q).all() and torch.all(q<=0))
            torch.testing.assert_close(q,model(states,actions))
            if arm!='shared_no_congestion':
                self.assertEqual(details['costs'].shape,(3,7))
                self.assertTrue(torch.all(details['costs'][:,3:].min(1).values>=0) if arm!='shared_joint_bound' else True)
            with self.assertRaises(ValueError):model(states,torch.full((3,32),5,dtype=torch.long))

    def test_gradient_flow_and_intentional_zero_load_ablation(self):
        x=self.states(6);actions=(torch.arange(32)[None]+torch.arange(6)[:,None])%5
        for arm in m.SHARED_ARMS:
            model=m.SharedCostNetwork(arm);model(x,actions).sum().backward()
            for name,p in model.named_parameters():
                self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(float(model.global_head.weight.grad[0].abs().sum()),0.)
            load_grad=float(model.global_head.weight.grad[1:].abs().sum())
            if arm=='shared_no_congestion':self.assertEqual(load_grad,0.)
            else:self.assertGreater(load_grad,0.)

    def test_zero_congestion_has_exact_independent_minimum(self):
        model=m.SharedCostNetwork('shared_no_congestion');x=self.states();u,w,L,beta,motion,base=model.tables(x)
        actions,q,_=model.greedy(x)
        torch.testing.assert_close(q,-(base+u.min(2).values.sum(1)+motion.min(2).values.sum(1)))
        self.assertTrue(torch.equal(actions[:,:30],u.argmin(2)))

    def test_exact_marginals_and_retained_candidate_dominance(self):
        gen=torch.Generator().manual_seed(99004)
        u=torch.rand(2,4,5,generator=gen,dtype=torch.float64)
        w=torch.rand(2,4,5,generator=gen,dtype=torch.float64);w[:,:,0]=0
        L=torch.rand(2,5,generator=gen,dtype=torch.float64);L[:,0]=0
        beta=torch.rand(2,5,generator=gen,dtype=torch.float64);beta[:,0]=0
        motion=torch.zeros(2,2,5,dtype=torch.float64);base=torch.zeros(2,dtype=torch.float64)
        action,value,details=m.decode_cost_tables(u,w,L,beta,motion,base)
        self.assertTrue(torch.all(value<=details['costs'][:,:3].min(1).values+1e-10))
        for b in range(2):
            choices=[]
            for a in itertools.product(range(5),repeat=4):
                choices.append(float(m.full_cost(torch.tensor([a]),u[b:b+1],w[b:b+1],L[b:b+1],beta[b:b+1],base[b:b+1])))
            self.assertGreaterEqual(float(value[b])+1e-10,min(choices))
        selected=action[:,:4];p=m.marginal_prices(selected,w,L,beta)
        old_cost=m.full_cost(selected,u,w,L,beta,base)
        for i in range(4):
            for d in range(5):
                trial=selected.clone();trial[:,i]=d
                delta=u[:,i,d]-u[:,i].gather(1,selected[:,i,None]).squeeze(1)+p[:,i,d]-p[:,i].gather(1,selected[:,i,None]).squeeze(1)
                torch.testing.assert_close(m.full_cost(trial,u,w,L,beta,base)-old_cost,delta)

    def test_symmetry_oscillation_is_repaired(self):
        u=torch.tensor([[[10.,0.,0.,20.,20.],[10.,0.,0.,20.,20.]]])
        w=torch.ones_like(u);w[:,:,0]=0;L=torch.zeros(1,5);beta=torch.ones(1,5);beta[:,0]=0
        a,c,details=m.decode_cost_tables(u,w,L,beta,torch.zeros(1,2,5),torch.zeros(1))
        self.assertAlmostEqual(float(details['costs'][0,0]),.5)
        self.assertAlmostEqual(float(c[0]),.25);self.assertNotEqual(int(a[0,0]),int(a[0,1]))
        self.assertGreater(int(details['repairs_accepted'].sum()),0)

    def test_correct_outer_bound_and_separate_target_evaluation(self):
        class Fixed(torch.nn.Module):
            def __init__(self,value):super().__init__();self.value=value
            def greedy(self,x):return torch.ones(len(x),32,dtype=torch.long),torch.zeros(len(x)),{}
            def forward(self,x,a):
                if not torch.all(a==1):raise AssertionError('target used wrong action')
                return torch.full((len(x),),self.value)
        a=dict(arm='shared_joint_bound',online=Fixed(-3.),target=Fixed(-7.),cfg={'gamma':1.})
        x=torch.zeros(2,533);x[1]=float('nan')
        y=r.targets(a,torch.tensor([-2.,-5.]),x,torch.tensor([0.,1.]))
        torch.testing.assert_close(y,torch.tensor([-9.,-5.]));self.assertFalse(y.requires_grad)
        model=m.SharedCostNetwork('shared_joint_bound');states=self.states();act,q,_=model.greedy(states)
        u,w,L,beta,motion,base=model.tables(states)
        cost=m.full_cost(act[:,:30],u,w,L,beta,base+motion.gather(2,act[:,-2:,None]).squeeze(2).sum(1))
        torch.testing.assert_close(q,-F.softplus(cost))

    def test_all_terminal_targets_skip_both_models(self):
        class Bomb(torch.nn.Module):
            def forward(self,*args):raise AssertionError('terminal forward call')
            def greedy(self,*args):raise AssertionError('terminal selection')
        a=dict(arm='shared_congestion_cost',online=Bomb(),target=Bomb(),cfg={'gamma':1.})
        torch.testing.assert_close(r.targets(a,torch.tensor([-2.]),torch.full((1,533),float('nan')),torch.ones(1)),torch.tensor([-2.]))

    def test_bdq_trunk_gradient_rescale(self):
        model=m.AdaptedBDQ();clone=copy.deepcopy(model);x=self.states()
        model(x).sum().backward()
        h=clone.trunk(x);adv=clone.advantage(h).reshape(-1,32,5)
        q=clone.value(h)[:,:,None]+adv-adv.mean(2,keepdim=True);q.sum().backward()
        for p,c in zip(model.trunk.parameters(),clone.trunk.parameters()):torch.testing.assert_close(p.grad*33,c.grad,rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(model.value.weight.grad,clone.value.weight.grad)

    def test_scope_excludes_cloud_charging_and_security(self):
        result=audit.exclusion_check();self.assertEqual(result['cloud_nodes'],0);self.assertEqual(result['charging_stations'],0)
        self.assertFalse(result['security_trust_enabled']);self.assertEqual(len(result['execution_destinations']),5)

    def test_cli_blocks_research_even_with_confirmation(self):
        for mode in ['development','final']:
            for confirmed in [False,True]:
                with tempfile.TemporaryDirectory() as td:
                    out=Path(td)/'must_not_start'
                    command=[sys.executable,str(r.ROOT/'run_controller_redesign_engineering.py'),'--mode',mode,'--output',str(out)]
                    if confirmed:command.append('--confirm-'+mode+'-run')
                    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('disabled' if confirmed else '--confirm-'+mode+'-run',result.stderr)
                    self.assertFalse(out.exists())

    def test_cost_screen_does_not_hide_latency_failure(self):
        fake={arm:dict(parameter_bytes=194944,median_ms=.1,p95_ms=.2) for arm in r.ARMS}
        fake['shared_congestion_cost']=dict(parameter_bytes=16208,median_ms=.11,p95_ms=.3)
        screens=r.cost_screens(fake,r.spec());self.assertTrue(screens['online_storage']);self.assertFalse(screens['complete_action_median'])

    def test_own_trajectory_calibration_arithmetic(self):
        row=dict(chosen_q=[-2.,-1.],target_q=[-2.,-1.],realized_return_to_go=[-3.,-2.],slot_rewards=[-1.,-2.])
        v=r.values_summary([row]);self.assertAlmostEqual(v['normalized_rmse'],1/np.sqrt(6.5))
        self.assertEqual(v['absolute_bias'],1.);self.assertEqual(v['terminal_abs_error_p95'],1.)
        self.assertEqual(v['near_zero_large_return_fraction'],0.)

class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='SYNTHETIC_NOT_RESEARCH_controller_v2_')
        cls.folder=Path(cls.temp.name)/'fixture'
        try:
            r.run_fixture(cls.folder);cls.audit=audit.audit_fixture(cls.folder)
            FIXTURE_EVIDENCE.update(cls.audit)
        except Exception:
            cls.temp.cleanup();raise
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_end_to_end_eight_arm_fixture_and_audit(self):
        self.assertEqual(self.audit['state'],'passed_synthetic_saved_artifact_audit')
        self.assertEqual(self.audit['checkpoints_checked'],8);self.assertEqual(self.audit['optimizer_rows_checked'],264)
        self.assertEqual(self.audit['simulator_steps_added'],0)
        self.assertEqual(len(audit.read(self.folder/'comparisons_and_values.json')['comparisons']),10)

    def test_auditor_rejects_modified_artifact(self):
        p=self.folder/'shared_congestion_cost'/'optimizer.jsonl';raw=p.read_bytes()
        try:
            p.write_bytes(raw+b'\n')
            with self.assertRaises(AssertionError):audit.audit_fixture(self.folder)
        finally:p.write_bytes(raw)

    def test_auditor_rejects_unlisted_artifact(self):
        p=self.folder/'unlisted_checkpoint.pt';p.write_bytes(b'not research')
        try:
            with self.assertRaises(AssertionError):audit.audit_fixture(self.folder)
        finally:p.unlink()

if __name__=='__main__':unittest.main(verbosity=2)
