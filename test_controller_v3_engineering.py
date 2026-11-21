"""V3 deterministic synthetic mechanics tests; no optimizer steps/rollouts."""
import copy,itertools,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import controller_v3_amortized as m
import run_controller_v3_engineering as g
import audit_controller_v3_engineering as audit

class Mechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):g.r.configure()
    def state(self,count=4,n=30):
        return torch.tensor(np.random.default_rng(99025).uniform(.01,.9,(count,17*n+23)),dtype=torch.float32)

    def test_parameter_macs_and_buffer_accounting(self):
        for arm,expected in zip(m.NEW_ARMS,[17066,17066,16841]):
            model=m.AmortizedPriceQ(arm);self.assertEqual(sum(v.numel() for v in model.parameters()),expected)
            self.assertEqual(g.r.parameter_bytes(model),4*expected)
            self.assertEqual(sum(b.numel()*b.element_size() for b in model.buffers()),20)
        model=m.AmortizedPriceQ();macs=[]
        handles=[v.register_forward_hook(lambda mod,inputs,out:macs.append(inputs[0].numel()*mod.out_features)) for v in model.modules() if isinstance(v,torch.nn.Linear)]
        model.tables(self.state(1))
        for h in handles:h.remove()
        self.assertEqual(sum(macs),16872)

    def test_exact_units_and_observation_offsets(self):
        model=m.AmortizedPriceQ();state=torch.zeros(1,533);state[0,6]=.8;state[0,270]=np.log1p(2.)
        _,_,_,d=model.tables(state)
        self.assertAlmostEqual(float(d['queue'][0,1]),2.,places=6)
        self.assertEqual(float(d['work'][0,0,0]),0.)
        self.assertAlmostEqual(float(d['work'][0,0,1]),400e6/10e9/.3,places=6)
        self.assertAlmostEqual(float(d['work'][0,0,3]),400e6/20e9/.3,places=6)
        self.assertTrue(torch.equal(d['work'][0,1:],torch.zeros(29,5)))
        self.assertTrue(torch.all(d['fractions']>=0));torch.testing.assert_close(d['fractions'].sum(1),torch.ones(1))

    def test_price_equation_matches_independent_loops(self):
        rng=np.random.default_rng(99025);w=torch.tensor(rng.uniform(0,1,(2,3,5)),dtype=torch.float64);w[:,:,0]=0
        L=torch.tensor(rng.uniform(0,1,(2,5)),dtype=torch.float64);L[:,0]=0
        beta=L.clone();pi=torch.softmax(L,1);actual=m.marginal_table(w,L,beta,pi)
        for b,i,d in itertools.product(range(2),range(3),range(5)):
            other=sum(float(w[b,j,d]) for j in range(3) if j!=i)
            expected=beta[b,d]*(2*(L[b,d]+pi[b,d]*other)*w[b,i,d]+w[b,i,d]**2)/12
            self.assertAlmostEqual(float(actual[b,i,d]),float(expected),places=12)

    def test_exhaustive_exact_decoder_all_three_new_critics(self):
        for arm in m.NEW_ARMS:
            torch.manual_seed(99021);model=m.AmortizedPriceQ(arm,n=2);x=self.state(1,n=2)
            action,q,_=model.greedy(x)
            candidates=torch.tensor(list(itertools.product(range(5),repeat=4)))
            with torch.inference_mode():scores=model(x.expand(len(candidates),-1),candidates)
            self.assertAlmostEqual(float(q[0]),float(scores.max()),places=6)
            self.assertTrue(torch.equal(action[0],candidates[scores.argmax()]))

    def test_ties_and_known_loss_of_actual_assignment_coordination(self):
        u=torch.tensor([[[10.,0.,0.,20.,20.],[10.,0.,0.,20.,20.]]]);w=torch.ones_like(u);w[:,:,0]=0
        L=torch.zeros(1,5);beta=torch.tensor([[0.,1.,1.,1.,1.]]);pi=torch.tensor([[0.,.5,.5,0.,0.]])
        a,_=m.decode(u+m.marginal_table(w,L,beta,pi),torch.zeros(1,2,5),torch.zeros(1))
        self.assertTrue(torch.equal(a,torch.tensor([[1,1,0,0]])))
        from controller_redesign_v2 import full_cost
        together=full_cost(a[:,:2],u,w,L,beta,torch.zeros(1))
        split=full_cost(torch.tensor([[1,2]]),u,w,L,beta,torch.zeros(1))
        self.assertAlmostEqual(float(together),.5);self.assertAlmostEqual(float(split),.25)

    def test_values_actions_and_nonnegative_tables(self):
        x=self.state()
        for arm in m.NEW_ARMS:
            model=m.AmortizedPriceQ(arm);a,q,_=model.greedy(x)
            self.assertEqual(tuple(a.shape),(4,32));self.assertTrue(torch.all((a>=0)&(a<5)))
            self.assertTrue(torch.isfinite(q).all() and torch.all(q<=0))
            torch.testing.assert_close(q,model(x,a))
            if arm!='plain24_nonpositive':
                u,v,b,d=model.tables(x);self.assertTrue(torch.all(u>=0) and torch.all(v>=0) and torch.all(b>=0))
            with self.assertRaises(ValueError):model(x,torch.full((4,32),5,dtype=torch.long))

    def test_all_intended_heads_have_finite_gradients(self):
        x=self.state(8);actions=(torch.arange(32)[None]+torch.arange(8)[:,None])%5
        for arm in m.NEW_ARMS:
            model=m.AmortizedPriceQ(arm);model(x,actions).sum().backward()
            self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()))
            grad=model.net[2].bias.grad
            self.assertGreater(float(grad[:161].abs().sum()),0.)
            if arm==m.NEW_ARMS[0]:
                self.assertGreater(float(grad[161:165].abs().sum()),0.);self.assertGreater(float(grad[165:].abs().sum()),0.)
            elif arm==m.NEW_ARMS[1]:self.assertEqual(float(grad[161:].abs().sum()),0.)

    def test_ablation_masks_only_price_and_keeps_weights(self):
        torch.manual_seed(99021);primary=m.AmortizedPriceQ();ablation=m.AmortizedPriceQ(m.NEW_ARMS[1]);ablation.load_state_dict(primary.state_dict())
        x=self.state();a,_,_,d=primary.tables(x);b,_,_,e=ablation.tables(x)
        torch.testing.assert_close(a-b,d['price']);self.assertEqual(float(e['price'].abs().sum()),0.)

    def test_ddqn_selects_online_evaluates_target_and_skips_terminal(self):
        class Online:
            def greedy(self,x):
                assert len(x)==1 and torch.isfinite(x).all()
                return torch.ones(1,32,dtype=torch.long),None,None
        class Target:
            def __call__(self,x,a):
                assert torch.all(a==1) and len(x)==1
                return torch.tensor([-7.],requires_grad=True)
        x=self.state(2);x[1]=float('nan')
        y=m.ddqn_targets(Online(),Target(),torch.tensor([-2.,-5.],requires_grad=True),x,torch.tensor([0.,1.]))
        torch.testing.assert_close(y,torch.tensor([-9.,-5.]));self.assertFalse(y.requires_grad)
        y=m.ddqn_targets(None,None,torch.tensor([-5.]),x[:1],torch.ones(1));self.assertEqual(float(y[0]),-5.)

    def test_target_parameters_no_gradient_and_state_roundtrip(self):
        model=m.AmortizedPriceQ();target=copy.deepcopy(model).requires_grad_(False);x=self.state();a=model.greedy(x)[0]
        y=m.ddqn_targets(model,target,torch.ones(4)*-1,x,torch.zeros(4));F.smooth_l1_loss(model(x,a),y).backward()
        self.assertTrue(all(v.grad is None for v in target.parameters()))
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'synthetic.pt';torch.save(model.state_dict(),path);clone=m.AmortizedPriceQ();clone.load_state_dict(torch.load(path,weights_only=True))
            torch.testing.assert_close(clone(x,a),model(x,a))

    def test_adapter_rejects_bad_inputs_and_equivalence_failure(self):
        for x in [np.zeros(532),np.zeros((1,533)),np.full(533,np.nan)]:
            with self.assertRaises(ValueError):g.compiled_values(None,x)
        with self.assertRaises(AssertionError):g.equivalent([[0]],[0.],[[1]],[0.])
        with self.assertRaises(AssertionError):g.equivalent([[0]],[np.nan],[[0]],[np.nan])
        with self.assertRaises(AssertionError):g.equivalent([[0]],[-1.],[[0]],[-2.])

    def test_auditor_rejects_corrupt_and_unlisted_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);(d/'a.txt').write_text('a');g.r.write(d/'run_manifest.json',dict(files={'a.txt':g.r.old.sha(d/'a.txt')}))
            audit.manifest_check(d);(d/'a.txt').write_text('b')
            with self.assertRaises(AssertionError):audit.manifest_check(d)
            (d/'a.txt').write_text('a');(d/'extra').write_text('extra')
            with self.assertRaises(AssertionError):audit.manifest_check(d)

    def test_latency_failure_cannot_be_overridden_by_storage(self):
        metrics={a:dict(parameter_bytes=194944,median_ms=.1,p95_ms=.2) for a in g.ARMS}
        metrics[m.NEW_ARMS[0]]=dict(parameter_bytes=68264,median_ms=.11,p95_ms=.3)
        s=g.screens(metrics);self.assertTrue(s['online_storage']);self.assertFalse(s['complete_action_median'])

    def test_research_cli_guard_with_and_without_confirmations(self):
        for mode in ['development','final']:
            for confirmed in [False,True]:
                with tempfile.TemporaryDirectory() as td:
                    out=Path(td)/'blocked';cmd=[sys.executable,str(g.ROOT/'run_controller_v3_engineering.py'),'--mode',mode,'--output',str(out)]
                    if confirmed:cmd.append('--confirm-'+mode+'-run')
                    proc=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
                    self.assertNotEqual(proc.returncode,0);self.assertIn('disabled',proc.stderr);self.assertFalse(out.exists())

    def test_scope_exclusions(self):
        result=audit.exclusion_check();self.assertEqual(result['cloud_nodes'],0);self.assertEqual(result['charging_stations'],0)
        self.assertFalse(result['security_trust_enabled'])

if __name__=='__main__':
    # Optional evidence path; hash the exact code that has just been tested.
    output=Path(sys.argv[1]) if len(sys.argv)>1 else None
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Mechanics)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if output:
        g.r.write(output,dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
            optimizer_updates=0,new_simulator_steps=0,source_sha256={n:g.r.old.sha(g.ROOT/n) for n in g.SOURCES}))
    sys.exit(0 if result.wasSuccessful() else 1)
