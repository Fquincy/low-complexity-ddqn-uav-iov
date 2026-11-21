"""Synthetic tests for conservative intervals and consistent screened DDQN."""
import copy,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
import deadline_screen_prototype as s

def observations(n):
    rng=np.random.default_rng(997310001);x=rng.uniform(0,1,(n,533)).astype(np.float32)
    features=x[:,:240].reshape(n,30,8);features[:,:,6]=rng.uniform(.6,1,(n,30));features[:,:,7]=rng.uniform(5/6,1,(n,30))
    x[:,-1]=(np.arange(n)%40)/40
    return torch.from_numpy(x)

class Checks(unittest.TestCase):
    def test_random_interval_containment_and_no_false_exclusion(self):
        rng=np.random.default_rng(997310011);c=rng.uniform(300e6,500e6,100000);d=rng.uniform(.250,.300,100000)
        cz=(c/500e6).astype(np.float32);dz=(d/.300).astype(np.float32)
        cl,ch=s.intervals(cz,500e6);dl,dh=s.intervals(dz,.300)
        self.assertTrue(np.all((cl<=c)&(c<=ch)&(dl<=d)&(d<=dh)))
        mask=np.nextafter(cl/1.5e9,-np.inf)>np.nextafter(dh+1e-10,np.inf)
        self.assertTrue(np.all((c[mask]/1.5e9)>d[mask]+1e-10))
        # Explicit tight equality/one-ulp perturbations around the true time boundary.
        d=np.linspace(.250,.300,1000);center=(d+1e-10)*1.5e9
        for c in [np.nextafter(center,-np.inf),center,np.nextafter(center,np.inf)]:
            cl,_=s.intervals((c/500e6).astype(np.float32),500e6);_,dh=s.intervals((d/.3).astype(np.float32),.3)
            excluded=np.nextafter(cl/1.5e9,-np.inf)>np.nextafter(dh+1e-10,np.inf)
            self.assertFalse(excluded.any())
    def test_mask_membership_nonempty_and_exploration_rng(self):
        x=observations(128);mask=s.allowed(x)
        self.assertTrue(mask[:,:,1:].all());self.assertTrue(mask[:,30:,:].all());self.assertTrue((mask.sum(2)>0).all())
        rng=np.random.default_rng(997310012);u=rng.random((128,32));before=copy.deepcopy(rng.bit_generator.state)
        a=s.explore(x,u);self.assertEqual(before,rng.bit_generator.state)
        np.testing.assert_array_equal(a,s.explore(x,u))
        self.assertTrue(np.take_along_axis(mask,a[:,:,None],2).all())
        np.testing.assert_array_equal(s.explore(x,u,False),(u*5).astype(np.int64))
    def test_eager_native_and_chosen_values_all_families(self):
        x=observations(32)
        for arm in s.SUPPORTED:
            model=s.Critic(arm).eval();original=s.existing.make_model(arm,99731)
            self.assertEqual(s.existing.base.parameter_bytes(model),s.existing.base.parameter_bytes(original))
            with torch.inference_mode():
                # Batch-one semantics on both paths, preserving near-tie behavior.
                for state in x:
                    a,q=model.greedy(state[None]);b,v=model.runtime(state.numpy())
                    np.testing.assert_array_equal(a.numpy(),b);np.testing.assert_allclose(q.numpy(),v,atol=1e-6,rtol=1e-5)
                    np.testing.assert_allclose(q.numpy(),model(state[None],a).numpy(),atol=1e-5,rtol=1e-5)
                    self.assertTrue(np.take_along_axis(s.allowed(state[None]),a.numpy()[:,:,None],2).all())
    def test_ties_and_no_mask_ablation(self):
        x=observations(1);x[:,:240].reshape(1,30,8)[:,:,6]=1.;x[:,:240].reshape(1,30,8)[:,:,7]=5/6
        model=s.Critic('no_price_both');
        for p in model.parameters():p.data.zero_()
        a,_=model.greedy(x);b,_=model.runtime(x[0].numpy())
        self.assertTrue(torch.all(a[:,:30]==1));self.assertTrue(torch.all(a[:,30:]==0));np.testing.assert_array_equal(a.numpy(),b)
        model.screened=False;a,q=model.greedy(x);self.assertTrue(torch.all(a==0))
        old_a,old_q,_=s.existing.greedy(model.model,x,model.arm)
        np.testing.assert_array_equal(a.numpy(),old_a.numpy());np.testing.assert_array_equal(q.detach().numpy(),old_q.detach().numpy())
    def test_next_state_selection_target_evaluation_and_terminal(self):
        x=observations(2);x[0,-1]=.5;x[1,-1]=1.
        online=s.Critic('price_both');target=copy.deepcopy(online);r=torch.tensor([-1.,-2.]);done=torch.tensor([0.,1.])
        selected=torch.ones((1,32),dtype=torch.long)
        with patch.object(online,'greedy',return_value=(selected,torch.tensor([-999.]))) as choice,patch.object(target,'forward',return_value=torch.tensor([-3.])) as value:
            y=s.targets(online,target,r,x,done)
            np.testing.assert_array_equal(y.numpy(),[-4.,-2.]);np.testing.assert_array_equal(choice.call_args.args[0].numpy(),x[:1].numpy())
            np.testing.assert_array_equal(value.call_args.args[1].numpy(),selected.numpy())
        with patch.object(online,'greedy',side_effect=AssertionError('terminal decode')):
            np.testing.assert_array_equal(s.targets(online,target,r[1:],x[1:],done[1:]).numpy(),[-2.])
        with self.assertRaises(ValueError):s.targets(online,target,r,x,torch.zeros(2))
    def test_objective_gradient_and_terminal_anchor(self):
        online=s.Critic('price_both');target=copy.deepcopy(online).eval().requires_grad_(False)
        x=observations(4);x[:,-1]=39/40;ns=x.clone();ns[:,-1]=1
        a=online.greedy(x)[0];batch=(x,a,-torch.ones(4),ns,torch.ones(4),np.arange(4))
        loss,detail=s.objective(online,target,batch,batch);loss.backward()
        self.assertAlmostEqual(float(loss),detail['uniform']+detail['anchor'],places=5)
        self.assertTrue(any(p.grad is not None for p in online.parameters()));self.assertTrue(all(p.grad is None for p in target.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in online.parameters()))
        self.assertTrue(torch.all(online(ns,a)==0))
    def test_invalid_inputs_and_terminal_rejection(self):
        x=observations(1)
        for bad in [x.double(),torch.full_like(x,float('nan')),x[:,:532]]:
            with self.assertRaises(ValueError):s.allowed(bad)
        bad=x.clone();bad[0,6]=1.1
        with self.assertRaises(ValueError):s.allowed(bad)
        x[:,-1]=1.;m=s.Critic('price_both')
        with self.assertRaises(ValueError):m.greedy(x)
        with self.assertRaises(ValueError):m.runtime(x[0].numpy())
        with self.assertRaises(ValueError):s.explore(x,np.zeros((1,32)))

if __name__=='__main__':
    s.existing.configure();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    out=Path('validation/deadline_screen_engineering_2026-11-11_01');out.mkdir(exist_ok=False,parents=True)
    (out/'tests.json').write_text(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),new_simulator_slots=0,research_launched=False),indent=2),encoding='utf-8')
    sys.exit(0 if result.wasSuccessful() else 1)
