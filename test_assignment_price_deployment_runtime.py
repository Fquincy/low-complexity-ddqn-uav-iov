"""Equivalence and validation tests; no simulator or optimizer updates."""
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import torch
import assignment_price_prototype as original
import assignment_price_fast_runtime as helper
import assignment_price_deployment_runtime as fast
fast.validate_mask=helper.validate_mask
from deadline_screen_prototype import allowed
import run_horizon_loss_research as r

BASE=Path('validation/assignment_price_fused_2026-11-15_01/extended_tests')
OLD=Path('validation/assignment_price_prototype_2026-11-15_01/cost_gate')

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r.configure()
        cls.states=np.load(r.ROOT/'validation/controller_redesign_engineering_2026-11-05_01/cost_gate/common_observations.npz')['observations']

    def test_saved_models_heads_actions_values(self):
        for arm in original.ARMS:
            for fit in range(2):
                saved=torch.load(OLD/f'{arm}_{fit}.pt',weights_only=True)
                model=original.Critic(arm,saved['seed']).eval();model.load_state_dict(saved['online'])
                runtime=fast.Runtime(model,BASE/f'test_{arm}_{fit}.onnx')
                before=r.models.original.existing.base.old.digest(model.state_dict())
                for state in self.states:
                    x=torch.from_numpy(state)[None]
                    with torch.inference_mode():
                        heads,_=model.heads(x)
                        actual_heads=runtime.session.run(None,{'state':x.numpy()})
                        for a,b in zip([heads[j] for j in (0,1,2,5,6)],actual_heads):
                            np.testing.assert_allclose(a.numpy().reshape(-1),b.reshape(-1),rtol=1e-5,atol=1e-7)
                        a,q=model.greedy(x)
                    actual,value=runtime.values(state)
                    np.testing.assert_array_equal(a.numpy(),actual)
                    np.testing.assert_allclose(q.numpy(),value,rtol=1e-5,atol=1e-6)
                self.assertEqual(before,r.models.original.existing.base.old.digest(model.state_dict()))

    def test_masks_boundaries_and_remaining(self):
        rng=np.random.default_rng(99883)
        for k in range(500):
            x=self.states[k%128].copy();tasks=x[:240].reshape(30,8)
            tasks[:,6:8]=rng.random((30,2),dtype=np.float32)
            if k%2==0:
                tasks[:,7]=tasks[:,6]*np.float32(500e6/1.5e9/.3)
                tasks[:,7]=np.minimum(1,tasks[:,7])
                tasks[:10,7]=np.nextafter(tasks[:10,7],np.float32(np.inf))
                tasks[10:20,7]=np.nextafter(tasks[10:20,7],np.float32(-np.inf))
                tasks[:,7]=np.clip(tasks[:,7],0,1)
            x[-1]=np.float32((k%40)/40)
            mask,h=fast.validate_mask(x)
            np.testing.assert_array_equal(mask,allowed(x[None])[0,:30])
            self.assertEqual(h,float(original.remaining(torch.from_numpy(x)[None])[0]))

    def test_invalid_and_noncontiguous_inputs(self):
        model=original.Critic();slow=original.Runtime(model);quick=fast.Runtime(model,BASE/('test_'+self._testMethodName+'.onnx'))
        cases=[]
        for index,value in [(0,np.nan),(1,np.inf),(-1,1.),(-1,.1234),(-1,-1.),(6,-.1),(6,1.1),(7,-.1),(7,1.1),(270,-1.),(270,100.)]:
            x=self.states[0].copy();x[index]=value;cases.append(x)
        cases += [self.states[0].astype(np.float64),self.states[0][:-1]]
        for x in cases:
            for runtime in [slow,quick]:
                with self.assertRaises(ValueError):runtime.values(x)
        holder=np.zeros(1066,np.float32);holder[::2]=self.states[0]
        for x in [holder[::2],self.states[0]]:
            a,q=slow.values(x);b,v=quick.values(x)
            np.testing.assert_array_equal(a,b);np.testing.assert_allclose(q,v,rtol=1e-5,atol=1e-6)

    def test_equal_heads_ties_snapshot_and_output_ownership(self):
        model=original.Critic()
        with torch.no_grad():
            for p in model.parameters():p.zero_()
        quick=fast.Runtime(model,BASE/('test_'+self._testMethodName+'.onnx'));x=self.states[0].copy()
        a,q=original.Runtime(model).values(x);b,v=quick.values(x)
        np.testing.assert_array_equal(a,b);np.testing.assert_allclose(q,v,rtol=1e-5,atol=1e-6)
        b[:]=4;v[:]=100
        b2,v2=quick.values(x)
        np.testing.assert_array_equal(a,b2);np.testing.assert_allclose(q,v2,rtol=1e-5,atol=1e-6)
        with torch.no_grad():model.core.net[-1].bias.add_(1.)
        b3,v3=quick.values(x)
        np.testing.assert_array_equal(b2,b3);np.testing.assert_array_equal(v2,v3)

    def test_queue_near_zero_and_near_tie_heads(self):
        model=original.Critic()
        with torch.no_grad():
            for p in model.parameters():p.zero_()
            # Adjacent FP32 bias values: no changed decoder tie tolerance.
            model.core.net[-1].bias[1]=torch.nextafter(torch.tensor(1.),torch.tensor(float('inf')))
            model.core.net[-1].bias[0]=1.
        runtime=fast.Runtime(model,BASE/'test_near_ties.onnx')
        slow=original.Runtime(model)
        for q in [0.,np.nextafter(np.float32(0),np.float32(1)),1e-12,1e-8,1e-5,.1,10.]:
            state=self.states[0].copy();state[270:274]=q
            a,v=slow.values(state);b,w=runtime.values(state)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_allclose(v,w,rtol=1e-5,atol=1e-6)

if __name__=='__main__':
    BASE.mkdir(exist_ok=False)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),saved_model_state_checks=1024,randomized_mask_checks=500,new_optimizer_updates=0,new_simulator_steps=0)
    (BASE/'tests_extended.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    manifest={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['assignment_price_deployment_runtime.py','test_assignment_price_deployment_runtime.py','profile_assignment_price_fast_route.py']}
    (BASE/'tested_source_freeze_extended.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    raise SystemExit(not result.wasSuccessful())
