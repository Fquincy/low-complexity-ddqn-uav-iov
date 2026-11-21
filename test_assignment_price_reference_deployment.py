"""Reference backend equivalence; no simulator or training."""
import json
from pathlib import Path
import unittest
import numpy as np
import torch
import assignment_price_reference_deployment as compiled
import run_horizon_loss_research as r

BASE=Path('validation/assignment_price_matched_deployment_2026-11-15_01')
OLD=Path('validation/assignment_price_prototype_2026-11-15_01/cost_gate')
ARMS=['screened_price24','unmasked_compact64','screened_wide128','unmasked_wide128']

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r.configure();cls.defs={x['id']:x for x in r.protocol()['learned_arms']}
        cls.states=np.load(r.ROOT/'validation/controller_redesign_engineering_2026-11-05_01/cost_gate/common_observations.npz')['observations']

    def model(self,arm,fit):
        saved=torch.load(OLD/f'{arm}_{fit}.pt',weights_only=True)
        model=r.models.Critic(self.defs[arm],saved['seed']);model.load_state_dict(saved['online'])
        return model.eval()

    def test_saved_models_actions_values(self):
        for arm in ARMS:
            for fit in range(2):
                model=self.model(arm,fit);runtime=compiled.Runtime(model,BASE/f'{arm}_{fit}.onnx')
                before=r.models.original.existing.base.old.digest(model.state_dict())
                old=r.models.Runtime(model)
                for state in self.states:
                    a,q=old.values(state);b,v=runtime.values(state)
                    np.testing.assert_array_equal(a,b);np.testing.assert_allclose(q,v,rtol=1e-5,atol=1e-6)
                    with torch.inference_mode():eager_a,eager_q=model.greedy(torch.from_numpy(state)[None])
                    np.testing.assert_array_equal(eager_a.numpy(),b)
                    np.testing.assert_allclose(eager_q.numpy(),v,rtol=1e-5,atol=1e-6)
                self.assertEqual(before,r.models.original.existing.base.old.digest(model.state_dict()))

    def test_ties_adjacent_bias_and_ownership(self):
        for arm in ARMS:
            model=self.model(arm,0)
            with torch.no_grad():
                for p in model.parameters():p.zero_()
            runtime=compiled.Runtime(model,BASE/f'ties_{arm}.onnx');old=r.models.Runtime(model)
            state=self.states[0].copy()
            a,q=old.values(state);b,v=runtime.values(state)
            np.testing.assert_array_equal(a,b);np.testing.assert_allclose(q,v,rtol=1e-5,atol=1e-6)
            b[:]=4;v[:]=999
            b,v=runtime.values(state);np.testing.assert_array_equal(a,b)
            np.testing.assert_allclose(q,v,rtol=1e-5,atol=1e-6)
            with torch.no_grad():
                parameters=[p for p in model.parameters()]
                parameters[-1].reshape(-1)[1]=torch.nextafter(torch.tensor(0.),torch.tensor(1.))
            adjacent=compiled.Runtime(model,BASE/f'adjacent_{arm}.onnx')
            a,q=old.values(state);b,v=adjacent.values(state)
            np.testing.assert_array_equal(a,b);np.testing.assert_allclose(q,v,rtol=1e-5,atol=1e-6)

    def test_validation_and_noncontiguous(self):
        for arm in ARMS:
            model=self.model(arm,0);old=r.models.Runtime(model)
            runtime=compiled.Runtime(model,BASE/f'validation_{arm}.onnx')
            for index,value in [(0,np.nan),(1,np.inf),(-1,1.),(-1,.1234),(-1,-1.),(6,-.1),(7,1.1)]:
                state=self.states[0].copy();state[index]=value
                for route in [old,runtime]:
                    with self.assertRaises(ValueError):route.values(state)
            for state in [self.states[0].astype(np.float64),self.states[0][:-1]]:
                with self.assertRaises(ValueError):runtime.values(state)
            holder=np.zeros(1066,np.float32);holder[::2]=self.states[0]
            a,q=old.values(holder[::2]);b,v=runtime.values(holder[::2])
            np.testing.assert_array_equal(a,b);np.testing.assert_allclose(q,v,rtol=1e-5,atol=1e-6)

if __name__=='__main__':
    BASE.mkdir(exist_ok=False)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    r.write(BASE/'tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),saved_model_state_checks=1024,new_simulator_steps=0,new_optimizer_updates=0))
    r.write(BASE/'tested_source_freeze.json',{name:r.sha(r.ROOT/name) for name in ['assignment_price_reference_deployment.py','test_assignment_price_reference_deployment.py']})
    raise SystemExit(not result.wasSuccessful())
