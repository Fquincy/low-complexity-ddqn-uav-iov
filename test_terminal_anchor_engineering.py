import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import terminal_anchor_prototype as p
import run_terminal_anchor_engineering as runner

def states():
    x=torch.zeros(40,533);x[:,-1]=torch.arange(40)/40
    return x

class Mechanics(unittest.TestCase):
    def test_time_grid_terminal_boundary_and_bad_inputs(self):
        x=states();torch.testing.assert_close(p.remaining(x),torch.arange(40,0,-1).float())
        x[0,-1]=1.;model=p.Critic('price_both')
        with torch.no_grad():self.assertEqual(float(model(x[:1],torch.zeros(1,32,dtype=torch.long))[0]),0.)
        with self.assertRaises(ValueError):model.greedy(x[:1])
        x[0,-1]=.013
        with self.assertRaises(ValueError):p.remaining(x)
        x[0,-1]=float('nan')
        with self.assertRaises(ValueError):p.remaining(x)

    def test_actions_parameter_counts_and_scaling(self):
        x=states()
        for arm in p.ARMS:
            model=p.Critic(arm)
            with torch.no_grad():
                a,q=model.greedy(x);ba,bq,_=p.existing.greedy(model.model,x,model.base_arm)
            torch.testing.assert_close(a,ba,rtol=0,atol=0)
            torch.testing.assert_close(q,bq*(p.remaining(x) if model.scaled else 1.),rtol=0,atol=0)
            self.assertEqual(sum(v.numel() for v in model.parameters()),sum(v.numel() for v in model.model.parameters()))
        self.assertEqual(sum(v.numel() for v in p.Critic('price_both').parameters()),17066)

    def test_terminal_targets_skip_decoder_and_detach(self):
        model=p.Critic('price_both');target=copy.deepcopy(model)
        ns=torch.zeros(3,533);ns[:,-1]=1.;r=torch.tensor([-1.,-2.,-3.],requires_grad=True)
        with patch.object(model,'greedy',side_effect=AssertionError('terminal decoder called')):
            y=p.targets(model,target,r,ns,torch.ones(3))
        torch.testing.assert_close(y,r);self.assertFalse(y.requires_grad)
        with self.assertRaises(ValueError):p.targets(model,target,r,ns,torch.zeros(3))

    def test_nonterminal_ddqn_selection_and_evaluation(self):
        model=p.Critic('price_both');target=copy.deepcopy(model)
        ns=states()[:2];r=torch.tensor([-1.,-2.]);done=torch.zeros(2)
        with torch.no_grad():expected=r+target(ns,model.greedy(ns)[0])
        torch.testing.assert_close(p.targets(model,target,r,ns,done),expected)

    def test_replay_eviction_and_terminal_distribution(self):
        replay=p.Replay(4);x=np.zeros(533,np.float32);a=np.zeros(32,np.int64)
        for done in [True,False,True,False]:replay.add((x,a,np.float32(-1),x,done))
        sample=replay.sample(10000,np.random.default_rng(99462),True)
        self.assertEqual(set(sample[-1]),{0,2});self.assertLess(abs(np.mean(sample[-1]==0)-.5),.03)
        replay.add((x,a,np.float32(-1),x,False));self.assertEqual(replay.terminal,{2})
        replay.add((x,a,np.float32(-1),x,False));replay.add((x,a,np.float32(-1),x,False))
        self.assertIsNone(replay.sample(32,np.random.default_rng(99463),True))

    def test_combined_gradient_and_empty_anchor(self):
        model=p.Critic('price_both');target=copy.deepcopy(model).requires_grad_(False)
        s=states()[-1:];ns=s.clone();ns[:,-1]=1.;a=torch.zeros(1,32,dtype=torch.long)
        b=(s,a,torch.tensor([-1.]),ns,torch.ones(1),np.array([0]))
        empty,info=p.objective(model,target,b,None);self.assertTrue(info['empty_anchor'])
        total,info=p.objective(model,target,b,b)
        params=list(model.parameters())
        g=torch.autograd.grad(total,params,retain_graph=True);u=torch.autograd.grad(empty,params)
        for x,y in zip(g,u):torch.testing.assert_close(x,2*y)
        self.assertTrue(all(v.grad is None for v in target.parameters()))

    def test_cli_blocks_research_before_output(self):
        for mode in ['development','final']:
            with tempfile.TemporaryDirectory() as td:
                out=Path(td)/'blocked'
                proc=subprocess.run([sys.executable,str(runner.ROOT/'run_terminal_anchor_engineering.py'),'--mode',mode,'--output',str(out)],capture_output=True,text=True,timeout=60)
                self.assertNotEqual(proc.returncode,0);self.assertFalse(out.exists());self.assertIn('disabled',proc.stderr)

    def test_full_fixture_and_artifact_tampering(self):
        out=RETAIN
        audit=runner.fixture(out);self.assertEqual(audit['updates'],438)
        path=out/'predictions.json';original=path.read_bytes();path.write_bytes(original+b' ')
        try:
            with self.assertRaises(AssertionError):runner.audit(out)
        finally:path.write_bytes(original)
        self.assertEqual(runner.audit(out)['checkpoints'],6)

if __name__=='__main__':
    p.existing.configure()
    RETAIN=Path(sys.argv[1])
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    runner.write(runner.ROOT/'validation/terminal_anchor_engineering_tests_2026-11-06.json',dict(
        passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        sources={name:runner.sha(runner.ROOT/name) for name in runner.SOURCES},
        fixture=str(RETAIN),research_launched=False,thesis_changed=False))
    sys.exit(0 if result.wasSuccessful() else 1)
