"""Full isolated integration, saved-auditor negative checks and CLI guards."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import run_assignment_price_fixture as w
import audit_assignment_price_fixture as auditor

BASE=w.r.ROOT/'validation/assignment_price_iov_fixture_2026-11-15_01'

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=Path(tempfile.mkdtemp(prefix='adp_engineering_fixture_'))
        w.r.write(BASE/'temporary_output_location.json',dict(path=str(cls.temporary),kind=w.KIND))
        cls.run=w.run(cls.temporary/'run')

    def test_complete_fixture_and_saved_auditor(self):
        report=auditor.audit(self.run);self.assertTrue(report['passed'])
        w.r.write(BASE/'independent_audit.json',report)
        self.assertEqual(report['optimizer_rows'],584)
        self.assertEqual(report['replay_terminal_evictions_checked'],8)

    def test_auditor_rejects_manifest_and_bellman_corruption(self):
        path=self.run/'counters.json';saved=path.read_bytes()
        try:
            path.write_bytes(saved+b' ')
            with self.assertRaises(AssertionError):auditor.audit(self.run)
        finally:path.write_bytes(saved)
        folder=self.run/'coupled_sweep';history=[]
        for ep in range(2):
            row=auditor.load(folder/f'train_{ep}.json.gz');arrays=np.load(folder/f'train_{ep}.npz')
            history.extend(dict(**slot,state=arrays['states'][i],next=arrays['next_states'][i]) for i,slot in enumerate(row['slots']))
        row=json.loads((folder/'optimizer.jsonl').read_text().splitlines()[0]);pl=w.plan();seed=pl['model_seed']
        def verify(record):return auditor.check_update(record,history,pl,True,True,np.random.default_rng(seed+200000),np.random.default_rng(seed+300000))
        verify(row)
        changed=copy.deepcopy(row);changed['bellman_targets'][0]+=1
        with self.assertRaises(AssertionError):verify(changed)
        changed=copy.deepcopy(row);changed['next_actions'][0][0]=-1
        with self.assertRaises(AssertionError):verify(changed)

    def test_research_cli_and_api_rejected_before_output(self):
        logs=[]
        for mode in ['development','final','unknown']:
            output=self.temporary/f'blocked_{mode}'
            with self.assertRaises(RuntimeError):w.run(output,mode)
            self.assertFalse(output.exists())
            result=subprocess.run([sys.executable,str(w.r.ROOT/'run_assignment_price_fixture.py'),'--mode',mode,'--output',str(output)],
                capture_output=True,text=True,cwd=w.r.ROOT)
            self.assertNotEqual(result.returncode,0);self.assertIn('fixture only',result.stderr);self.assertFalse(output.exists())
            logs.append(dict(mode=mode,returncode=result.returncode,stderr=result.stderr,output_created=output.exists()))
        w.r.write(BASE/'cli_guard_evidence.json',logs)

    def test_terminal_ddqn_and_target_gradient_isolation(self):
        pl=w.plan();agent=w.build('coupled_sweep',pl)
        row=auditor.load(self.run/'coupled_sweep/train_0.json.gz')
        arrays=np.load(self.run/'coupled_sweep/train_0.npz')
        state=torch.from_numpy(arrays['states'][-1:]);next_state=torch.from_numpy(arrays['next_states'][-1:])
        action=torch.tensor([row['slots'][-1]['action']]);reward=torch.tensor([row['slot_rewards'][-1]],dtype=torch.float32)
        with patch.object(agent['online'],'greedy',side_effect=AssertionError('terminal decoded')):
            target=w.adp.targets(agent['online'],agent['target'],reward,next_state,torch.ones(1))
        torch.testing.assert_close(target,reward.double())
        loss,detail=w.adp.objective(agent['online'],agent['target'],(state,action,reward,next_state,torch.ones(1)),(state,action,reward))
        agent['opt'].zero_grad(set_to_none=True);loss.backward()
        self.assertTrue(all(p.grad is None for p in agent['target'].parameters()))
        self.assertTrue(any(p.grad is not None for p in agent['online'].parameters()))
        # No optimizer step: this check only verifies gradient routing.

if __name__=='__main__':
    BASE.mkdir(exist_ok=False)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    w.r.write(BASE/'tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),
        kind=w.KIND,research_launched=False,extra_test_optimizer_steps=0))
    raise SystemExit(not result.wasSuccessful())
