"""End-to-end engineering fixture and artifact-corruption tests. No pilot seeds."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_heuristic_initialization_pilot as runner
from audit_heuristic_initialization_pilot import audit
from heuristic_pilot_teacher_adapter import build_teacher


class PilotChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output=Path(sys.argv[1])
        cls.receipt=runner.run(cls.output,'fixture')

    def test_entire_orchestration(self):
        result=self.receipt
        self.assertTrue(result['passed'])
        self.assertEqual(result['simulation_slots_recorded'],440)
        self.assertEqual(result['historical_ddqn_updates'],128)
        self.assertEqual(result['historical_bc_updates'],4)
        self.assertEqual(len(result['value_distributions']),4)
        self.assertEqual(result['new_optimizer_updates'],0)
        self.assertEqual(len(list((self.output/'run').rglob('*.pt'))),4)

    def test_launch_guard_precedes_models_output_and_tasks(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/'must_not_exist'
            with patch.object(runner.baseline,'TeacherInitializedCompactDDQN',side_effect=AssertionError('constructed')):
                for confirmation in (None,runner.CONFIRM):
                    with self.assertRaises(RuntimeError):runner.run(output,'pilot',confirmation)
                with self.assertRaises(RuntimeError):runner.run(output,'final')
            self.assertFalse(output.exists())
            process=subprocess.run([sys.executable,str(runner.ROOT/'run_heuristic_initialization_pilot.py'),
                '--mode','pilot','--output',str(output),'--confirmation',runner.CONFIRM],capture_output=True,text=True,timeout=40)
            self.assertNotEqual(process.returncode,0)
            self.assertIn('disabled pending',process.stderr)
            self.assertFalse(output.exists())

    def modified(self, mutator, rehash=True):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'run';shutil.copytree(self.output/'run',root)
            path=mutator(root)
            if rehash:
                manifest=runner.read(root/'manifest.json');manifest[path.relative_to(root).as_posix()]=runner.sha(path)
                runner.write(root/'manifest.json',manifest)
            with self.assertRaises(ValueError):audit(root)

    def test_comparison_tamper_even_after_rehash(self):
        def change(root):
            p=root/'analysis.json';v=runner.read(p);v['promising']=not v['promising'];runner.write(p,v);return p
        self.modified(change)

    def test_bellman_loss_tamper_even_after_rehash(self):
        def change(root):
            p=root/'teacher_bc_then_ddqn'/'fit_0'/'optimizer.json';v=runner.read(p);v[0]['loss']+=1;runner.write(p,v);return p
        self.modified(change)

    def test_unlisted_and_corrupt_files(self):
        def unlisted(root):
            p=root/'extra.txt';p.write_text('not listed');return p
        self.modified(unlisted,False)
        def corrupt(root):
            p=root/'CHECKPOINT_FREEZE.json';p.write_text('{}');return p
        self.modified(corrupt,False)

    def test_checkpoint_freeze_event_required(self):
        def change(root):
            p=root/'events.json';v=runner.read(p)
            for event in v:
                if event['stage']=='checkpoints_frozen':event['stage']='skipped_freeze'
            runner.write(p,v);return p
        self.modified(change)

    def test_paired_workload_tamper_after_rehash(self):
        def change(root):
            p=root/'common_demo_no_bc'/'fit_0'/'train_000.json'
            value=runner.read(p);value['records'][0]['input_bits']+=1;runner.write(p,value);return p
        self.modified(change)

    def test_teacher_adapter_against_preserved_reference(self):
        from test_heuristic_initialized_ddqn_baseline import teacher_reference,observations
        adapter=build_teacher(runner.ROOT,runner.Chapter3Config);reference=teacher_reference()
        for s in observations(3):
            a,count=adapter.values(s);b,checks=reference.values(s)
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(count,checks)
        # This verifies Python function extraction, not old compiled equivalence.


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Provide a NEW engineering output directory')
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(PilotChecks)
    text=io.StringIO()
    result=unittest.TextTestRunner(stream=text,verbosity=2).run(suite)
    print(text.getvalue())
    output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=True)
    (output/'test_log.txt').write_text(text.getvalue(),encoding='utf-8')
    runner.write(output/'tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),
        research_launch_authorized=False,pilot_seed_use=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
