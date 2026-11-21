"""Seven-arm workflow, formula, launch-guard and semantic-corruption checks."""
import copy
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
from torch.nn import functional as F
import run_centered_linear_comparison as runner
import centered_linear_comparison_models as models
from audit_centered_linear_comparison import audit


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output=Path(sys.argv[1]);cls.receipt=runner.run(cls.output,'fixture')

    def test_complete_seven_arm_workflow(self):
        v=self.receipt
        self.assertTrue(v['passed']);self.assertEqual(v['simulation_slots_recorded'],1160)
        self.assertEqual(v['historical_ddqn_updates'],384);self.assertEqual(v['historical_bc_updates'],20)
        self.assertEqual(len(v['value_distributions']),14)
        self.assertEqual(len(list((self.output/'run').rglob('*.pt'))),14)
        self.assertEqual(len(runner.read(self.output/'run/timing.json')),60)

    def test_formula_controls_and_old_baseline_equivalence(self):
        torch.manual_seed(957290001)
        raw=torch.randn(3,161);p=raw[:,:150].reshape(3,30,5)/30;m=raw[:,150:160].reshape(3,2,5)/2;v=raw[:,160]
        a=torch.randint(0,5,(3,32));cost=p.gather(2,a[:,:30,None]).sum((1,2))+m.gather(2,a[:,30:,None]).sum((1,2))
        linear=v-cost+p.mean(2).sum(1)+m.mean(2).sum(1)
        expected={'old_softplus_bc_ddqn':-F.softplus(v+cost),'centered_softplus_bc_ddqn':-F.softplus(-linear),'uncentered_linear_bc_ddqn':v-cost}
        for arm in models.ARMS:
            model=models.Critic(arm)
            torch.testing.assert_close(model.value(p,m,v,a),expected.get(arm,linear))
            if arm=='old_softplus_bc_ddqn':
                torch.testing.assert_close(model.value(p,m,v,a),model.student.value_from_tables(p,m,v,a),rtol=0,atol=0)

    def test_development_and_final_guards(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/'forbidden'
            with patch.object(runner.variants,'Critic',side_effect=AssertionError('constructed')):
                for confirmation in [None,runner.CONFIRM]:
                    with self.assertRaises(RuntimeError):runner.run(output,'development',confirmation)
                with self.assertRaises(RuntimeError):runner.run(output,'final')
            self.assertFalse(output.exists())
            p=subprocess.run([sys.executable,str(Path(runner.__file__)),'--mode','development','--confirmation',runner.CONFIRM,'--output',str(output)],capture_output=True,text=True,timeout=35)
            self.assertNotEqual(p.returncode,0);self.assertIn('disabled pending',p.stderr);self.assertFalse(output.exists())

    def mutate(self,func,rehash=True):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'run';shutil.copytree(self.output/'run',root)
            path=func(root)
            if rehash:
                v=runner.read(root/'manifest.json');v[path.relative_to(root).as_posix()]=runner.sha(path);runner.write(root/'manifest.json',v)
            with self.assertRaises(ValueError):audit(root)

    def test_policy_or_value_screen_tamper(self):
        def change(root):
            p=root/'analysis.json';v=runner.read(p);v['value_screen']=not v['value_screen'];runner.write(p,v);return p
        self.mutate(change)

    def test_primary_BC_only_weight_hash_mismatch(self):
        def change(root):
            p=root/'CHECKPOINT_FREEZE.json';v=runner.read(p);v['bc_end_hashes']['centered_linear_bc_only/0']='x';runner.write(p,v);return p
        self.mutate(change)

    def test_no_demo_preload_and_BC_only_update_tamper(self):
        for arm,field,value in [('centered_linear_no_demo_no_bc','preloaded_demo_hash','teacher-data'),('centered_linear_bc_only','updates',1)]:
            def change(root,arm=arm,field=field,value=value):
                p=root/arm/'fit_0/counts.json';v=runner.read(p);v[field]=value;runner.write(p,v);return p
            self.mutate(change)

    def test_recorded_TD_arithmetic_tamper(self):
        def change(root):
            p=root/'centered_linear_bc_ddqn/fit_0/optimizer.json';v=runner.read(p);v[0]['target'][0]+=10;runner.write(p,v);return p
        self.mutate(change)

    def test_unlisted_checkpoint_or_modified_workload(self):
        def unlisted(root):
            p=root/'intermediate.pt';p.write_bytes(b'not final');return p
        self.mutate(unlisted,False)
        def workload(root):
            p=root/'centered_linear_demo_no_bc/fit_0/train_000.json';v=runner.read(p);v['records'][0]['input_bits']+=1;runner.write(p,v);return p
        self.mutate(workload)

    def test_missing_global_freeze(self):
        def change(root):
            p=root/'events.json';v=runner.read(p)
            for row in v:
                if row['stage']=='checkpoints_frozen':row['stage']='missing'
            runner.write(p,v);return p
        self.mutate(change)


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Provide new fixture output directory')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(stream.getvalue());output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=True)
    (output/'test_log.txt').write_text(stream.getvalue(),encoding='utf-8')
    runner.write(output/'tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),research_seed_use=False,research_launch_authorized=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
