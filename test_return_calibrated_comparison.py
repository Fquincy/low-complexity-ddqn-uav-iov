"""Integration evidence and semantic corruption checks on isolated fixtures only."""
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

import run_return_calibrated_comparison as runner
from audit_return_calibrated_comparison import audit


class Workflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output=Path(sys.argv[1]);cls.root=cls.output/'run'
        cls.receipt=runner.read(cls.output/'saved_artifact_audit.json')
        cls.A,cls.B,cls.C,cls.D,cls.E,cls.F,cls.G=list(runner.h.ARMS)

    def corrupt(self,change):
        with tempfile.TemporaryDirectory(prefix='handoff_corruption_') as td:
            root=Path(td)/'run';shutil.copytree(self.root,root)
            names=change(root)
            if isinstance(names,str):names=[names]
            manifest=runner.read(root/'manifest.json')
            for name in names:
                if (root/name).exists():manifest[name]=runner.sha(root/name)
                else:manifest.pop(name,None)
            runner.write(root/'manifest.json',manifest)
            with self.assertRaises(ValueError):audit(root)

    def edit(self,root,name,transform):
        p=root/name;v=runner.read(p);transform(v);runner.write(p,v);return name

    def test_01_complete_same_path_workflow_and_counts(self):
        r=self.receipt
        self.assertTrue(r['passed']);self.assertEqual(r['kind'],'ENGINEERING_FIXTURE_NOT_RESEARCH')
        self.assertEqual(r['simulation_slots_recorded'],1880)
        self.assertEqual(r['historical_bc_updates'],24);self.assertEqual(r['historical_ddqn_updates'],384)
        self.assertEqual(r['retention_batches'],128);self.assertEqual(r['ridge_solves'],6)
        self.assertEqual(r['final_checkpoints'],14)
        self.assertEqual(len(runner.read(self.root/'timing.json')),60)
        self.assertTrue(r['teacher_reference_recomputations']>0)
        self.assertEqual(len(list(self.root.rglob('*.pt'))),14)
        for fit in range(2):
            cal=runner.read(self.root/self.A/f'fit_{fit}/calibration.json')
            self.assertEqual(cal['fitting_episodes'],list(range(8)))
            self.assertEqual(cal['withheld_return_episodes'],[8,9])
        self.assertFalse(runner.read(self.output/'completion.json')['research_launch_authorized'])

    def test_02_research_guards_before_models_outputs_and_draws(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'blocked'
            with patch.object(runner.h,'Critic',side_effect=AssertionError('constructed')):
                for confirmation in [None,runner.CONFIRM]:
                    with self.assertRaises(RuntimeError):runner.run(out,'development',confirmation)
                with self.assertRaises(RuntimeError):runner.run(out,'final',runner.CONFIRM)
            self.assertFalse(out.exists())
            cmd=[sys.executable,runner.__file__,'--mode','development','--output',str(out),'--confirmation',runner.CONFIRM]
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=35)
            self.assertNotEqual(p.returncode,0);self.assertIn('disabled pending',p.stderr);self.assertFalse(out.exists())

    def test_03_return_split_corruption(self):
        def change(root):
            return self.edit(root,f'{self.E}/fit_0/calibration.json',lambda v:v.update(fitting_episodes=list(range(9))))
        self.corrupt(change)

    def test_04_retention_draw_and_target_corruption(self):
        for field in ['retention_indices','target']:
            def change(root,field=field):
                return self.edit(root,f'{self.A}/fit_0/optimizer.json',lambda v:v[0][field].__setitem__(0,v[0][field][0]+1))
            self.corrupt(change)

    def test_05_hash_parity_and_scientific_screen_corruption(self):
        def parity(root):
            return self.edit(root,'CHECKPOINT_FREEZE.json',lambda v:v['handoff_hashes'].__setitem__(f'{self.B}/0','wrong'))
        self.corrupt(parity)
        def decision(root):return self.edit(root,'analysis.json',lambda v:v.update(promising=not v['promising']))
        self.corrupt(decision)

    def test_06_global_checkpoint_freeze_and_unknown_artifacts(self):
        def freeze(root):
            def change(v):
                for e in v:
                    if e['stage']=='checkpoints_frozen':e['stage']='removed'
            return self.edit(root,'events.json',change)
        self.corrupt(freeze)
        def extra(root):
            (root/'undeclared.json').write_text('{}',encoding='utf-8');return 'undeclared.json'
        self.corrupt(extra)
        def checkpoint(root):
            (root/'selected_best.pt').write_bytes(b'invalid');return 'selected_best.pt'
        self.corrupt(checkpoint)

    def test_07_plain_control_must_not_train_on_demonstrations(self):
        def change(root):
            return self.edit(root,f'{self.G}/fit_0/bc.json',lambda v:v.append(dict(step=0,indices=[0,1,2,3],loss=0,gradient_norm=0)))
        self.corrupt(change)

    def test_08_budget_and_fixture_seed_override_rejected(self):
        for field in ['warmup','seeds']:
            def change(root,field=field):
                pl=runner.read(root/'runtime_plan.json')
                if field=='warmup':pl['warmup']+=1
                else:pl['seeds']['model_seeds'][0]=runner.protocol()['seeds']['model_seeds'][0]
                runner.write(root/'runtime_plan.json',pl)
                freeze=runner.read(root/'PRE_RUN_FREEZE.json');freeze['plan_sha256']=runner.sha(root/'runtime_plan.json')
                runner.write(root/'PRE_RUN_FREEZE.json',freeze)
                return ['runtime_plan.json','PRE_RUN_FREEZE.json']
            self.corrupt(change)

    def test_09_frozen_decision_math_all_fits_and_interaction(self):
        a,b,c,d,e,f,g=list(runner.h.ARMS);per={};values={}
        for i in range(3):
            absolute={arm:dict(return_value=-10.,deadline=.9) for arm in runner.h.ARMS}
            absolute[a]=dict(return_value=-9.,deadline=.91)
            per[str(i)]=dict(absolute=absolute,contrasts={arm:{m:absolute[a][m]-v[m] for m in v} for arm,v in absolute.items() if arm!=a})
            for arm in runner.h.ARMS:values[f'{arm}/{i}']=dict(nrmse=.5 if arm==a else 1.,bias=.5 if arm==a else 1.)
        result=runner.summarize(per,values);self.assertTrue(result['promising'])
        self.assertAlmostEqual(result['factorial_interaction']['0']['return_value'],1.)
        bad=copy.deepcopy(per);bad['2']['contrasts'][e]['deadline']=-.001
        self.assertFalse(runner.summarize(bad,values)['policy_screen'])
        worse=copy.deepcopy(values);worse[f'{a}/1']['nrmse']=worse[f'{d}/1']['nrmse']
        self.assertFalse(runner.summarize(per,worse)['value_screen'])

    def test_10_ridge_and_withheld_warning_corruption(self):
        for field in ['ridge','withheld_fit_worsened_vs_after_bc']:
            def change(root,field=field):
                def modify(v):
                    if field=='ridge':v['ridge']['beta'][0]+=1.
                    else:v[field]=not v[field]
                return self.edit(root,f'{self.E}/fit_0/calibration.json',modify)
            self.corrupt(change)

    def test_11_gradient_partition_and_first_identity_corruption(self):
        for field in ['gradient_norms','first_update_decomposition']:
            def change(root,field=field):
                def modify(v):
                    if field=='gradient_norms':v[0][field]['branch_output']+=1.
                    else:v[0][field]['same_state_gap'][0]+=1.
                return self.edit(root,f'{self.A}/fit_0/optimizer.json',modify)
            self.corrupt(change)

    def test_12_same_size_storage_and_timing_outputs(self):
        for v in self.receipt['storage'].values():
            self.assertEqual(v['parameter_bytes'],67364);self.assertEqual(v['buffer_bytes'],20)
        self.assertEqual(len(self.receipt['value_distributions']),14)
        self.assertTrue(all('per_round' in v for v in self.receipt['timing'].values()))
        analysis=runner.read(self.root/'analysis.json')
        self.assertEqual(len(analysis['equal_fit_means']),7)
        self.assertEqual(len(analysis['value_contrasts']),2)


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Provide completed isolated fixture output directory')
    out=Path(sys.argv[1])
    if (out/'tests.json').exists():raise SystemExit('Preserve prior test records')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Workflow))
    print(stream.getvalue());(out/'test_log.txt').write_text(stream.getvalue(),encoding='utf-8')
    runner.write(out/'tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),kind='ENGINEERING_FIXTURE_NOT_RESEARCH',reserved_research_seeds_used=False,research_launch_authorized=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
