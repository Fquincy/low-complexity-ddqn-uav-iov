"""Eleven-arm research-runner readiness tests; research CLI never launches."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np,torch
import run_v3_research_preparation as runner
import audit_v3_research_preparation as auditor
import run_controller_redesign_engineering as base

class Protocol(unittest.TestCase):
    def test_frozen_eleven_arm_budget_and_comparison_contract(self):
        p=runner.protocol();self.assertEqual(len(p['arms']),11)
        self.assertEqual(p['development']['training_slots'],440000)
        self.assertEqual(p['development']['optimizer_updates'],383735)
        self.assertEqual(p['final']['status'],'reserved_untouched_no_launch')
        self.assertFalse(p['scope']['cloud']);self.assertFalse(p['scope']['charging_stations']);self.assertFalse(p['scope']['security_trust'])
        self.assertIn('plain24_nonpositive',p['comparisons']['primary_against_each_of'])
        self.assertIn('amortized_no_price24',p['arms'])

    def test_research_modes_blocked_with_and_without_confirmation(self):
        for mode in ['development','final']:
            for confirmation in [False,True]:
                with tempfile.TemporaryDirectory() as td:
                    output=Path(td)/'blocked';cmd=[sys.executable,str(runner.ROOT/'run_v3_research_preparation.py'),'--mode',mode,'--output',str(output)]
                    if confirmation:cmd.append('--confirm-'+mode+'-run')
                    result=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
                    self.assertNotEqual(result.returncode,0);self.assertIn('disabled',result.stderr);self.assertFalse(output.exists())

    def test_quality_screens_keep_every_fit_and_mechanism_margin(self):
        comparisons={}
        refs=runner.protocol()['comparisons']['primary_against_each_of']
        for name in [*refs,'amortized_no_price24','all_local','nearest_rsu','uav_selective']:
            comparisons[name]=dict(per_fit=[dict(return_difference=2.,deadline_difference=.02) for _ in range(5)],
                mean_return_difference=2.,mean_deadline_difference=.02)
        values={arm:[dict(normalized_rmse=.1,absolute_bias=.1,near_zero_large_return_fraction=0.,terminal_abs_error_p95=.1) for _ in range(5)] for arm in runner.ARMS}
        screens=runner.quality_screens(comparisons,values,5)
        self.assertTrue(all(screens.values()))
        comparisons['amortized_no_price24']['per_fit'][0]['return_difference']=-.1
        self.assertFalse(runner.quality_screens(comparisons,values,5)['mechanism_each_fit'])
        self.assertFalse(runner.quality_screens(comparisons,values,4)['evaluated'])
        bootstrap=dict(return_difference_one_sided_97_5_lower=-2.,deadline_difference_one_sided_97_5_lower=-.01)
        self.assertTrue(runner.quality_screens(comparisons,values,5,bootstrap,'final')['final_crossed_bootstrap'])
        bootstrap['deadline_difference_one_sided_97_5_lower']=-.02
        self.assertFalse(runner.quality_screens(comparisons,values,5,bootstrap,'final')['final_crossed_bootstrap'])

    def test_order_independent_comparison_rejects_unpaired_workloads(self):
        row=lambda seed,workload,ret:dict(seed=seed,workload_sha256=workload,return_sum=ret,deadline_success=.9)
        fresh={arm:[row(99131,'same',1.)] for arm in runner.ARMS};heuristics={'all_local':[row(99131,'same',0.)]}
        runner.compare_all(fresh,heuristics)
        fresh['plain24_nonpositive'][0]['workload_sha256']='changed'
        with self.assertRaises(ValueError):runner.compare_all(fresh,heuristics)

    def test_locked_final_bootstrap_resamples_fits_and_paired_workloads(self):
        primary=[];reference=[]
        for fit in [51001,51011,51017,51023,51031]:
            for task in range(200):
                row=dict(model_seed=fit,seed=520000000+task,workload_sha256=f'work-{task}',return_sum=4.,deadline_success=.91)
                primary.append(row);reference.append(dict(row,return_sum=1.,deadline_success=.90))
        first=runner.crossed_bootstrap(primary,reference);again=runner.crossed_bootstrap(primary,reference)
        self.assertEqual(first,again);self.assertEqual(first['draws'],5000)
        self.assertEqual(first['fit_resampling'],'paired_with_replacement_shared_across_arms')
        self.assertAlmostEqual(first['return_difference_one_sided_97_5_lower'],3.)
        self.assertAlmostEqual(first['deadline_difference_one_sided_97_5_lower'],.01)

class FullFixture(unittest.TestCase):
    def test_complete_synthetic_orchestration_then_independent_audit(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_NOT_RESEARCH_v3_') as td:
            out=Path(td)/'fixture';runner.patch_training_core();result=runner.training_fixture(out)
            self.assertEqual(result['training_slots'],440);self.assertEqual(result['optimizer_updates'],363)
            self.assertEqual(result['target_copies'],44);self.assertTrue(result['checkpoints_frozen_before_evaluation'])
            audit=auditor.audit_fixture(out)
            self.assertEqual(audit['state'],'passed_v3_synthetic_orchestration_audit')
            self.assertEqual(audit['checkpoints_checked'],11);self.assertEqual(audit['optimizer_rows_checked'],363)
            manifest=json.loads((out/'run_manifest.json').read_text())['files']
            for name,digest in manifest.items():self.assertEqual(base.old.sha(out/name),digest)
            # The auditor's own output is independently covered by its contents.
            saved=json.loads((out/'saved_artifact_audit.json').read_text())
            self.assertEqual(saved['training_slots_checked'],440);self.assertFalse(saved['research_quality_evidence'])

    def test_research_cli_never_touches_user_output(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/'should_not_exist'
            proc=subprocess.run([sys.executable,str(runner.ROOT/'run_v3_research_preparation.py'),'--mode','development','--confirm-development-run','--output',str(output)],capture_output=True,text=True,timeout=40)
            self.assertNotEqual(proc.returncode,0);self.assertIn('disabled',proc.stderr);self.assertFalse(output.exists())

if __name__=='__main__':
    runner.configure()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if len(sys.argv)>1:
        base.write(Path(sys.argv[1]),dict(passed=result.wasSuccessful(),tests=7,failures=len(result.failures),errors=len(result.errors),
            protocol_sha256=base.old.sha(runner.PROTOCOL),source_sha256={name:base.old.sha(runner.ROOT/name) for name in runner.SOURCES},
            optimizer_updates=0,new_research_simulator_steps=0,research_training_launched=False))
    sys.exit(0 if result.wasSuccessful() else 1)
