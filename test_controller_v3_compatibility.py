"""Preserved numerical counterexample, route fairness and launch guard checks."""
import subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import controller_v3_runtime_compatibility as c
import run_controller_v3_compatibility_gate as runner
import test_controller_v3_fused_runtime as previous

class Compatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):runner.g.r.configure();c.native.numba.set_num_threads(1)

    def test_preserved_complete_action_counterexample(self):
        model=runner.g.build('shared_congestion_cost',99023)
        with tempfile.TemporaryDirectory() as td:
            runtime,meta=c.export(model,'shared_congestion_cost',99023,Path(td)/'whole.onnx')
            result=runner.regression(runtime)
            self.assertTrue(result['passed']);self.assertEqual(result['expected_destination'],2)
            self.assertTrue(meta['reused_validated_full_graph'])
            original=c.ORIGINAL/'exports/shared_congestion_cost_99023.onnx'
            self.assertEqual(meta['graph_sha256'],runner.g.r.old.sha(original))

    def test_entire_historical_family_has_fixed_preserved_route(self):
        for arm in c.WHOLE_POLICY_ARMS:
            self.assertEqual(c.route(arm),'validated_complete_ORT_policy')
            model=runner.g.build(arm,99021)
            with tempfile.TemporaryDirectory() as td:
                runtime,meta=c.export(model,arm,99021,Path(td)/'whole.onnx')
                self.assertEqual(meta['graph_sha256'],runner.g.r.old.sha(c.ORIGINAL/'exports'/f'{arm}_99021.onnx'))
                self.assertTrue(meta['reused_validated_full_graph'])

    def test_main_references_receive_same_native_treatment(self):
        runner.check_routes(runner.g.read(runner.PROTOCOL))
        for arm in c.NATIVE_ARMS:
            model=runner.g.build(arm,99021)
            with patch.object(c.native,'export',return_value=(None,{})) as export:
                runtime,meta=c.export(model,arm,99021,Path('unused.onnx'))
                export.assert_called_once();self.assertEqual(meta['route'],'native_core_and_decoder')
        with self.assertRaises(ValueError):c.route('unknown')

    def test_wrong_weights_cannot_reuse_retained_graph(self):
        model=runner.g.build('shared_congestion_cost',99029)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(AssertionError):c.export(model,'shared_congestion_cost',99023,Path(td)/'wrong.onnx')

    def test_cost_screen_uses_native_wide_reference(self):
        metrics={a:dict(parameter_bytes=194944,median_ms=.03,p95_ms=.05) for a in runner.g.ARMS}
        metrics['amortized_workload_price24']=dict(parameter_bytes=68264,median_ms=.031,p95_ms=.04)
        screens=runner.g.screens(metrics);self.assertTrue(screens['online_storage']);self.assertFalse(screens['complete_action_median'])

    def test_research_modes_remain_blocked_with_or_without_confirmation(self):
        for mode in ['development','final']:
            for confirmed in [False,True]:
                with tempfile.TemporaryDirectory() as td:
                    output=Path(td)/'blocked';cmd=[sys.executable,str(runner.ROOT/'run_controller_v3_compatibility_gate.py'),'--mode',mode,'--output',str(output)]
                    if confirmed:cmd.append('--confirm-'+mode+'-run')
                    result=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
                    self.assertNotEqual(result.returncode,0);self.assertIn('disabled',result.stderr);self.assertFalse(output.exists())

if __name__=='__main__':
    output=Path(sys.argv[1]);suite=unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(previous.Mechanics))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(Compatibility))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    runner.g.r.write(output,dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        source_sha256={n:runner.g.r.old.sha(runner.ROOT/n) for n in runner.SOURCES},new_simulator_steps=0,optimizer_updates=0))
    sys.exit(0 if result.wasSuccessful() else 1)
