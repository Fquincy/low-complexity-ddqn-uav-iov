"""Seventeen-arm research preparation: reduced plan only, never research."""
import copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_anchor_research_route_v3 as r
import audit_anchor_research_route_v3 as a

class Checks(unittest.TestCase):
    def test_declared_route_is_installed_and_terminal_rejected(self):
        self.assertFalse(r.RESEARCH_ENABLED)
        x=np.zeros(533,np.float32);x[-1]=1.
        with self.assertRaises(ValueError):r.runtime_values(None,'price_both',x)
        with patch.object(r.runtime_adapter,'values',return_value='route_marker') as call:
            self.assertEqual(r.runtime_values(None,'price_both',x),'route_marker')
            call.assert_called_once()
    def test_protocol_budget_and_prospective_scope(self):
        p=r.protocol();self.assertEqual(len(p['arms']),17)
        self.assertEqual(p['development']['training_slots'],680000)
        self.assertEqual(p['development']['optimizer_updates'],593045)
        self.assertEqual(p['development']['target_copies'],2890)
        self.assertEqual(p['value_screen_scope']['gated_arms'],['price_both'])
        self.assertEqual(p['comparisons']['no_price_arm'],'no_price_both')
        self.assertEqual(p['final']['status'],'reserved_untouched_no_launch')
    def test_original_replay_indices_and_dtypes_preserved(self):
        old=r.base.old.Replay(8);new=r.TerminalBuffer(8)
        for i in range(10):
            row=(np.zeros(533,np.float32),np.zeros(32,np.int64),-.3,np.zeros(533,np.float32),i in [0,9])
            old.add(*row);new.add(*row)
        first=old.sample(12,np.random.default_rng(99605));second=new.sample(12,np.random.default_rng(99605))
        for x,y in zip(first,second):np.testing.assert_array_equal(x,y);self.assertEqual(x.dtype,y.dtype)
        self.assertEqual(new.terminal,{1})
    def test_candidate_value_gate_and_cost_failure(self):
        p=r.protocol();names=p['comparisons']['primary_against_each_of']+p['comparisons']['heuristics']+['no_price_both']+p['comparisons']['correction_ablations']
        comparisons={n:dict(per_fit=[dict(seed=i,return_difference=3.,deadline_difference=.02) for i in range(5)],mean_return_difference=3.,mean_deadline_difference=.02) for n in names}
        good=dict(normalized_rmse=.1,absolute_bias=.1,near_zero_large_return_fraction=0.,terminal_abs_error_p95=.1)
        values={arm:[dict(good) for _ in range(5)] for arm in r.ARMS}
        values['additive128_unbounded'][0]['normalized_rmse']=100
        screens=r.quality_screens(comparisons,values,5);self.assertTrue(all(screens.values()))
        self.assertFalse(r.aggregate_screens(screens,dict(all_fit_cost_screens_pass=False)))
        values['price_both'][0]['normalized_rmse']=100
        self.assertFalse(r.quality_screens(comparisons,values,5)['value_each_fit'])
        comparisons['price_horizon_only']['per_fit'][0]['return_difference']=-1
        self.assertFalse(r.quality_screens(comparisons,values,5)['correction_ablations_each_fit'])
    def test_research_guards(self):
        for mode in ['development','final']:
            for confirmed in [False,True]:
                with tempfile.TemporaryDirectory() as td:
                    out=Path(td)/'blocked';cmd=[sys.executable,str(r.ROOT/'run_anchor_research_route_v3.py'),'--mode',mode,'--output',str(out)]
                    if confirmed:cmd.append('--confirm-'+mode+'-run')
                    proc=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
                    self.assertNotEqual(proc.returncode,0);self.assertIn('disabled',proc.stderr);self.assertFalse(out.exists())
                    with self.assertRaises(RuntimeError):r.run_plan(out,phase=mode)
                    self.assertFalse(out.exists())
    def test_complete_shared_workflow_and_saved_artifact_audit(self):
        result=r.run_plan(OUTPUT,fixture=True,multi_fixture=True)
        self.assertEqual(result['training_slots'],2720);self.assertEqual(result['optimizer_updates'],2482)
        self.assertEqual(result['target_copies'],306);self.assertEqual(result['checkpoint_count'],34)
        audit=a.audit_research(OUTPUT);self.assertEqual(audit['episodes_checked'],244)
        self.assertEqual(audit['final_inference']['complete_actions_replayed'],4352)
        self.assertEqual(audit['final_inference']['timing_rows_recomputed'],21760)
        path=OUTPUT/'ALL_FINAL_CHECKPOINTS_FREEZE.json';original=path.read_bytes();bad=r.read(path);bad['count']+=1;r.write(path,bad)
        try:
            with patch.object(a,'verify_manifest',return_value={}):
                with self.assertRaises(AssertionError):a.audit_research(OUTPUT)
        finally:path.write_bytes(original)
        a.verify_manifest(OUTPUT)

if __name__=='__main__':
    r.configure();OUTPUT=Path(sys.argv[1])
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    r.write(r.ROOT/'validation/anchor_research_route_v3_tests_2026-11-06.json',dict(passed=result.wasSuccessful(),
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        source_sha256={name:r.base.old.sha(r.ROOT/name) for name in r.SOURCES},
        protocol_sha256=r.base.old.sha(r.PROTOCOL),research_launched=False))
    sys.exit(0 if result.wasSuccessful() else 1)
