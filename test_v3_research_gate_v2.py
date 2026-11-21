"""Engineering only: shared research orchestration, auditing and failure checks."""
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_v3_research_gate_v2 as runner
import audit_v3_research_gate_v2 as auditor

RETAIN = None
EVIDENCE = {}

class Components(unittest.TestCase):
    def test_archive_mapping_preserves_historical_evidence(self):
        result=runner.verify_prior_engineering()
        self.assertEqual(len(result['files']),114)
        self.assertNotEqual(runner.base.old.sha(runner.ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json'),
                            result['files']['LOW_COMPLEXITY_PROJECT_STATUS.json'])

    def test_synthetic_schedule_disjoint_and_frozen_research_unchanged(self):
        plan=runner.synthetic_multifit_plan();runner.validate_plan(plan)
        self.assertEqual(runner.base.old.sha(runner.PROTOCOL),
            '5feece24ac058b547140b0513f4991a3f8035536b59b4002893f0ae1fc8d96d9')
        self.assertEqual(runner.protocol()['development']['training_slots'],440000)
        unsafe=copy.deepcopy(plan);unsafe['model_seeds'][0]=41001
        with self.assertRaises(ValueError):runner.validate_plan(unsafe)
        duplicate=copy.deepcopy(plan);duplicate['evaluation_seeds']*=2
        with self.assertRaises(ValueError):runner.validate_plan(duplicate)

    def test_cost_failure_prevents_aggregate_success(self):
        self.assertTrue(runner.aggregate_screens({'return':True,'value':True},{'all_fit_cost_screens_pass':True}))
        self.assertFalse(runner.aggregate_screens({'return':True,'value':True},{'all_fit_cost_screens_pass':False}))
        self.assertFalse(runner.aggregate_screens({'return':False},{'all_fit_cost_screens_pass':True}))
        self.assertFalse(runner.aggregate_screens({'evaluated':False},{'all_fit_cost_screens_pass':True}))

    def test_duplicate_missing_or_unpaired_rows_rejected(self):
        def row(fit,seed):return dict(model_seed=fit,seed=seed,workload_sha256=str(seed),return_sum=1.,deadline_success=.9)
        evaluations={arm:[row(f,s) for f in [99261,99267] for s in [992690100,992690101]] for arm in runner.ARMS}
        heuristics={'all_local':[dict(row(None,s),return_sum=0.) for s in [992690100,992690101]]}
        comparisons=runner.compare_all(evaluations,heuristics)
        self.assertEqual(len(comparisons['plain24_nonpositive']['per_fit']),2)
        for mode in ['duplicate','missing','hash']:
            broken=copy.deepcopy(evaluations)
            if mode=='duplicate':broken['plain24_nonpositive'].append(broken['plain24_nonpositive'][0])
            if mode=='missing':broken['plain24_nonpositive'].pop()
            if mode=='hash':broken['plain24_nonpositive'][0]['workload_sha256']='changed'
            with self.assertRaises(ValueError):runner.compare_all(broken,heuristics)

    def test_quality_screens_require_all_five_fit_identities(self):
        p=runner.protocol()
        names=p['comparisons']['primary_against_each_of']+p['comparisons']['heuristics']+['amortized_no_price24']
        comparisons={name:dict(per_fit=[dict(seed=i,return_difference=2.,deadline_difference=.02) for i in range(5)],
             mean_return_difference=2.,mean_deadline_difference=.02) for name in names}
        values={arm:[dict(normalized_rmse=.1,absolute_bias=.1,near_zero_large_return_fraction=0.,terminal_abs_error_p95=.1) for _ in range(5)] for arm in runner.ARMS}
        self.assertTrue(all(runner.quality_screens(comparisons,values,5).values()))
        comparisons['plain24_nonpositive']['per_fit'].pop()
        with self.assertRaises(ValueError):runner.quality_screens(comparisons,values,5)

    def test_cli_and_direct_api_block_research_before_output(self):
        guards=[]
        for phase in ['development','final']:
            for confirmed in [False,True]:
                with tempfile.TemporaryDirectory() as td:
                    out=Path(td)/'blocked'
                    cmd=[sys.executable,str(runner.ROOT/'run_v3_research_gate_v2.py'),'--mode',phase,'--output',str(out)]
                    if confirmed:cmd.append('--confirm-'+phase+'-run')
                    proc=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
                    self.assertNotEqual(proc.returncode,0);self.assertFalse(out.exists())
                    self.assertIn('disabled',proc.stderr)
                    guards.append(dict(phase=phase,confirmed=confirmed,output_created=False,exit_code=proc.returncode))
                    with self.assertRaises(RuntimeError):runner.run_plan(out,phase=phase)
                    self.assertFalse(out.exists())
        EVIDENCE['research_guards']=guards

class Integration(unittest.TestCase):
    def test_shared_multifit_orchestration_and_auditor(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_NOT_RESEARCH_v3_multifit_') as td:
            out=Path(td)/'fixture'
            try:
                result=runner.run_plan(out,fixture=True,multi_fixture=True)
                self.assertEqual(result['training_slots'],1760)
                self.assertEqual(result['optimizer_updates'],1606)
                self.assertEqual(result['target_copies'],198)
                self.assertEqual(result['checkpoint_count'],22)
                audit=auditor.audit_research(out)
                self.assertEqual(audit['state'],'passed_v3_synthetic_shared_research_audit')
                self.assertEqual(audit['episodes_checked'],160)
                self.assertEqual(audit['final_inference']['complete_actions_replayed'],2816)
                self.assertEqual(audit['final_inference']['timing_rows_recomputed'],14080)
                self.assertFalse(audit['research_quality_evidence'])
                EVIDENCE['integration_result']=result;EVIDENCE['integration_audit']=audit
                # Bypass outer manifest only for deliberate semantic corruption:
                # this proves the independent checks reject bad content too.
                freeze_path=out/'ALL_FINAL_CHECKPOINTS_FREEZE.json'
                original=freeze_path.read_bytes();bad=runner.read(freeze_path);bad['count']+=1
                runner.write(freeze_path,bad)
                try:
                    with patch.object(auditor,'verify_manifest',return_value={}):
                        with self.assertRaises(AssertionError):auditor.audit_research(out)
                finally:freeze_path.write_bytes(original)
                folder=out/'frozen_checkpoint_inference';stage_path=folder/'TIMING_STAGE_ORDER.json'
                original=stage_path.read_bytes();bad=runner.read(stage_path);bad['first_timed_call_ns']=0
                runner.write(stage_path,bad)
                try:
                    with self.assertRaises(AssertionError):auditor.audit_timing_freeze(folder)
                finally:stage_path.write_bytes(original)
                extra=next((folder/'exports').glob('*.extra.npy'));original=extra.read_bytes()
                extra.write_bytes(original+b'corrupt')
                try:
                    with self.assertRaises(AssertionError):auditor.audit_timing_freeze(folder)
                finally:extra.write_bytes(original)
                verify=out/'verification.json';original=verify.read_bytes();verify.write_bytes(original+b' ')
                try:
                    with self.assertRaises(AssertionError):auditor.verify_manifest(out)
                finally:verify.write_bytes(original)
                auditor.verify_manifest(out)
                EVIDENCE['semantic_corruption_checks']=['checkpoint_inventory','freeze_after_first_call','decoder_payload_change','manifest_tampering']
            finally:
                if RETAIN and out.exists():shutil.copytree(out,RETAIN)

if __name__=='__main__':
    runner.configure()
    RETAIN=Path(sys.argv[1]) if len(sys.argv)>1 else None
    if RETAIN and RETAIN.exists():raise FileExistsError(RETAIN)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    EVIDENCE.update(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        research_training_launched=False,thesis_changed=False,protocol_sha256=runner.base.old.sha(runner.PROTOCOL),
        source_sha256={name:runner.base.old.sha(runner.ROOT/name) for name in runner.SOURCES})
    path=runner.ROOT/'validation/v3_engineering_gate_v2_tests_2026-11-06.json'
    runner.write(path,EVIDENCE)
    sys.exit(0 if result.wasSuccessful() else 1)
