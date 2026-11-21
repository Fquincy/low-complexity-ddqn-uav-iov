"""Research binding, launch guards, independent diagnostics and full fixture."""
import copy
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import run_horizon_loss_research as r
import run_horizon_loss_iov_fixture as tested
import horizon_loss_research_workflow as workflow
import audit_horizon_loss_research as auditor
import audit_horizon_loss_research_reports as diagnostics

OUT = r.ROOT / 'validation/horizon_loss_research_binding_2026-11-14_01'
TEMP = Path(tempfile.mkdtemp(prefix='horizon_loss_research_binding_'))
FIXTURE = TEMP / 'workflow'

class Checks(unittest.TestCase):
    def test_01_protocol_budget_and_unchanged_learning(self):
        p = r.protocol()
        plan = r.plan(p, 'development')  # Read schedule only; do not initialize any model.
        fits = len(plan['model_seeds']) * len(p['learned_arms'])
        self.assertEqual(fits, 70)
        self.assertEqual(fits * plan['cfg']['episodes_per_fit'] * 40, 560000)
        self.assertEqual(fits * (8000-1024+1), 488390)
        self.assertEqual(fits * ((8000-1024+1)//200), 2380)
        self.assertEqual((fits+len(p['heuristics']))*len(plan['evaluation_seeds'])*40, 140160)
        self.assertEqual(fits*640, 44800)
        self.assertEqual(p['comparisons'], tested.protocol()['comparisons'])
        for name in ('update', 'episode', 'build', 'compact'):
            self.assertEqual(inspect.getsource(getattr(r, name)), inspect.getsource(getattr(tested, name)))
        self.assertFalse(r.DEVELOPMENT_RELEASED)
        self.assertFalse(p['authorization']['launch_authorized'])

    def test_02_api_cli_and_release_guards(self):
        for mode, token, message in [('development', None, 'confirmation'),
                                     ('development', 'WRONG', 'confirmation'),
                                     ('development', r.CONFIRM, 'disabled'),
                                     ('final', r.CONFIRM, 'disabled')]:
            path = TEMP / ('denied_'+mode+str(token))
            with self.assertRaisesRegex(RuntimeError, message):
                workflow.run(path, mode, token)
            self.assertFalse(path.exists())
            args = [sys.executable, str(r.ROOT/'horizon_loss_research_workflow.py'),
                    '--mode', mode, '--output', str(path)]
            if token is not None:
                args += ['--confirm-development', token]
            proc = subprocess.run(args, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(proc.returncode, 0)
            self.assertFalse(path.exists())
        # Even a future release must still require the correct explicit token.
        with patch.object(r, 'DEVELOPMENT_RELEASED', True), patch.object(r, 'protocol') as checked:
            with self.assertRaisesRegex(RuntimeError, 'confirmation'):
                r.run(TEMP/'still_denied', 'development')
            checked.assert_not_called()

    def test_03_complete_bound_workflow(self):
        result = workflow.run(FIXTURE)
        self.assertTrue(result['artifact_audit_passed'])
        self.assertTrue(result['deployment_storage_audit']['passed'])
        self.assertFalse(result['quality_evidence'])
        self.assertFalse(result['research_launched'])
        v = r.read(FIXTURE/'run/verification.json')
        self.assertEqual([v[k] for k in ('checkpoints','training_slots','optimizer_updates','target_copies','evaluation_slots','raw_timing_rows')],
                         [28,2240,2044,252,2480,17920])
        a = r.read(FIXTURE/'run/saved_artifact_audit.json')
        self.assertEqual(a['diagnostic_reports_independently_verified']['training_fit_reports'], 28)
        self.assertEqual(a['diagnostic_reports_independently_verified']['policy_reports'], 31)
        self.assertEqual((a['episodes'],a['task_records']),(118,141600))
        report = r.read(FIXTURE/'run/analysis.json')
        self.assertEqual(len(report['comparisons']),16)
        self.assertEqual(len(report['auxiliary_contrasts']),9)
        self.assertIsNone(report['all_scientific_screens_passed'])
        self.assertFalse(report['screens_applied'])

    def test_04_rehashed_diagnostic_tamper_and_output_reuse(self):
        out = FIXTURE/'run'
        plan, p = r.read(out/'runtime_plan.json'), r.protocol()
        for name, mutate in [
            ('training_residuals_by_horizon.json', lambda x: x['fits']['screened_price24/0'][next(iter(x['fits']['screened_price24/0']))]['normalized'].update(mean=999)),
            ('remote_decision_diagnostics.json', lambda x: x['policies']['screened_price24/0']['misses_by_time_destination'][0].__setitem__(0,999))]:
            path, manifest = out/name, out/'run_manifest.json'
            original, original_manifest = path.read_bytes(), manifest.read_bytes()
            try:
                value = r.read(path); mutate(value); r.write(path,value)
                files = r.read(manifest); files['files'][name] = r.sha(path); r.write(manifest,files)
                with self.assertRaises(AssertionError): diagnostics.audit(out, plan, p)
            finally:
                path.write_bytes(original); manifest.write_bytes(original_manifest)
        extra = out/'unrecorded.txt'; extra.write_text('unexpected')
        try:
            with self.assertRaises(AssertionError): auditor.audit(out,False)
        finally: extra.unlink()
        with self.assertRaises(FileExistsError): workflow.run(FIXTURE)

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=False)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if FIXTURE.exists(): shutil.copytree(FIXTURE, OUT/'workflow')
    r.write(OUT/'tests.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
        errors=len(result.errors), failures=len(result.failures), temporary_workflow=str(FIXTURE),
        research_launched=False, reserved_development_seeds_consumed=False, quality_evidence=False))
    r.write(OUT/'evidence_manifest.json', {x.relative_to(OUT).as_posix():r.sha(x) for x in OUT.rglob('*') if x.is_file()})
    raise SystemExit(0 if result.wasSuccessful() else 1)
