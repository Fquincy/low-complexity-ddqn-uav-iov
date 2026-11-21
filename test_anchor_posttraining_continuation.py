"""Meaningful continuation regressions; isolated existing synthetic weights only."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
import run_anchor_posttraining_continuation as c
import audit_anchor_posttraining_continuation as a
OUT=Path(sys.argv[1])
class Checks(unittest.TestCase):
    def test_complete_continuation_and_full_audit_without_training(self):
        with patch.object(c,'build',side_effect=AssertionError('training build forbidden')),patch.object(torch.optim,'Adam',side_effect=AssertionError('optimizer construction forbidden')),patch.object(c.base,'update',side_effect=AssertionError('optimizer update forbidden')):
            result=c.run_continuation(c.FIXTURE_SOURCE,OUT,fixture=True)
        self.assertEqual(result['new_training_slots'],0);self.assertEqual(result['new_optimizer_updates'],0)
        self.assertEqual(result['checkpoint_count'],34)
        audit=a.audit_research(OUT)
        self.assertEqual(audit['checkpoints_checked'],34);self.assertEqual(audit['episodes_checked'],244)
        self.assertEqual(audit['optimizer_updates_checked'],2482)
        self.assertEqual(audit['new_optimizer_updates'],0)
        self.assertEqual(audit['final_inference']['complete_actions_replayed'],4352)
        self.assertEqual(audit['final_inference']['timing_rows_recomputed'],21760)
        c.audit_continuation_inputs(OUT)
    def test_research_and_final_cli_and_api_guards(self):
        for mode in ['development','final']:
            for confirm in [False,True]:
                with tempfile.TemporaryDirectory() as td:
                    target=Path(td)/'blocked'
                    cmd=[sys.executable,str(c.ROOT/'run_anchor_posttraining_continuation.py'),'--mode',mode,'--source',str(c.RESEARCH_SOURCE),'--output',str(target)]
                    if confirm:cmd.append('--confirm-development-run')
                    result=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
                    self.assertNotEqual(result.returncode,0);self.assertIn('disabled',result.stderr);self.assertFalse(target.exists())
                    with self.assertRaises(RuntimeError):c.run_continuation(c.RESEARCH_SOURCE,target)
                    self.assertFalse(target.exists())
    def test_reserved_source_rejected_in_fixture_mode(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'forbidden'
            with self.assertRaisesRegex(ValueError,'fixture source required'):c.run_continuation(c.RESEARCH_SOURCE,out,fixture=True)
            self.assertFalse(out.exists())
    def test_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileExistsError):c.run_continuation(c.FIXTURE_SOURCE,Path(td),fixture=True)
    def test_summary_tampering_rejected_without_retiming(self):
        path=OUT/'verification.json';original=path.read_bytes();bad=c.read(path);bad['new_optimizer_updates']=1;c.write(path,bad)
        try:
            with self.assertRaises(AssertionError):a.audit_research(OUT)
        finally:path.write_bytes(original)
        a.verify_manifest(OUT)
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    c.write(c.ROOT/'validation/anchor_posttraining_continuation_tests_2026-11-11.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),research_launched=False,
        source_sha256={n:c.base.old.sha(c.ROOT/n) for n in [*c.CONTINUATION_SOURCES,'test_anchor_posttraining_continuation.py']}))
    sys.exit(0 if result.wasSuccessful() else 1)
