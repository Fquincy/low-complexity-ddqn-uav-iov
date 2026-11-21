"""Launcher wiring tests using temporary files/mocks, never research training."""
import argparse
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import launch_value_gradient_isolation_comparison_2026_11_21 as worker
import supervise_value_gradient_isolation_launch_2026_11_21 as supervisor


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def small_archive_source(root):
    out=root/'source';out.mkdir();(out/'run').mkdir();(out/'run/fixture.txt').write_text('SYNTHETIC_NOT_RESEARCH')
    for name,value in [('completion.json',dict(audit_passed=True)),('saved_artifact_audit.json',dict(passed=True)),
        ('launch_authorization.json',dict(kind='SYNTHETIC_NOT_RESEARCH')),('launch_binding.json',{}),('audit_phase_timing.json',dict(wall_ns=1))]:
        write(out/name,value)
    return out


class Launch(unittest.TestCase):
    def test_01_confirmation_before_preflight_or_outputs(self):
        with patch.object(worker,'preflight',side_effect=AssertionError('preflight called')):
            for confirmation in [None,'wrong']:
                with self.assertRaisesRegex(RuntimeError,'confirmation'):worker.execute(confirmation)
        with self.assertRaisesRegex(RuntimeError,'confirmation'):supervisor.execute(None)

    def test_02_source_binding_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);write(root/'binding.json',dict(authorization=worker.AUTHORIZATION,sha256={'source.py':'wrong'}))
            (root/'source.py').write_text('SYNTHETIC_NOT_RESEARCH')
            with patch.object(worker,'ROOT',root),patch.object(worker,'BINDING',root/'binding.json'):
                with self.assertRaisesRegex(RuntimeError,'Launch-bound file changed'):worker.preflight()

    def test_03_archive_hash_and_inventory_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=small_archive_source(root);archive=root/'archive'
            inventory=worker.archive_completed(out,archive,1,time.perf_counter_ns())
            self.assertGreater(len(inventory),6)
            for name,digest in inventory.items():self.assertEqual(worker.runner.sha(archive/name),digest)
            self.assertEqual(worker.runner.sha(out/'archive_verification.json'),worker.runner.sha(archive/'archive_verification.json'))

    def test_04_failed_audit_cannot_archive(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=small_archive_source(root);write(out/'completion.json',dict(audit_passed=False))
            with self.assertRaisesRegex(RuntimeError,'audit must pass'):worker.archive_completed(out,root/'archive',1,time.perf_counter_ns())
            self.assertFalse((root/'archive').exists())

    def test_05_archive_copy_corruption_is_rejected_and_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=small_archive_source(root);archive=root/'archive';original=worker.shutil.copy2
            def corrupt(source,destination):
                result=original(source,destination)
                if Path(source).name=='completion.json':Path(destination).write_text('corrupted')
                return result
            with patch.object(worker.shutil,'copy2',corrupt):
                with self.assertRaisesRegex(RuntimeError,'hash mismatch'):worker.archive_completed(out,archive,1,time.perf_counter_ns())
            self.assertTrue(archive.exists());self.assertFalse((archive/'archive_verification.json').exists())

    def test_06_supervisor_fixed_command_and_warning_exit_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);script=root/'launch_value_gradient_isolation_comparison_2026_11_21.py'
            script.write_text("import sys; print('SYNTHETIC_CHILD_NOT_RESEARCH'); print('synthetic warning',file=sys.stderr)\n")
            binding=dict(authorization=worker.AUTHORIZATION,sha256={script.name:worker.runner.sha(script)},
                python_executable=sys.executable,monitoring_output='monitor',output='no_research_output')
            write(root/'binding.json',binding)
            with patch.object(supervisor,'ROOT',root),patch.object(supervisor,'BINDING',root/'binding.json'):
                result=supervisor.execute(supervisor.CONFIRM)
            self.assertEqual(result['child_exit_code'],0);self.assertGreater(result['stderr_bytes'],0)
            self.assertFalse((root/'no_research_output').exists())

    def successful_or_failed_worker(self,fail):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);checks=root/'checks';binding=dict(output='out',archive=str(root/'archive'),user_instruction='SYNTHETIC_NOT_RESEARCH')
            write(root/'binding.json',binding);original=worker.auditor.audit
            def fake_run(out,mode,confirmation):
                self.assertTrue(worker.runner.DEVELOPMENT_RELEASED);self.assertEqual(mode,'development')
                Path(out).mkdir();(Path(out)/'run').mkdir();(Path(out)/'run/fixture.txt').write_text('SYNTHETIC_NOT_RESEARCH')
                if fail:raise ValueError('synthetic worker failure')
                receipt=worker.auditor.audit(Path(out)/'run')
                write(Path(out)/'completion.json',dict(audit_passed=True));write(Path(out)/'saved_artifact_audit.json',receipt)
                return receipt
            with patch.object(worker,'ROOT',root),patch.object(worker,'BINDING',root/'binding.json'),\
                 patch.object(worker,'preflight',return_value=(binding,checks)),\
                 patch.object(worker.runner,'run',side_effect=fake_run),patch.object(worker.auditor,'audit',return_value=dict(passed=True)):
                before=worker.auditor.audit
                if fail:
                    with self.assertRaisesRegex(ValueError,'synthetic worker failure'):worker.execute(worker.runner.CONFIRM)
                    self.assertTrue((root/'out/failure.txt').exists());self.assertFalse((root/'archive').exists())
                else:
                    result=worker.execute(worker.runner.CONFIRM)
                    self.assertEqual(result['status'],'AUDITED_AND_ARCHIVED_COMPLETE')
                    self.assertTrue((root/'out/audit_phase_timing.json').exists())
                    self.assertTrue(worker.runner.read(root/'archive/archive_verification.json')['passed'])
                self.assertFalse(worker.runner.DEVELOPMENT_RELEASED);self.assertIs(worker.auditor.audit,before)
            self.assertIs(worker.auditor.audit,original)

    def test_07_worker_success_enables_only_memory_and_restores(self):self.successful_or_failed_worker(False)
    def test_08_worker_failure_preserves_data_and_restores(self):self.successful_or_failed_worker(True)

    def test_09_existing_archive_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=small_archive_source(root);archive=root/'archive';archive.mkdir();(archive/'keep.txt').write_text('keep')
            with self.assertRaises(FileExistsError):worker.archive_completed(out,archive,1,time.perf_counter_ns())
            self.assertEqual((archive/'keep.txt').read_text(),'keep')

    def test_10_defaults_and_final_mode_remain_closed(self):
        self.assertFalse(worker.runner.DEVELOPMENT_RELEASED)
        self.assertFalse(worker.runner.vgi.DEVELOPMENT_RELEASED);self.assertFalse(worker.runner.vgi.FINAL_RELEASED)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'blocked'
            with self.assertRaisesRegex(RuntimeError,'disabled'):worker.runner.run(out,'final',worker.runner.CONFIRM)
            self.assertFalse(out.exists())


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Launch))
    (out/'test_log.txt').write_text(stream.getvalue(),encoding='utf-8');print(stream.getvalue())
    names=[Path(worker.__file__).name,Path(supervisor.__file__).name,Path(__file__).name]
    source=out/'source_snapshot';source.mkdir()
    for name in names:(source/name).write_bytes((worker.ROOT/name).read_bytes())
    write(out/'verification.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        kind='LAUNCH_WRAPPER_SYNTHETIC_NOT_RESEARCH',source_sha256={name:worker.runner.sha(worker.ROOT/name) for name in names},
        new_research_simulator_steps=0,new_research_optimizer_updates=0,new_teacher_calls=0,new_models_created=0,
        harmless_synthetic_python_children=1,reserved_research_seeds_used=False,research_launched=False,word_thesis_changed=False))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');raise SystemExit(main(parser.parse_args().output))
