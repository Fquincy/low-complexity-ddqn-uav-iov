"""Read-only progress and harmless subprocess tests; no model imports."""
import argparse
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import value_gradient_isolation_progress as progress
import value_gradient_isolation_process_monitor as process

ROOT=Path(__file__).resolve().parent
FIXTURE=ROOT/'validation/value_gradient_isolation_comparison_fixture_2026-11-20_01'
SPEC=ROOT/'VALUE_GRADIENT_ISOLATION_COMPARISON_PROTOCOL_2026-11-20.json'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


class Monitoring(unittest.TestCase):
    def synthetic(self,base):
        spec=progress.read(SPEC)
        plan=dict(mode='development',kind='SYNTHETIC_MONITOR_PLAN_NOT_RESEARCH',fits=3,
            online_episodes=50,eval_episodes=16,warmup=128,bc_steps=200)
        write(base/'run/runtime_plan.json',plan)
        return spec

    def test_01_complete_fixture_counters(self):
        row=progress.snapshot(FIXTURE)
        self.assertEqual(row['saved'],row['expected']);self.assertTrue(row['audit_complete'])
        self.assertFalse(row['archive_receipt_present'])
        self.assertEqual(row['saved']['total_slots'],2200)
        self.assertEqual(row['saved']['deployment_exports'],18)

    def test_02_development_denominators_without_simulation(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);self.synthetic(base);row=progress.snapshot(base)
            expected=dict(demo_slots=1200,online_slots=48000,evaluation_slots=17920,
                total_slots=67120,final_checkpoints=27,deployment_exports=27,
                completed_fit_bc_updates=4800,completed_fit_td_updates=44928,
                completed_fit_retention_batches=22464)
            self.assertEqual(row['expected'],expected);self.assertTrue(all(v==0 for v in row['saved'].values()))

    def test_03_partial_saved_counts_and_text(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);self.synthetic(base)
            write(base/'run/demo_0/episode_000.json',{})
            write(base/'run/P/fit_0/train_000.json',{})
            write(base/'run/evaluation/P/0/episode_000.json',{})
            write(base/'run/P/fit_0/counts.json',dict(bc_updates=200,updates=1872,retention_batches=1872))
            row=progress.snapshot(base)
            self.assertEqual(row['saved']['total_slots'],120)
            self.assertEqual(row['saved']['completed_fit_td_updates'],1872)
            self.assertIn('Student training slots: 40 / 48000',progress.format_text(row))

    def test_04_separate_audit_archive_and_failure_stages(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);self.synthetic(base);write(base/'progress.json',dict(stage='artifacts_complete'))
            self.assertEqual(progress.snapshot(base)['stage'],'FULL_ARTIFACT_AUDIT_RUNNING_OR_PENDING')
            write(base/'completion.json',dict(audit_passed=True))
            self.assertEqual(progress.snapshot(base)['stage'],'AUDITED_COMPLETE_ARCHIVE_NOT_VERIFIED')
            write(base/'archive_verification.json',dict(passed=True))
            self.assertEqual(progress.snapshot(base)['stage'],'AUDITED_AND_ARCHIVED_COMPLETE')
            (base/'failure.txt').write_text('Synthetic failure')
            self.assertEqual(progress.snapshot(base)['stage'],'STOPPED_INSPECT_FAILURE_RECORD')

    def test_05_warning_stderr_does_not_change_zero_exit(self):
        result=process.supervise([sys.executable,'-c',"import warnings; warnings.warn('synthetic monitoring warning'); print('child completed')"],
            self.output/'warning_child',progress_output=FIXTURE,poll_seconds=.01)
        self.assertEqual(result['child_exit_code'],0);self.assertTrue(result['child_exited_successfully'])
        self.assertGreater(result['stderr_bytes'],0);self.assertFalse(result['audit_pass_claim'])
        self.assertIn('Python child exit code: 0',(self.output/'warning_child/LIVE_STATUS.txt').read_text())

    def test_06_nonzero_child_is_failure_even_without_stderr(self):
        result=process.supervise([sys.executable,'-c','raise SystemExit(7)'],self.output/'nonzero_child',poll_seconds=.01)
        self.assertEqual(result['child_exit_code'],7);self.assertFalse(result['child_exited_successfully'])
        self.assertEqual(result['stderr_bytes'],0)

    def test_07_child_start_failure_is_preserved(self):
        with self.assertRaises(OSError):process.supervise([str(self.output/'nonexistent_python.exe')],self.output/'start_failure')
        self.assertFalse(progress.read(self.output/'start_failure/child_start_failure.json')['child_created'])

    def test_08_existing_monitor_directory_cannot_be_reused(self):
        path=self.output/'existing';path.mkdir();(path/'preserved.txt').write_text('keep')
        with self.assertRaises(FileExistsError):process.supervise([sys.executable,'-c','pass'],path)
        self.assertEqual((path/'preserved.txt').read_text(),'keep')

    def test_09_monitor_rejects_writes_inside_immutable_run(self):
        target=FIXTURE/'run/forbidden_monitor_output'
        with self.assertRaises(ValueError):process.supervise([sys.executable,'-c','pass'],target,progress_output=FIXTURE)
        self.assertFalse(target.exists())

    def test_10_unavailable_progress_still_records_actual_exit(self):
        result=process.supervise([sys.executable,'-c',"print('no simulation')"],self.output/'no_progress',
            progress_output=self.output/'missing_plan',poll_seconds=.01)
        self.assertEqual(result['child_exit_code'],0)
        self.assertIsNotNone(progress.read(self.output/'no_progress/LIVE_STATUS.json')['progress_read_error'])


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);Monitoring.output=out
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Monitoring))
    (out/'test_log.txt').write_text(stream.getvalue(),encoding='utf-8');print(stream.getvalue())
    write(out/'tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        kind='MONITORING_AND_PROCESS_FIXTURES_NOT_RESEARCH',new_models_created=0,new_simulator_steps=0,new_optimizer_updates=0,
        new_teacher_calls=0,reserved_research_seeds_used=False,research_launch_authorized=False))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');raise SystemExit(main(parser.parse_args().output))
