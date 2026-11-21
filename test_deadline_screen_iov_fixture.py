"""End-to-end synthetic integration and launch-boundary tests."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
import run_deadline_screen_iov_fixture as r
import audit_deadline_screen_iov_fixture as a
OUT=Path(sys.argv[1])
class Checks(unittest.TestCase):
    def test_complete_fixture_and_independent_audit(self):
        result=r.run(OUT);self.assertEqual(result['optimizer_updates'],1022)
        audit=a.audit(OUT);self.assertEqual(audit['episodes'],56);self.assertTrue(audit['evaluation_invariance'])
    def test_development_final_guards_before_output(self):
        for mode in ['development','final']:
            with tempfile.TemporaryDirectory() as td:
                out=Path(td)/'forbidden'
                with self.assertRaises(RuntimeError):r.run(out,mode)
                proc=subprocess.run([sys.executable,str(r.R/'run_deadline_screen_iov_fixture.py'),'--mode',mode,'--output',str(out)],capture_output=True,text=True,timeout=60)
                self.assertNotEqual(proc.returncode,0);self.assertIn('disabled',proc.stderr);self.assertFalse(out.exists())
    def test_output_and_summary_tamper_guards(self):
        with self.assertRaises(FileExistsError):r.run(OUT)
        path=OUT/'verification.json';original=path.read_bytes();bad=r.read(path);bad['optimizer_updates']+=1;r.write(path,bad)
        try:
            with self.assertRaises(AssertionError):a.audit(OUT)
        finally:path.write_bytes(original)
        self.assertEqual(r.sha(path),r.read(OUT/'run_manifest.json')['files']['verification.json'])
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    r.write(r.R/'validation/deadline_screen_iov_fixture_tests_2026-11-11.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),research_launched=False,
        source_sha256={n:r.sha(r.R/n) for n in r.SOURCES}))
    sys.exit(0 if result.wasSuccessful() else 1)
