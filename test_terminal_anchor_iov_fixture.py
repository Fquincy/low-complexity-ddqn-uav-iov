import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import run_terminal_anchor_iov_fixture as r

class Engineering(unittest.TestCase):
    def test_research_guards(self):
        for mode in ['development','final']:
            with tempfile.TemporaryDirectory() as td:
                out=Path(td)/'blocked'
                proc=subprocess.run([sys.executable,str(r.ROOT/'run_terminal_anchor_iov_fixture.py'),'--mode',mode,'--output',str(out)],capture_output=True,text=True,timeout=60)
                self.assertNotEqual(proc.returncode,0);self.assertFalse(out.exists());self.assertIn('disabled',proc.stderr)
    def test_deployed_terminal_rejection(self):
        x=np.zeros(533,np.float32);x[-1]=1
        with self.assertRaises(ValueError):r.deployed(None,'price_both',x)
        x[-1]=.0123
        with self.assertRaises(ValueError):r.deployed(None,'price_both',x)
    def test_paired_iov_fixture_audit_and_tamper(self):
        result=r.fixture(OUTPUT)
        self.assertEqual(result['episodes_checked'],54);self.assertEqual(result['updates_checked'],876)
        path=OUTPUT/'inference'/'timings.json';original=path.read_bytes();path.write_bytes(original+b' ')
        try:
            with self.assertRaises(AssertionError):r.audit(OUTPUT)
        finally:path.write_bytes(original)

if __name__=='__main__':
    OUTPUT=Path(sys.argv[1]);r.p.existing.configure()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    r.io.write(r.ROOT/'validation/terminal_anchor_iov_fixture_tests_2026-11-06.json',dict(
        passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        source_sha256={name:r.io.sha(r.ROOT/name) for name in dict.fromkeys(r.SOURCES)},
        research_launched=False,word_thesis_changed=False))
    sys.exit(0 if result.wasSuccessful() else 1)
