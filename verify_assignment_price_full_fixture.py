"""Saved full fixture audit, rejection tests and disabled launch checks."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
import run_assignment_price_full_fixture as w
import audit_assignment_price_full_fixture as audit

BASE=w.r.ROOT/'validation/assignment_price_full_engineering_2026-11-15_01'
OUT=Path(w.r.read(BASE/'temporary_output_location.json')['path'])

class Checks(unittest.TestCase):
    def test_full_artifact_analysis_and_values(self):
        report=audit.audit(OUT);self.assertTrue(report['passed'])
        self.assertEqual(report['optimizer_rows'],1095);self.assertEqual(report['task_records'],58800)
        w.r.write(BASE/'independent_audit.json',report)

    def test_auditor_rejects_artifact_and_diagnostic_corruption(self):
        path=OUT/'counters.json';saved=path.read_bytes()
        try:
            path.write_bytes(saved+b' ')
            with self.assertRaises(AssertionError):audit.audit(OUT)
        finally:path.write_bytes(saved)
        path=OUT/'fixture_analysis.json';saved=path.read_bytes()
        try:
            value=json.loads(saved);value['comparisons']['coupled_no_sweep']['per_fit'][0]['return_difference']+=1
            w.r.write(path,value)
            with self.assertRaises(AssertionError):audit.check_analysis(OUT,w.protocol())
        finally:path.write_bytes(saved)

    def test_api_cli_guards_before_output_and_seeds(self):
        evidence=[]
        for mode,confirmation in [('development',None),('development',w.CONFIRM),('final',None),('unknown',None)]:
            output=OUT.parent/f'blocked_{mode}_{confirmation}'
            with self.assertRaises(RuntimeError):w.run(output,mode,confirmation)
            self.assertFalse(output.exists())
            cmd=[sys.executable,str(w.r.ROOT/'run_assignment_price_full_fixture.py'),'--output',str(output),'--mode',mode]
            if confirmation:cmd+=['--confirm-development',confirmation]
            process=subprocess.run(cmd,capture_output=True,text=True,cwd=w.r.ROOT)
            self.assertNotEqual(process.returncode,0);self.assertFalse(output.exists())
            self.assertTrue(any(s in process.stderr for s in ['confirmation required','pending separate launch review','fixture only']))
            evidence.append(dict(mode=mode,confirmation=confirmation,returncode=process.returncode,stderr=process.stderr,output_created=False))
        w.r.write(BASE/'cli_guard_evidence.json',evidence)

    def test_saved_heuristic_choices_reproduced_from_observations(self):
        folder=OUT;policy='observed_load_radio';row=audit.load(folder/f'heuristic_{policy}.json.gz')
        arrays=np.load(folder/f'heuristic_{policy}.npz');runtime=w.load_radio.Runtime()
        for i,x in enumerate(arrays['states']):
            action,checks=runtime.values(x)
            np.testing.assert_array_equal(action[0],row['slots'][i]['action'])
            mask=w.adp.allowed(x[None])[0,:30].copy()
            if x[528]<=0:mask[:,1]=False
            if x[531]<=0:mask[:,2]=False
            self.assertEqual(checks[0],2*mask.sum())
        self.assertEqual(len(row['slots']),40)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    w.r.write(BASE/'integration_tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        new_simulator_steps=0,new_optimizer_updates=0,research_launched=False))
    raise SystemExit(not result.wasSuccessful())
