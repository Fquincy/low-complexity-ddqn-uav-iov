"""Read-only completion on saved fixture, including negative evidence tests."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import assignment_price_audited_completion as c

OUT=c.HEURISTIC_EVIDENCE
BIND=c.ROOT/'validation/assignment_price_research_binding_2026-11-15_01'

class Checks(unittest.TestCase):
    def test_actual_completion_audits_fixture_without_retraining_or_retiming(self):
        run=BIND/'fixture_evidence';result=c.complete(run,OUT/'verified_fixture_completion')
        self.assertEqual(c.runner.r.read(result/'completion.json')['status'],'AUDITED_COMPLETE')
        self.assertTrue(c.runner.r.read(result/'saved_artifact_audit.json')['passed'])
    def test_bad_cost_report_stops_before_audit_or_completion_write(self):
        original=c.runner.r.read
        def changed(path):
            value=original(path)
            if Path(path).name=='heuristic_cost_summary.json':value=copy.deepcopy(value);value['median_ms']+=1
            return value
        blocked=OUT/'blocked_bad_cost_completion'
        with patch.object(c.runner.r,'read',side_effect=changed),patch.object(c.audit,'audit',side_effect=AssertionError('must not audit')) as audit:
            with self.assertRaises(AssertionError):c.complete(BIND/'fixture_evidence',blocked)
            audit.assert_not_called()
        self.assertFalse(blocked.exists())

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    c.runner.r.write(OUT/'completion_tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),
        simulator_steps=0,optimizer_steps=0,acceptance_retiming=False,research_launched=False))
    raise SystemExit(not result.wasSuccessful())
