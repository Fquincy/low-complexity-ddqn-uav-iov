"""Focused provenance checks; no simulator, optimizer or research execution."""
import ast,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run_anchor_research_route_v3 as r
import test_anchor_research_route_v3 as inherited
class BoundaryStop(Exception):pass
class Provenance(unittest.TestCase):
    def test_snapshot_before_first_model(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'fixture'
            def stop(*args,**kwargs):
                plan=r.read(out/'runtime_plan.json')
                self.assertEqual(set(plan['source_sha256']),set(r.SOURCES))
                for name,digest in plan['source_sha256'].items():
                    self.assertEqual(r.base.old.sha(out/'source_snapshot'/name),digest)
                    self.assertEqual(r.base.old.sha(ROOT/name),digest)
                self.assertIn('run_v3_research_gate_v2.py',plan['source_sha256'])
                raise BoundaryStop()
            with patch.object(r,'build',side_effect=stop):
                with self.assertRaises(BoundaryStop):r.run_plan(out,fixture=True,multi_fixture=True)
            self.assertFalse(list(out.rglob('*.pt')))
            self.assertFalse(list(out.rglob('train_*.json.gz')))
    def test_changed_factory_rejected_before_output(self):
        real=r.base.old.sha
        def digest(path):
            return 'tampered' if Path(path)==ROOT/'run_v3_research_gate_v2.py' else real(path)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'rejected'
            with patch.object(r.base.old,'sha',side_effect=digest):
                with self.assertRaisesRegex(ValueError,'factory changed'):r.run_plan(out,fixture=True)
            self.assertFalse(out.exists())
    def test_all_shared_scientific_functions_unchanged(self):
        def functions(path):
            return {x.name:x for x in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
        old=functions(ROOT/'run_anchor_research_route_v2.py');new=functions(ROOT/'run_anchor_research_route_v3.py')
        for name,node in old.items():
            if name=='main':continue
            if name=='run_plan':
                candidate=new[name]
                candidate.body=[x for x in candidate.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Name) and x.value.func.id=='verify_factory_provenance')]
            self.assertEqual(ast.dump(node,include_attributes=False),ast.dump(new[name],include_attributes=False),name)
if __name__=='__main__':
    r.configure();suite=unittest.TestSuite()
    for name in unittest.defaultTestLoader.getTestCaseNames(inherited.Checks):
        if name!='test_complete_shared_workflow_and_saved_artifact_audit':suite.addTest(inherited.Checks(name))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(Provenance))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    r.write(ROOT/'validation/anchor_launch_readiness_tests_2026-11-06.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),new_simulator_slots=0,optimizer_updates=0,research_launched=False,source_sha256={name:r.base.old.sha(ROOT/name) for name in r.SOURCES}))
    sys.exit(0 if result.wasSuccessful() else 1)
