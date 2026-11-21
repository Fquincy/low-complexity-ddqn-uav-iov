"""Exercise v2 fixture, actual frozen-source preflight and launch boundaries."""
import ast,copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
import run_anchor_posttraining_continuation_v2 as c
import audit_anchor_posttraining_continuation_v2 as a
import test_anchor_posttraining_continuation as previous
previous.c=c;previous.a=a;previous.OUT=Path(sys.argv[1])
class Ready(unittest.TestCase):
    def test_actual_research_source_preflight_without_execution(self):
        with patch.object(c.base,'episode',side_effect=AssertionError('no simulator in preflight')),patch.object(torch.optim,'Adam',side_effect=AssertionError('no optimizer')):
            plan,freeze=c.preflight(c.RESEARCH_SOURCE,False)
        self.assertEqual(freeze['count'],85);self.assertEqual(len(plan['evaluation_seeds']),48)
        self.assertFalse(c.CONTINUATION_RESEARCH_ENABLED)
    def test_frozen_plan_rejects_seed_configuration_and_phase_changes(self):
        plan=c.read(c.RESEARCH_SOURCE/'runtime_plan.json')
        for key,value in [('evaluation_seeds',[990000001]),('model_seeds',[99111]),('phase','final'),('cfg',dict(plan['cfg'],learning_rate=.01))]:
            bad=copy.deepcopy(plan);bad[key]=value
            with self.assertRaises((ValueError,AssertionError)):c.validate_research_plan(bad)
    def test_shared_orchestration_unchanged(self):
        def functions(name):return {n.name:n for n in ast.parse((c.ROOT/name).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
        old=functions('run_anchor_posttraining_continuation.py');new=functions('run_anchor_posttraining_continuation_v2.py')
        for name,node in old.items():
            if name=='preflight':continue
            if name=='run_continuation':node.body=node.body[1:];new[name].body=new[name].body[1:]
            if name=='main':continue
            self.assertEqual(ast.dump(node,include_attributes=False),ast.dump(new[name],include_attributes=False),name)
    def test_launcher_requires_confirmation_and_rejects_final(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'out';control=Path(td)/'control'
            cmd=[sys.executable,str(c.ROOT/'launch_anchor_evaluation.py'),'--mode','development','--source',str(c.RESEARCH_SOURCE),'--output',str(out),'--control',str(control)]
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
            self.assertNotEqual(result.returncode,0);self.assertIn('confirmation' if False else '--confirm-development-evaluation required',result.stderr)
            cmd[cmd.index('development')]='final'
            result=subprocess.run(cmd+['--confirm-development-evaluation'],capture_output=True,text=True,timeout=60)
            self.assertNotEqual(result.returncode,0);self.assertIn('Final execution disabled',result.stderr)
            self.assertFalse(out.exists());self.assertFalse(control.exists())
if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(previous.Checks),unittest.defaultTestLoader.loadTestsFromTestCase(Ready)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    c.write(c.ROOT/'validation/anchor_evaluation_readiness_tests_2026-11-11.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),research_evaluation_started=False,
        source_sha256={n:c.base.old.sha(c.ROOT/n) for n in [*c.CONTINUATION_SOURCES,'test_anchor_evaluation_readiness.py']}))
    sys.exit(0 if result.wasSuccessful() else 1)
