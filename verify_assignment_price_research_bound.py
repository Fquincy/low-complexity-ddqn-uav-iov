"""Multi-fit guards/plans and complete workflow audit using engineering data."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import run_assignment_price_research_bound as w
import audit_assignment_price_research_bound_v2 as auditor

BASE=w.r.ROOT/'validation/assignment_price_research_binding_2026-11-15_01'

class Plans(unittest.TestCase):
    def test_development_plan_exactly_matches_declaration(self):
        pl=w.plan('development');p=w.base.protocol();d=p['development'];arms=len(p['learned_arms']);fits=len(pl['model_seeds'])
        self.assertEqual(pl['model_seeds'],d['model_seeds']);self.assertEqual(pl['train_blocks'],d['train_blocks'])
        self.assertEqual(pl['evaluation_seeds'],list(range(d['evaluation_seeds']['start'],d['evaluation_seeds']['start']+d['evaluation_seeds']['count'])))
        cfg=pl['cfg'];self.assertEqual(arms*fits*cfg['slots_per_fit'],600000)
        self.assertEqual(arms*fits*(cfg['slots_per_fit']-cfg['warmup']+1),523275)
        self.assertEqual((arms*fits+4)*len(pl['evaluation_seeds'])*40,151680)
        self.assertEqual(arms*fits*pl['timing']['rounds']*pl['timing']['states'],48000)
        self.assertEqual(cfg['anchor'],p['training']['anchor'])
    def test_api_cli_guard_precedes_plan_seed_output(self):
        temp=Path(tempfile.mkdtemp(prefix='adp_bound_guard_'));evidence=[]
        for mode,confirmation in [('development',None),('development',w.CONFIRM),('final',None),('unknown',None)]:
            output=temp/f'{mode}_{confirmation}'
            with patch.object(w,'plan',side_effect=AssertionError('plan must not be reached')):
                with self.assertRaises(RuntimeError):w.run(output,mode,confirmation)
            self.assertFalse(output.exists())
            cmd=[sys.executable,str(w.r.ROOT/'run_assignment_price_research_bound.py'),'--mode',mode,'--output',str(output)]
            if confirmation:cmd+=['--confirm-development',confirmation]
            result=subprocess.run(cmd,capture_output=True,text=True,cwd=w.r.ROOT)
            self.assertNotEqual(result.returncode,0);self.assertFalse(output.exists())
            self.assertTrue(any(x in result.stderr for x in ['confirmation required','pending separate launch review','mode disabled']))
            evidence.append(dict(mode=mode,confirmation=confirmation,returncode=result.returncode,stderr=result.stderr,output_created=False))
        w.r.write(BASE/'guard_evidence.json',evidence)

class Saved(unittest.TestCase):
    def test_full_multifit_artifact_and_analysis_audit(self):
        output=Path(w.r.read(BASE/'temporary_output.json')['path']);report=auditor.audit(output)
        self.assertTrue(report['passed']);self.assertEqual(report['fits'],2);self.assertEqual(report['checkpoint_graphs'],30)
        self.assertEqual(report['optimizer_rows'],2190);self.assertEqual(report['episodes'],128)
        w.r.write(BASE/'independent_audit.json',report)
    def test_rejects_manifest_tamper(self):
        output=Path(w.r.read(BASE/'temporary_output.json')['path']);path=output/'storage.json';original=path.read_bytes()
        try:
            path.write_bytes(original+b' ')
            with self.assertRaises(AssertionError):auditor.audit(output)
        finally:path.write_bytes(original)

if __name__=='__main__':
    parser=__import__('argparse').ArgumentParser();parser.add_argument('--saved',action='store_true');args=parser.parse_args()
    BASE.mkdir(exist_ok=args.saved)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Saved if args.saved else Plans))
    w.r.write(BASE/('saved_tests_verified.json' if args.saved else 'plan_tests_verified.json'),dict(passed=result.wasSuccessful(),tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),new_simulator_steps=0,new_optimizer_updates=0,research_launched=False))
    raise SystemExit(not result.wasSuccessful())
