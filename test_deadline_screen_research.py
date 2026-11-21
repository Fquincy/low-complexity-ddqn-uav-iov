"""Integration evidence using only the reserved synthetic fixture, no research seeds."""
import copy,json,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import run_deadline_screen_research as r
import audit_deadline_screen_research as auditor

EVIDENCE=r.ROOT/'validation/deadline_screen_research_engineering_2026-11-14_01'
TEMP=Path(tempfile.mkdtemp(prefix='deadline_screen_research_fixture_'))
OUT=TEMP/'fixture'

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):r.configure()

    def test_01_protocol_and_adapter_semantics(self):
        p=r.protocol();plan=r.plan(p,'integration-fixture');state=np.load(r.ROOT/p['action_cost']['common_inputs'])['observations'][0].copy()
        # Force impossible-local tasks; unmasked models must remain able to choose local.
        state[6:240:8]=1.;state[7:240:8]=np.float32(.25/.3)
        require_mask=r.models.original.allowed(state[None])[0]
        self.assertFalse(require_mask[:30,0].any());self.assertTrue(require_mask[:,1:].all())
        for definition in p['learned_arms']:
            agent=r.build(definition,99921,plan['cfg']);model=agent['online'];runtime=r.models.Runtime(model)
            x=torch.from_numpy(state)[None]
            with torch.inference_mode():a,q=model.greedy(x)
            b,v=runtime.values(state);np.testing.assert_array_equal(a.numpy(),b);np.testing.assert_allclose(q.numpy(),v,rtol=1e-5,atol=1e-6)
            # Deterministic ties: zero weights must retain first-index tie behavior.
            for parameter in model.parameters():parameter.data.zero_()
            with torch.inference_mode():a,q=model.greedy(x)
            b,v=runtime.values(state);np.testing.assert_array_equal(a.numpy(),b);np.testing.assert_allclose(q.numpy(),v,rtol=1e-5,atol=1e-6)
            if definition['screened']:self.assertTrue(np.all(a.numpy()[0,:30]!=0))
            else:self.assertTrue(np.all(a.numpy()[0,:30]==0))
            terminal=state.copy();terminal[-1]=1.
            with self.assertRaises(ValueError):model.greedy(torch.from_numpy(terminal)[None])
            with self.assertRaises(ValueError):runtime.values(terminal)
        # Independent target construction tests next-state masking and terminal bypass.
        definition=p['learned_arms'][0];agent=r.build(definition,99921,plan['cfg']);s=torch.from_numpy(state)[None]
        terminal=s.clone();terminal[:,-1]=1.;rewards=torch.tensor([-2.,-3.]);states=torch.cat((s,terminal));dones=torch.tensor([0.,1.])
        with torch.no_grad():
            act=agent['online'].greedy(s)[0];expected=-2.+agent['target'](s,act)[0]
            targets=r.models.original.targets(agent['online'],agent['target'],rewards,states,dones)
        np.testing.assert_allclose(targets.numpy(),[float(expected),-3.],rtol=1e-6,atol=1e-6)

    def test_02_cli_and_api_launch_guards(self):
        for mode,confirmation in [('development',None),('development',r.CONFIRM),('final',None)]:
            out=TEMP/f'forbidden_{mode}_{bool(confirmation)}'
            with self.assertRaises(RuntimeError):r.run(out,mode,confirmation)
            self.assertFalse(out.exists())
            cmd=[sys.executable,str(r.ROOT/'run_deadline_screen_research.py'),'--mode',mode,'--output',str(out)]
            if confirmation:cmd+=['--confirm-development',confirmation]
            proc=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
            self.assertNotEqual(proc.returncode,0);self.assertFalse(out.exists())
            self.assertIn('confirmation' if mode=='development' and not confirmation else 'disabled',proc.stderr)

    def test_03_complete_fixture_and_independent_audit(self):
        result=r.run(OUT)
        self.assertEqual((result['training_slots'],result['evaluation_slots'],result['optimizer_updates'],result['target_copies']),(1760,2000,1606,198))
        self.assertFalse(result['quality_evidence']);self.assertFalse(result['research_launched'])
        saved=auditor.audit(OUT)
        self.assertEqual((saved['episodes'],saved['task_records'],saved['runtime_engineering_actions'],saved['raw_timing_rows']),(94,112800,2816,14080))
        self.assertTrue(saved['comparisons_and_bootstrap_recomputed'])

    def test_04_existing_output_and_manifest_guards(self):
        self.assertTrue((OUT/'saved_artifact_audit.json').exists(),'fixture did not complete')
        with self.assertRaises(FileExistsError):r.run(OUT)
        path=OUT/'verification.json';original=path.read_bytes()
        try:
            obj=r.read(path);obj['optimizer_updates']+=1;r.write(path,obj)
            with self.assertRaises(AssertionError):auditor.audit(OUT,False)
        finally:path.write_bytes(original)
        # Extra file and missing episode rejection must happen before heavy recomputation.
        extra=OUT/'unexpected.txt';extra.write_text('bad')
        try:
            with self.assertRaises(AssertionError):auditor.audit(OUT,False)
        finally:extra.unlink()
        path=next(OUT.glob('*/fit_00/evaluation_*.json.gz'));temp=path.with_suffix('.held')
        path.rename(temp)
        try:
            with self.assertRaises(AssertionError):auditor.audit(OUT,False)
        finally:temp.rename(path)

    def test_05_rehashed_analysis_tamper_rejected(self):
        self.assertTrue((OUT/'saved_artifact_audit.json').exists(),'fixture did not complete')
        path=OUT/'analysis.json';mp=OUT/'run_manifest.json';original=path.read_bytes();manifest=mp.read_bytes()
        try:
            obj=r.read(path);obj['comparisons']['unmasked_price24']['per_fit'][0]['return_difference']+=10.;r.write(path,obj)
            obj=r.read(mp);obj['files']['analysis.json']=r.sha(path);r.write(mp,obj)
            with self.assertRaisesRegex(AssertionError,'comparison recomputation'):auditor.audit(OUT,False)
        finally:path.write_bytes(original);mp.write_bytes(manifest)


if __name__=='__main__':
    if EVIDENCE.exists():raise SystemExit('Refusing existing evidence directory')
    EVIDENCE.mkdir(parents=True)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if OUT.exists():shutil.copytree(OUT,EVIDENCE/'fixture')
    r.write(EVIDENCE/'tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        temporary_fixture_output=str(OUT),preserved_fixture='fixture',research_launched=False))
    print('Evidence:',EVIDENCE,flush=True)
    sys.exit(0 if result.wasSuccessful() else 1)
