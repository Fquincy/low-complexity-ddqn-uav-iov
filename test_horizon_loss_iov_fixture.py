"""Full isolated workflow; research modes remain disabled regardless of token."""
import copy,json,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import run_horizon_loss_iov_fixture as r
import horizon_loss_iov_workflow as workflow
import audit_horizon_loss_iov_fixture as auditor
import horizon_normalized_loss as original_loss

OUT=r.ROOT/'validation/horizon_loss_iov_integration_2026-11-14_01'
TEMP=Path(tempfile.mkdtemp(prefix='horizon_loss_iov_fixture_'));FIXTURE=TEMP/'workflow'

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):r.configure()

    def test_01_instrumented_adapter_matches_tested_updates(self):
        p=r.protocol();cfg=r.plan(p,'integration-fixture')['cfg'];states=np.load(r.ROOT/p['action_cost']['common_inputs'])['observations'][:4].copy()
        states[:,-1]=np.array([0.,.5,.95,.975],np.float32);following=states.copy();following[:,-1]=np.array([.025,.525,.975,1.],np.float32)
        for d in p['learned_arms'][-3:]:
            a=r.build(d,99941,cfg);a['slots']=8
            for i in range(4):a['replay'].add(states[i],np.ones(32,np.int64),float(-i-1),following[i],i==3)
            b=copy.deepcopy(a);expected=original_loss.update(b);actual=r.update(a)
            for k,v in expected.items():
                if k!='optimizer_ns':self.assertEqual(actual[k],v,k)
            for k,v in a['online'].state_dict().items():self.assertTrue(torch.equal(v,b['online'].state_dict()[k]))

    def test_02_cli_and_api_guards(self):
        for mode in ('development','final'):
            path=TEMP/('forbidden_'+mode)
            with self.assertRaises(RuntimeError):workflow.run(path,mode,r.CONFIRM)
            self.assertFalse(path.exists())
            proc=subprocess.run([sys.executable,str(r.ROOT/'horizon_loss_iov_workflow.py'),'--mode',mode,'--output',str(path),'--confirm-development',r.CONFIRM],capture_output=True,text=True,timeout=60)
            self.assertNotEqual(proc.returncode,0);self.assertIn('disabled',proc.stderr);self.assertFalse(path.exists())

    def test_03_complete_workflow_and_independent_audits(self):
        p=r.protocol();self.assertEqual(len(p['learned_arms']),14);self.assertEqual(len(p['heuristics']),3)
        result=workflow.run(FIXTURE)
        self.assertTrue(result['artifact_audit_passed']);self.assertTrue(result['deployment_storage_audit']['passed']);self.assertFalse(result['quality_evidence'])
        v=r.read(FIXTURE/'run/verification.json');self.assertEqual((v['checkpoints'],v['training_slots'],v['optimizer_updates'],v['target_copies'],v['evaluation_slots'],v['raw_timing_rows']),(28,2240,2044,252,2480,17920))
        a=r.read(FIXTURE/'run/saved_artifact_audit.json');self.assertEqual((a['episodes'],a['task_records'],a['optimizer_rows'],a['runtime_engineering_actions']),(118,141600,2044,3584))
        decision=r.read(FIXTURE/'run/analysis.json');self.assertFalse(decision['screens_applied']);self.assertIsNone(decision['all_scientific_screens_passed'])
        self.assertEqual(len(decision['comparisons']),16)

    def test_04_manifest_and_normalization_tamper(self):
        self.assertTrue((FIXTURE/'workflow_verification.json').exists())
        with self.assertRaises(FileExistsError):workflow.run(FIXTURE)
        saved=FIXTURE/'run';arm='normalized_screened_price24';path=saved/arm/'fit_00'/'optimizer.jsonl';mp=saved/'run_manifest.json'
        original=path.read_bytes();manifest=mp.read_bytes()
        try:
            lines=path.read_text().splitlines();x=json.loads(lines[0]);x['current_h'][0]+=1;lines[0]=json.dumps(x);path.write_text('\n'.join(lines)+'\n')
            obj=r.read(mp);obj['files'][path.relative_to(saved).as_posix()]=r.sha(path);r.write(mp,obj)
            with self.assertRaises(AssertionError):auditor.audit(saved,False)
        finally:path.write_bytes(original);mp.write_bytes(manifest)
        extra=saved/'unrecorded.txt';extra.write_text('corruption')
        try:
            with self.assertRaises(AssertionError):auditor.audit(saved,False)
        finally:extra.unlink()
        self.assertEqual(r.sha(path),r.read(mp)['files'][path.relative_to(saved).as_posix()])

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if FIXTURE.exists():shutil.copytree(FIXTURE,OUT/'workflow')
    r.write(OUT/'tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),
        temporary_workflow=str(FIXTURE),research_launched=False,quality_evidence=False,oracle_optimizer_steps=3,reference_clone_optimizer_steps=3))
    r.write(OUT/'evidence_manifest.json',{x.relative_to(OUT).as_posix():r.sha(x) for x in OUT.rglob('*') if x.is_file()})
    print('Evidence:',OUT,flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
