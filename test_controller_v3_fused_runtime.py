"""Native mechanics and rejection tests before any timing; zero updates/slots."""
import itertools,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np,torch
import controller_v3_fused_runtime as f
import run_controller_v3_fused_gate as runner
import audit_controller_v3_fused_gate as auditor
import controller_redesign_v2 as old

class Mechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):runner.g.r.configure();f.numba.set_num_threads(1)

    def test_softplus_stability_and_bounds(self):
        for x in [-1000.,-100.,-20.,-1e-8,0.,1e-8,20.,21.,100.,1000.]:
            v=f.softplus(np.float32(x));q=float(torch.nn.functional.softplus(torch.tensor(x)))
            np.testing.assert_allclose(v,q,atol=1e-6,rtol=1e-5);self.assertTrue(np.isfinite(v) and v>=0)

    def test_ties_and_plain_branch_decoders(self):
        for mode in [0,1,2,3]:
            a,q=f.flat_decode(np.zeros(161,np.float32),mode)
            np.testing.assert_array_equal(a,np.zeros(32,np.int64));self.assertTrue(np.isfinite(q))
        raw=np.arange(160,dtype=np.float32);a,_=f.flat_decode(raw,0);self.assertTrue(np.all(a==4))
        a,_=f.flat_decode(raw,1);self.assertTrue(np.all(a==0))

    def test_v3_matches_tables_and_ablation_on_synthetic_logits(self):
        rng=np.random.default_rng(99025)
        for enabled in [True,False]:
            for unused in range(8):
                raw=rng.uniform(-2,2,170).astype(np.float32);state=rng.uniform(0,1,533).astype(np.float32)
                model=runner.g.build('amortized_workload_price24' if enabled else 'amortized_no_price24',99021)
                with torch.no_grad():
                    model.net[0].weight.zero_();model.net[0].bias.zero_();model.net[2].weight.zero_();model.net[2].bias.copy_(torch.from_numpy(raw))
                    a,q,_=model.greedy(torch.from_numpy(state)[None])
                b,v=f.v3_decode(raw,state,enabled);np.testing.assert_array_equal(a[0].numpy(),b)
                np.testing.assert_allclose(float(q[0]),v,atol=1e-6,rtol=1e-5)

    def test_v2_symmetric_repairs_and_threshold_fixtures(self):
        u=np.array([[[10.,0.,0.,20.,20.],[10.,0.,0.,20.,20.]]],np.float32)
        w=np.ones_like(u);w[:,:,0]=0;L=np.zeros((1,5),np.float32);beta=np.array([[0,1,1,1,1]],np.float32)
        motion=np.zeros((1,2,5),np.float32);base=np.zeros(1,np.float32)
        for delta in [None,.5e-6,.9e-6,1e-6,1.1e-6,2e-6]:
            unary=u.copy();b=beta.copy()
            if delta is not None:unary[:,:,2]=.375-delta;b[:,2]=0
            ta,tq,_=old.decode_cost_tables(*(torch.from_numpy(x) for x in [unary,w,L,b,motion,base]))
            a,q=f.shared_decode(*(x[0] for x in [unary,w,L,b,motion,base]),True,False)
            np.testing.assert_array_equal(ta[0].numpy(),a);np.testing.assert_allclose(-float(tq[0]),q,atol=1e-6,rtol=1e-5)

    def test_exact_marginal_and_load_identities(self):
        rng=np.random.default_rng(99025);u=rng.uniform(0,1,(3,5)).astype(np.float32);w=rng.uniform(0,1,(3,5)).astype(np.float32)
        w[:,0]=0;L=rng.uniform(0,1,5).astype(np.float32);beta=L.copy();beta[0]=0
        a=np.array([1,2,3],np.int64);prices=f.price_table(a,w,L,beta);base=f.score(a,u,w,L,beta,np.float32(.3))
        for i,d in itertools.product(range(3),range(5)):
            trial=a.copy();trial[i]=d;delta=u[i,d]-u[i,a[i]]+prices[i,d]-prices[i,a[i]]
            np.testing.assert_allclose(f.score(trial,u,w,L,beta,np.float32(.3))-base,delta,atol=1e-6,rtol=1e-5)

    def test_invalid_input_and_corrupt_artifact_rejection(self):
        for state in [np.zeros(532),np.full(533,np.nan),np.zeros((1,533))]:
            with self.assertRaises(ValueError):f.values(None,'plain24_nonpositive',state)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'a').write_text('a');runner.g.r.write(p/'run_manifest.json',dict(files={'a':runner.g.r.old.sha(p/'a')}))
            auditor.manifest_check(p);(p/'a').write_text('bad')
            with self.assertRaises(AssertionError):auditor.manifest_check(p)

    def test_no_fastmath_parallel_or_object_mode(self):
        f.softplus(np.float32(0.))
        for fn in [f.softplus,f.flat_decode,f.v3_decode,f.shared_decode,f.emc_decode,f.bdq_decode]:
            self.assertFalse(fn.targetoptions['fastmath']);self.assertFalse(fn.targetoptions['parallel']);self.assertTrue(fn.targetoptions['boundscheck'])
        self.assertTrue(f.softplus.nopython_signatures)

    def test_cli_blocks_both_research_modes(self):
        for mode in ['development','final']:
            for confirmed in [False,True]:
                with tempfile.TemporaryDirectory() as td:
                    p=Path(td)/'blocked';cmd=[sys.executable,str(runner.ROOT/'run_controller_v3_fused_gate.py'),'--mode',mode,'--output',str(p)]
                    if confirmed:cmd.append('--confirm-'+mode+'-run')
                    result=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
                    self.assertNotEqual(result.returncode,0);self.assertIn('disabled',result.stderr);self.assertFalse(p.exists())

if __name__=='__main__':
    path=Path(sys.argv[1]);result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Mechanics))
    runner.g.r.write(path,dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        source_sha256={n:runner.g.r.old.sha(runner.ROOT/n) for n in runner.SOURCES},optimizer_updates=0,new_simulator_steps=0))
    sys.exit(0 if result.wasSuccessful() else 1)
