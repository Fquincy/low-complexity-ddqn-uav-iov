"""Isolated gradients, independent assignment oracle, DDQN and runtime tests."""
import copy
from itertools import product
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
import assignment_price_prototype as p
import run_horizon_loss_research as r

OUT=r.ROOT/'validation/assignment_price_prototype_2026-11-15_01'

def oracle_cost(g,v,b,w,q,beta,eta,a):
    # Independent pairwise interaction expansion, not cached workload sums.
    n=len(g);c=b+sum(g[i,a[i]] for i in range(n))+sum(v[j,a[n+j]] for j in range(len(v)))
    for i in range(n):
        d=a[i];wi=w[i,d]
        c+=beta[d]*(2*q[d]*wi+wi*wi)/(4*n)
        for j in range(i):
            if a[j]==d:c+=beta[d]*2*eta[d]*wi*w[j,d]/(4*n)
    return c

def oracle_decode(args,mask,cycles,sweep):
    g,v,b,w,q,beta,eta=args;n,D=g.shape
    old=g+beta[None,:]*(2*(q[None,:]+eta[None,:]*(w.sum(0)[None,:]-w))*w+w*w)/(4*n)
    a=np.concatenate([np.where(mask,old,np.inf).argmin(1),v.argmin(1)])
    if sweep:
        for i in sorted(range(n),key=lambda j:(-cycles[j],j)):
            scores={}
            for d in range(D):
                if mask[i,d]:
                    candidate=a.copy();candidate[i]=d;scores[d]=oracle_cost(*args,candidate)
            best=min(scores.values())
            # Full-cost sums can round at 1e-16; random fixtures avoid exact ties.
            a[i]=a[i] if scores[a[i]]==best else min(scores,key=scores.get)
    return a

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r.configure();spec=r.protocol()
        cls.states=np.load(r.ROOT/spec['action_cost']['common_inputs'])['observations'].copy()

    def test_01_exact_counts_and_chosen_values(self):
        for arm in p.ARMS:
            m=p.Critic(arm);self.assertEqual(sum(x.numel()*x.element_size() for x in m.parameters()),67364 if arm=='fixed_price_plain_sweep' else 68264)
            self.assertEqual(sum(x.numel()*x.element_size() for x in m.buffers()),20)
            self.assertTrue(all(x.dtype==torch.float32 for x in m.parameters()))
            s=torch.from_numpy(self.states[:3]);a,q=m.greedy(s)
            self.assertEqual(q.dtype,torch.float64);torch.testing.assert_close(q,m(s,a),rtol=1e-12,atol=1e-12)
            self.assertTrue((q<=0).all());self.assertTrue(torch.from_numpy(p.allowed(s)).gather(2,a[:,:,None]).all())
            core=m.core;old=r.models.original.existing.base.old.digest(core.state_dict())
            m.greedy(s);self.assertEqual(old,r.models.original.existing.base.old.digest(core.state_dict()))

    def test_02_independent_pairwise_potential_and_gradients(self):
        torch.manual_seed(99871)
        g=torch.rand(2,3,3,dtype=torch.double,requires_grad=True)
        v=torch.rand(2,2,3,dtype=torch.double,requires_grad=True)
        b=torch.rand(2,dtype=torch.double,requires_grad=True)
        w=torch.rand(2,3,3,dtype=torch.double,requires_grad=True)
        q=torch.rand(2,3,dtype=torch.double,requires_grad=True)
        rawbeta=torch.rand(2,3,dtype=torch.double,requires_grad=True)
        raweta=torch.rand(2,3,dtype=torch.double,requires_grad=True)
        a=torch.tensor([[1,1,2,0,2],[2,2,2,1,0]])
        def fn(g,v,b,w,q,beta,eta):return p.potential(g,v,b,w,q,F.softplus(beta),F.softmax(eta,1),a)
        self.assertTrue(torch.autograd.gradcheck(fn,(g,v,b,w,q,rawbeta,raweta),eps=1e-6,atol=1e-5,rtol=1e-4))
        tensors=(g,v,b,w,q,F.softplus(rawbeta),F.softmax(raweta,1))
        expected=np.array([oracle_cost(*(x.detach().numpy()[i] for x in tensors),a[i].numpy()) for i in range(2)])
        np.testing.assert_allclose(fn(g,v,b,w,q,rawbeta,raweta).detach(),expected,rtol=1e-12,atol=1e-12)

    def test_03_decoder_oracle_and_bound(self):
        rng=np.random.default_rng(99873)
        for case in range(24):
            n,D=4,3;g=rng.random((n,D));v=rng.random((2,D));b=.3
            w=rng.random((n,D));w[:,0]=0;q=rng.random(D);q[0]=0
            beta=rng.random(D);beta[0]=0;eta=np.array([.2,.3,.5]);mask=np.ones((n,D),bool)
            mask[case%n,0]=False;cycles=rng.random(n);args=(g,v,b,w,q,beta,eta)
            a,diag=p.decode(*args,mask,cycles,True)
            np.testing.assert_array_equal(a,oracle_decode(args,mask,cycles,True))
            self.assertLessEqual(diag[1],diag[0]+1e-12);self.assertEqual(diag[2],mask.sum());self.assertEqual(diag[3],mask.sum())
            choices=[np.flatnonzero(row).tolist() for row in mask]
            exact=min(oracle_cost(*args,np.r_[x,v.argmin(1)]) for x in product(*choices))
            self.assertGreaterEqual(diag[1]+1e-12,exact)
        # Deliberate suboptimal fixed-point example from the exact design gate.
        w=np.array([[0,z/100,z/100] for z in [8,7,6,5]],np.float64)
        args=(np.ones((4,3))*.01,np.zeros((2,3)),0.,w,np.zeros(3),np.array([0.,1.,1.]),np.ones(3)/3)
        mask=np.ones((4,3),bool);mask[:,0]=False;a,diag=p.decode(*args,mask,np.array([8.,7.,6.,5.]),True)
        best=min(oracle_cost(*args,np.r_[x,[0,0]]) for x in product([1,2],repeat=4))
        self.assertGreater(diag[1],best)

    def test_04_ties_boundaries_and_invalid_states(self):
        for n,D in [(1,2),(4,3)]:
            args=(np.zeros((n,D)),np.zeros((2,D)),0.,np.zeros((n,D)),np.zeros(D),np.zeros(D),np.ones(D)/D)
            a,diag=p.decode(*args,np.ones((n,D),bool),np.ones(n),True)
            np.testing.assert_array_equal(a,np.zeros(n+2,np.int64));self.assertEqual(diag[1],0)
            # One ULP distinction must not become a configurable tie tolerance.
            g=np.zeros((n,D));g[:,0]=np.nextafter(0.,1.);args=(g,*args[1:])
            a,_=p.decode(*args,np.ones((n,D),bool),np.ones(n),True);np.testing.assert_array_equal(a[:n],np.ones(n,np.int64))
        m=p.Critic();x=self.states[0].copy();x[-1]=1.
        with self.assertRaises(ValueError):m.greedy(torch.from_numpy(x)[None])
        x[-1]=.12345
        with self.assertRaises(ValueError):p.Runtime(m).values(x)
        with self.assertRaises(ValueError):p.Runtime(m).values(self.states[0].astype(np.float64))
        # Adjacent task/deadline encodings must keep identical existing masks.
        for direction in [np.float32(-np.inf),np.float32(np.inf)]:
            x=self.states[0].copy();x[6]=np.nextafter(np.float32(.84),direction);x[7]=np.float32(.93333334)
            a,_=m.greedy(torch.from_numpy(x)[None]);self.assertTrue(p.allowed(x[None])[0,np.arange(32),a[0].numpy()].all())

    def test_05_ddqn_terminal_anchor_and_synthetic_update(self):
        s=self.states[:4].copy();s[:,-1]=[0.,.5,.95,.975];ns=s.copy();ns[:,-1]=[.025,.525,.975,1.]
        s,ns=torch.from_numpy(s),torch.from_numpy(ns);rw=torch.tensor([-1.,-2.,-3.,-4.]);done=torch.tensor([0.,0.,0.,1.])
        for arm in p.ARMS:
            online=p.Critic(arm);target=p.Critic(arm,99869).eval().requires_grad_(False)
            act=online.greedy(s)[0];expected=rw.double().clone()
            with torch.no_grad():na=online.greedy(ns[:3])[0];expected[:3]+=target(ns[:3],na)
            with patch.object(target,'greedy',side_effect=AssertionError('target must not select')):
                y=p.targets(online,target,rw,ns,done)
            torch.testing.assert_close(y,expected);self.assertFalse(y.requires_grad)
            with patch.object(online,'greedy',side_effect=AssertionError('terminal decoder called')):
                torch.testing.assert_close(p.targets(online,target,rw[3:],ns[3:],done[3:]),rw[3:].double())
            opt=torch.optim.Adam(online.parameters(),lr=1e-4)
            loss,parts=p.objective(online,target,(s,act,rw,ns,done),(s[3:],act[3:],rw[3:]))
            expected_anchor=F.smooth_l1_loss(online(s[3:],act[3:]),rw[3:].double())
            torch.testing.assert_close(parts['anchor'],expected_anchor)
            torch.testing.assert_close(loss,parts['uniform']+expected_anchor)
            before=copy.deepcopy(online.state_dict());opt.zero_grad();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(online.parameters(),10.,error_if_nonfinite=True);opt.step()
            self.assertTrue(torch.isfinite(norm));self.assertTrue(all(x.grad is None for x in target.parameters()))
            self.assertTrue(any(not torch.equal(v,before[k]) for k,v in online.state_dict().items()))

    def test_06_eager_native_and_runtime_equivalence(self):
        for arm in p.ARMS:
            model=p.Critic(arm);runtime=p.Runtime(model)
            for state in self.states:
                a,q=model.greedy(torch.from_numpy(state)[None],compiled=False)
                na,nq=runtime.values(state)
                np.testing.assert_array_equal(a.numpy(),na);np.testing.assert_allclose(q.numpy(),nq,atol=1e-6,rtol=1e-5)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    r.write(OUT/'tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),synthetic_optimizer_steps=4,simulator_steps=0,research_launched=False))
    for name in ['assignment_price_prototype.py','test_assignment_price_prototype.py','ASSIGNMENT_DEPENDENT_PRICE_DESIGN_BRIEF_2026-11-15.txt']:
        shutil.copy2(r.ROOT/name,OUT/name)
    r.write(OUT/'tests_manifest.json',{x.name:r.sha(x) for x in OUT.iterdir() if x.is_file()})
    raise SystemExit(0 if result.wasSuccessful() else 1)
