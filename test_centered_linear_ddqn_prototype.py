"""Synthetic equations, gradients and DDQN semantics; no optimizer or rollout."""
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

import centered_linear_ddqn_prototype as c

torch.set_num_threads(1)


def states(n=4):
    rng=np.random.default_rng(937200001)
    x=rng.uniform(.1,.9,(n,533)).astype(np.float32)
    x[:,:240].reshape(n,30,8)[:,:,6]=.9
    x[:,:240].reshape(n,30,8)[:,:,7]=.9
    x[:,-1]=.5
    return torch.from_numpy(x)


class Mechanics(unittest.TestCase):
    def setUp(self):torch.manual_seed(937200011)

    def test_storage_and_all_five_centering(self):
        model=c.CenteredLinearDDQN();x=states()
        self.assertEqual(sum(p.numel() for p in model.parameters()),16841)
        self.assertEqual(sum(p.numel()*p.element_size() for p in model.parameters()),67364)
        self.assertEqual(sum(b.numel()*b.element_size() for b in model.buffers()),20)
        p,m,v,_=model.tables(x)
        self.assertTrue(torch.allclose((p-p.mean(2,keepdim=True)).mean(2),torch.zeros(4,30),atol=1e-8))
        a,q,_=model.greedy(x)
        torch.testing.assert_close(q,model(x,a))
        # Explicitly excluded local still contributes to the five-action mean.
        self.assertTrue(torch.all(a[:,:30]!=0))
        hand=v-(p.gather(2,a[:,:30,None]).squeeze(2)-p.sum(2)/5).sum(1)-(m.gather(2,a[:,30:,None]).squeeze(2)-m.sum(2)/5).sum(1)
        torch.testing.assert_close(q,hand)

    def test_offset_invariance_and_equivalent_linear_function_class(self):
        p=torch.randn(2,30,5,dtype=torch.float64);m=torch.randn(2,2,5,dtype=torch.float64)
        v=torch.randn(2,dtype=torch.float64);a=torch.randint(0,5,(2,32))
        pp=p+torch.randn(2,30,1,dtype=torch.float64);mm=m+torch.randn(2,2,1,dtype=torch.float64)
        torch.testing.assert_close(c.joint_value(p,m,v,a),c.joint_value(pp,mm,v,a))
        adjusted_v=v+p.mean(2).sum(1)+m.mean(2).sum(1)
        torch.testing.assert_close(c.joint_value(p,m,v,a),c.uncentered_linear_reference(p,m,adjusted_v,a))

    def test_exhaustive_reduced_joint_maximum_and_ties(self):
        p=torch.zeros(1,30,5);m=torch.zeros(1,2,5);v=torch.tensor([.7])
        p[0,:2]=torch.tensor([[3.,2.,-1.,4.,.2],[1.,-2.,3.,0.,5.]])
        m[0,0]=torch.tensor([2.,1.,0.,-3.,4.])
        mask=np.ones((32,5),bool);mask[0,2]=False
        greedy=np.array([int(np.argmin(p[0,i].numpy()+np.where(mask[i],0,np.inf))) for i in range(30)]+[int(m[0,j].argmin()) for j in range(2)])
        qg=float(c.joint_value(p,m,v,torch.tensor(greedy)[None])[0])
        values=[]
        for a,b,z in itertools.product(range(5),repeat=3):
            if not mask[0,a]:continue
            action=torch.zeros(1,32,dtype=torch.long);action[0,0]=a;action[0,1]=b;action[0,30]=z
            values.append(float(c.joint_value(p,m,v,action)[0]))
        self.assertAlmostEqual(qg,max(values),places=6)
        model=c.CenteredLinearDDQN()
        for param in model.parameters():param.data.zero_()
        action,q,_=model.greedy(states(1))
        self.assertTrue(torch.all(action[:,:30]==1));self.assertTrue(torch.all(action[:,30:]==0))

    def test_analytic_raw_output_gradients(self):
        raw=torch.randn(1,161,dtype=torch.float64,requires_grad=True)
        a=torch.arange(32)[None]%5
        q=c.joint_value(raw[:,:150].reshape(1,30,5)/30,raw[:,150:160].reshape(1,2,5)/2,raw[:,160],a)
        q.sum().backward();expected=torch.empty_like(raw)
        for b in range(32):
            expected[0,b*5:b*5+5]=.2/(30 if b<30 else 2)
            expected[0,b*5+int(a[0,b])]-=1/(30 if b<30 else 2)
        expected[0,160]=1.
        torch.testing.assert_close(raw.grad,expected)

    def test_scalar_saturation_factor_removed(self):
        value=torch.tensor([40.],dtype=torch.float64,requires_grad=True)
        p=torch.zeros(1,30,5,dtype=torch.float64);m=torch.zeros(1,2,5,dtype=torch.float64);a=torch.zeros(1,32,dtype=torch.long)
        old=c.centered_softplus_reference(p,m,value,a)
        old_grad=torch.autograd.grad(old.sum(),value)[0]
        linear=c.joint_value(p,m,value,a);linear_grad=torch.autograd.grad(linear.sum(),value)[0]
        self.assertLess(float(old_grad),1e-16);self.assertEqual(float(linear_grad),1.)
        self.assertGreater(float(linear),0.) # signed outputs are deliberately allowed

    def test_bc_gradient_paths_preferences_and_bc_only_freeze(self):
        model=c.CenteredLinearDDQN();x=states();labels=torch.ones(4,32,dtype=torch.long);labels[:,30:]=0
        loss=c.imitation_loss(model,x,labels);loss.backward()
        self.assertTrue(torch.all(model.student.net[-1].weight.grad[160]==0))
        self.assertEqual(float(model.student.net[-1].bias.grad[160]),0.)
        self.assertGreater(float(model.student.net[0].weight.grad.abs().sum()),0.)
        frozen=copy.deepcopy(model).eval().requires_grad_(False)
        before={k:v.clone() for k,v in frozen.state_dict().items()}
        with torch.no_grad():frozen.greedy(x)
        for k,v in frozen.state_dict().items():torch.testing.assert_close(v,before[k])
        p,m,v,_=model.tables(x)
        mask=torch.from_numpy(c.previous.allowed(x)[:,:30])
        raw=(-p).masked_fill(~mask,-torch.inf)
        centered=(-(p-p.mean(2,keepdim=True))).masked_fill(~mask,-torch.inf)
        torch.testing.assert_close(F.softmax(raw,2),F.softmax(centered,2))

    def test_ddqn_online_selection_target_eval_and_terminal_bypass(self):
        model=c.CenteredLinearDDQN();target=copy.deepcopy(model).requires_grad_(False)
        x=states(2);ns=x.clone();ns[1,-1]=1.;a=torch.ones(2,32,dtype=torch.long)
        selected=torch.full((1,32),3,dtype=torch.long)
        with patch.object(model,'forward',return_value=torch.tensor([-4.,-2.],requires_grad=True)),patch.object(model,'greedy',return_value=(selected,None,None)) as choose,patch.object(target,'forward',return_value=torch.tensor([-3.])) as evaluate:
            loss=c.ddqn_loss(model,target,x,a,torch.tensor([-1.,-2.]),ns,torch.tensor([False,True]))
            self.assertEqual(float(loss.detach()),0.)
            self.assertTrue(torch.equal(choose.call_args.args[0],ns[:1]))
            self.assertTrue(torch.equal(evaluate.call_args.args[1],selected))
        ns[:,-1]=1.
        with patch.object(model,'greedy',side_effect=AssertionError('terminal decode')):
            c.ddqn_loss(model,target,x,a,-torch.ones(2),ns,torch.ones(2,dtype=torch.bool))

    def test_td_gradients_and_invalid_actions(self):
        model=c.CenteredLinearDDQN();target=copy.deepcopy(model).requires_grad_(False)
        x=states();ns=x.clone();ns[:,-1]=1.;a=model.greedy(x)[0]
        loss=c.ddqn_loss(model,target,x,a,-5*torch.ones(4),ns,torch.ones(4,dtype=torch.bool));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(model.student.net[-1].weight.grad[160].abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in target.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaises(ValueError):model(x,torch.full((4,32),5,dtype=torch.long))
        with self.assertRaises(ValueError):model.greedy(ns)
        with self.assertRaises(ValueError):model.greedy(x.double())

    def test_checkpoint_deployment_and_research_guard(self):
        model=c.CenteredLinearDDQN().eval();x=states(1)[0].numpy()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'final.pt';torch.save(model.state_dict(),p)
            reloaded=c.CenteredLinearDDQN().eval();reloaded.load_state_dict(torch.load(p,weights_only=True))
            with patch.object(c.previous,'demonstration_actions',side_effect=AssertionError('teacher invoked')):
                np.testing.assert_array_equal(c.deploy_actions(model,x),c.deploy_actions(reloaded,x))
        for mode in ['development','pilot','final']:
            result=subprocess.run([sys.executable,str(Path(c.__file__)),'--mode',mode,'--confirmation','anything'],capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0);self.assertIn('disabled',result.stderr)


if __name__=='__main__':
    output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=False)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Mechanics))
    record=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                new_optimizer_updates=0,new_simulator_steps=0,research_seeds_consumed=False,research_launch_authorized=False)
    (output/'verification.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
