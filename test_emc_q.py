"""Mechanics only; backward checks do not call an optimizer or simulator."""
import itertools,io,json,hashlib
from pathlib import Path
import unittest
import numpy as np
import torch
from emc_q import EMCQ,decode_tables,chosen_tables,frozen_target,ddqn_targets


class Mechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def test_exact_small_exhaustive(self):
        for seed in range(20):
            torch.manual_seed(seed)
            u=torch.randn(3,2,2);m=torch.randn(3,2,2);f=torch.randn(3,2,2,2);g=torch.randn(3,2);e=torch.randn(2,2,2)
            actions,val=decode_tables(u,m,f,g,e)
            all_actions=torch.tensor(list(itertools.product(range(2),repeat=4)))
            for b in range(3):
                n=len(all_actions)
                scores=chosen_tables(u[b:b+1].expand(n,-1,-1),m[b:b+1].expand(n,-1,-1),f[b:b+1].expand(n,-1,-1,-1),g[b:b+1].expand(n,-1),e,all_actions)
                torch.testing.assert_close(val[b],scores.max())
            torch.testing.assert_close(chosen_tables(u,m,f,g,e,actions),val)

    def test_batch_single_and_count(self):
        torch.manual_seed(42);model=EMCQ();s=torch.randn(7,533)
        a,q=model.greedy(s)
        self.assertEqual(tuple(a.shape),(7,32));self.assertTrue(torch.all((a>=0)&(a<5)))
        torch.testing.assert_close(q,model(s,a))
        for i in range(7):
            ai,qi=model.greedy(s[i:i+1]);torch.testing.assert_close(ai[0],a[i]);torch.testing.assert_close(qi[0],q[i])
        self.assertEqual(sum(p.numel() for p in model.parameters()),69361)

    def test_ties_and_ablation(self):
        u=torch.tensor([[[1.,2.],[3.,0.]]]);m=torch.zeros(1,2,2);f=torch.zeros(1,2,2,2);g=torch.ones(1,2);e=torch.ones(2,2,2)
        a,q=decode_tables(u,m,f,g,e)
        torch.testing.assert_close(a,torch.tensor([[1,0,0,0]]));torch.testing.assert_close(q,torch.tensor([5.]))
        a,q=decode_tables(u*0,m,f,g,e);torch.testing.assert_close(a,torch.zeros_like(a))
        # Explicit coupling: preferred destination changes with UAV movement.
        u=u*0;f[0,0,1,0]=2;e.zero_();e[0,1,0]=1
        a,q=decode_tables(u,m,f,g,e)
        self.assertEqual(a[0,0],1);self.assertEqual(a[0,-2],1)

    def test_all_head_blocks_and_embeddings_receive_gradient(self):
        torch.manual_seed(51);model=EMCQ();s=torch.randn(100,533);a=torch.randint(0,5,(100,32))
        model(s,a).square().mean().backward()
        grad=model.head.weight.grad
        for start,end in ((0,150),(150,175),(175,475),(475,477)):
            self.assertGreater(float(grad[start:end].abs().sum()),0)
        self.assertGreater(float(model.encoder[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.movement_embeddings.grad.abs().sum()),0)

    def test_target_detached_and_terminal_skipped(self):
        torch.manual_seed(99);online=EMCQ();target=frozen_target(online)
        successor=torch.randn(4,533,requires_grad=True)
        with torch.no_grad():successor[1]=float('nan');successor[3]=float('nan')
        reward=torch.tensor([-1.,-2.,-3.,-4.]);done=torch.tensor([0.,1.,0.,1.])
        y=ddqn_targets(online,target,reward,successor,done)
        self.assertFalse(y.requires_grad);torch.testing.assert_close(y[[1,3]],reward[[1,3]])
        active=torch.tensor([0,2]);a,_=online.greedy(successor[active])
        with torch.no_grad():expected=reward[active]+target(successor[active],a)
        torch.testing.assert_close(y[active],expected)
        online(torch.randn(4,533),torch.zeros(4,32,dtype=torch.long)).sub(y).square().mean().backward()
        self.assertIsNone(successor.grad)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in target.parameters()))

    def test_one_encoder_and_invalid_actions(self):
        model=EMCQ();calls=[];handle=model.encoder.register_forward_hook(lambda *args:calls.append(1))
        model.greedy(torch.zeros(1,533));handle.remove();self.assertEqual(len(calls),1)
        with self.assertRaises(ValueError):model(torch.zeros(1,533),torch.full((1,32),5,dtype=torch.long))


if __name__=='__main__':
    out=Path('validation/emc_mechanics_2026-11-04_01');out.mkdir(parents=True,exist_ok=False)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Mechanics))
    (out/'tests.txt').write_text(stream.getvalue(),encoding='utf-8')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (out/'verification.json').write_text(json.dumps(dict(state='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,source_sha256={p:sha(p) for p in ('emc_q.py','test_emc_q.py')},optimizer_updates=0,simulator_steps=0),indent=2),encoding='utf-8')
    print(stream.getvalue())
    if not result.wasSuccessful():raise SystemExit(1)
