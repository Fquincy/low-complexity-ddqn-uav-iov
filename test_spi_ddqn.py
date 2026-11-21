"""Mechanics gate for the single-pass prototype."""
import io
from pathlib import Path
import unittest
import numpy as np
import torch
from spi_ddqn import SPIDDQN,select


class SPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.manual_seed(20261101);cls.model=SPIDDQN().eval();cls.state=np.linspace(0,1,533,dtype=np.float32)

    def test_shapes_and_parameters(self):
        x=torch.zeros(3,533)
        self.assertEqual(tuple(self.model.logits(x).shape),(3,32,5))
        self.assertGreater(self.model.parameter_bytes(),0)
        self.assertTrue(np.isfinite(self.model.parameter_bytes()))

    def test_joint_score_depends_on_actions(self):
        emb=self.model.encode(torch.zeros(2,533))
        a=torch.zeros(2,32,dtype=torch.long);b=a.clone();b[:,0]=1
        self.assertFalse(torch.equal(self.model.score_joint(emb,a),self.model.score_joint(emb,b)))

    def test_selection_legal_and_bounded(self):
        action,info=select(self.model,self.state,2,8)
        self.assertEqual(action.shape,(32,));self.assertTrue(np.all((action>=0)&(action<5)))
        self.assertEqual(info['candidate_count'],8);self.assertEqual(info['encoder_calls'],1)

    def test_gradient_reaches_encoder_and_joint_parameters(self):
        torch.manual_seed(20261102)
        states=torch.randn(4,533);actions=torch.randint(0,5,(4,32));target=torch.ones(4)
        loss=(self.model.joint_values(states,actions)-target).pow(2).mean();loss.backward()
        self.assertGreater(float(self.model.encoder[0].weight.grad.abs().sum()),0.)
        self.assertGreater(float(self.model.action_embedding.grad.abs().sum()),0.)

    def test_deterministic_decode(self):
        a,ia=select(self.model,self.state);b,ib=select(self.model,self.state)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(ia['joint_scores'],ib['joint_scores'])


if __name__=='__main__':
    out=Path('validation/spi_ddqn_mechanics_2026-11-01_01');out.mkdir(parents=True,exist_ok=False)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SPITests))
    (out/'tests.txt').write_text(stream.getvalue(),encoding='utf-8')
    (out/'verification.json').write_text(__import__('json').dumps(dict(state='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,parameter_bytes=SPITests.model.parameter_bytes()),indent=2),encoding='utf-8')
    print(stream.getvalue())
    if not result.wasSuccessful():raise SystemExit(1)
