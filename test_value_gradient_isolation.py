"""Synthetic mechanics checks, not policy evidence. No simulator/teacher/seeds."""
import copy
import io
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

import return_calibrated_handoff as old
import heuristic_initialized_ddqn_baseline as baseline
import test_return_calibrated_handoff as fixtures
import value_gradient_isolation as vgi

COUNTS = dict(synthetic_BC_optimizer_steps=0, synthetic_TD_optimizer_steps=0,
    independent_reference_optimizer_steps=0, inherited_baseline_optimizer_steps=0,
    synthetic_ridge_solves=0, artifact_TD_optimizer_steps=0)


def model(arm=vgi.PRIMARY):
    with torch.random.fork_rng(devices=[]): m = vgi.Critic(arm)
    m.load_state_dict(fixtures.model(vgi.BASE if arm in vgi.TD_RULES else arm).state_dict())
    return m


def counted(optimizer, key):
    def record(*unused): COUNTS[key] += 1
    return optimizer.register_step_post_hook(record)


def calibrate(m):
    result = old.calibrate(m, fixtures.episodes())
    COUNTS['synthetic_ridge_solves'] += 1
    return result


def reference_tables(m, s, rule='ordinary'):
    """Independent detach/identity-gradient construction, never custom affine."""
    first, _, last = m.student.net
    h = F.relu(F.linear(s, first.weight, first.bias))
    td_h = h.detach()+(h-h.detach())/math.sqrt(2.) if rule == 'rescale_shared_td' else h
    branch = F.linear(td_h, last.weight[:160], last.bias[:160])
    value_h = h.detach() if rule == 'isolate_value' else td_h
    value = F.linear(value_h, last.weight[160:], last.bias[160:])[:, 0]
    return branch[:, :150].reshape(-1, 30, 5)/30, branch[:, 150:].reshape(-1, 2, 5)/2, value


def reference_q(m, s, a, rule='ordinary'):
    p, motion, value = reference_tables(m, s, rule)
    return value-(p.gather(2, a[:, :30, None])[:, :, 0]-p.mean(2)).sum(1)-(motion.gather(2, a[:, 30:, None])[:, :, 0]-motion.mean(2)).sum(1)


def reference_greedy(m, s):
    p, motion, _ = reference_tables(m, s)
    mask = torch.from_numpy(baseline.allowed(s))
    return torch.cat([p.masked_fill(~mask[:, :30], torch.inf).argmin(2), motion.argmin(2)], 1)


def reference_keep(m, s, a, alpha):
    p, motion, _ = reference_tables(m, s)
    mask = torch.from_numpy(baseline.allowed(s))
    vehicle_logits = (-p/alpha).masked_fill(~mask[:, :30], -torch.inf)
    move_logits = -motion/alpha
    vehicle = torch.logsumexp(vehicle_logits, 2)-vehicle_logits.gather(2, a[:, :30, None])[:, :, 0]
    move = torch.logsumexp(move_logits, 2)-move_logits.gather(2, a[:, 30:, None])[:, :, 0]
    return vehicle.mean()+move.mean()


def reference_losses(m, target, data, demo, alpha):
    s, a, rewards, ns, done = data
    with torch.no_grad():
        y = rewards.clone()
        if (~done).any(): y[~done] += reference_q(target, ns[~done], reference_greedy(m, ns[~done]))
    q = reference_q(m, s, a, vgi.TD_RULES.get(m.handoff_arm, 'ordinary'))
    td = F.smooth_l1_loss(q, y)
    keep = reference_keep(m, *demo, alpha) if m.definition.retention else td.new_zeros(())
    return td, keep, y


def demo(n=8, adversarial=False):
    x = fixtures.states(n)
    a = fixtures.batch(n)[1].clone()
    if adversarial: a[:, :30] = 4; a[:, 30:] = 1
    return x, a


class Mechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_01_full_affine_forward_is_bit_exact(self):
        for dtype in [torch.float32, torch.float64]:
            h = (torch.arange(24, dtype=dtype).reshape(4, 6)[:, ::2]/19).detach().requires_grad_()
            w = (torch.arange(48, dtype=dtype).reshape(8, 6)[:, ::2]/37).detach().requires_grad_()
            b = torch.arange(8, dtype=dtype)/31
            for rule in vgi.TD_RULES.values():
                self.assertTrue(torch.equal(vgi.td_affine(h, w, b, rule), F.linear(h, w, b)))

    def test_02_pure_value_blocks_only_feature_gradient(self):
        for rule in vgi.TD_RULES.values():
            h = torch.arange(12, dtype=torch.float64).reshape(4, 3).requires_grad_()
            w = (torch.arange(21, dtype=torch.float64).reshape(7, 3)/23).requires_grad_()
            b = torch.zeros(7, dtype=torch.float64, requires_grad=True)
            out = vgi.td_affine(h, w, b, rule)
            gh, gw, gb = torch.autograd.grad(out[:, -1].sum(), [h, w, b])
            expected = torch.zeros_like(h) if rule == 'isolate_value' else w[-1].expand_as(h)/math.sqrt(2)
            torch.testing.assert_close(gh, expected, rtol=0, atol=1e-14)
            torch.testing.assert_close(gw[-1], h.sum(0), rtol=0, atol=0)
            self.assertEqual(float(gb[-1]), 4.); self.assertEqual(float(gw[:-1].abs().sum()), 0.)

    def test_03_branch_gradient_preserved_or_declared_rescale(self):
        for rule in vgi.TD_RULES.values():
            h = torch.arange(12, dtype=torch.float64).reshape(4, 3).requires_grad_()
            w = (torch.arange(21, dtype=torch.float64).reshape(7, 3)/23).requires_grad_()
            b = torch.zeros(7, dtype=torch.float64, requires_grad=True)
            up = torch.arange(28, dtype=torch.float64).reshape(4, 7)/17; up[:, -1] = 0
            actual = torch.autograd.grad(vgi.td_affine(h, w, b, rule), [h, w, b], up)
            reference = torch.autograd.grad(F.linear(h, w, b), [h, w, b], up)
            factor = 1. if rule == 'isolate_value' else 1/math.sqrt(2)
            torch.testing.assert_close(actual[0], reference[0]*factor, rtol=1e-13, atol=1e-13)
            for a, r in zip(actual[1:], reference[1:]): torch.testing.assert_close(a, r, rtol=0, atol=0)

    def test_04_bad_affine_inputs_rejected(self):
        h = torch.ones(2, 3);w = torch.ones(7, 3);b = torch.zeros(7)
        for args in [(h, w, b, 'bad'), (h.double(), w, b, 'isolate_value'),
                     (h, w, b[:3], 'isolate_value'), (h[:0], w, b, 'isolate_value'),
                     (h*float('nan'), w, b, 'isolate_value')]:
            with self.assertRaises(ValueError): vgi.td_affine(*args)

    def test_05_all_nine_forward_state_and_storage_match(self):
        self.assertEqual(len(old.ARMS), 7);self.assertEqual(len(vgi.ARMS), 9)
        for arm in vgi.ARMS:
            m = model(arm);base = fixtures.model(vgi.BASE if arm in vgi.TD_RULES else arm)
            self.assertEqual(set(m.state_dict()), set(base.state_dict()))
            self.assertEqual(sum(p.numel() for p in m.parameters()), 16841)
            self.assertEqual(baseline.parameter_bytes(m), 67364)
            self.assertEqual(sum(b.numel()*b.element_size() for b in m.buffers()), 20)
            for n in [1, 8, 40]:
                x = fixtures.states(n);a = fixtures.batch(n)[1]
                self.assertTrue(torch.equal(m(x, a), base(x, a)))
                self.assertTrue(torch.equal(m.td_prediction(x, a), base(x, a)))
                for got, want in zip(m.greedy(x)[:2], base.greedy(x)[:2]):self.assertTrue(torch.equal(got, want))

    def test_06_mask_ties_and_terminal_decoding(self):
        for arm in [vgi.PRIMARY, vgi.RESCALE]:
            m=model(arm)
            with torch.no_grad():m.student.net[2].weight[:160].zero_();m.student.net[2].bias[:160].zero_()
            x=fixtures.states(8);a,q,_=m.greedy(x)
            self.assertTrue((a[:, ::2][:, :15] == 1).all())
            self.assertTrue((a[:, 1:30:2] == 0).all());self.assertTrue((a[:, 30:] == 0).all())
            self.assertTrue(torch.equal(m.td_prediction(x,a),q))
            terminal=x.clone();terminal[:,-1]=1
            with self.assertRaises(ValueError):m.greedy(terminal)
            self.assertTrue(torch.isfinite(m(terminal,a)).all())

    def test_07_mixed_gradients_match_independent_detached_reference(self):
        for arm in [vgi.PRIMARY,vgi.RESCALE]:
            m=model(arm);ref=copy.deepcopy(m);data=fixtures.batch();examples=demo();alpha=.05
            target,_=vgi.td_setup(m);rt=copy.deepcopy(target)
            s,a,_,_,_=data
            td,keep,y=reference_losses(ref,rt,data,examples,alpha)
            actual_td=F.smooth_l1_loss(m.td_prediction(s,a),y)
            actual_keep=old.imitation_loss(m,*examples,alpha)
            (actual_td+actual_keep).backward();(td+keep).backward()
            torch.testing.assert_close(actual_td,td,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(actual_keep,keep,rtol=1e-5,atol=1e-6)
            for p,r in zip(m.parameters(),ref.parameters()):torch.testing.assert_close(p.grad,r.grad,rtol=3e-5,atol=2e-6)
            self.assertTrue(all(p.grad is None for p in target.parameters()))

    def test_08_BC_optimizer_behavior_unchanged(self):
        candidates=[model(vgi.PRIMARY),model(vgi.RESCALE),fixtures.model()]
        x,a=demo();digests=[]
        for m in candidates:
            optimizer=torch.optim.Adam(m.parameters(),lr=.001)
            handle=counted(optimizer,'synthetic_BC_optimizer_steps')
            loss=old.bc_loss(m,x,a);optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),10.);optimizer.step();handle.remove()
            digests.append(old.tensor_hash(m))
        self.assertEqual(len(set(digests)),1)

    def test_09_calibration_handoff_parity_and_no_mutation(self):
        hashes=[];reports=[]
        for arm in [vgi.PRIMARY,vgi.RESCALE,vgi.BASE]:
            m=model(arm);before=old.tensor_hash(m);result=calibrate(m)
            self.assertEqual(old.tensor_hash(m),before);self.assertEqual(result.model.handoff_arm,arm)
            hashes.append(old.tensor_hash(result.model));reports.append(result.report)
        self.assertEqual(len(set(hashes)),1);self.assertEqual(reports[0],reports[1]);self.assertEqual(reports[0],reports[2])

    def test_10_one_joint_step_matches_reference_and_detaches_target(self):
        for arm in [vgi.PRIMARY,vgi.RESCALE]:
            m=model(arm);target,opt=vgi.td_setup(m);ref=copy.deepcopy(m);rt=copy.deepcopy(target)
            ropt=torch.optim.Adam(ref.parameters(),lr=.0001,betas=(.9,.999),eps=1e-8,weight_decay=0)
            handle=counted(opt,'synthetic_TD_optimizer_steps');rh=counted(ropt,'independent_reference_optimizer_steps')
            data=fixtures.batch();examples=demo();alpha=.05;target_hash=old.tensor_hash(target)
            with patch.object(opt,'step',wraps=opt.step) as step_spy:
                report=vgi.ddqn_retention_step(m,target,opt,data,alpha,examples)
                self.assertEqual(step_spy.call_count,1)
            td,keep,y=reference_losses(ref,rt,data,examples,alpha)
            ropt.zero_grad(set_to_none=True);(td+keep).backward();torch.nn.utils.clip_grad_norm_(ref.parameters(),10.);ropt.step()
            handle.remove();rh.remove()
            for p,r in zip(m.parameters(),ref.parameters()):torch.testing.assert_close(p,r,rtol=1e-5,atol=2e-7)
            np.testing.assert_allclose(report['target'],y.numpy(),rtol=1e-5,atol=1e-6)
            self.assertEqual(report['optimizer_steps'],1);self.assertFalse(report['target_copy_performed'])
            self.assertEqual(old.tensor_hash(target),target_hash);self.assertTrue(all(p.grad is None for p in target.parameters()))
            self.assertTrue(all(float(state['step'])==1. for state in opt.state.values()))
            self.assertTrue(all(not torch.isinf(p).any() for p in m.parameters()))

    def test_11_all_terminal_targets_never_decode_next_states(self):
        for arm in [vgi.PRIMARY,vgi.RESCALE]:
            m=model(arm);target,opt=vgi.td_setup(m);handle=counted(opt,'synthetic_TD_optimizer_steps')
            original=m.greedy
            def guarded(s):
                self.assertTrue((s[:,-1]<1).all());return original(s)
            with patch.object(m,'greedy',side_effect=guarded),patch.object(target,'forward',side_effect=AssertionError('terminal target evaluated')):
                report=vgi.ddqn_retention_step(m,target,opt,fixtures.batch(all_terminal=True),.05,demo())
            handle.remove();self.assertTrue(np.all(np.array(report['next_actions'])==-1))
            self.assertEqual(report['target'],report['rewards']);self.assertTrue(np.all(np.array(report['target_q'])==0))

    def test_12_clipping_and_large_retention_step_match_reference(self):
        m=model();target,opt=vgi.td_setup(m);ref=copy.deepcopy(m);rt=copy.deepcopy(target)
        ropt=torch.optim.Adam(ref.parameters(),lr=.0001,betas=(.9,.999),eps=1e-8,weight_decay=0)
        handle=counted(opt,'synthetic_TD_optimizer_steps');rh=counted(ropt,'independent_reference_optimizer_steps')
        data=fixtures.batch();examples=demo(adversarial=True);alpha=.0001
        report=vgi.ddqn_retention_step(m,target,opt,data,alpha,examples)
        td,keep,_=reference_losses(ref,rt,data,examples,alpha);ropt.zero_grad(set_to_none=True);(td+keep).backward()
        expected_norm=torch.nn.utils.clip_grad_norm_(ref.parameters(),10.);ropt.step();handle.remove();rh.remove()
        self.assertTrue(report['clipped']);self.assertGreater(report['clip_norm'],10)
        self.assertAlmostEqual(report['clip_norm']/float(expected_norm),1.,places=5)
        self.assertLessEqual(float(torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in m.parameters()]))),10.00001)
        for p,r in zip(m.parameters(),ref.parameters()):torch.testing.assert_close(p,r,rtol=1e-5,atol=2e-7)

    def test_13_invalid_steps_and_aliases_fail_before_updates(self):
        m=model();target,opt=vgi.td_setup(m);before=old.tensor_hash(m)
        wrong,_=vgi.td_setup(model(vgi.RESCALE))
        alias=copy.deepcopy(target);alias.student.net[2].weight.data=m.student.net[2].weight.data
        buffer_alias=copy.deepcopy(target);buffer_alias.student.cycle_to_work=m.student.cycle_to_work
        examples=demo();data=fixtures.batch()
        for t,alpha,ex in [(m,.1,examples),(wrong,.1,examples),(alias,.1,examples),
                           (buffer_alias,.1,examples),(target,0.,examples),(target,.1,None)]:
            with self.assertRaises(ValueError):vgi.ddqn_retention_step(m,t,opt,data,alpha,ex)
        broken=list(data);broken[-1]=~broken[-1]
        with self.assertRaises(ValueError):vgi.ddqn_retention_step(m,target,opt,tuple(broken),.1,examples)
        broken=list(data);broken[1]=data[1].clone();broken[1][:,0]=0
        with self.assertRaises(ValueError):vgi.ddqn_retention_step(m,target,opt,tuple(broken),.1,examples)
        opt.param_groups[0]['lr']=.01
        with self.assertRaises(ValueError):vgi.ddqn_retention_step(m,target,opt,data,.1,examples)
        self.assertEqual(old.tensor_hash(m),before);self.assertFalse(opt.state)
        with self.assertRaises(ValueError):model('unrecognized')

    def test_14_nonfinite_gradient_stops_before_Adam(self):
        m=model();target,opt=vgi.td_setup(m);before=old.tensor_hash(m)
        hook=m.student.net[2].weight.register_hook(lambda grad:torch.full_like(grad,float('nan')))
        try:
            with patch.object(opt,'step',wraps=opt.step) as spy:
                with self.assertRaisesRegex(ValueError,'nonfinite'):vgi.ddqn_retention_step(m,target,opt,fixtures.batch(),.05,demo())
                self.assertEqual(spy.call_count,0)
        finally:hook.remove()
        self.assertEqual(old.tensor_hash(m),before);self.assertFalse(opt.state)

    def test_15_ordinary_ablation_step_matches_preserved_implementation(self):
        m=model(vgi.BASE);original=fixtures.model();target,opt=vgi.td_setup(m);ot,oo=old.td_setup(original)
        handle=counted(opt,'synthetic_TD_optimizer_steps');oh=counted(oo,'inherited_baseline_optimizer_steps')
        got=vgi.ddqn_retention_step(m,target,opt,fixtures.batch(),.05,demo())
        expected=old.ddqn_retention_step(original,ot,oo,fixtures.batch(),.05,demo());handle.remove();oh.remove()
        self.assertEqual(old.tensor_hash(m),old.tensor_hash(original))
        self.assertEqual(got['gradient_norms'],expected['gradient_norms']);self.assertEqual(got['target'],expected['target'])

    def test_16_export_is_original_class_weights_and_independent_storage(self):
        for arm in [vgi.PRIMARY,vgi.RESCALE,vgi.BASE]:
            m=model(arm);exported=vgi.deployment_model(m)
            self.assertIs(type(exported),old.Critic);self.assertFalse(hasattr(exported,'td_prediction'))
            self.assertEqual(old.tensor_hash(m),old.tensor_hash(exported))
            self.assertTrue(all(not p.requires_grad for p in exported.parameters()))
            storage={v.untyped_storage().data_ptr() for v in m.state_dict().values()}
            self.assertFalse(any(v.untyped_storage().data_ptr() in storage for v in exported.state_dict().values()))
            buff=io.BytesIO();torch.save(exported.state_dict(),buff);buff.seek(0)
            reload=fixtures.model();reload.load_state_dict(torch.load(buff,weights_only=True))
            for n in [1,8,40]:
                x=fixtures.states(n)
                for left,right in zip(m.greedy(x)[:2],reload.greedy(x)[:2]):self.assertTrue(torch.equal(left,right))

    def test_17_routing_does_not_leak_into_BC_calibration_or_inference(self):
        for arm in [vgi.PRIMARY,vgi.RESCALE]:
            m=model(arm);data=fixtures.batch();m.td_prediction(data[0],data[1])
            with patch.object(vgi,'td_affine',side_effect=AssertionError('TD routing leaked')):
                m(data[0],data[1]);m.greedy(data[0]);old.bc_loss(m,*demo());calibrate(m)

    def test_18_research_and_final_CLI_disabled_even_with_confirmation(self):
        self.assertFalse(vgi.DEVELOPMENT_RELEASED);self.assertFalse(vgi.FINAL_RELEASED)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'research-must-not-exist'
            for mode in ['development','final']:
                for confirm in [[],['--confirmation','LAUNCH_FROZEN_VALUE_GRADIENT_ISOLATION_COMPARISON']]:
                    result=subprocess.run([sys.executable,str(Path(vgi.__file__)),'--mode',mode,'--output',str(out)]+confirm,
                        capture_output=True,text=True,timeout=60)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('Research modes disabled',result.stderr);self.assertFalse(out.exists())


if __name__=='__main__':unittest.main(verbosity=2)
