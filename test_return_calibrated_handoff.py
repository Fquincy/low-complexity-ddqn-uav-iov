"""Synthetic mechanics only. No simulator/teacher or reserved research seeds."""
import copy
import hashlib
import io
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

import return_calibrated_handoff as h

COUNTS = dict(synthetic_bc_optimizer_steps=0, synthetic_td_optimizer_steps=0,
              independent_reference_optimizer_steps=0)


def model(arm='calibrated_retained_ddqn'):
    # Fork restores ambient RNG; every parameter is replaced deterministically.
    # No explicit research or engineering seed is selected or consumed.
    with torch.random.fork_rng():
        m = h.Critic(arm)
    with torch.no_grad():
        for i, p in enumerate(m.parameters()):
            v = torch.arange(p.numel(), dtype=torch.float32).reshape(p.shape)
            p.copy_(torch.sin(v*.037+i)*.025)
        m.student.net[0].bias.add_(.15)
        m.student.net[2].bias[:150].copy_(torch.tensor([.8, .4, .2, -.3, -.1]*30))
        m.student.net[2].bias[150:160].copy_(torch.tensor([-2., 1., 1.5, 2., 2.5]*2))
    return m


def states(n=40):
    x = torch.from_numpy(((np.arange(n*533).reshape(n, 533)%37)/80).astype(np.float32))
    task = x[:, :240].reshape(n, 30, 8)
    task[:, :, 6] = torch.where(torch.arange(30)%2 == 0, .9, .6)
    task[:, :, 7] = .9
    x[:, -1] = (torch.arange(n)%40).float()/40
    return x


def episodes():
    out = []
    for ep in range(10):
        x = states()+ep*.00001
        x[:, -1] = torch.arange(40).float()/40
        a = torch.full((40, 32), 3, dtype=torch.long)
        a[:, 1:30:2] = 0
        a[:, 30:] = 0
        r = -.25-torch.arange(40).float()*.0001-ep*.001
        d = torch.zeros(40, dtype=torch.bool); d[-1] = True
        out.append(h.DemoEpisode(x, a, r, d))
    return out


def batch(n=8, all_terminal=False):
    x = states(n);ns = x.clone();ns[:, -1] += .025
    done = torch.zeros(n, dtype=torch.bool)
    done[:] = all_terminal
    if not all_terminal: done[-1] = True
    ns[done, -1] = 1.
    a = torch.full((n, 32), 3, dtype=torch.long);a[:, 30:] = 0
    r = -torch.linspace(.3, 1., n)
    return x, a, r, ns, done


def step(*args, **kwargs):
    report = h.ddqn_retention_step(*args, **kwargs)
    COUNTS['synthetic_td_optimizer_steps'] += 1
    return report


class Mechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.episodes = episodes()

    def test_01_episode_returns_are_separate_and_terminal(self):
        for e in self.episodes: e.validate()
        np.testing.assert_allclose(h.episode_returns([-1., -2., -3.]), [-6., -5., -3.])
        e = self.episodes[0]
        broken = h.DemoEpisode(e.states, e.actions, e.rewards, torch.zeros(40, dtype=torch.bool))
        with self.assertRaises(ValueError): broken.validate()

    def test_02_scaling_actions_ties_bound_and_temperature(self):
        m = model();before = h.tensor_hash(m);result = h.calibrate(m, self.episodes)
        r = result.report;beforeprobe = r['stages']['after_bc'];scaled = r['stages']['after_scale']
        self.assertGreater(result.alpha, 0);self.assertLessEqual(result.alpha, 1)
        self.assertEqual(before, h.tensor_hash(m));self.assertEqual(beforeprobe['actions'], scaled['actions'])
        self.assertEqual(scaled['actions'], r['stages']['after_value_fit']['actions'])
        self.assertLessEqual(max(scaled['all_action_ranges'][:320]), r['reward_scale']+1e-5)
        self.assertAlmostEqual(beforeprobe['imitation_loss'], scaled['imitation_loss'], places=5)
        # Scaling must leave hidden parameters and unsupervised V row intact.
        with torch.no_grad():
            allx = torch.cat([e.states for e in self.episodes])
            greedy = result.model.greedy(allx)[0]
            for n, p in m.named_parameters():
                if n.startswith('student.net.0.'):
                    torch.testing.assert_close(result.model.state_dict()[n], p, rtol=0, atol=0)
            original_p, original_m, _, _ = m.tables(allx)
            new_p, new_m, _, _ = result.model.tables(allx)
            torch.testing.assert_close(new_p, original_p*result.alpha, rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(new_m, original_m*result.alpha, rtol=1e-5, atol=1e-7)
            cost = (new_p.gather(2,self.episodes[0].actions[:1,:30,None].expand(400,-1,-1)).squeeze(2)-new_p.mean(2)).sum(1)
            self.assertTrue(torch.isfinite(cost).all())
        # Exact output ties preserve inherited first-index convention.
        tied = model()
        with torch.no_grad():
            tied.student.net[2].weight[:160].zero_(); tied.student.net[2].bias[:160].zero_()
        tied_result = h.calibrate(tied, self.episodes)
        self.assertEqual(tied_result.alpha, 1.)
        acts = np.array(tied_result.report['stages']['after_scale']['actions'])
        self.assertTrue((acts[:, 1:30:2] == 0).all());self.assertTrue((acts[:, 30:] == 0).all())

    def test_03_return_targets_withheld_do_not_affect_fit(self):
        first = h.calibrate(model(), self.episodes)
        changed = list(self.episodes)
        for i in [8, 9]:
            e = changed[i]
            changed[i] = h.DemoEpisode(e.states, e.actions, e.rewards*20, e.dones)
        second = h.calibrate(model(), changed)
        self.assertEqual(h.tensor_hash(first.model), h.tensor_hash(second.model))
        self.assertEqual(first.alpha, second.alpha)
        self.assertNotEqual(first.report['residuals']['withheld_returns'], second.report['residuals']['withheld_returns'])

    def test_04_ridge_matches_independent_augmented_solver(self):
        x = (np.arange(80*24).reshape(80,24)%29)/19.
        x[:,0] = 1.;x[:,1] = x[:,2]
        g = np.linspace(-12,-.3,80);t = g+.04*np.arange(80)
        w,b,report = h.ridge_value_coefficients(x,t,g)
        mu = x.mean(0);sig = np.maximum(1e-6,np.sqrt(np.mean((x-mu)**2,0)));X=(x-mu)/sig
        sg=max(1e-6,np.sqrt(np.mean(g*g)));u=t/sg
        # Solve augmented regression, rather than sharing the implementation's normal equations.
        aug=np.vstack([X/np.sqrt(len(x)),np.sqrt(.001)*np.eye(24)])
        rhs=np.concatenate([(u-u.mean())/np.sqrt(len(x)),np.zeros(24)])
        beta=np.linalg.lstsq(aug,rhs,rcond=None)[0]
        expected=sg*(X@beta+u.mean())
        np.testing.assert_allclose(x.astype(np.float32)@w+b,expected,rtol=1e-5,atol=1e-4)
        self.assertLess(report['rank'],24)
        const=np.ones((40,24));w,b,r=h.ridge_value_coefficients(const,np.full(40,-3.),np.full(40,-3.))
        self.assertTrue((w==0).all());self.assertAlmostEqual(float(b),-3.)

    def test_05_invalid_handoff_is_atomic(self):
        m=model();digest=h.tensor_hash(m)
        for data in [self.episodes[:9], [self.episodes[0]]*10]:
            with self.assertRaises(ValueError):h.calibrate(m,data)
        e=self.episodes[0];bad_rewards=e.rewards.clone();bad_rewards[0]=float('nan')
        bad=list(self.episodes);bad[0]=h.DemoEpisode(e.states,e.actions,bad_rewards,e.dones)
        with self.assertRaises(ValueError):h.calibrate(m,bad)
        actual=h.probe
        with patch.object(h,'probe',wraps=actual) as probe:
            def flip(*args,**kwargs):
                value=actual(*args,**kwargs)
                if kwargs.get('temperature', args[3] if len(args)>3 else 1.) != 1.:
                    value['actions'][0][0]=(value['actions'][0][0]+1)%5
                return value
            probe.side_effect=flip
            with self.assertRaisesRegex(ValueError,'changed greedy'):h.calibrate(m,self.episodes)
        self.assertEqual(h.tensor_hash(m),digest)

    def test_06_independent_bc_and_handoff_hash_parity(self):
        # Three independent one-step synthetic BC optimizers; no research training.
        results=[];initial=[]
        for arm in ['calibrated_retained_ddqn','calibrated_no_retention_ddqn','calibrated_bc_only']:
            m=model(arm);initial.append(h.tensor_hash(m))
            opt=torch.optim.Adam(m.parameters(),lr=.001,betas=(.9,.999),eps=1e-8)
            opt.zero_grad();h.bc_loss(m,self.episodes[0].states,self.episodes[0].actions).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),10.);opt.step()
            COUNTS['synthetic_bc_optimizer_steps']+=1
            results.append(h.handoff(m,self.episodes))
        self.assertEqual(len(set(initial)),1)
        self.assertEqual(len({h.tensor_hash(r.model) for r in results}),1)
        self.assertEqual(len({r.alpha for r in results}),1)
        target,opt=h.td_setup(results[0].model)
        self.assertFalse(opt.state)
        self.assertEqual(h.tensor_hash(target),h.tensor_hash(results[0].model))
        self.assertTrue(all(not p.requires_grad for p in target.parameters()))

    def test_07_all_arm_formulas_and_information_boundaries(self):
        hashes=[]
        for name,definition in h.ARMS.items():
            m=model(name);hashes.append(h.tensor_hash(m))
            x,a,_,_,_=batch()
            with torch.no_grad():
                p,motion,v,_=m.tables(x)
                cost=p.gather(2,a[:,:30,None]).sum((1,2))+motion.gather(2,a[:,30:,None]).sum((1,2))
                expected=-F.softplus(v+cost) if definition.softplus else v-cost+p.mean(2).sum(1)+motion.mean(2).sum(1)
                torch.testing.assert_close(m(x,a),expected,rtol=1e-5,atol=1e-6)
            if not definition.calibration:
                handoff=h.handoff(m,None if not definition.bc else self.episodes)
                self.assertEqual(h.tensor_hash(m),h.tensor_hash(handoff.model));self.assertEqual(handoff.alpha,1.)
            if not definition.bc:
                with self.assertRaises(ValueError):h.bc_loss(m,x,a)
                with self.assertRaises(ValueError):h.handoff(m,self.episodes)
            if not definition.ddqn:
                with self.assertRaises(ValueError):h.td_setup(m)
                with self.assertRaises(ValueError):h.draw_indices(m,20,np.random.default_rng(17))
        self.assertEqual(len(set(hashes)),1)

    def test_08_rng_streams_and_no_retention_draws(self):
        m=model();r=np.random.default_rng(17);keep=np.random.default_rng(23)
        same=np.random.default_rng(17)
        idx,aux=h.draw_indices(m,100,r,keep,40,batch=64)
        np.testing.assert_array_equal(idx,same.integers(0,100,64))
        self.assertEqual(len(aux),64)
        before=copy.deepcopy(r.bit_generator.state)
        with self.assertRaises(ValueError):h.draw_indices(m,100,r,r,40)
        self.assertEqual(before,r.bit_generator.state)
        no=model('calibrated_no_retention_ddqn')
        _,nothing=h.draw_indices(no,100,r)
        self.assertIsNone(nothing)
        with self.assertRaises(ValueError):h.draw_indices(no,100,r,keep,40)

    def test_09_combined_step_matches_independent_gradient_and_adam(self):
        result=h.calibrate(model(),self.episodes);m=result.model
        target,opt=h.td_setup(m);reference=copy.deepcopy(m)
        reference_target=copy.deepcopy(target)
        refopt=torch.optim.Adam(reference.parameters(),lr=.0001,betas=(.9,.999),eps=1e-8,weight_decay=0)
        data=batch();s,a,r,ns,done=data;demo=(self.episodes[0].states[:8],self.episodes[0].actions[:8])
        # Independent expression of the masked teacher loss and DDQN target.
        with torch.no_grad():
            y=r.clone();selected=reference.greedy(ns[~done])[0]
            y[~done]+=reference_target(ns[~done],selected)
        td=F.smooth_l1_loss(reference(s,a),y)
        p,motion,_,_=reference.tables(demo[0]);mask=torch.from_numpy(h.baseline.allowed(demo[0]))
        keep=(F.cross_entropy((-p/result.alpha).masked_fill(~mask[:,:30],-torch.inf).reshape(-1,5),demo[1][:,:30].reshape(-1))+
              F.cross_entropy((-motion/result.alpha).reshape(-1,5),demo[1][:,30:].reshape(-1)))
        refopt.zero_grad();(td+keep).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(),10.);refopt.step()
        COUNTS['independent_reference_optimizer_steps']+=1
        target_hash=h.tensor_hash(target)
        report=step(m,target,opt,data,result.alpha,demo)
        self.assertAlmostEqual(report['total_loss'],float((td+keep).detach()),places=5)
        for p,q in zip(m.parameters(),reference.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
        self.assertEqual(target_hash,h.tensor_hash(target));self.assertTrue(all(p.grad is None for p in target.parameters()))
        self.assertTrue(all(int(v['step'])==1 for v in opt.state.values()))
        self.assertTrue(np.isfinite(list(report['gradient_norms'].values())).all())

    def test_10_every_TD_arm_objective_and_zero_target_terminal(self):
        for name,d in h.ARMS.items():
            if not d.ddqn:continue
            m=model(name);target,opt=h.td_setup(m)
            demo=(self.episodes[0].states[:8],self.episodes[0].actions[:8]) if d.retention else None
            report=step(m,target,opt,batch(),alpha=.1 if d.calibration else 1.,demonstration=demo)
            self.assertEqual(report['optimizer_steps'],1)
            self.assertEqual(report['retention_loss']>0,d.retention)
            self.assertEqual(report['next_actions'][-1],[-1]*32)
            self.assertEqual(report['target_q'][-1],0.)
            self.assertEqual(report['target'][-1],report['rewards'][-1])

    def test_11_all_terminal_rows_bypass_next_decode(self):
        m=model('uncalibrated_no_retention_ddqn');target,opt=h.td_setup(m)
        original=m.greedy
        def no_terminal(x,*args,**kwargs):
            self.assertTrue((x[:,-1]<1).all());return original(x,*args,**kwargs)
        with patch.object(m,'greedy',side_effect=no_terminal),patch.object(target,'forward',side_effect=AssertionError('terminal evaluated')):
            report=step(m,target,opt,batch(all_terminal=True))
        self.assertEqual(report['target'],report['rewards'])
        self.assertTrue(all(a==[-1]*32 for a in report['next_actions']))

    def test_12_target_alias_optimizer_and_auxiliary_misuse_rejected(self):
        m=model();target,opt=h.td_setup(m);demo=(self.episodes[0].states[:8],self.episodes[0].actions[:8])
        digest=h.tensor_hash(m)
        with self.assertRaises(ValueError):h.ddqn_retention_step(m,m,opt,batch(),.1,demo)
        with self.assertRaises(ValueError):h.ddqn_retention_step(m,target,opt,batch(),.1,None)
        with self.assertRaises(ValueError):h.ddqn_retention_step(m,target,opt,batch(),0,demo)
        wrong=torch.optim.Adam(model().parameters(),lr=.0001)
        with self.assertRaises(ValueError):h.ddqn_retention_step(m,target,wrong,batch(),.1,demo)
        alias=copy.deepcopy(target);alias.student.net[2].weight.data=m.student.net[2].weight.data
        with self.assertRaisesRegex(ValueError,'aliases'):h.ddqn_retention_step(m,alias,opt,batch(),.1,demo)
        data=list(batch());data[-1]=~data[-1]
        with self.assertRaisesRegex(ValueError,'Terminal flags'):h.ddqn_retention_step(m,target,opt,tuple(data),.1,demo)
        self.assertEqual(digest,h.tensor_hash(m));self.assertFalse(opt.state)
        no=model('uncalibrated_no_retention_ddqn');t,o=h.td_setup(no)
        with self.assertRaises(ValueError):h.ddqn_retention_step(no,t,o,batch(),demonstration=demo)
        only=model('calibrated_bc_only')
        with self.assertRaises(ValueError):h.ddqn_retention_step(only,target,opt,batch())

    def test_13_first_update_identity_and_stage_records(self):
        result=h.calibrate(model(),self.episodes);m=result.model;target,opt=h.td_setup(m)
        s,a,r,ns,done=batch();demo=(self.episodes[0].states[:8],self.episodes[0].actions[:8])
        with torch.no_grad():
            pred=m(s,a);current=m.greedy(s)[1];nq=m.greedy(ns[~done])[1]
            delta=r[~done]+current[~done]-pred[~done]+nq-current[~done]
        report=step(m,target,opt,(s,a,r,ns,done),result.alpha,demo)
        np.testing.assert_allclose(delta.numpy(),(np.array(report['target'])-report['prediction'])[~done.numpy()],rtol=1e-5,atol=1e-5)
        self.assertEqual(set(result.report['stages']),{'after_bc','after_scale','after_value_fit'})
        self.assertEqual(set(result.report['residuals']),{'fitting','withheld_returns'})

    def test_14_deployed_storage_unchanged_and_no_extra_buffers(self):
        before=model();after=h.calibrate(before,self.episodes).model
        self.assertEqual(sum(p.numel() for p in after.parameters()),16841)
        self.assertEqual(sum(p.numel()*p.element_size() for p in after.parameters()),67364)
        self.assertEqual(sum(v.numel()*v.element_size() for v in after.buffers()),20)
        self.assertEqual(set(before.state_dict()),set(after.state_dict()))

    def test_15_research_CLI_rejects_without_output(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'forbidden'
            for mode in ['development','final']:
                for confirmation in [None,'LAUNCH_FROZEN_RETURN_CALIBRATED_HANDOFF']:
                    cmd=[sys.executable,str(Path(h.__file__)),'--mode',mode,'--output',str(out)]
                    if confirmation:cmd+=['--confirmation',confirmation]
                    p=subprocess.run(cmd,capture_output=True,text=True,timeout=35)
                    self.assertNotEqual(p.returncode,0);self.assertIn('Research modes disabled',p.stderr)
                    self.assertFalse(out.exists())


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Provide a new engineering-evidence directory')
    out=Path(sys.argv[1])
    out.mkdir(parents=True,exist_ok=False)
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Mechanics))
    log=stream.getvalue();print(log)
    (out/'test_log.txt').write_text(log,encoding='utf-8')
    record=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                kind='SYNTHETIC_MECHANICS_NOT_RESEARCH',**COUNTS,
                new_simulator_steps=0,new_teacher_calls=0,reserved_research_seeds_used=False,
                research_launch_authorized=False,word_thesis_changed=False)
    (out/'tests.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    if result.wasSuccessful():
        sample=h.calibrate(model(),episodes())
        (out/'synthetic_handoff.json').write_text(json.dumps(sample.report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
