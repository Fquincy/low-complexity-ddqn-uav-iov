"""Algebra, gradient, actual-update and artifact checks on synthetic transitions."""
import copy,json,sys,unittest
from pathlib import Path
import numpy as np
import torch
import horizon_normalized_loss as n

r=n.base
OUT=r.ROOT/'validation/horizon_normalized_loss_engineering_2026-11-14_01'

def states():
    p=r.protocol();x=np.load(r.ROOT/p['action_cost']['common_inputs'])['observations'][:4].copy()
    x[:,-1]=np.array([0.,.5,.95,.975],np.float32)
    nx=x.copy();nx[:,-1]=np.array([.025,.525,.975,1.],np.float32)
    return x,nx

def scalar_huber(delta):
    x=np.abs(np.asarray(delta,np.float64));return float(np.where(x<1,.5*x*x,x-.5).mean())

def independent_anchor_stats(model,anchor):
    if anchor is None:return 0.
    s,a,rw,_,_,_=anchor
    with torch.no_grad():q=model(s,a).numpy()
    return scalar_huber(q-rw.numpy())

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):r.configure()

    def test_01_scalar_values_gradients_and_division_location(self):
        x,_=states();s=torch.from_numpy(x);h=torch.tensor([40.,20.,2.,1.],dtype=torch.float64)
        # Explicit chain rule for Q=-h*c. Include quadratic and linear Huber regimes.
        c=torch.tensor([.2,.7,2.,.4],dtype=torch.float64,requires_grad=True)
        q=-h*c;y=torch.tensor([-16.,-2.,-12.,-1.],dtype=torch.float64,requires_grad=True)
        loss=n.uniform_loss(q,y,s);residual=(q.detach()-y.detach())/h
        self.assertAlmostEqual(float(loss.detach()),scalar_huber(residual.numpy()),places=12)
        loss.backward();expected=-torch.clamp(residual,-1,1)/4
        torch.testing.assert_close(c.grad,expected,rtol=1e-12,atol=1e-12);self.assertIsNone(y.grad)
        wrong=torch.nn.functional.smooth_l1_loss(q.detach(),y.detach(),reduction='none')/h
        self.assertFalse(np.isclose(float(wrong.mean()),float(loss.detach())))
        terminal=s.clone();terminal[0,-1]=1.
        with self.assertRaises(ValueError):n.uniform_loss(q,y,terminal)
        bad=s.clone();bad[0,-1]=.013
        with self.assertRaises(ValueError):n.uniform_loss(q,y,bad)

    def test_02_targets_terminal_anchor_and_full_gradient_oracle(self):
        x,nx=states();s=torch.from_numpy(x);ns=torch.from_numpy(nx);a=torch.ones((4,32),dtype=torch.long)
        rewards=torch.tensor([-1.,-2.,-3.,-4.]);dones=torch.tensor([0.,0.,0.,1.]);h=torch.tensor([40.,20.,2.,1.])
        batch=(s,a,rewards,ns,dones,np.arange(4))
        for definition in n.definitions()[-3:]:
            cfg=r.plan(r.protocol(),'integration-fixture')['cfg'];agent=r.build(definition,99921,cfg)
            oracle=copy.deepcopy(agent['online']);target=agent['target']
            with torch.no_grad():
                for i,p in enumerate(target.parameters()):p.add_(.002*(i+1))
            def forbidden(*args):raise AssertionError('target must evaluate, never select')
            target.greedy=forbidden
            anchor=(s[3:].repeat(32,1),a[3:].repeat(32,1),rewards[3:].repeat(32),ns[3:].repeat(32,1),dones[3:].repeat(32),np.array([3]*32))
            with torch.no_grad():
                na=oracle.greedy(ns[:3])[0];y=rewards.clone();y[:3]+=target(ns[:3],na)
            predicted=oracle(s,a);normalized=(predicted-y)/h
            # Separate explicit Huber expression, not implementation helper.
            absolute=normalized.abs();uniform=torch.where(absolute<1,.5*normalized.square(),absolute-.5).mean()
            terminal_residual=oracle(anchor[0],anchor[1])-anchor[2];abs_terminal=terminal_residual.abs()
            terminal=torch.where(abs_terminal<1,.5*terminal_residual.square(),abs_terminal-.5).mean()
            expected=uniform+terminal;expected.backward()
            loss,detail=n.objective(agent['online'],target,batch,anchor);loss.backward()
            self.assertAlmostEqual(float(loss.detach()),float(expected.detach()),places=5)
            np.testing.assert_allclose(detail['bellman_targets'],y.numpy(),rtol=1e-6,atol=1e-6)
            self.assertEqual(detail['current_h'],[40.,20.,2.,1.]);self.assertEqual(detail['bellman_targets'][-1],-4.)
            self.assertAlmostEqual(detail['anchor_loss'],independent_anchor_stats(agent['online'],anchor),places=5)
            for left,right in zip(agent['online'].parameters(),oracle.parameters()):torch.testing.assert_close(left.grad,right.grad,rtol=2e-5,atol=2e-6)
            self.assertTrue(all(p.grad is None for p in target.parameters()))
            _,empty=n.objective(agent['online'],target,batch,None);self.assertEqual(empty['anchor_loss'],0.)
            incorrect=ns.clone();incorrect[0,-1]=.05
            with self.assertRaises(ValueError):n.objective(agent['online'],target,(s,a,rewards,incorrect,dones,np.arange(4)))

    def test_03_control_updates_exact_and_paired_normalized_fixture(self):
        x,nx=states();cfg=copy.deepcopy(r.plan(r.protocol(),'integration-fixture')['cfg'])
        cfg.update(warmup=4,replay_capacity=4,target_interval_updates=4)
        freeze={};counts={};outputs={};actor_sources={};sampling={};updates=0
        for seed in (99921,99929):
            for definition in n.definitions():
                arm=definition['id'];folder=OUT/arm/f'seed_{seed}';folder.mkdir(parents=True)
                agent=r.build(definition,seed,cfg);logs=[];transitions=[]
                for i in range(8):
                    index=i%4;u=float(agent['action_rng'].random());uniforms=agent['action_rng'].random(32)
                    # Engineering-only exploration exercises identical source draws/remapping.
                    action=r.models.original.explore(x[index:index+1],uniforms[None],definition['screened'])[0]
                    agent['actor_history'].append(dict(epsilon_uniform=u,branch_uniforms=uniforms.tolist()))
                    transition=(x[index],action,float(-index-1),nx[index],index==3)
                    agent['replay'].add(*transition);agent['slots']+=1
                    if agent['slots']<4:self.assertIsNone(n.update(agent));continue
                    # Clone BEFORE every update to verify exact unchanged controls and normalized step.
                    reference=copy.deepcopy(agent)
                    expected_log=r.update(reference) if not definition.get('normalized_horizon_loss') else None
                    if definition.get('normalized_horizon_loss'):
                        raw=reference['replay'].sample(cfg['batch_size'],reference['replay_rng'])
                        bt=(*map(torch.tensor,raw[:5]),raw[-1]);an=reference['replay'].terminal_sample(32,reference['anchor_rng'])
                        ss,aa,rr,nn,dd,_=bt;hh=r.models.original.remaining(ss)
                        with torch.no_grad():
                            yy=rr.clone();active=~dd.bool()
                            if active.any():
                                sel=reference['online'].greedy(nn[active])[0];yy[active]+=reference['target'](nn[active],sel)
                        prediction=reference['online'](ss,aa);delta=(prediction-yy)/hh
                        absolute=delta.abs();uniform=torch.where(absolute<1,.5*delta**2,absolute-.5).mean()
                        ar=reference['online'](an[0],an[1])-an[2];ab=ar.abs();anchor_loss=torch.where(ab<1,.5*ar**2,ab-.5).mean()
                        expected_loss=uniform+anchor_loss;reference['opt'].zero_grad(set_to_none=True);expected_loss.backward()
                        torch.nn.utils.clip_grad_norm_(reference['online'].parameters(),10,error_if_nonfinite=True);reference['opt'].step()
                        reference['updates']+=1
                        if reference['updates']%4==0:reference['target'].load_state_dict(reference['online'].state_dict());reference['copies']+=1
                    log=n.update(agent);logs.append(log);updates+=1
                    for k,v in agent['online'].state_dict().items():torch.testing.assert_close(v,reference['online'].state_dict()[k],rtol=1e-5,atol=1e-6)
                    if expected_log is not None:
                        for key in expected_log:
                            if key!='optimizer_ns':self.assertEqual(log[key],expected_log[key],(arm,key))
                    else:
                        self.assertAlmostEqual(log['loss'],float(expected_loss.detach()),places=5)
                        self.assertEqual(log['current_h'],hh.tolist())
                    self.assertEqual(agent['replay_rng'].bit_generator.state,reference['replay_rng'].bit_generator.state)
                    self.assertEqual(agent['anchor_rng'].bit_generator.state,reference['anchor_rng'].bit_generator.state)
                self.assertEqual((agent['slots'],agent['updates'],agent['copies']),(8,5,1))
                payload=dict(definition=definition,seed=seed,kind='SYNTHETIC_TRANSITIONS_NOT_RESEARCH',slots=8,updates=5,copies=1,
                    online=agent['online'].state_dict(),target=agent['target'].state_dict(),optimizer=agent['opt'].state_dict())
                torch.save(payload,folder/'final.pt');r.write(folder/'updates.json',logs)
                key=f'{arm}/{seed}';freeze[key]=r.sha(folder/'final.pt');counts[key]=dict(slots=8,updates=5,copies=1)
                actor_sources[key]=agent['actor_history'];sampling[key]=[v['replay_indices'] for v in logs]
        self.assertEqual(updates,140)
        r.write(OUT/'ALL_FINAL_CHECKPOINTS_FREEZE.json',freeze);r.write(OUT/'fixture_counts.json',counts)
        for seed in (99921,99929):
            keys=[f"{d['id']}/{seed}" for d in n.definitions()]
            self.assertTrue(all(actor_sources[k]==actor_sources[keys[0]] for k in keys))
            self.assertTrue(all(sampling[k]==sampling[keys[0]] for k in keys))
        r.write(OUT/'fixture_pairing.json',dict(action_source_uniforms_equal=True,replay_indices_equal=True,models=28,optimizer_updates=140,synthetic_transition_insertions=224,simulator_steps=0))

    def test_04_final_checkpoints_and_runtime_equivalence(self):
        self.assertTrue((OUT/'ALL_FINAL_CHECKPOINTS_FREEZE.json').exists())
        freeze=r.read(OUT/'ALL_FINAL_CHECKPOINTS_FREEZE.json');p=r.protocol();common=np.load(r.ROOT/p['action_cost']['common_inputs'])['observations'];checked=0
        for definition in n.definitions():
            for seed in (99921,99929):
                folder=OUT/definition['id']/f'seed_{seed}';path=folder/'final.pt'
                self.assertEqual(r.sha(path),freeze[f"{definition['id']}/{seed}"])
                ck=torch.load(path,weights_only=False,map_location='cpu');model=r.models.Critic(definition,seed).eval();model.load_state_dict(ck['online']);runtime=r.models.Runtime(model)
                self.assertEqual(sum(p.numel()*p.element_size() for p in model.parameters()),definition['expected_fp32_parameter_bytes'])
                for state in common:
                    with torch.inference_mode():a,q=model.greedy(torch.from_numpy(state)[None])
                    b,v=runtime.values(state);np.testing.assert_array_equal(a.numpy(),b);np.testing.assert_allclose(q.numpy(),v,rtol=1e-5,atol=1e-6);checked+=1
        r.write(OUT/'runtime_verification.json',dict(passed=True,models=28,action_checks=checked,parameters_unchanged=True,no_retiming=True))

    def test_05_saved_normalized_loss_independent_audit(self):
        rows=0
        for definition in n.definitions()[-3:]:
            for seed in (99921,99929):
                logs=r.read(OUT/definition['id']/f'seed_{seed}'/'updates.json')
                for entry in logs:
                    h=np.array(entry['current_h']);q=np.array(entry['chosen_Q']);y=np.array(entry['bellman_targets'])
                    np.testing.assert_allclose(entry['TD_residuals'],q-y,rtol=1e-5,atol=1e-6)
                    np.testing.assert_allclose(entry['normalized_TD_residuals'],(q-y)/h,rtol=1e-5,atol=1e-6)
                    self.assertAlmostEqual(entry['uniform_loss'],scalar_huber((q-y)/h),places=5)
                    anchor=scalar_huber(np.array(entry['anchor_Q'])-entry['anchor_rewards'])
                    self.assertAlmostEqual(entry['anchor_loss'],anchor,places=5)
                    self.assertAlmostEqual(entry['loss'],entry['uniform_loss']+entry['anchor_loss'],places=5)
                    self.assertEqual(entry['target_copy'],entry['step']%4==0);rows+=1
        self.assertEqual(rows,30);r.write(OUT/'loss_artifact_audit.json',dict(passed=True,normalized_update_rows=rows,original_unit_residuals_retained=True,independent_Huber_recomputation=True))

    def test_06_research_guards(self):
        for mode in ('development','research','final'):
            with self.assertRaises(RuntimeError):n.authorize(mode)
        n.authorize('engineering-fixture')

if __name__=='__main__':
    n.authorize(sys.argv[1] if len(sys.argv)>1 else 'engineering-fixture')
    OUT.mkdir(parents=True,exist_ok=False)
    sources=['horizon_normalized_loss.py','test_horizon_normalized_loss.py',n.AMENDMENT.name,'DEADLINE_SCREEN_HORIZON_LOSS_PROSPECTIVE_BRIEF_2026-11-14.txt']
    import shutil
    for name in sources:shutil.copy2(r.ROOT/name,OUT/name)
    r.write(OUT/'PRE_FIXTURE_FREEZE.json',dict(kind='SYNTHETIC_TRANSITIONS_NOT_RESEARCH',sources={name:r.sha(r.ROOT/name) for name in sources},
        base_protocol_sha256=r.sha(r.SPEC),model_seeds=[99921,99929],task_seeds=[],simulator_execution=False,optimizer_updates_planned=140))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    r.write(OUT/'tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        research_launched=False,quality_evidence=False,new_simulator_steps=0,research_seeds_reserved=False))
    r.write(OUT/'manifest.json',{p.relative_to(OUT).as_posix():r.sha(p) for p in OUT.rglob('*') if p.is_file()})
    raise SystemExit(0 if result.wasSuccessful() else 1)
