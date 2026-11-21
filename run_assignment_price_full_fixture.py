"""Isolated engineering workflow only. Research/final modes are disabled."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
import torch
import assignment_price_prototype as adp
import assignment_price_native_heads_runtime as deployed
import assignment_price_all_reference_deployment as references
import assignment_price_load_radio_heuristic as load_radio
import run_horizon_loss_research as r

SPEC=r.ROOT/'ASSIGNMENT_PRICE_RESEARCH_PROTOCOL_2026-11-15.json'
SPEC_SHA='6213fe3e77946fec0b142f020577274084a795807993f140f39af0c7cbae6fef'
CONFIRM='LAUNCH_FROZEN_ADP_DEVELOPMENT'
RESEARCH_RELEASED=False
def protocol():
    if r.sha(SPEC)!=SPEC_SHA:raise ValueError('ADP protocol changed')
    p=r.read(SPEC)
    for name,h in p['definition_sha256'].items():
        if r.sha(r.ROOT/name)!=h:raise ValueError('frozen definition changed: '+name)
    return p

ARMS=[d['id'] for d in protocol()['learned_arms']]
HEURISTICS=protocol()['heuristics']
KIND='ENGINEERING_FIXTURE_NOT_RESEARCH_OR_POLICY_EVIDENCE'

def plan():
    cfg=copy.deepcopy(r.protocol()['training'])
    cfg.update(episodes_per_fit=2,slots_per_fit=80,warmup=8,batch_size=4,replay_capacity=32,
               target_interval_updates=8,epsilon_decay=80,optimizer_updates_per_fit=73,target_copies_per_fit=9)
    cfg['anchor']['batch']=4
    return dict(kind=KIND,arms=ARMS,heuristics=HEURISTICS,model_seed=867250001,
        train_seeds=[867251000,867251001],evaluation_seed=867252000,order_seed=867253000,
        cfg=cfg,training_slots=1200,evaluation_slots=760,optimizer_updates=1095,target_copies=135,
        final_checkpoints_only=True,quality_evidence=False,cost_acceptance=False,research_disabled=True)

def build(arm,pl):
    cfg=pl['cfg'];seed=pl['model_seed']
    if arm not in adp.ARMS:
        definition=next(d for d in r.protocol()['learned_arms'] if d['id']==arm)
        return r.build(definition,seed,cfg)
    online=adp.Critic(arm,seed);target=copy.deepcopy(online).eval().requires_grad_(False)
    return dict(online=online,target=target,seed=seed,cfg=cfg,slots=0,updates=0,copies=0,
        definition=dict(id=arm,screened=True),replay=r.models.original.existing.TerminalBuffer(cfg['replay_capacity']),
        opt=torch.optim.Adam(online.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay']),
        action_rng=np.random.default_rng(seed+100000),replay_rng=np.random.default_rng(seed+200000),
        anchor_rng=np.random.default_rng(seed+300000),actor_history=[])

def update(a):
    if a['definition']['id'] not in adp.ARMS:return r.update(a)
    cfg=a['cfg']
    if a['slots']<cfg['warmup']:return None
    begin=time.perf_counter_ns()
    raw=a['replay'].sample(cfg['batch_size'],a['replay_rng'])
    s,act,rewards,ns,dones=map(torch.tensor,raw[:5])
    anchor=a['replay'].terminal_sample(cfg['anchor']['batch'],a['anchor_rng'])
    na=np.full((len(dones),32),-1,np.int64);nv=np.zeros(len(dones),np.float64)
    with torch.no_grad():
        active=~dones.bool()
        if active.any():
            selected=a['online'].greedy(ns[active])[0]
            na[active.numpy()]=selected.numpy();nv[active.numpy()]=a['target'](ns[active],selected).numpy()
    loss,parts=adp.objective(a['online'],a['target'],(s,act,rewards,ns,dones),None if anchor is None else anchor[:3])
    q=a['online'](s,act);y=parts['targets']
    np.testing.assert_allclose(y.detach().numpy(),rewards.double().numpy()+nv,rtol=1e-12,atol=1e-12)
    aq=[] if anchor is None else a['online'](anchor[0],anchor[1]).detach().tolist()
    a['opt'].zero_grad(set_to_none=True);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(a['online'].parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
    assert all(p.grad is None and not p.requires_grad for p in a['target'].parameters())
    a['opt'].step();a['updates']+=1
    assert all(torch.isfinite(p).all() for p in a['online'].parameters())
    copied=a['updates']%cfg['target_interval_updates']==0
    if copied:a['target'].load_state_dict(a['online'].state_dict());a['copies']+=1
    return dict(step=a['updates'],physical_slots=a['slots'],target_copy=copied,loss=float(loss.detach()),
        uniform_loss=float(parts['uniform'].detach()),anchor_loss=float(parts['anchor'].detach()),
        anchor_Q=aq,anchor_rewards=[] if anchor is None else anchor[2].tolist(),empty_anchor=anchor is None,
        anchor_indices=[] if anchor is None else anchor[-1].tolist(),terminal_pool=sorted(a['replay'].terminal),
        replay_indices=raw[-1].tolist(),sample_rewards=rewards.tolist(),sample_dones=dones.tolist(),
        next_actions=na.tolist(),next_target_values=nv.tolist(),bellman_targets=y.detach().tolist(),
        chosen_Q=q.detach().tolist(),TD_residuals=(q.detach()-y).tolist(),
        grad_norm_before_clip=float(norm),clipped=bool(norm>cfg['gradient_clip']),optimizer_ns=time.perf_counter_ns()-begin)

def episode(a,seed,training=False,policy=None,stream=None):
    env=r.Chapter3IoVEnv(seed=seed);states=[];next_states=[];slots=[];qs=[];tqs=[]
    for t in range(40):
        state=env._observation().copy();states.append(state)
        tasks=r.models.original.existing.base.old.workload(env)
        before=env.uav_positions.copy();vehicle=env.vehicle_positions.copy()
        mask=adp.allowed(state[None])[0]
        if policy:
            action=load_radio.Runtime().values(state)[0][0] if policy=='observed_load_radio' else np.r_[r.select_destinations(env,policy),np.zeros(2,np.int64)]
        elif training:
            cfg=a['cfg'];warm=a['slots']<cfg['warmup'];epsilon=1. if warm else 1-.95*min(a['slots']/cfg['epsilon_decay'],1.)
            u=float(a['action_rng'].random());uniforms=a['action_rng'].random(32);exploring=bool(warm or u<epsilon)
            if exploring:action=adp.explore(state[None],uniforms[None],a['definition']['screened'])[0]
            else:
                with torch.inference_mode():action=a['online'].greedy(torch.from_numpy(state)[None])[0][0].numpy().copy()
            a['actor_history'].append(dict(warmup=warm,epsilon=epsilon,explore=exploring,epsilon_uniform=u,branch_uniforms=uniforms.tolist()))
        else:
            with torch.inference_mode():
                acts,q=a['online'].greedy(torch.from_numpy(state)[None]);tq=a['target'](torch.from_numpy(state)[None],acts)
            native,value=a['runtime'].values(state)
            np.testing.assert_array_equal(acts.numpy(),native);np.testing.assert_allclose(q.numpy(),value,rtol=1e-5,atol=1e-6)
            action=native[0].copy();qs.append(float(q[0]));tqs.append(float(tq[0]))
        assert action.shape==(32,) and np.all((action>=0)&(action<5))
        if a is not None and a['definition']['screened']:assert mask[np.arange(32),action].all()
        ns,reward,done,info=env.step(r.to_action_dict(action,30));next_states.append(ns.copy())
        assert done==(t==39) and np.isfinite(reward) and np.isfinite(ns).all()
        if training:
            a['replay'].add(state,action,reward,ns,done);a['slots']+=1;row=update(a)
            if row is not None:stream.write(json.dumps(row,allow_nan=False)+'\n')
        after=env.uav_positions.copy()
        slots.append(dict(slot=t,action=action.tolist(),reward=reward,done=done,reward_components=info['reward_components'],
            energy_j=info['energy_j'],tasks_arrived=tasks,vehicle_positions=vehicle.tolist(),uav_before=before.tolist(),
            uav_after=after.tolist(),actual_distance_m=np.linalg.norm(after-before,axis=1).tolist(),excluded_local=(~mask[:30,0]).tolist()))
    summary=env.episode_summary();rewards=[s['reward'] for s in slots]
    return dict(seed=seed,run_kind=KIND,training=training,policy=policy,return_sum=sum(rewards),
        deadline_success=summary['deadline_success_rate'],tasks_generated=summary['tasks_generated'],tasks_on_time=summary['tasks_on_time'],
        slot_rewards=rewards,summary=summary,slots=slots,tasks=env.export_task_records(),chosen_q=qs,target_q=tqs,
        realized_return_to_go=np.cumsum(rewards[::-1])[::-1].tolist(),workload_sha256=r.models.original.existing.base.old.digest([s['tasks_arrived'] for s in slots]),
        observations=np.stack(states),next_observations=np.stack(next_states))

def run(output,mode='fixture',confirmation=None):
    if mode=='development':
        if confirmation!=CONFIRM:raise RuntimeError('Explicit development confirmation required')
        if not RESEARCH_RELEASED:raise RuntimeError('Research disabled pending separate launch review')
    if mode!='fixture':raise RuntimeError('Final/research execution disabled; fixture only')
    protocol();pl=plan();r.configure();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    r.write(out/'plan.json',pl)
    files=set(r.source_files(r.protocol()))|{Path(__file__).name,'assignment_price_prototype.py','assignment_price_native_heads_runtime.py',
        'assignment_price_deployment_runtime.py','assignment_price_fast_runtime.py','assignment_price_reference_deployment.py',
        'audit_assignment_price_full_fixture.py','test_assignment_price_full_engineering.py','assignment_price_all_reference_deployment.py','assignment_price_load_radio_heuristic.py',SPEC.name,'assignment_price_full_fixture_analysis.py'}
    for name in files:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(r.ROOT/name,dest)
    r.write(out/'PRE_RUN_FREEZE.json',dict(kind=KIND,plan_sha256=r.sha(out/'plan.json'),sources={n:r.sha(r.ROOT/n) for n in sorted(files)}))
    order=list(ARMS);np.random.default_rng(pl['order_seed']).shuffle(order);r.write(out/'arm_order.json',order)
    frozen={};counts={};reports={}
    for arm in order:
        r.write(out/'progress.json',dict(stage='fixture_training',arm=arm,kind=KIND))
        folder=out/arm;folder.mkdir();a=build(arm,pl)
        with (folder/'optimizer.jsonl').open('w',encoding='utf-8') as stream:
            for i,seed in enumerate(pl['train_seeds']):r.save_episode(folder/f'train_{i}.json.gz',episode(a,seed,True,stream=stream))
        checkpoint=dict(kind=KIND,arm=arm,model_seed=pl['model_seed'],plan_sha256=r.sha(out/'plan.json'),
            online=copy.deepcopy(a['online'].state_dict()),target=copy.deepcopy(a['target'].state_dict()),optimizer=a['opt'].state_dict(),
            slots=a['slots'],updates=a['updates'],copies=a['copies'],actor_history=a['actor_history'])
        torch.save(checkpoint,folder/'final.pt');frozen[arm]=r.sha(folder/'final.pt')
        counts[arm]={key:a[key] for key in ('slots','updates','copies')}
        reloaded=build(arm,pl);reloaded['online'].load_state_dict(checkpoint['online']);reloaded['target'].load_state_dict(checkpoint['target'])
        reloaded['opt'].load_state_dict(checkpoint['optimizer'])
        np.testing.assert_equal(r.models.original.existing.base.old.digest(reloaded['online'].state_dict()),r.models.original.existing.base.old.digest(a['online'].state_dict()))
        a=reloaded;a['online'].eval();a['target'].eval()
        a['runtime']=deployed.Runtime(a['online'],folder/'final.onnx') if arm in adp.ARMS else references.Runtime(a['online'],folder/'final.onnx')
        before=r.models.original.existing.base.old.digest(a['online'].state_dict())
        r.write(folder/'export_warnings.json',a['runtime'].warnings)
        # Evaluation is deferred until all final checkpoint artifacts are frozen.
        reports[arm]=a
    r.write(out/'FINAL_CHECKPOINT_FREEZE.json',frozen)
    for arm in order:
        a=reports[arm];before=r.models.original.existing.base.old.digest(a['online'].state_dict())
        row=episode(a,pl['evaluation_seed']);r.save_episode(out/arm/'evaluation.json.gz',row)
        assert before==r.models.original.existing.base.old.digest(a['online'].state_dict())
        assert r.sha(out/arm/'final.pt')==frozen[arm]
    for policy in HEURISTICS:r.save_episode(out/f'heuristic_{policy}.json.gz',episode(None,pl['evaluation_seed'],policy=policy))
    r.write(out/'counters.json',counts)
    r.write(out/'verification.json',dict(kind=KIND,quality_evidence=False,research_launched=False,
        training_slots=sum(v['slots'] for v in counts.values()),evaluation_slots=760,
        updates=sum(v['updates'] for v in counts.values()),copies=sum(v['copies'] for v in counts.values()),
        deployment_equivalence_states=600,final_checkpoint_only=True,evaluation_no_updates=True))
    import assignment_price_full_fixture_analysis as reporting
    reporting.report(out,protocol())
    r.write(out/'manifest.json',{p.relative_to(out).as_posix():r.sha(p) for p in out.rglob('*') if p.is_file()})
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--mode',default='fixture');parser.add_argument('--confirm-development')
    args=parser.parse_args();run(args.output,args.mode,args.confirm_development)
