"""Prospective normalized loss adapter. No research execution entry point."""
import copy,time
import numpy as np
import torch
from torch.nn import functional as F
import run_deadline_screen_research_v2 as base

AMENDMENT=base.ROOT/'DEADLINE_SCREEN_HORIZON_LOSS_COMPARISON_AMENDMENT_2026-11-14.json'

def definitions():
    p=base.protocol();amend=base.read(AMENDMENT)
    base.require(base.sha(base.SPEC)==amend['base_protocol_sha256'],'base protocol drift')
    result=copy.deepcopy(p['learned_arms'])
    for item in amend['new_arms']:
        d=copy.deepcopy(item['base_definition']);d['id']=item['id'];d['normalized_horizon_loss']=True
        base.require(d['horizon_scaling'] and d['terminal_anchor'] and d['screened'],'unsupported normalization')
        result.append(d)
    return result

def uniform_loss(q,y,current_states):
    h=base.models.original.remaining(current_states)
    base.require(q.shape==y.shape==h.shape,'loss shape mismatch')
    base.require(bool(torch.all(h>=1)),'executed-state horizon must be positive')
    base.require(bool(torch.isfinite(q).all() and torch.isfinite(y).all()),'nonfinite normalized operands')
    return F.smooth_l1_loss(q/h,y.detach()/h,beta=1.)

def objective(online,target,batch,anchor=None):
    states,actions,rewards,next_states,dones,_=batch
    h=base.models.original.remaining(states);nh=base.models.original.remaining(next_states)
    base.require(bool(torch.all(h>=1) and torch.all(nh==h-1)),'transition remaining horizon mismatch')
    base.require(bool(torch.equal(dones.bool(),nh==0)),'terminal mask/time mismatch')
    y=base.models.original.targets(online,target,rewards,next_states,dones)
    # Extra detached audit evidence; selection/evaluation arithmetic is unchanged.
    na=np.full((len(dones),32),-1,np.int64);nv=np.zeros(len(dones),np.float32)
    with torch.no_grad():
        active=~dones.bool()
        if active.any():
            selected=online.greedy(next_states[active])[0]
            na[active.numpy()]=selected.numpy();nv[active.numpy()]=target(next_states[active],selected).numpy()
    q=online(states,actions);uniform=uniform_loss(q,y,states)
    terminal=uniform*0;aq=[];ar=[]
    if anchor is not None:
        ts,ta,tr,tn,td,_=anchor
        base.require(bool(td.bool().all() and torch.all(base.models.original.remaining(ts)==1)),'invalid terminal anchor')
        v=online(ts,ta);terminal=F.smooth_l1_loss(v,tr.detach(),beta=1.)
        aq=v.detach().tolist();ar=tr.tolist()
    return uniform+terminal,dict(uniform_loss=float(uniform.detach()),anchor_loss=float(terminal.detach()),
        chosen_Q=q.detach().tolist(),bellman_targets=y.detach().tolist(),current_h=h.tolist(),
        TD_residuals=(q.detach()-y).tolist(),normalized_TD_residuals=((q.detach()-y)/h).tolist(),anchor_Q=aq,anchor_rewards=ar,sample_rewards=rewards.tolist(),sample_dones=dones.tolist(),next_actions=na.tolist(),next_target_values=nv.tolist(),branch_values=[])

def update(agent):
    if not agent['definition'].get('normalized_horizon_loss',False):
        return base.update(agent)  # Original controls, including BDQ, retain their exact update.
    cfg=agent['cfg']
    if agent['slots']<cfg['warmup']:return None
    base.require(agent['online'].scaled and agent['online'].anchored,'normalization only declared for anchored scaled arms')
    base.require(cfg['gamma']==cfg['anchor']['weight']==cfg['huber_beta']==1.,'undeclared objective settings')
    start=time.perf_counter_ns();raw=agent['replay'].sample(cfg['batch_size'],agent['replay_rng'])
    batch=(*map(torch.tensor,raw[:5]),raw[-1]);anchor=agent['replay'].terminal_sample(cfg['anchor']['batch'],agent['anchor_rng'])
    loss,details=objective(agent['online'],agent['target'],batch,anchor)
    agent['opt'].zero_grad(set_to_none=True);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(agent['online'].parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
    base.require(all(p.grad is None and not p.requires_grad for p in agent['target'].parameters()),'target gradient')
    agent['opt'].step();agent['updates']+=1
    base.require(all(torch.isfinite(p).all() for p in agent['online'].parameters()),'nonfinite weights')
    copied=agent['updates']%cfg['target_interval_updates']==0
    if copied:agent['target'].load_state_dict(agent['online'].state_dict());agent['copies']+=1
    return dict(**details,loss=float(loss.detach()),step=agent['updates'],physical_slots=agent['slots'],
        grad_norm_before_clip=float(norm),clipped=bool(norm>cfg['gradient_clip']),target_copy=copied,
        replay_indices=raw[-1].tolist(),terminal_pool=sorted(agent['replay'].terminal),
        anchor_indices=[] if anchor is None else anchor[-1].tolist(),empty_anchor=anchor is None,
        optimizer_ns=time.perf_counter_ns()-start)

def authorize(mode):
    if mode!='engineering-fixture':raise RuntimeError('Research/development/final execution disabled: no new research protocol or authorization')
