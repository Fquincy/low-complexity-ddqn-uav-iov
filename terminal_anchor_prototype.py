"""Isolated finite-horizon/terminal-anchor prototype; no research launcher."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import run_v3_research_gate_v2 as existing

ARMS={
 'price_both':('amortized_workload_price24',True,True),
 'no_price_both':('amortized_no_price24',True,True),
 'price_horizon_only':('amortized_workload_price24',True,False),
 'price_anchor_only':('amortized_workload_price24',False,True),
 'plain_both':('plain24_nonpositive',True,True),
 'compact_both':('additive64_nonpositive',True,True)}

def remaining(states):
    time=states[:,-1]*40
    if not torch.isfinite(states).all() or torch.any(time<-.0001) or torch.any(time>40.0001):
        raise ValueError('invalid time/state')
    rounded=time.round()
    if torch.any((time-rounded).abs()>.0001):raise ValueError('time is not on slot grid')
    return 40-rounded

class Critic(nn.Module):
    def __init__(self,arm,seed=99461):
        super().__init__()
        self.arm=arm;self.base_arm,self.scaled,self.anchored=ARMS[arm]
        self.model=existing.make_model(self.base_arm,seed)
    def forward(self,states,actions):
        h=remaining(states);q=states.new_zeros(len(states));active=h>0
        if active.any():
            values=existing.chosen(self.model,states[active],actions[active],self.base_arm)
            q[active]=values*(h[active] if self.scaled else 1.)
        return q
    def greedy(self,states):
        h=remaining(states)
        if torch.any(h==0):raise ValueError('do not decode terminal states')
        actions,q,_=existing.greedy(self.model,states,self.base_arm)
        return actions,q*(h if self.scaled else 1.)

@torch.no_grad()
def targets(online,target,rewards,next_states,dones):
    if not torch.all((dones==0)|(dones==1)):raise ValueError('binary mask')
    h=remaining(next_states)
    if not torch.equal(dones.bool(),h==0):raise ValueError('done/time mismatch')
    y=rewards.detach().clone();active=~dones.bool()
    if active.any():
        actions=online.greedy(next_states[active])[0]
        y[active]+=target(next_states[active],actions)
    return y

class Replay:
    def __init__(self,capacity):
        if capacity<1:raise ValueError('capacity')
        self.capacity=capacity;self.rows=[];self.position=0;self.terminal=set()
    def add(self,row):
        index=self.position;self.terminal.discard(index)
        if len(self.rows)<self.capacity:self.rows.append(row)
        else:self.rows[index]=row
        if row[4]:self.terminal.add(index)
        self.position=(index+1)%self.capacity
    def sample(self,count,rng,terminal=False):
        choices=sorted(self.terminal) if terminal else list(range(len(self.rows)))
        if not choices:return None
        indices=rng.choice(choices,size=count,replace=True)
        tensors=[torch.tensor(np.stack([self.rows[i][j] for i in indices])) for j in range(5)]
        return (*tensors,indices)

def objective(online,target,batch,anchor):
    s,a,r,ns,d,_=batch
    y=targets(online,target,r,ns,d);q=online(s,a)
    uniform=F.smooth_l1_loss(q,y,beta=1.)
    terminal=uniform*0
    if online.anchored and anchor is not None:
        ts,ta,tr,tn,td,_=anchor
        if not td.bool().all() or not torch.all(remaining(ts)==1):raise ValueError('invalid terminal anchor')
        terminal=F.smooth_l1_loss(online(ts,ta),tr.detach(),beta=1.)
    return uniform+terminal,dict(uniform=float(uniform.detach()),anchor=float(terminal.detach()),
        empty_anchor=bool(online.anchored and anchor is None),target=y.detach().tolist())
