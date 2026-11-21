"""Observed-state necessary local-feasibility screen. No research launcher."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import run_anchor_development_v4 as existing
from terminal_anchor_prototype import remaining,targets,objective
from controller_v3_amortized import NEW_ARMS
from controller_v3_fused_runtime import native

SUPPORTED=('price_both','no_price_both','plain_both','compact_both',
           'plain24_nonpositive','additive128_unbounded','bdq64_uniform_adapted')

def intervals(features,scale):
    """Enclose source FP64 values normalized then rounded into FP32 features."""
    z=np.asarray(features)
    if z.dtype!=np.float32 or not np.isfinite(z).all() or np.any(z<0) or np.any(z>1):
        raise ValueError('normalized finite FP32 values in [0,1] required')
    lower=np.maximum(0.,np.nextafter(z,np.float32(-np.inf)).astype(np.float64))
    upper=np.minimum(1.,np.nextafter(z,np.float32(np.inf)).astype(np.float64))
    return np.nextafter(lower*scale,-np.inf),np.nextafter(upper*scale,np.inf)

def allowed(states):
    if isinstance(states,torch.Tensor):
        if states.device.type!='cpu' or states.dtype!=torch.float32:raise ValueError('CPU FP32 states required')
        x=states.detach().numpy()
    else:x=np.asarray(states)
    if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=533 or not np.isfinite(x).all():
        raise ValueError('finite batch of533 FP32 observations required')
    tasks=x[:,:240].reshape(-1,30,8)
    c_lo,_=intervals(tasks[:,:,6],500e6)
    _,d_hi=intervals(tasks[:,:,7],.300)
    lower_service=np.nextafter(c_lo/1.5e9,-np.inf)
    upper_deadline=np.nextafter(d_hi+1e-10,np.inf)
    mask=np.ones((len(x),32,5),dtype=bool)
    mask[:,:30,0]=~(lower_service>upper_deadline)
    return mask

@native
def native_select(scores,mask):
    actions=np.empty((scores.shape[0],32),np.int64)
    for batch in range(scores.shape[0]):
        for branch in range(32):
            best=-1;value=np.float32(0.)
            for d in range(5):
                if mask[batch,branch,d] and (best<0 or scores[batch,branch,d]<value):
                    best=d;value=scores[batch,branch,d]
            if best<0:raise ValueError('empty allowed branch')
            actions[batch,branch]=best
    return actions

class Critic(nn.Module):
    def __init__(self,arm,seed=99731,screened=True):
        super().__init__()
        if arm not in SUPPORTED:raise ValueError('unsupported screened arm')
        self.arm=arm;self.screened=screened
        self.model=existing.make_model(arm,seed)
        self.scaled=arm in existing.CORRECTED and existing.anchor.ARMS[arm][1]
        self.anchored=arm in existing.CORRECTED and existing.anchor.ARMS[arm][2]
        self.base_arm=existing.anchor.ARMS[arm][0] if arm in existing.CORRECTED else arm

    def forward(self,states,actions):
        # Off-policy action values remain defined, even for a newly screened action.
        return existing.chosen(self.model,states,actions,self.arm)

    def tables(self,states):
        core=self.model.model if self.arm in existing.CORRECTED else self.model
        if self.base_arm in NEW_ARMS:
            placement,motion,bias,_=core.tables(states)
            return torch.cat((placement,motion),1),('price',placement,motion,bias,core)
        branch=core(states)
        # Compact additive output is interpreted as a cost before outer softplus.
        if self.base_arm=='additive64_nonpositive':return branch,('compact',branch)
        return -branch,('wide_or_bdq',branch)

    def value(self,states,actions,context):
        if context[0]=='price':
            _,p,m,b,core=context;q=core.value_from_tables(p,m,b,actions)
        else:
            branch=context[1];selected=branch.gather(2,actions[:,:,None]).squeeze(2)
            if context[0]=='compact':q=-F.softplus(selected.sum(1))
            elif self.base_arm=='bdq64_uniform_adapted':q=selected.mean(1)
            else:q=selected.sum(1)
        return q*(remaining(states) if self.scaled else 1.)

    def greedy(self,states):
        if torch.any(remaining(states)==0):raise ValueError('do not decode terminal states')
        mask=torch.from_numpy(allowed(states)) if self.screened else torch.ones((len(states),32,5),dtype=torch.bool)
        scores,context=self.tables(states)
        actions=scores.masked_fill(~mask,float('inf')).argmin(2)
        return actions,self.value(states,actions,context)

    @torch.no_grad()
    def runtime(self,state):
        x=np.asarray(state)
        mask=allowed(x[None]) if self.screened else np.ones((1,32,5),bool)
        tensor=torch.from_numpy(x)[None]
        if torch.any(remaining(tensor)==0):raise ValueError('terminal decoding forbidden')
        scores,context=self.tables(tensor)
        actions=native_select(scores.numpy(),mask)
        q=self.value(tensor,torch.from_numpy(actions),context)
        return actions,q.numpy()

def explore(states,uniforms,screened=True):
    mask=allowed(states)
    tensor=states if isinstance(states,torch.Tensor) else torch.from_numpy(states)
    if torch.any(remaining(tensor)==0):raise ValueError('terminal exploration forbidden')
    if not screened:mask[:]=True
    u=np.asarray(uniforms)
    if u.shape!=mask.shape[:2] or not np.isfinite(u).all() or np.any(u<0) or np.any(u>=1):
        raise ValueError('one uniform in [0,1) per branch required')
    counts=mask.sum(2);rank=(u*counts).astype(np.int64)
    # Ordered allowed choices, same supplied random uniform for paired arms.
    order=np.where(mask,np.arange(5),5);order.sort(axis=2)
    return np.take_along_axis(order,rank[:,:,None],axis=2)[:,:,0]
