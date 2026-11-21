"""Equivalent batch-one ADP inference; training critic remains unchanged."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from assignment_price_prototype import decode
from controller_v3_fused_runtime import native


class Heads(torch.nn.Module):
    def __init__(self,critic):
        super().__init__()
        self.core=copy.deepcopy(critic.core).eval()
        self.fixed=critic.arm=='fixed_price_plain_sweep'

    def forward(self,states):
        raw=self.core.net(states)
        positive=F.softplus(raw[:,:161 if self.fixed else 165])
        g=positive[:,:150].reshape(-1,30,5)/30
        v=positive[:,150:160].reshape(-1,2,5)/2
        b=positive[:,160]
        if self.fixed:
            beta=states.new_ones((1,5));beta[:,0]=0
            eta=states.new_full((1,5),.2)
        else:
            beta=F.pad(positive[:,161:165],(1,0))
            eta=F.softmax(raw[:,165:170],dim=1)
        w=states[:,:240].reshape(-1,30,8)[:,:,6,None]*self.core.cycle_to_work
        q=F.pad(torch.expm1(states[:,270:274]),(1,0))
        return g.double(),v.double(),b.double(),w.double(),q.double(),beta.double(),eta.double()


@native
def validate_mask(x):
    for z in x:
        if not np.isfinite(z):raise ValueError('finite state required')
    time=np.float32(x[-1]*np.float32(40.))
    if time<-.0001 or time>40.0001:raise ValueError('invalid time')
    rounded=np.rint(time)
    if abs(np.float32(time-rounded))>.0001:raise ValueError('off-grid time')
    h=40.-rounded
    if h<=0:raise ValueError('terminal decoding forbidden')
    mask=np.ones((30,5),np.bool_)
    for i in range(30):
        cycles=x[i*8+6];deadline=x[i*8+7]
        if cycles<0 or cycles>1 or deadline<0 or deadline>1:raise ValueError('invalid task features')
        c=max(0.,float(np.nextafter(cycles,np.float32(-np.inf))))
        d=min(1.,float(np.nextafter(deadline,np.float32(np.inf))))
        c_lo=np.nextafter(c*500e6,-np.inf)
        d_hi=np.nextafter(d*.300,np.inf)
        lower_service=np.nextafter(c_lo/1.5e9,-np.inf)
        upper_deadline=np.nextafter(d_hi+1e-10,np.inf)
        mask[i,0]=not (lower_service>upper_deadline)
    return mask,h


@native
def evaluate(g,v,b,w,q,beta,eta,mask,cycles,sweep,separable):
    for d in range(5):
        if not np.isfinite(q[d]) or q[d]<0:raise ValueError('invalid workload')
    a,diagnostics=decode(g,v,b,w,q,beta,eta,mask,cycles,sweep)
    cost=diagnostics[1]
    if separable:
        sums=w.sum(0);cost=b
        for i in range(30):
            d=a[i]
            cost+=g[i,d]+beta[d]*(2*(q[d]+eta[d]*(sums[d]-w[i,d]))*w[i,d]+w[i,d]*w[i,d])/120
        for j in range(2):cost+=v[j,a[30+j]]
    return a,cost


class Runtime:
    """Immutable inference snapshot; rebuild after changing training weights."""
    def __init__(self,critic):
        self.arm=critic.arm
        with torch.inference_mode():
            self.heads=torch.jit.freeze(torch.jit.script(Heads(critic).eval()))
        self.sweep=self.arm not in ('coupled_no_sweep','separable64')
        self.separable=self.arm=='separable64'

    @torch.inference_mode()
    def values(self,state):
        x=np.asarray(state)
        if x.dtype!=np.float32 or x.shape!=(533,):raise ValueError('one FP32 state required')
        mask,h=validate_mask(x)
        arrays=[t.numpy()[0] for t in self.heads(torch.from_numpy(x)[None])]
        a,cost=evaluate(*arrays,mask,x[:240].reshape(30,8)[:,6].copy(),self.sweep,self.separable)
        return a[None].copy(),np.array([-h*cost],np.float64)
