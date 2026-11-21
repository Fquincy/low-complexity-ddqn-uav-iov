"""Isolated ADP critic and bounded decoder. No simulator/training launcher."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from controller_v3_fused_runtime import native
from controller_v3_amortized import AmortizedPriceQ
from deadline_screen_prototype import allowed,remaining,explore

ARMS=('coupled_sweep','coupled_no_sweep','separable64','fixed_price_plain_sweep')

def potential(g,v,b,w,q,beta,eta,actions):
    """Differentiable chosen-action cost, FP64 scalar arithmetic."""
    g,v,b,w,q,beta,eta=[x.double() for x in (g,v,b,w,q,beta,eta)]
    n=g.shape[1]
    assigned=F.one_hot(actions[:,:n],g.shape[2]).double()
    W=(w*assigned).sum(1);S=(w.square()*assigned).sum(1)
    P=(beta*(2*q*W+eta*W.square()+(1-eta)*S)).sum(1)/(4*n)
    return b+g.gather(2,actions[:,:n,None]).sum((1,2))+v.gather(2,actions[:,n:,None]).sum((1,2))+P

@native
def scalar_cost(g,v,b,w,q,beta,eta,a):
    n,D=g.shape;W=np.zeros(D,np.float64);S=np.zeros(D,np.float64);cost=b
    for i in range(n):
        d=a[i];cost+=g[i,d];W[d]+=w[i,d];S[d]+=w[i,d]*w[i,d]
    for j in range(v.shape[0]):cost+=v[j,a[n+j]]
    for d in range(D):cost+=beta[d]*(2*q[d]*W[d]+eta[d]*W[d]*W[d]+(1-eta[d])*S[d])/(4*n)
    return cost

@native
def decode(g,v,b,w,q,beta,eta,mask,cycles,sweep=True):
    n,D=g.shape;a=np.empty(n+v.shape[0],np.int64)
    total=w.sum(0);initial_evaluations=0;sweep_evaluations=0;changes=0
    for i in range(n):
        best=-1;value=np.inf
        for d in range(D):
            if mask[i,d]:
                candidate=g[i,d]+beta[d]*(2*(q[d]+eta[d]*(total[d]-w[i,d]))*w[i,d]+w[i,d]**2)/(4*n)
                initial_evaluations+=1
                if candidate<value:best=d;value=candidate
        if best<0:raise ValueError('empty allowed destination set')
        a[i]=best
    for j in range(v.shape[0]):a[n+j]=np.argmin(v[j])
    initial=scalar_cost(g,v,b,w,q,beta,eta,a)
    if sweep:
        order=np.argsort(-cycles,kind='mergesort')
        W=np.zeros(D,np.float64)
        for i in range(n):W[a[i]]+=w[i,a[i]]
        for index in range(n):
            i=order[index];old=a[i];W[old]-=w[i,old]
            best=old;value=g[i,old]+beta[old]*(2*(q[old]+eta[old]*W[old])*w[i,old]+w[i,old]**2)/(4*n)
            for d in range(D):
                if mask[i,d]:
                    candidate=g[i,d]+beta[d]*(2*(q[d]+eta[d]*W[d])*w[i,d]+w[i,d]**2)/(4*n)
                    sweep_evaluations+=1
                    if candidate<value or (candidate==value and best!=old and d<best):best=d;value=candidate
            a[i]=best;changes+=int(best!=old);W[best]+=w[i,best]
    final=scalar_cost(g,v,b,w,q,beta,eta,a)
    if final>initial+1e-10*max(1.,abs(initial)):raise ValueError('potential increased')
    return a,np.array([initial,final,initial_evaluations,sweep_evaluations,changes],np.float64)

class Critic(nn.Module):
    def __init__(self,arm='coupled_sweep',seed=99861):
        super().__init__()
        if arm not in ARMS:raise ValueError('unknown engineering arm')
        self.arm=arm
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.core=AmortizedPriceQ('plain24_nonpositive' if arm=='fixed_price_plain_sweep' else 'amortized_workload_price24')

    def heads(self,states):
        if states.dtype!=torch.float32 or states.device.type!='cpu' or states.ndim!=2 or states.shape[1]!=533:
            raise ValueError('CPU FP32 states required')
        h=remaining(states)
        raw=self.core.net(states);n=30;positive=F.softplus(raw[:,:165 if self.arm!='fixed_price_plain_sweep' else 161])
        g=positive[:,:150].reshape(-1,n,5)/n;v=positive[:,150:160].reshape(-1,2,5)/2;b=positive[:,160]
        if self.arm=='fixed_price_plain_sweep':
            beta=states.new_ones((len(states),5));beta[:,0]=0;eta=states.new_full((len(states),5),.2)
        else:
            beta=F.pad(positive[:,161:165],(1,0));eta=F.softmax(raw[:,165:170],dim=1)
        w=states[:,:240].reshape(-1,30,8)[:,:,6,None]*self.core.cycle_to_work
        q=F.pad(torch.expm1(states[:,270:274]),(1,0))
        if torch.any(w<0) or torch.any(q<0) or not torch.isfinite(q).all():raise ValueError('invalid workload')
        return (g.double(),v.double(),b.double(),w.double(),q.double(),beta.double(),eta.double()),h

    def forward(self,states,actions):
        if actions.dtype!=torch.int64 or actions.shape!=(len(states),32) or torch.any(actions<0) or torch.any(actions>=5):
            raise ValueError('32 valid int64 actions required')
        heads,h=self.heads(states);g,v,b,w,q,beta,eta=heads
        if self.arm=='separable64':
            other=w.sum(1,keepdim=True)-w
            price=beta[:,None,:]*(2*(q[:,None,:]+eta[:,None,:]*other)*w+w.square())/120
            c=b+(g+price).gather(2,actions[:,:30,None]).sum((1,2))+v.gather(2,actions[:,30:,None]).sum((1,2))
        else:c=potential(*heads,actions)
        return -h.double()*c

    @torch.no_grad()
    def greedy(self,states,compiled=True,diagnostics=False):
        heads,h=self.heads(states)
        if torch.any(h<=0):raise ValueError('terminal states must not be decoded')
        masks=allowed(states);arrays=[x.detach().numpy() for x in heads];actions=[];records=[]
        fn=decode if compiled else decode.py_func
        for i in range(len(states)):
            a,diag=fn(*(x[i] for x in arrays),masks[i,:30],states[i,:240].numpy().reshape(30,8)[:,6].copy(),self.arm not in ('coupled_no_sweep','separable64'))
            actions.append(a);records.append(diag)
        act=torch.from_numpy(np.stack(actions))
        # Reuse heads: no second neural pass to evaluate returned actions.
        if self.arm=='separable64':
            g,v,b,w,q,beta,eta=heads
            price=beta[:,None,:]*(2*(q[:,None,:]+eta[:,None,:]*(w.sum(1,keepdim=True)-w))*w+w.square())/120
            cost=b+(g+price).gather(2,act[:,:30,None]).sum((1,2))+v.gather(2,act[:,30:,None]).sum((1,2))
        else:cost=potential(*heads,act)
        result=(act,-h.double()*cost)
        return (*result,np.stack(records)) if diagnostics else result

class Runtime:
    def __init__(self,critic):self.critic=critic.eval()
    @torch.inference_mode()
    def values(self,state):
        x=np.asarray(state)
        if x.dtype!=np.float32 or x.shape!=(533,):raise ValueError('one FP32 state required')
        a,q=self.critic.greedy(torch.from_numpy(x)[None])
        return a.numpy().copy(),q.numpy().copy()

@torch.no_grad()
def targets(online,target,rewards,next_states,dones):
    if not torch.all((dones==0)|(dones==1)):raise ValueError('binary done')
    if not torch.equal(dones.bool(),remaining(next_states)==0):raise ValueError('done/time mismatch')
    y=rewards.double().clone();active=~dones.bool()
    if active.any():
        a=online.greedy(next_states[active])[0];y[active]+=target(next_states[active],a)
    return y

def objective(online,target,batch,anchor=None):
    states,actions,rewards,next_states,dones=batch
    if not torch.all(remaining(states)==remaining(next_states)+1):raise ValueError('invalid transition horizon')
    y=targets(online,target,rewards,next_states,dones)
    uniform=F.smooth_l1_loss(online(states,actions),y,beta=1.)
    terminal=uniform*0
    if anchor is not None:
        s,a,r=anchor
        if not torch.all(remaining(s)==1):raise ValueError('nonterminal anchor')
        terminal=F.smooth_l1_loss(online(s,a),r.double(),beta=1.)
    return uniform+terminal,dict(uniform=uniform,anchor=terminal,targets=y)
