"""Prospective aggregate-radio DDQN synthetic prototype; no research launcher."""
import copy
from pathlib import Path
import warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import rq24_prototype as prior
from controller_v3_fused_runtime import native

ARMS=('ar24','ar24_no_radio','ar24_no_deadline_weight','ar24_homogeneous_geometry',
      'ar24_no_cpu_coupling','ar24_no_sweep')


def flags(arm):
    if arm not in ARMS:raise ValueError('unknown AR24 variant')
    return (arm!='ar24_no_radio',arm!='ar24_no_deadline_weight',
            arm=='ar24_homogeneous_geometry',arm!='ar24_no_cpu_coupling',arm!='ar24_no_sweep')


@native
def features(state,scale,deadline_weight,homogeneous):
    x=np.empty((30,2),np.float64);y=np.empty((30,2),np.float64)
    w=np.empty((30,5),np.float64);q=np.zeros(5,np.float64)
    for d in range(4):
        q[d+1]=np.expm1(np.float64(state[270+d]))
        if q[d+1]<0 or not np.isfinite(q[d+1]):raise ValueError('invalid compute queue')
    for j in range(2):
        if state[526+3*j]<0 or state[526+3*j]>1 or state[527+3*j]<0 or state[527+3*j]>1:raise ValueError('invalid UAV geometry')
    for i in range(30):
        px=np.float64(state[8*i]);py=np.float64(state[8*i+1])
        b=np.float64(state[8*i+5]);t=np.float64(state[8*i+7]);cycles=np.float64(state[8*i+6])
        if px<0 or px>1 or py<0 or py>1 or b<0 or b>1 or t<=0 or t>1:raise ValueError('invalid task features')
        for d in range(5):w[i,d]=cycles*scale[d]
        for j in range(2):
            dx=(px-np.float64(state[526+3*j]))*1000.
            dy=(py-np.float64(state[527+3*j]))*1000.
            g=1. if homogeneous else 2500./(dx*dx+dy*dy+2500.)
            x[i,j]=(b/t if deadline_weight else b)/g;y[i,j]=g
            if not np.isfinite(x[i,j]):raise ValueError('nonfinite radio feature')
    return x,y,w,q


@native
def heads(raw):
    for z in raw:
        if not np.isfinite(z):raise ValueError('nonfinite raw head')
    g=raw[:150].astype(np.float64).reshape(30,5)/30.
    v=raw[150:160].astype(np.float64).reshape(2,5)/2.
    beta=np.zeros(5,np.float64);eta=np.empty(5,np.float64);lam=np.empty(2,np.float64)
    for d in range(4):beta[d+1]=prior.softplus64(np.float64(raw[161+d]))
    top=np.float64(np.max(raw[165:170]));total=0.
    for d in range(5):eta[d]=np.exp(np.float64(raw[165+d])-top);total+=eta[d]
    for d in range(5):eta[d]/=total
    for d in range(2):lam[d]=prior.softplus64(np.float64(raw[170+d]))
    return g,v,np.float64(raw[160]),beta,eta,lam


@native
def radio_prefix(x,y,a,lam):
    X=np.zeros(2,np.float64);Y=np.zeros(2,np.float64);pair=np.zeros(2,np.float64)
    for i in range(len(x)):
        d=a[i]
        if d==1 or d==2:
            j=d-1;pair[j]+=x[i,j]*Y[j]+y[i,j]*X[j];X[j]+=x[i,j];Y[j]+=y[i,j]
    return (lam[0]*pair[0]+lam[1]*pair[1])/len(x)


@native
def scalar(g,v,b,w,q,beta,eta,x,y,lam,a,radio,cpu):
    # Disabling CPU changes only its term, leaving residuals/head tensors intact.
    cost=prior.adp.scalar_cost(g,v,b,w,q,beta if cpu else np.zeros(5,np.float64),eta,a)
    if radio:cost+=radio_prefix(x,y,a,lam)
    return cost


@native
def marginal(i,d,W,X,Y,g,w,q,beta,eta,x,y,lam,radio,cpu):
    n=len(g);z=g[i,d]
    if cpu:z+=beta[d]*(2*(q[d]+eta[d]*W[d])*w[i,d]+w[i,d]**2)/(4*n)
    if radio and (d==1 or d==2):
        j=d-1;z+=lam[j]*(x[i,j]*Y[j]+y[i,j]*X[j])/n
    return z


@native
def decode(g,v,b,w,q,beta,eta,x,y,lam,mask,cycles,radio,cpu,sweep):
    n,D=g.shape;a=np.empty(n+len(v),np.int64);total=w.sum(0)
    for i in range(n):
        best=-1;value=np.inf
        for d in range(D):
            if mask[i,d]:
                z=g[i,d]
                if cpu:z+=beta[d]*(2*(q[d]+eta[d]*(total[d]-w[i,d]))*w[i,d]+w[i,d]**2)/(4*n)
                if z<value:best=d;value=z
        if best<0:raise ValueError('empty allowed set')
        a[i]=best
    for j in range(len(v)):a[n+j]=np.argmin(v[j])
    initial=scalar(g,v,b,w,q,beta,eta,x,y,lam,a,radio,cpu)
    W=np.zeros(D,np.float64);X=np.zeros(2,np.float64);Y=np.zeros(2,np.float64);count=np.zeros(D,np.int64)
    for i in range(n):
        d=a[i];count[d]+=1;W[d]+=w[i,d]
        if d==1 or d==2:X[d-1]+=x[i,d-1];Y[d-1]+=y[i,d-1]
    changes=0
    if sweep:
        order=np.argsort(-cycles,kind='mergesort')
        for ii in range(n):
            i=order[ii];old=a[i];count[old]-=1;W[old]-=w[i,old]
            if old==1 or old==2:X[old-1]-=x[i,old-1];Y[old-1]-=y[i,old-1]
            if count[old]==0:
                W[old]=0.
                if old==1 or old==2:X[old-1]=0.;Y[old-1]=0.
            elif W[old]<0 or ((old==1 or old==2) and (X[old-1]<0 or Y[old-1]<0)):raise ValueError('negative aggregate cache')
            best=old;value=marginal(i,old,W,X,Y,g,w,q,beta,eta,x,y,lam,radio,cpu)
            for d in range(D):
                if mask[i,d]:
                    z=marginal(i,d,W,X,Y,g,w,q,beta,eta,x,y,lam,radio,cpu)
                    if z<value or (z==value and best!=old and d<best):best=d;value=z
            changes+=int(best!=old);a[i]=best;count[best]+=1;W[best]+=w[i,best]
            if best==1 or best==2:X[best-1]+=x[i,best-1];Y[best-1]+=y[i,best-1]
    final=scalar(g,v,b,w,q,beta,eta,x,y,lam,a,radio,cpu)
    if final>initial+1e-12*max(1.,abs(initial)):raise ValueError('surrogate increased')
    return a,np.array([initial,final,changes],np.float64)


@native
def finish(raw,state,scale,mask,h,radio,deadline,homogeneous,cpu,sweep):
    x,y,w,q=features(state,scale,deadline,homogeneous);g,v,b,beta,eta,lam=heads(raw)
    a,diag=decode(g,v,b,w,q,beta,eta,x,y,lam,mask,state[:240].reshape(30,8)[:,6],radio,cpu,sweep)
    return a[None].copy(),np.array([-h*prior.softplus64(diag[1])]),diag


class Critic(nn.Module):
    def __init__(self,arm='ar24',seed=906196101):
        super().__init__();self.arm=arm;self.flags=flags(arm)
        with torch.random.fork_rng():
            torch.manual_seed(seed);self.net=nn.Sequential(nn.Linear(533,24),nn.ReLU(),nn.Linear(24,172))
        self.register_buffer('cycle_to_work',torch.tensor([0.,500e6/10e9/.3,500e6/10e9/.3,500e6/20e9/.3,500e6/20e9/.3]))

    def chosen(self,states,actions,raw):
        if states.dtype!=torch.float32 or states.ndim!=2 or states.shape[1]!=533 or not torch.isfinite(states).all():raise ValueError('finite FP32 state required')
        if actions.dtype!=torch.int64 or actions.shape!=(len(states),32) or torch.any(actions<0) or torch.any(actions>=5):raise ValueError('legal actions required')
        h=prior.adp.remaining(states);outputs=[]
        radio,deadline,homogeneous,cpu,sweep=self.flags
        for i in range(len(states)):
            if h[i]==0:outputs.append(raw[i].double().sum()*0);continue
            prior.validate_mask(states[i].detach().numpy())
            x,y,w,q=features(states[i].detach().numpy(),self.cycle_to_work.detach().numpy().astype(np.float64),deadline,homogeneous)
            z=raw[i].double();g=z[:150].reshape(30,5)/30;v=z[150:160].reshape(2,5)/2
            beta=F.pad(F.softplus(z[161:165]),(1,0));exp=torch.exp(z[165:170]-z[165:170].max())
            total=exp[0]
            for d in range(1,5):total=total+exp[d]
            eta=exp/total;lam=F.softplus(z[170:172]);act=actions[i]
            cost=z[160]+g.gather(1,act[:30,None]).sum()+v.gather(1,act[30:,None]).sum()
            if cpu:
                assigned=F.one_hot(act[:30],5).double();wt=torch.from_numpy(w);qt=torch.from_numpy(q)
                W=(wt*assigned).sum(0);S=(wt.square()*assigned).sum(0)
                cost=cost+(beta*(2*qt*W+eta*W.square()+(1-eta)*S)).sum()/120
            if radio:
                pair=[0.,0.];X=[0.,0.];Y=[0.,0.]
                for j in range(30):
                    d=int(act[j])
                    if d in (1,2):
                        u=d-1;pair[u]+=x[j,u]*Y[u]+y[j,u]*X[u];X[u]+=x[j,u];Y[u]+=y[j,u]
                cost=cost+(lam[0]*pair[0]+lam[1]*pair[1])/30
            outputs.append(-h[i].double()*F.softplus(cost))
        return torch.stack(outputs)

    def forward(self,states,actions):return self.chosen(states,actions,self.net(states))

    @torch.no_grad()
    def greedy(self,states):
        raw=self.net(states);acts=[]
        for i in range(len(states)):
            state=states[i].numpy();mask,h=prior.validate_mask(state)
            a,_,_=finish(raw[i].numpy(),state,self.cycle_to_work.numpy().astype(np.float64),mask,h,*self.flags)
            acts.append(a[0])
        actions=torch.from_numpy(np.stack(acts));return actions,self.chosen(states,actions,raw)


class Runtime:
    def __init__(self,model,path):
        self.path=Path(path)
        if self.path.exists():raise FileExistsError(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with warnings.catch_warnings(record=True),torch.inference_mode():
            torch.onnx.export(copy.deepcopy(model.net).eval(),torch.zeros(1,533),str(path),input_names=['state'],output_names=['raw'],opset_version=17,dynamo=False)
        self.session=prior.deployed.session(path);self.scale=model.cycle_to_work.numpy().astype(np.float64);self.flags=model.flags

    def values(self,state):
        state=np.asarray(state)
        if state.dtype!=np.float32 or state.shape!=(533,):raise ValueError('one FP32 observation required')
        mask,h=prior.validate_mask(state);raw=self.session.run(None,{'state':state[None]})[0][0]
        a,q,_=finish(raw,state,self.scale,mask,h,*self.flags);return a,q
