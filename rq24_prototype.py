"""RQ24 design-v1 engineering prototype. No training or research launcher."""
import copy
from pathlib import Path
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import assignment_price_prototype as adp
import assignment_price_load_radio_heuristic as physical
import compiled_controller_inference as deployed
from assignment_price_fast_runtime import validate_mask
from controller_v3_fused_runtime import native

ARMS=('rq24','rq24_no_radio_pair','rq24_no_physical_unary_or_radio_pair','rq24_no_sweep')


@native
def phi(t,deadline):
    return .5*t/.300+max(t/deadline-1.,0.)


@native
def tx(bits,gain,interference,c):
    rate=c[7]*np.log1p(c[8]*gain/(c[9]+interference))/np.log(2.)
    if rate<=0 or not np.isfinite(rate):raise ValueError('invalid physical rate')
    result=bits/rate
    if not np.isfinite(result):raise ValueError('invalid transmission duration')
    return result


@native
def edge_pairs(gain,k):
    """Top-(K+1) cache; unique undirected edges without a dense N*N table."""
    n,U=gain.shape;k=min(k,max(0,n-1));count=0
    ei=np.empty(U*n*k,np.int64);ej=np.empty(U*n*k,np.int64);ed=np.empty(U*n*k,np.int64)
    for d in range(U):
        top=np.full(min(n,k+1),-1,np.int64)
        for j in range(n):
            for pos in range(len(top)):
                if top[pos]<0 or gain[j,d]>gain[top[pos],d]:
                    for z in range(len(top)-1,pos,-1):top[z]=top[z-1]
                    top[pos]=j;break
        donors=np.empty((n,k),np.int64)
        for i in range(n):
            pos=0
            for j in top:
                if j!=i and pos<k:donors[i,pos]=j;pos+=1
        for i in range(n):
            for pos in range(k):
                j=donors[i,pos];reverse=False
                if j<i:
                    for z in range(k):
                        if donors[j,z]==i:reverse=True
                if not reverse:
                    ei[count]=min(i,j);ej[count]=max(i,j);ed[count]=d+1;count+=1
    return ei[:count].copy(),ej[:count].copy(),ed[:count].copy()


@native
def features(x,c,k,radio):
    bits,cycles,deadline,local,remote,backlog,gain,battery=physical.prepare(x,c)
    if np.any(deadline<=0) or np.any(bits<0) or np.any(cycles<0):raise ValueError('invalid task')
    n=len(bits);T=np.empty((n,5),np.float64);unary=np.empty((n,5),np.float64)
    reference=np.empty((n,2),np.float64)
    for i in range(n):
        for d in range(2):reference[i,d]=tx(bits[i],gain[i,d],0.,c)
        T[i,0]=local[i]+cycles[i]/c[2]
        wait=0.
        for d in range(4):
            if d<2:
                rate=c[7]*np.log1p(c[8]*gain[i,d]/c[9])/np.log(2.)
            else:rate=c[16]
            wait+=backlog[i,d]/rate
        for d in range(1,5):
            upload=reference[i,d-1] if d<=2 else bits[i]/c[16]
            T[i,d]=wait+upload+remote[d]+cycles[i]/c[2+d]
        for d in range(5):
            unary[i,d]=phi(T[i,d],deadline[i])
            if not np.isfinite(unary[i,d]):raise ValueError('nonfinite unary')
    if radio:
        ei,ej,ed=edge_pairs(gain,k);coeff=np.empty(len(ei),np.float64)
        for e in range(len(ei)):
            i=ei[e];j=ej[e];d=ed[e];u=d-1
            di=tx(bits[i],gain[i,u],c[8]*gain[j,u],c)-reference[i,u]
            dj=tx(bits[j],gain[j,u],c[8]*gain[i,u],c)-reference[j,u]
            coeff[e]=max(0.,phi(T[i,d]+di,deadline[i])-unary[i,d]+phi(T[j,d]+dj,deadline[j])-unary[j,d])
            if not np.isfinite(coeff[e]):raise ValueError('nonfinite edge')
    else:
        ei=np.empty(0,np.int64);ej=ei.copy();ed=ei.copy();coeff=np.empty(0,np.float64)
    return unary,ei,ej,ed,coeff


@native
def softplus64(x):
    return max(x,0.)+np.log1p(np.exp(-abs(x)))


@native
def heads(raw):
    g=raw[:150].astype(np.float64).reshape(30,5)/30.
    v=raw[150:160].astype(np.float64).reshape(2,5)/2.
    beta=np.zeros(5,np.float64)
    for d in range(1,5):beta[d]=softplus64(float(raw[160+d]))
    eta=np.exp(raw[165:170].astype(np.float64)-float(np.max(raw[165:170])))
    eta/=eta.sum()
    return g,v,float(raw[160]),beta,eta,softplus64(float(raw[170]))


@native
def scalar(g,v,b,w,q,beta,eta,a,ei,ej,ed,coeff,lam):
    cost=adp.scalar_cost(g,v,b,w,q,beta,eta,a)
    for e in range(len(ei)):
        if a[ei[e]]==ed[e] and a[ej[e]]==ed[e]:cost+=lam*coeff[e]/len(g)
    return cost


@native
def decode(g,v,b,w,q,beta,eta,mask,cycles,ei,ej,ed,coeff,lam,sweep):
    n,D=g.shape;a=np.empty(n+len(v),np.int64);total=w.sum(0)
    for i in range(n):
        best=-1;value=np.inf
        for d in range(D):
            if mask[i,d]:
                z=g[i,d]+beta[d]*(2*(q[d]+eta[d]*(total[d]-w[i,d]))*w[i,d]+w[i,d]**2)/(4*n)
                if z<value:best=d;value=z
        if best<0:raise ValueError('empty mask')
        a[i]=best
    for j in range(len(v)):a[n+j]=np.argmin(v[j])
    initial=scalar(g,v,b,w,q,beta,eta,a,ei,ej,ed,coeff,lam)
    # Linked adjacency and per-node/destination radio costs: O(ND+E) scratch.
    first=np.full(n,-1,np.int64);other=np.empty(2*len(ei),np.int64)
    edge=np.empty(2*len(ei),np.int64);following=np.empty(2*len(ei),np.int64)
    rc=np.zeros((n,D),np.float64)
    for e in range(len(ei)):
        i=ei[e];j=ej[e];d=ed[e];weight=lam*coeff[e]/n
        for z in range(2):
            node=i if z==0 else j;neighbor=j if z==0 else i;pos=2*e+z
            other[pos]=neighbor;edge[pos]=e;following[pos]=first[node];first[node]=pos
        if a[j]==d:rc[i,d]+=weight
        if a[i]==d:rc[j,d]+=weight
    W=np.zeros(D,np.float64)
    for i in range(n):W[a[i]]+=w[i,a[i]]
    changes=0
    if sweep:
        order=np.argsort(-cycles,kind='mergesort')
        for index in range(n):
            i=order[index];old=a[i];W[old]-=w[i,old]
            best=old
            value=g[i,old]+beta[old]*(2*(q[old]+eta[old]*W[old])*w[i,old]+w[i,old]**2)/(4*n)+rc[i,old]
            for d in range(D):
                if mask[i,d]:
                    z=g[i,d]+beta[d]*(2*(q[d]+eta[d]*W[d])*w[i,d]+w[i,d]**2)/(4*n)+rc[i,d]
                    if z<value or (z==value and best!=old and d<best):best=d;value=z
            if best!=old:
                changes+=1;pos=first[i]
                while pos>=0:
                    e=edge[pos];d=ed[e];weight=lam*coeff[e]/n;neighbor=other[pos]
                    if old==d:rc[neighbor,d]-=weight
                    if best==d:rc[neighbor,d]+=weight
                    pos=following[pos]
            a[i]=best;W[best]+=w[i,best]
    final=scalar(g,v,b,w,q,beta,eta,a,ei,ej,ed,coeff,lam)
    if final>initial+1e-10*max(1.,abs(initial)):raise ValueError('surrogate increased')
    return a,np.array([initial,final,changes,len(ei)],np.float64)


@native
def finish(raw,x,scale,c,mask,h,k,radio,physical_unary,sweep):
    unary,ei,ej,ed,coeff=features(x,c,k,radio)
    g,v,b,beta,eta,lam=heads(raw)
    if physical_unary:g+=unary/30.
    w=x[:240].astype(np.float64).reshape(30,8)[:,6:7]*scale
    q=np.zeros(5,np.float64);q[1:]=np.expm1(x[270:274].astype(np.float64))
    a,diag=decode(g,v,b,w,q,beta,eta,mask,x[:240].reshape(30,8)[:,6],ei,ej,ed,coeff,lam,sweep)
    return a[None].copy(),np.array([-h*softplus64(diag[1])],np.float64),diag


class Critic(nn.Module):
    def __init__(self,arm='rq24',seed=906190001):
        super().__init__()
        if arm not in ARMS:raise ValueError('unknown RQ24 arm')
        self.arm=arm;self.radio=arm not in (ARMS[1],ARMS[2]);self.physical=arm!=ARMS[2];self.sweep=arm!=ARMS[3]
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.net=nn.Sequential(nn.Linear(533,24),nn.ReLU(),nn.Linear(24,171))
        self.register_buffer('cycle_to_work',torch.tensor([0.,500e6/10e9/.3,500e6/10e9/.3,500e6/20e9/.3,500e6/20e9/.3]))
        self.c=physical.constants();self.k=4

    def chosen(self,states,actions,raw):
        if states.dtype!=torch.float32 or states.ndim!=2 or states.shape[1]!=533 or not torch.isfinite(states).all():raise ValueError('finite FP32 states required')
        if actions.dtype!=torch.int64 or actions.shape!=(len(states),32) or torch.any(actions<0) or torch.any(actions>=5):raise ValueError('legal actions required')
        h=adp.remaining(states);outputs=[]
        for i in range(len(states)):
            if h[i]==0:outputs.append(raw[i].sum().double()*0);continue
            x=states[i].detach().numpy();validate_mask(x)
            unary,ei,ej,ed,coeff=features(x,self.c,self.k,self.radio)
            g=raw[i:i+1,:150].double().reshape(1,30,5)/30
            if self.physical:g=g+torch.from_numpy(unary)[None]/30
            v=raw[i:i+1,150:160].double().reshape(1,2,5)/2
            beta=F.pad(F.softplus(raw[i:i+1,161:165].double()),(1,0))
            eta=F.softmax(raw[i:i+1,165:170].double(),dim=1)
            w=states[i:i+1,:240].double().reshape(1,30,8)[:,:,6,None]*self.cycle_to_work.double()
            q=F.pad(torch.expm1(states[i:i+1,270:274].double()),(1,0))
            cost=adp.potential(g,v,raw[i:i+1,160].double(),w,q,beta,eta,actions[i:i+1])[0]
            if len(ei):
                matches=(actions[i,ei]==torch.from_numpy(ed))&(actions[i,ej]==torch.from_numpy(ed))
                cost=cost+F.softplus(raw[i,170].double())*(torch.from_numpy(coeff)*matches).sum()/30
            outputs.append(-h[i].double()*F.softplus(cost))
        return torch.stack(outputs)

    def forward(self,states,actions):return self.chosen(states,actions,self.net(states))

    @torch.no_grad()
    def greedy(self,states):
        raw=self.net(states);acts=[]
        for i in range(len(states)):
            x=states[i].numpy();mask,h=validate_mask(x)
            a,_,_=finish(raw[i].numpy(),x,self.cycle_to_work.numpy().astype(np.float64),self.c,mask,h,self.k,self.radio,self.physical,self.sweep)
            acts.append(a[0])
        actions=torch.from_numpy(np.stack(acts))
        return actions,self.chosen(states,actions,raw)


class Runtime:
    def __init__(self,model,path):
        self.path=Path(path)
        if self.path.exists():raise FileExistsError(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with warnings.catch_warnings(record=True),torch.inference_mode():
            torch.onnx.export(copy.deepcopy(model.net).eval(),torch.zeros(1,533),str(path),
                              input_names=['state'],output_names=['raw'],opset_version=17,dynamo=False)
        self.session=deployed.session(path);self.scale=model.cycle_to_work.numpy().astype(np.float64)
        self.c=model.c.copy();self.k=model.k;self.radio=model.radio;self.physical=model.physical;self.sweep=model.sweep

    def values(self,state):
        x=np.asarray(state)
        if x.dtype!=np.float32 or x.shape!=(533,):raise ValueError('one FP32 state required')
        mask,h=validate_mask(x)
        raw=self.session.run(None,{'state':x[None]})[0][0]
        a,q,_=finish(raw,x,self.scale,self.c,mask,h,self.k,self.radio,self.physical,self.sweep)
        return a,q
