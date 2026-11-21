"""Version-2 shared congestion-cost models. No simulator stepping or training here."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

SHARED_ARMS = ('shared_congestion_cost', 'shared_no_congestion',
               'untied_congestion_cost', 'shared_joint_bound')

def mlp(a, b, c):
    return nn.Sequential(nn.Linear(a,b), nn.ReLU(), nn.Linear(b,c))

def split_observation(states, n=30):
    if states.ndim != 2 or states.shape[1] != 17*n+23:
        raise ValueError('batch x (17N+23) observations required')
    features=states[:,:8*n].reshape(-1,n,8)
    workload=states[:,8*n:9*n+4]
    radio=states[:,9*n+4:17*n+4].reshape(-1,n,8)
    local=torch.cat((features,workload[:,:n,None],radio),dim=2)
    global_=torch.cat((workload[:,n:],states[:,17*n+4:]),dim=1)
    return local,global_

def loads(actions,w):
    return torch.zeros((len(actions),5),dtype=w.dtype,device=w.device).scatter_add(
        1,actions,w.gather(2,actions[:,:,None]).squeeze(2))

def marginal_prices(actions,w,L,beta):
    A=loads(actions,w)
    H=L[:,None,:]+A[:,None,:]-F.one_hot(actions,5)*w
    return beta[:,None,:]*(2*H*w+w.square())/(4*actions.shape[1])

def full_cost(actions,u,w,L,beta,constant):
    A=loads(actions,w)
    # Algebraically equivalent expanded form avoids subtracting large squares.
    penalty=(beta*(2*L*A+A.square())).sum(1)/(4*actions.shape[1])
    return constant+u.gather(2,actions[:,:,None]).squeeze(2).sum(1)+penalty

def decode_cost_tables(u,w,L,beta,motion,constant,enabled=True):
    """Fixed 2 price passes + 4 exact single-move repairs; approximate global min."""
    batch,n,_=u.shape
    moves=motion.argmin(2)
    c=constant+motion.gather(2,moves[:,:,None]).squeeze(2).sum(1)
    a=u.argmin(2)
    if not enabled:
        cost=full_cost(a,u,w,L,beta,c)
        return torch.cat((a,moves),1),cost,{'costs':cost[:,None]}
    candidates=[a]
    for _ in range(2):
        a=(u+marginal_prices(a,w,L,beta)).argmin(2)
        candidates.append(a)
    costs=torch.stack([full_cost(a,u,w,L,beta,c) for a in candidates],1)
    best=costs.argmin(1)
    rows=torch.arange(batch,device=u.device)
    a=torch.stack(candidates,1)[rows,best]
    current=costs[rows,best]
    scores=[costs[:,i] for i in range(3)]
    accepted=[];predicted=[];actual=[]
    for _ in range(4):
        p=marginal_prices(a,w,L,beta)
        old=(u+p).gather(2,a[:,:,None])
        delta=u+p-old
        index=delta.flatten(1).argmin(1)
        vehicle=index//5;destination=index%5
        change=delta.flatten(1)[rows,index]
        trial=a.clone()
        trial[rows,vehicle]=torch.where(change < -1e-6,destination,a[rows,vehicle])
        trial_cost=full_cost(trial,u,w,L,beta,c)
        accept=(change < -1e-6)&(trial_cost < current-1e-6)
        a=torch.where(accept[:,None],trial,a)
        new=torch.where(accept,trial_cost,current)
        scores.append(trial_cost)
        accepted.append(accept);predicted.append(change);actual.append(trial_cost-current)
        current=new
    return torch.cat((a,moves),1),current,dict(costs=torch.stack(scores,1),
        initial_best=best,repairs_accepted=torch.stack(accepted,1),
        repair_predicted_delta=torch.stack(predicted,1),repair_trial_delta=torch.stack(actual,1))

class SharedCostNetwork(nn.Module):
    def __init__(self,arm='shared_congestion_cost',n=30,cycle_max=500e6,
                 frequencies=(10e9,10e9,20e9,20e9),normalizer=.3):
        super().__init__()
        if arm not in SHARED_ARMS:raise ValueError('unknown shared arm')
        self.arm=arm;self.n=n
        self.untied=arm=='untied_congestion_cost'
        self.joint=arm=='shared_joint_bound'
        self.congestion=arm!='shared_no_congestion'
        # Construction order: phi, rho, placement, mobility, global head.
        self.phi=nn.ModuleList([mlp(17,32,16) for _ in range(n)]) if self.untied else mlp(17,32,16)
        self.rho=mlp(39,32,16)
        self.placement=nn.ModuleList([mlp(32,16,5) for _ in range(n)]) if self.untied else mlp(32,16,5)
        self.mobility=mlp(16,16,10)
        self.global_head=nn.Linear(16,5)
        self.register_buffer('cycle_to_work',torch.tensor([0.]+[cycle_max/f/normalizer for f in frequencies]))

    def tables(self,states):
        x,g=split_observation(states,self.n)
        if self.untied:z=torch.stack([module(x[:,i]) for i,module in enumerate(self.phi)],1)
        else:z=self.phi(x)
        h=self.rho(torch.cat((z.mean(1),g),1))
        p=torch.cat((z,h[:,None,:].expand(-1,self.n,-1)),2)
        if self.untied:l=torch.stack([module(p[:,i]) for i,module in enumerate(self.placement)],1)
        else:l=self.placement(p)
        m=self.mobility(h).reshape(-1,2,5)
        global_values=self.global_head(h)
        beta=F.pad(F.softplus(global_values[:,1:]),(1,0))
        if not self.congestion:beta=beta*0.
        L=F.pad(torch.expm1(g[:,:4]),(1,0))
        w=x[:,:,6,None]*self.cycle_to_work[None,None,:]
        u=(l if self.joint else F.softplus(l))/self.n
        motion=(m if self.joint else F.softplus(m))/2
        base=global_values[:,0] if self.joint else F.softplus(global_values[:,0])
        return u,w,L,beta,motion,base

    def forward(self,states,actions):
        u,w,L,beta,m,base=self.tables(states)
        if actions.shape!=(len(states),self.n+2) or actions.dtype!=torch.long:
            raise ValueError('int64 complete joint actions required')
        if torch.any(actions<0) or torch.any(actions>=5):raise ValueError('illegal action')
        c=base+m.gather(2,actions[:,-2:,None]).squeeze(2).sum(1)
        cost=full_cost(actions[:,:self.n],u,w,L,beta,c)
        return -F.softplus(cost) if self.joint else -cost

    @torch.no_grad()
    def greedy(self,states):
        tables=self.tables(states)
        actions,cost,details=decode_cost_tables(*tables,enabled=self.congestion)
        q=-F.softplus(cost) if self.joint else -cost
        return actions,q,details

class _BDQScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):return x.view_as(x)
    @staticmethod
    def backward(ctx,gradient):return gradient/33.

class AdaptedBDQ(nn.Module):
    """Disclosed 64-64 uniform/Huber adaptation; not the original paper's full agent."""
    def __init__(self):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(533,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.value=nn.Linear(64,1);self.advantage=nn.Linear(64,160)
    def forward(self,states):
        h=_BDQScale.apply(self.trunk(states))
        advantages=self.advantage(h).reshape(-1,32,5)
        return self.value(h)[:,:,None]+advantages-advantages.mean(2,keepdim=True)

def parameter_bytes(model):return sum(p.numel()*p.element_size() for p in model.parameters())
