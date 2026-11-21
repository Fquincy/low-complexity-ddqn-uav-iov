"""Predeclared v3 separable workload-price critic. No training launcher."""
import torch
from torch import nn
from torch.nn import functional as F

NEW_ARMS=('amortized_workload_price24','amortized_no_price24','plain24_nonpositive')

def marginal_table(work,queue,beta,fractions):
    """All values use normalized service-time units; work is [batch,N,5]."""
    other=(work.sum(1,keepdim=True)-work).clamp_min(0.)
    expected=queue[:,None,:]+fractions[:,None,:]*other
    return beta[:,None,:]*(2*expected*work+work.square())/(4*work.shape[1])

def decode(placement,motion,bias):
    a=placement.argmin(2);b=motion.argmin(2)
    cost=bias+placement.gather(2,a[:,:,None]).squeeze(2).sum(1)+motion.gather(2,b[:,:,None]).squeeze(2).sum(1)
    return torch.cat((a,b),1),cost

class AmortizedPriceQ(nn.Module):
    def __init__(self,arm='amortized_workload_price24',n=30):
        super().__init__()
        if arm not in NEW_ARMS or n<1:raise ValueError('invalid arm or vehicle count')
        self.arm=arm;self.n=n;self.state_size=17*n+23
        size=5*n+20 if arm!='plain24_nonpositive' else 5*n+11
        self.net=nn.Sequential(nn.Linear(self.state_size,24),nn.ReLU(),nn.Linear(24,size))
        self.register_buffer('cycle_to_work',torch.tensor([0.,500e6/10e9/.3,500e6/10e9/.3,500e6/20e9/.3,500e6/20e9/.3]))

    def tables(self,states):
        if states.ndim!=2 or states.shape[1]!=self.state_size:raise ValueError('invalid observation shape')
        out=self.net(states);n=self.n;split=5*n
        placement=out[:,:split].reshape(-1,n,5)
        motion=out[:,split:split+10].reshape(-1,2,5);bias=out[:,split+10]
        if self.arm=='plain24_nonpositive':return placement/n,motion/2,bias,{}
        beta=F.pad(F.softplus(out[:,split+11:split+15]),(1,0))
        fractions=F.softmax(out[:,split+15:split+20],dim=1)
        work=states[:,:8*n].reshape(-1,n,8)[:,:,6,None]*self.cycle_to_work[None,None,:]
        queue=F.pad(torch.expm1(states[:,9*n:9*n+4]),(1,0))
        price=marginal_table(work,queue,beta,fractions)
        if self.arm=='amortized_no_price24':price=price*0.
        return F.softplus(placement)/n+price,F.softplus(motion)/2,F.softplus(bias),dict(price=price,beta=beta,fractions=fractions,work=work,queue=queue)

    def value_from_tables(self,placement,motion,bias,actions):
        cost=bias+placement.gather(2,actions[:,:self.n,None]).squeeze(2).sum(1)+motion.gather(2,actions[:,self.n:,None]).squeeze(2).sum(1)
        return -F.softplus(cost) if self.arm=='plain24_nonpositive' else -cost

    def forward(self,states,actions):
        if actions.shape!=(len(states),self.n+2) or actions.dtype!=torch.long:raise ValueError('int64 joint actions required')
        if torch.any(actions<0) or torch.any(actions>=5):raise ValueError('illegal action')
        placement,motion,bias,_=self.tables(states)
        return self.value_from_tables(placement,motion,bias,actions)

    def greedy(self,states):
        placement,motion,bias,details=self.tables(states)
        actions,cost=decode(placement,motion,bias)
        q=-F.softplus(cost) if self.arm=='plain24_nonpositive' else -cost
        return actions,q,details

@torch.no_grad()
def ddqn_targets(online,target,rewards,next_states,dones,gamma=1.):
    if not torch.all((dones==0)|(dones==1)):raise ValueError('binary terminal mask required')
    y=rewards.detach().clone();active=~dones.bool()
    if active.any():
        a=online.greedy(next_states[active])[0]
        y[active]+=gamma*target(next_states[active],a)
    return y
