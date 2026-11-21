"""Seven fixed adapters for the prospective correction comparison."""
import torch
from torch.nn import functional as F
import centered_linear_ddqn_prototype as corrected
import heuristic_initialized_ddqn_baseline as old

ARMS=('centered_linear_bc_ddqn','old_softplus_bc_ddqn',
      'centered_softplus_bc_ddqn','uncentered_linear_bc_ddqn',
      'centered_linear_bc_only','centered_linear_demo_no_bc','centered_linear_no_demo_no_bc')


class Critic(corrected.CenteredLinearDDQN):
    def __init__(self,arm):
        if arm not in ARMS:raise ValueError('unknown comparison arm')
        super().__init__();self.arm=arm

    def value(self,p,m,v,a):
        if self.arm=='old_softplus_bc_ddqn':
            # Exactly the preserved baseline's value formula and summation order.
            return self.student.value_from_tables(p,m,v,a)
        if self.arm=='centered_softplus_bc_ddqn':
            return corrected.centered_softplus_reference(p,m,v,a)
        if self.arm=='uncentered_linear_bc_ddqn':
            return corrected.uncentered_linear_reference(p,m,v,a)
        return corrected.joint_value(p,m,v,a)

    def forward(self,s,a):
        old.validate_states(s)
        p,m,v,_=self.tables(s)
        # Reference functions must reject invalid actions just like the primary.
        if a.dtype!=torch.long or a.shape!=(len(s),32) or torch.any(a<0) or torch.any(a>4):
            raise ValueError('invalid action')
        q=self.value(p,m,v,a)
        if not torch.isfinite(q).all():raise ValueError('nonfinite Q')
        return q

    def greedy(self,s,screened=None):
        old.validate_states(s,nonterminal=True)
        if screened is not None and screened is not True:raise ValueError('screen required')
        p,m,v,details=self.tables(s);mask=torch.from_numpy(old.allowed(s))
        a=torch.cat((p.masked_fill(~mask[:,:30],torch.inf).argmin(2),m.argmin(2)),1)
        q=self.value(p,m,v,a)
        if not torch.isfinite(q).all():raise ValueError('nonfinite Q')
        return a,q,details

    def scalar_diagnostics(self,s,a):
        p,m,v,_=self.tables(s)
        cost=p.gather(2,a[:,:30,None]).squeeze(2).sum(1)+m.gather(2,a[:,30:,None]).squeeze(2).sum(1)
        centered=cost-p.mean(2).sum(1)-m.mean(2).sum(1)
        if self.arm=='old_softplus_bc_ddqn':
            pre=v+cost;slope=torch.sigmoid(pre)
        elif self.arm=='centered_softplus_bc_ddqn':
            pre=-v+centered;slope=torch.sigmoid(pre)
        else:
            pre=self.value(p,m,v,a);slope=torch.ones_like(pre)
        return dict(scalar_input=pre,scalar_slope=slope,value_output=v,
                    placement_mean=p.mean(2),motion_mean=m.mean(2))


def analyze(per_fit,values):
    primary='centered_linear_bc_ddqn'
    policy=all(row['contrasts'][arm][metric]>=0
               for row in per_fit.values() for arm in ['old_softplus_bc_ddqn','centered_linear_bc_only']
               for metric in ['return_value','deadline'])
    policy=policy and all(sum(row['contrasts'][arm][metric] for row in per_fit.values())/len(per_fit)>0
               for arm in ['old_softplus_bc_ddqn','centered_linear_bc_only'] for metric in ['return_value','deadline'])
    value=all(values[f'{primary}/{fit}']['nrmse']<values[f'old_softplus_bc_ddqn/{fit}']['nrmse']
              and abs(values[f'{primary}/{fit}']['bias'])<abs(values[f'old_softplus_bc_ddqn/{fit}']['bias'])
              for fit in per_fit)
    return dict(policy_screen=bool(policy),value_screen=bool(value),promising=bool(policy and value))
