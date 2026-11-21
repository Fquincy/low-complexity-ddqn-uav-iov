"""Isolated dueling-style correction. No simulator or research launcher."""
import argparse

import numpy as np
import torch
from torch.nn import functional as F

import heuristic_initialized_ddqn_baseline as previous


def joint_value(placement, motion, value, actions):
    """Inputs are already scaled; center over all five declared actions."""
    if (placement.ndim != 3 or placement.shape[1:] != (30,5)
            or motion.shape != (len(placement),2,5) or value.shape != (len(placement),)):
        raise ValueError('invalid branch/value shape')
    if (actions.shape != (len(placement),32) or actions.dtype != torch.long
            or torch.any(actions < 0) or torch.any(actions >= 5)):
        raise ValueError('invalid joint action')
    return (value
            -(placement.gather(2,actions[:,:30,None]).squeeze(2)-placement.mean(2)).sum(1)
            -(motion.gather(2,actions[:,30:,None]).squeeze(2)-motion.mean(2)).sum(1))


def centered_softplus_reference(placement,motion,value,actions):
    """Equation C only: same centered scalar with a saturating transform."""
    return -F.softplus(-joint_value(placement,motion,value,actions))


def uncentered_linear_reference(placement,motion,value,actions):
    """Equation D only: no centering, same signed state-value convention."""
    return value-placement.gather(2,actions[:,:30,None]).squeeze(2).sum(1)-motion.gather(2,actions[:,30:,None]).squeeze(2).sum(1)


class CenteredLinearDDQN(previous.TeacherInitializedCompactDDQN):
    def forward(self,states,actions):
        previous.validate_states(states)
        p,m,v,_=self.tables(states)
        q=joint_value(p,m,v,actions)
        if not torch.isfinite(q).all():raise ValueError('nonfinite scalar value')
        return q

    def greedy(self,states,screened=None):
        previous.validate_states(states,nonterminal=True)
        if screened is not None and screened is not True:
            raise ValueError('prototype requires inherited screen')
        p,m,v,details=self.tables(states)
        mask=torch.from_numpy(previous.allowed(states))
        actions=torch.cat((p.masked_fill(~mask[:,:30],torch.inf).argmin(2),m.argmin(2)),1)
        q=joint_value(p,m,v,actions)
        if not torch.isfinite(q).all():raise ValueError('nonfinite scalar value')
        return actions,q,details


imitation_loss=previous.imitation_loss
ddqn_loss=previous.ddqn_loss
deploy_actions=previous.deploy_actions


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Prototype only; research execution disabled')
    parser.add_argument('--mode',choices=['development','pilot','final'],required=True)
    parser.add_argument('--confirmation')
    parser.parse_args()
    raise SystemExit('Research modes disabled: isolated mechanics prototype only')
