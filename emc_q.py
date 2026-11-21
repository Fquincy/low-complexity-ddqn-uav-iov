"""Restricted movement-conditioned Q with exact model argmax; no trainer."""
import torch
from torch import nn


def decode_tables(unary,motion,factors,gates,embeddings):
    """Supports small dimensions for exhaustive mechanics checks as well."""
    batch,vehicles,choices=unary.shape
    movements=motion.shape[1]
    context=embeddings[0,:,None,:]+embeddings[1,None,:,:]
    conditional=unary[:,None,None,:,:]+torch.einsum('bvdr,br,ijr->bijvd',factors,gates,context)
    values,destinations=conditional.max(-1)
    scores=motion+values.sum(-1)
    pair=scores.flatten(1).argmax(1)
    b1,b2=pair//movements,pair%movements
    rows=torch.arange(batch,device=unary.device)
    action=torch.cat((destinations[rows,b1,b2],b1[:,None],b2[:,None]),1)
    return action,scores[rows,b1,b2]


def chosen_tables(unary,motion,factors,gates,embeddings,actions):
    batch,vehicles,_=unary.shape
    if actions.shape!=(batch,vehicles+2) or actions.dtype!=torch.long:
        raise ValueError('int64 batch x(vehicles+2) actions required')
    if torch.any(actions<0) or torch.any(actions[:,:vehicles]>=unary.shape[2]) or torch.any(actions[:,vehicles:]>=motion.shape[1]):
        raise ValueError('illegal action')
    rows=torch.arange(batch,device=unary.device)
    vi=torch.arange(vehicles,device=unary.device)[None,:]
    d=actions[:,:vehicles];b1,b2=actions[:,-2],actions[:,-1]
    context=(embeddings[0,b1]+embeddings[1,b2])*gates
    interaction=(factors[rows[:,None],vi,d]*context[:,None,:]).sum(-1)
    return motion[rows,b1,b2]+(unary[rows[:,None],vi,d]+interaction).sum(1)


class EMCQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(533,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.head=nn.Linear(64,477)
        self.movement_embeddings=nn.Parameter(torch.empty(2,5,2))
        nn.init.normal_(self.movement_embeddings,std=.1)

    def tables(self,states):
        if states.ndim!=2 or states.shape[1]!=533:raise ValueError('batch x533 required')
        z=self.head(self.encoder(states));b=len(states)
        return z[:,:150].reshape(b,30,5),z[:,150:175].reshape(b,5,5),z[:,175:475].reshape(b,30,5,2),z[:,475:]

    def forward(self,states,actions):
        return chosen_tables(*self.tables(states),self.movement_embeddings,actions)

    @torch.no_grad()
    def greedy(self,states):
        return decode_tables(*self.tables(states),self.movement_embeddings)


def frozen_target(online):
    import copy
    target=copy.deepcopy(online).eval()
    target.requires_grad_(False)
    return target


@torch.no_grad()
def ddqn_targets(online,target,rewards,next_states,dones):
    if rewards.ndim!=1 or dones.shape!=rewards.shape or next_states.shape!=(len(rewards),533):raise ValueError('batch shapes')
    if not torch.all((dones==0)|(dones==1)):raise ValueError('binary masks')
    result=rewards.detach().clone();active=~dones.bool()
    if active.any():
        actions,_=online.greedy(next_states[active])
        result[active]+=target(next_states[active],actions)
    return result
