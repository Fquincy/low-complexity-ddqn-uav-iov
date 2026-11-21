"""Single-pass interaction-aware action selector prototype; no trainer."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

PHYSICAL_SIZE=533
COMPONENTS=32
CHOICES=5


class SPIDDQN(nn.Module):
    def __init__(self, width=96, rank=8, embedding=16):
        super().__init__()
        self.width=width; self.rank=rank; self.embedding=embedding
        self.encoder=nn.Sequential(nn.Linear(PHYSICAL_SIZE,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU())
        self.component_head=nn.Linear(width,COMPONENTS*CHOICES)
        self.action_embedding=nn.Parameter(torch.empty(COMPONENTS,CHOICES,embedding))
        nn.init.normal_(self.action_embedding,std=0.02)
        self.interaction=nn.Sequential(nn.Linear(embedding,rank),nn.Tanh())
        self.joint_scorer=nn.Sequential(nn.Linear(width+embedding+rank,32),nn.ReLU(),nn.Linear(32,1))

    def encode(self,states):
        if states.ndim!=2 or states.shape[1]!=PHYSICAL_SIZE: raise ValueError('batch x533 required')
        return self.encoder(states)

    def logits(self,states):
        return self.component_head(self.encode(states)).view(-1,COMPONENTS,CHOICES)

    def score_joint(self,embedding,actions):
        if embedding.ndim!=2 or actions.ndim!=2 or embedding.shape[0]!=actions.shape[0] or actions.shape[1]!=COMPONENTS:
            raise ValueError('embedding batch and action batch mismatch')
        component_index=torch.arange(COMPONENTS,device=actions.device)[None,:]
        selected=self.action_embedding[component_index,actions]
        pooled=selected.mean(1)
        interaction=self.interaction(pooled)
        return self.joint_scorer(torch.cat((embedding,pooled,interaction),1)).squeeze(1)

    def joint_values(self,states,actions):
        return self.score_joint(self.encode(states),actions)

    def parameter_bytes(self):
        return sum(p.numel()*p.element_size() for p in self.parameters())


@torch.inference_mode()
def select(model,state,top_k=2,beam_width=8):
    if not 1<=top_k<=CHOICES or beam_width<1: raise ValueError('invalid decoding limits')
    x=torch.as_tensor(np.asarray(state,dtype=np.float32)).reshape(1,PHYSICAL_SIZE)
    embedding=model.encode(x)
    logits=model.component_head(embedding).view(1,COMPONENTS,CHOICES)[0]
    top_values,top_actions=logits.topk(top_k,dim=1)
    # Beam candidates are expanded using parallel head scores; joint scoring is
    # applied once to the bounded complete candidate set.
    beams=[(0.,np.empty(0,dtype=np.int64))]
    for component in range(COMPONENTS):
        expanded=[]
        for additive_score,vector in beams:
            for j in range(top_k):
                choice=int(top_actions[component,j])
                expanded.append((additive_score+float(top_values[component,j]),np.append(vector,choice)))
        expanded.sort(key=lambda item:item[0],reverse=True)
        beams=expanded[:beam_width]
    actions=torch.as_tensor(np.stack([v for _,v in beams]),dtype=torch.long)
    repeated=embedding.expand(len(beams),-1)
    joint=model.score_joint(repeated,actions)
    best=int(torch.argmax(joint))
    return beams[best][1].copy(),dict(candidate_count=len(beams),top_k=top_k,beam_width=beam_width,
                                      additive_scores=np.array([x[0] for x in beams]),joint_scores=joint.cpu().numpy(),
                                      encoder_calls=1,action_logits_shape=tuple(logits.shape))
