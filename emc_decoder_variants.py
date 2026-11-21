"""Equivalent exact decoders; no change to learned parameters or training loss."""
import torch


def finish(unary,motion,interaction):
    batch,vehicles,choices=unary.shape
    movements=motion.shape[1]
    values,destinations=(interaction+unary[:,None,:,:]).max(-1)
    scores=motion.flatten(1)+values.sum(-1)
    pair=scores.argmax(1)
    rows=torch.arange(batch,device=unary.device)
    return torch.cat((destinations[rows,pair],(pair//movements)[:,None],(pair%movements)[:,None]),1),scores[rows,pair]


def decode_matmul(unary,motion,factors,gates,embeddings):
    b,v,d=unary.shape
    context=(embeddings[0,:,None,:]+embeddings[1,None,:,:]).flatten(0,1)
    # Flatten the destination axis; contract only the final rank dimension.
    term=torch.matmul((factors*gates[:,None,None,:]).reshape(b,v*d,-1),context.T)
    interaction=term.transpose(1,2).reshape(b,-1,v,d)
    return finish(unary,motion,interaction)


def decode_rank2(unary,motion,factors,gates,embeddings):
    if factors.shape[-1]!=2:raise ValueError('rank2 only')
    context=(embeddings[0,:,None,:]+embeddings[1,None,:,:]).flatten(0,1)
    interaction=(factors[:,None,:,:,0]*(gates[:,0,None]*context[None,:,0])[:,:,None,None]
                +factors[:,None,:,:,1]*(gates[:,1,None]*context[None,:,1])[:,:,None,None])
    return finish(unary,motion,interaction)
