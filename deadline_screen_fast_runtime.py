"""Single validation/interval pass; unchanged PyTorch FP32 cores and native decoding."""
import numpy as np,torch
import deadline_screen_prototype as original
from controller_v3_fused_runtime import native
import controller_v3_fused_runtime as decoder

@native
def validate_and_screen(x):
    if x.shape!=(533,):raise ValueError('533 FP32 observations required')
    for v in x:
        if not np.isfinite(v):raise ValueError('finite observations required')
    t=np.float32(x[-1]*np.float32(40.));slot=round(t)
    if abs(t-slot)>1e-4 or slot<0 or slot>=40:raise ValueError('nonterminal grid time required')
    excluded=np.empty(30,np.bool_)
    for i in range(30):
        c=x[8*i+6];d=x[8*i+7]
        if c<0 or c>1 or d<0 or d>1:raise ValueError('normalized task features required')
        cl=max(0.,np.float64(np.nextafter(c,np.float32(-np.inf))))
        dh=min(1.,np.float64(np.nextafter(d,np.float32(np.inf))))
        cl=np.nextafter(cl*500e6,-np.inf)
        dh=np.nextafter(dh*.300,np.inf)
        service=np.nextafter(cl/1.5e9,-np.inf)
        deadline=np.nextafter(dh+1e-10,np.inf)
        excluded[i]=service>deadline
    return excluded,40-slot

@native
def branch_values(raw,excluded,mode):
    # Input cost/value scores already computed in original PyTorch arithmetic.
    a=np.empty(32,np.int64);total=np.float32(0.)
    for i in range(32):
        best=-1;score=np.float32(0.)
        for d in range(5):
            if i<30 and d==0 and excluded[i]:continue
            value=raw[i*5+d]
            if best<0 or value>score:best=d;score=value
        a[i]=best;total=np.float32(total+score)
    return a,np.float32(total/np.float32(32.)) if mode==1 else total

class Runtime:
    def __init__(self,model):
        if not model.screened:raise ValueError('screened model required')
        self.arm=model.base_arm;self.scaled=model.scaled
        core=model.model.model if model.arm in original.existing.CORRECTED else model.model
        self.model=core.eval()
    @torch.inference_mode()
    def values(self,state):
        x=np.asarray(state)
        if x.dtype!=np.float32 or x.ndim!=1:raise ValueError('one FP32 state required')
        excluded,h=validate_and_screen(x)
        tensor=torch.from_numpy(x)[None]
        if self.arm in original.NEW_ARMS:
            raw=self.model.net(tensor).numpy()[0].copy()
            for i in np.flatnonzero(excluded):raw[5*i]=np.float32(np.inf)
            if self.arm=='plain24_nonpositive':a,q=decoder.flat_decode(raw,2)
            else:a,q=decoder.v3_decode(raw,x,self.arm=='amortized_workload_price24')
        elif self.arm=='additive64_nonpositive':
            raw=self.model(tensor).numpy()[0].reshape(-1).copy()
            for i in np.flatnonzero(excluded):raw[5*i]=np.float32(np.inf)
            a,q=decoder.flat_decode(raw,1)
        else:
            raw=self.model(tensor).numpy()[0].reshape(-1)
            a,q=branch_values(raw,excluded,1 if self.arm=='bdq64_uniform_adapted' else 0)
        return a[None],np.array([np.float32(q)*np.float32(h if self.scaled else 1.)],np.float32)
