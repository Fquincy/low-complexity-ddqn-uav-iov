"""Single declared allocation/rate-reuse attempt; critic and ONNX unchanged."""
import numpy as np
import rq24_prototype as original
from controller_v3_fused_runtime import native


@native
def prepare(x,c):
    """Scalar geometry loops avoid per-link array temporaries; same equations."""
    bits=np.empty(30,np.float64);cycles=np.empty(30,np.float64);deadline=np.empty(30,np.float64)
    local=np.empty(30,np.float64);remote=np.zeros(5,np.float64)
    backlog=np.empty((30,4),np.float64);gain=np.empty((30,2),np.float64)
    for d in range(4):remote[d+1]=np.expm1(np.float64(x[270+d]))*.300
    for i in range(30):
        bits[i]=np.float64(x[8*i+5])*c[1];cycles[i]=np.float64(x[8*i+6])*500e6
        deadline[i]=np.float64(x[8*i+7])*.300;local[i]=np.expm1(np.float64(x[240+i]))*.300
        for d in range(4):backlog[i,d]=np.expm1(np.float64(x[274+8*i+d]))*c[1]
        for j in range(2):
            dx=(np.float64(x[8*i])-np.float64(x[526+3*j]))*c[0]
            dy=(np.float64(x[8*i+1])-np.float64(x[527+3*j]))*c[0]
            distance=np.sqrt(dx*dx+dy*dy+c[11]**2)
            elevation=np.degrees(np.arcsin(c[11]/distance))
            los=1/(1+c[12]*np.exp(-c[13]*(elevation-c[12])))
            loss=20*np.log10(4*np.pi*c[10]/299792458.)+20*np.log10(distance)+los*c[14]+(1-los)*c[15]
            gain[i,j]=10**(-loss/10)
    return bits,cycles,deadline,local,remote,backlog,gain,np.array([np.float64(x[528]),np.float64(x[531])])


@native
def features(x,c,k,radio):
    bits,cycles,deadline,local,remote,backlog,gain,battery=prepare(x,c)
    if np.any(deadline<=0) or np.any(bits<0) or np.any(cycles<0):raise ValueError('invalid task')
    n=len(bits);T=np.empty((n,5),np.float64);unary=np.empty((n,5),np.float64)
    reference=np.empty((n,2),np.float64);rates=np.empty((n,2),np.float64)
    for i in range(n):
        for d in range(2):
            rate=c[7]*np.log1p(c[8]*gain[i,d]/c[9])/np.log(2.)
            if rate<=0 or not np.isfinite(rate):raise ValueError('invalid physical rate')
            rates[i,d]=rate;reference[i,d]=bits[i]/rate
            if not np.isfinite(reference[i,d]):raise ValueError('invalid transmission duration')
        T[i,0]=local[i]+cycles[i]/c[2]
        wait=0.
        for d in range(4):
            rate=rates[i,d] if d<2 else c[16]
            wait+=backlog[i,d]/rate
        for d in range(1,5):
            upload=reference[i,d-1] if d<=2 else bits[i]/c[16]
            T[i,d]=wait+upload+remote[d]+cycles[i]/c[2+d]
        for d in range(5):
            unary[i,d]=original.phi(T[i,d],deadline[i])
            if not np.isfinite(unary[i,d]):raise ValueError('nonfinite unary')
    if radio:
        # Preserve the exact proven donor/unique-edge routine and edge ordering.
        ei,ej,ed=original.edge_pairs(gain,k);coeff=np.empty(len(ei),np.float64)
        for e in range(len(ei)):
            i=ei[e];j=ej[e];d=ed[e];u=d-1
            di=original.tx(bits[i],gain[i,u],c[8]*gain[j,u],c)-reference[i,u]
            dj=original.tx(bits[j],gain[j,u],c[8]*gain[i,u],c)-reference[j,u]
            coeff[e]=max(0.,original.phi(T[i,d]+di,deadline[i])-unary[i,d]+original.phi(T[j,d]+dj,deadline[j])-unary[j,d])
            if not np.isfinite(coeff[e]):raise ValueError('nonfinite edge')
    else:
        ei=np.empty(0,np.int64);ej=ei.copy();ed=ei.copy();coeff=np.empty(0,np.float64)
    return unary,ei,ej,ed,coeff


@native
def finish(raw,x,scale,c,mask,h,k,radio,physical_unary,sweep):
    unary,ei,ej,ed,coeff=features(x,c,k,radio)
    g,v,b,beta,eta,lam=original.heads(raw)
    if physical_unary:g+=unary/30.
    # Scalar extraction avoids a 240-element copy and intermediate broadcast.
    # Match original FP64 multiplication and extraction order for each element.
    w=np.empty((30,5),np.float64)
    for i in range(30):
        for d in range(5):w[i,d]=np.float64(x[8*i+6])*scale[d]
    q=np.zeros(5,np.float64)
    for d in range(4):q[d+1]=np.expm1(np.float64(x[270+d]))
    a,diag=original.decode(g,v,b,w,q,beta,eta,mask,x[:240].reshape(30,8)[:,6],ei,ej,ed,coeff,lam,sweep)
    return a[None].copy(),np.array([-h*original.softplus64(diag[1])],np.float64),diag


class Runtime:
    """Wrap an immutable original runtime; scratch is local, no state cache."""
    def __init__(self,original_runtime):
        self.session=original_runtime.session
        self.scale=original_runtime.scale.copy();self.c=original_runtime.c.copy()
        self.k=original_runtime.k;self.radio=original_runtime.radio
        self.physical=original_runtime.physical;self.sweep=original_runtime.sweep

    def values(self,state):
        x=np.asarray(state)
        if x.dtype!=np.float32 or x.shape!=(533,):raise ValueError('one FP32 state required')
        mask,h=original.validate_mask(x)
        raw=self.session.run(None,{'state':x[None]})[0][0]
        a,q,_=finish(raw,x,self.scale,self.c,mask,h,self.k,self.radio,self.physical,self.sweep)
        return a,q
