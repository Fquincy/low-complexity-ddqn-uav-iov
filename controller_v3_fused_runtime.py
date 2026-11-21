"""Same critics/weights: ONNX neural cores and equally applied native decoders."""
import os,sys,time,warnings
from pathlib import Path
DEPS=Path(os.environ.get('TEMP',str(Path.home()/'AppData/Local/Temp')))/'controller_v3_fused_deps_2026_11_05'
sys.path.insert(0,str(DEPS))
import numba,llvmlite,numpy as np,torch,onnx
from numba import njit
import run_controller_v3_engineering as g

def native(fn):return njit(fastmath=False,parallel=False,boundscheck=True,cache=False)(fn)

@native
def softplus(x):
    x=np.float32(x)
    if x>np.float32(20.):return x
    return np.float32(np.log1p(np.exp(x)))

@native
def flat_decode(raw,mode):
    # 0 wide max; 1 compact min/outer-bound; 2 plain24 weighted min;
    # 3 BDQ branch scores supplied after centering.
    a=np.empty(32,np.int64);total=np.float32(0.)
    placement=np.float32(0.);movement=np.float32(0.)
    for i in range(32):
        best=0;score=np.float32(raw[i*5])
        if mode==2:score=np.float32(score/np.float32(30. if i<30 else 2.))
        for d in range(1,5):
            value=np.float32(raw[i*5+d])
            if mode==2:value=np.float32(value/np.float32(30. if i<30 else 2.))
            if (mode==0 or mode==3) and value>score or (mode==1 or mode==2) and value<score:
                score=value;best=d
        a[i]=best;total=np.float32(total+score)
        if i<30:placement=np.float32(placement+score)
        else:movement=np.float32(movement+score)
    if mode==1:total=-softplus(total)
    elif mode==2:total=-softplus(np.float32(np.float32(raw[160]+placement)+movement))
    elif mode==3:total=np.float32(total/np.float32(32.))
    return a,total

@native
def v3_decode(raw,state,enabled):
    beta=np.zeros(5,np.float32);frac=np.empty(5,np.float32);scale=np.array([0.,500e6/10e9/.3,500e6/10e9/.3,500e6/20e9/.3,500e6/20e9/.3],np.float32)
    w=np.empty((30,5),np.float32);sums=np.zeros(5,np.float32);L=np.zeros(5,np.float32)
    maxlog=raw[165]
    for d in range(1,5):
        if raw[165+d]>maxlog:maxlog=raw[165+d]
    denom=np.float32(0.)
    for d in range(5):frac[d]=np.exp(np.float32(raw[165+d]-maxlog));denom=np.float32(denom+frac[d])
    for d in range(5):frac[d]=np.float32(frac[d]/denom)
    for d in range(1,5):beta[d]=softplus(raw[160+d]);L[d]=np.float32(np.expm1(state[269+d]))
    for i in range(30):
        for d in range(5):w[i,d]=np.float32(state[i*8+6]*scale[d]);sums[d]=np.float32(sums[d]+w[i,d])
    a=np.empty(32,np.int64);placement=np.float32(0.);movement=np.float32(0.)
    for i in range(32):
        best=0;bestvalue=np.float32(0.)
        for d in range(5):
            value=np.float32(softplus(raw[i*5+d])/np.float32(30. if i<30 else 2.))
            if enabled and i<30:
                other=max(np.float32(0.),np.float32(sums[d]-w[i,d]))
                expected=np.float32(L[d]+np.float32(frac[d]*other))
                term=np.float32(np.float32(np.float32(2.)*expected)*w[i,d])+np.float32(w[i,d]*w[i,d])
                price=np.float32(np.float32(beta[d]*term)/np.float32(120.))
                value=np.float32(value+price)
            if d==0 or value<bestvalue:best=d;bestvalue=value
        a[i]=best
        if i<30:placement=np.float32(placement+bestvalue)
        else:movement=np.float32(movement+bestvalue)
    return a,-np.float32(np.float32(softplus(raw[160])+placement)+movement)

@native
def bdq_decode(value,adv):
    raw=np.empty(160,np.float32)
    for i in range(32):
        mean=np.float32(0.)
        for d in range(5):mean=np.float32(mean+adv[i*5+d])
        mean=np.float32(mean/np.float32(5.))
        for d in range(5):raw[i*5+d]=np.float32(np.float32(value+adv[i*5+d])-mean)
    return flat_decode(raw,3)

@native
def emc_decode(raw,emb):
    best=np.float32(0.);answer=np.empty(32,np.int64)
    for b1 in range(5):
        for b2 in range(5):
            trial=np.empty(32,np.int64);total=np.float32(0.)
            for i in range(30):
                bestd=0;minimum=np.float32(0.)
                for d in range(5):
                    interaction=np.float32(0.)
                    for k in range(2):
                        # Match einsum selection order: factor*gate, then context.
                        fg=np.float32(raw[175+(i*5+d)*2+k]*raw[475+k])
                        interaction=np.float32(interaction+np.float32(fg*np.float32(emb[0,b1,k]+emb[1,b2,k])))
                    cost=np.float32(raw[i*5+d]+interaction)
                    if d==0 or cost<minimum:minimum=cost;bestd=d
                trial[i]=bestd;total=np.float32(total+minimum)
            cost=np.float32(raw[150+b1*5+b2]+total)
            if b1==0 and b2==0 or cost<best:
                best=cost;trial[30]=b1;trial[31]=b2;answer=trial.copy()
    # Chosen-value evaluation uses the original forward order.
    total=np.float32(0.)
    for i in range(30):
        inter=np.float32(0.)
        for k in range(2):
            context=np.float32(np.float32(emb[0,answer[30],k]+emb[1,answer[31],k])*raw[475+k])
            inter=np.float32(inter+np.float32(raw[175+(i*5+answer[i])*2+k]*context))
        total=np.float32(total+np.float32(raw[i*5+answer[i]]+inter))
    return answer,-softplus(np.float32(raw[150+answer[30]*5+answer[31]]+total))

@native
def assignment_load(a,w):
    A=np.zeros(5,np.float32)
    for i in range(len(a)):A[a[i]]=np.float32(A[a[i]]+w[i,a[i]])
    return A

@native
def price_table(a,w,L,beta):
    n=len(a);A=assignment_load(a,w);p=np.empty((n,5),np.float32)
    for i in range(n):
        for d in range(5):
            H=np.float32(np.float32(L[d]+A[d])-(w[i,d] if a[i]==d else np.float32(0.)))
            term=np.float32(np.float32(np.float32(2.)*H)*w[i,d])+np.float32(w[i,d]*w[i,d])
            p[i,d]=np.float32(np.float32(beta[d]*term)/np.float32(4*n))
    return p

@native
def score(a,u,w,L,beta,constant):
    A=assignment_load(a,w);penalty=np.float32(0.);unary=np.float32(0.)
    for d in range(5):
        term=np.float32(np.float32(np.float32(2.)*L[d])*A[d])+np.float32(A[d]*A[d])
        penalty=np.float32(penalty+np.float32(beta[d]*term))
    for i in range(len(a)):unary=np.float32(unary+u[i,a[i]])
    return np.float32(np.float32(constant+unary)+np.float32(penalty/np.float32(4*len(a))))

@native
def shared_decode(u,w,L,beta,motion,base,enabled,joint):
    n=len(u);a=np.empty(n,np.int64);moves=np.empty(2,np.int64);constant=base
    for j in range(2):
        moves[j]=np.argmin(motion[j]);constant=np.float32(constant+motion[j,moves[j]])
    for i in range(n):a[i]=np.argmin(u[i])
    current=score(a,u,w,L,beta,constant)
    if enabled:
        best=a.copy();bestcost=current
        for unused in range(2):
            prices=price_table(a,w,L,beta)
            for i in range(n):a[i]=np.argmin(u[i]+prices[i])
            cost=score(a,u,w,L,beta,constant)
            if cost<bestcost:bestcost=cost;best=a.copy()
        a=best;current=bestcost
        for unused in range(4):
            prices=price_table(a,w,L,beta);delta_min=np.float32(0.);index=0
            for i in range(n):
                old=np.float32(u[i,a[i]]+prices[i,a[i]])
                for d in range(5):
                    delta=np.float32(np.float32(u[i,d]+prices[i,d])-old)
                    if i==0 and d==0 or delta<delta_min:delta_min=delta;index=i*5+d
            trial=a.copy()
            if delta_min<np.float32(-1e-6):trial[index//5]=index%5
            cost=score(trial,u,w,L,beta,constant)
            if delta_min<np.float32(-1e-6) and cost<np.float32(current-np.float32(1e-6)):a=trial;current=cost
    result=np.empty(n+2,np.int64);result[:n]=a;result[n:]=moves
    return result,-softplus(current) if joint else -current

class Core(torch.nn.Module):
    def __init__(self,model,arm):super().__init__();self.model=model;self.arm=arm
    def forward(self,x):
        if self.arm in g.NEW_ARMS:return self.model.net(x)
        if self.arm in g.r.SHARED_ARMS:return self.model.tables(x)[:6]
        if self.arm=='emc_q_nonpositive':return self.model.head(self.model.encoder(x))
        if self.arm=='bdq64_uniform_adapted':
            h=self.model.trunk(x);return self.model.value(h),self.model.advantage(h)
        return self.model(x)

def export(model,arm,path):
    before=g.r.old.digest(model.state_dict())
    torch.onnx.register_custom_op_symbolic('aten::expm1',lambda graph,x:graph.op('Sub',graph.op('Exp',x),
        graph.op('Constant',value_t=torch.tensor(1.,dtype=torch.float32))),17)
    count=6 if arm in g.r.SHARED_ARMS else 2 if arm=='bdq64_uniform_adapted' else 1
    names=[f'core{i}' for i in range(count)];start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught,torch.inference_mode():
        warnings.simplefilter('always')
        torch.onnx.export(Core(model,arm),torch.zeros(1,533),str(path),opset_version=17,dynamo=False,
            input_names=['states'],output_names=names,do_constant_folding=True)
    duration=time.perf_counter()-start;onnx.checker.check_model(onnx.load(path))
    start=time.perf_counter();sess=g.c.session(path);startup=time.perf_counter()-start
    extra=model.movement_embeddings.detach().numpy().copy() if arm=='emc_q_nonpositive' else np.zeros((2,5,2),np.float32)
    assert g.r.old.digest(model.state_dict())==before
    return (sess,extra),dict(graph_sha256=g.r.old.sha(path),graph_bytes=path.stat().st_size,
        parameter_hash=before,parameters=sum(p.numel() for p in model.parameters()),parameter_bytes=g.r.parameter_bytes(model),
        export_seconds=duration,startup_seconds=startup,warnings=sorted({str(w.message) for w in caught}))

def values(runtime,arm,state):
    x=np.asarray(state,dtype=np.float32)
    if x.shape!=(533,) or not np.isfinite(x).all():raise ValueError('one finite533 observation required')
    sess,extra=runtime;raw=sess.run(None,{'states':x[None]})
    if arm in g.NEW_ARMS:
        if arm=='plain24_nonpositive':a,q=flat_decode(raw[0][0],2)
        else:a,q=v3_decode(raw[0][0],x,arm=='amortized_workload_price24')
    elif arm in g.r.SHARED_ARMS:a,q=shared_decode(*(v[0] for v in raw),arm!='shared_no_congestion',arm=='shared_joint_bound')
    elif arm=='emc_q_nonpositive':a,q=emc_decode(raw[0][0],extra)
    elif arm=='bdq64_uniform_adapted':a,q=bdq_decode(raw[0][0,0],raw[1][0])
    else:a,q=flat_decode(raw[0][0].reshape(-1),1 if arm=='additive64_nonpositive' else 0)
    return a[None],np.array([q],np.float32)

def action(runtime,arm,state):return values(runtime,arm,state)[0][0].copy()
