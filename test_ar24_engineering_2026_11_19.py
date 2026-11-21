"""AR24 engineering-only algebra/autograd/ONNX and one matched synthetic gate."""
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

import ar24_prototype as ar
import test_rq24_engineering_2026_11_19 as synthetic_source
import run_assignment_price_research_bound as old

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation/ar24_engineering_2026-11-19_01'
SEEDS=(906196101,906196111,906196121)
REFS=('screened_plain24','screened_wide128','unmasked_wide128','unmasked_compact64')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def scalar_numpy(g,v,b,w,q,beta,eta,x,y,lam,a,radio,cpu):
    n=len(g);value=b+sum(g[i,a[i]] for i in range(n))+sum(v[j,a[n+j]] for j in range(len(v)))
    if cpu:
        for d in range(5):
            chosen=[w[i,d] for i in range(n) if a[i]==d];W=sum(chosen);S=sum(z*z for z in chosen)
            value+=beta[d]*(2*q[d]*W+eta[d]*W*W+(1-eta[d])*S)/(4*n)
    if radio:
        for d in (1,2):
            value+=lam[d-1]*sum(x[i,d-1]*y[j,d-1]+x[j,d-1]*y[i,d-1]
                for i in range(n) for j in range(i+1,n) if a[i]==d and a[j]==d)/n
    return value


def direct_sweep(args,mask,cycles,radio,cpu):
    g,v,b,w,q,beta,eta,x,y,lam=args
    a,_=ar.decode(*args,mask,cycles,radio,cpu,False)
    for i in np.argsort(-cycles,kind='stable'):
        previous=a[i];best=previous;value=scalar_numpy(*args,a,radio,cpu)
        for d in range(5):
            if mask[i,d]:
                trial=a.copy();trial[i]=d;z=scalar_numpy(*args,trial,radio,cpu)
                if z<value or (z==value and best!=previous and d<best):best=d;value=z
        a[i]=best
    return a


def algebra():
    rng=np.random.default_rng(906197101);counts=dict(random_systems=0,direct_delta_checks=0,prefix_pair_checks=0)
    for n in (1,3,7):
        for case in range(6):
            g=rng.normal(size=(n,5));v=rng.normal(size=(2,5));b=.13
            w=rng.uniform(0,.2,(n,5));w[:,0]=0.;q=rng.uniform(0,1,5)
            beta=rng.uniform(0,1,5);beta[0]=0;eta=rng.dirichlet(np.ones(5))
            x=rng.uniform(.1,10,(n,2));y=rng.uniform(.001,1,(n,2));lam=rng.uniform(0,1,2)
            mask=np.ones((n,5),bool);cycles=rng.uniform(0,1,n);args=(g,v,b,w,q,beta,eta,x,y,lam)
            for radio,cpu in ((True,True),(False,True),(True,False)):
                a,diag=ar.decode(*args,mask,cycles,radio,cpu,True)
                np.testing.assert_array_equal(a,direct_sweep(args,mask,cycles,radio,cpu))
                np.testing.assert_allclose(diag[1],scalar_numpy(*args,a,radio,cpu),rtol=1e-12,atol=1e-12)
                assert diag[1]<=diag[0]+1e-12*max(1.,abs(diag[0]))
                for assigned in (np.zeros(n+2,np.int64),np.ones(n+2,np.int64),a):
                    px=ar.radio_prefix(x,y,assigned,lam)
                    dense=sum(lam[d-1]*(x[i,d-1]*y[j,d-1]+x[j,d-1]*y[i,d-1])/n
                         for d in (1,2) for i in range(n) for j in range(i+1,n) if assigned[i]==d and assigned[j]==d)
                    np.testing.assert_allclose(px,dense,rtol=1e-12,atol=1e-12);counts['prefix_pair_checks']+=1
                for i in range(n):
                    W=np.zeros(5);X=np.zeros(2);Y=np.zeros(2)
                    for j in range(n):
                        if j==i:continue
                        d=a[j];W[d]+=w[j,d]
                        if d in (1,2):X[d-1]+=x[j,d-1];Y[d-1]+=y[j,d-1]
                    base=scalar_numpy(*args,a,radio,cpu)
                    old_m=ar.marginal(i,a[i],W,X,Y,g,w,q,beta,eta,x,y,lam,radio,cpu)
                    for d in range(5):
                        trial=a.copy();trial[i]=d
                        direct=scalar_numpy(*args,trial,radio,cpu)-base
                        cached=ar.marginal(i,d,W,X,Y,g,w,q,beta,eta,x,y,lam,radio,cpu)-old_m
                        np.testing.assert_allclose(cached,direct,rtol=1e-12,atol=1e-12);counts['direct_delta_checks']+=1
                counts['random_systems']+=1
    # Exact-zero tie case: lowest-ID initialization and movement; stable old-choice ties.
    n=3;g=np.zeros((n,5));v=np.zeros((2,5));w=np.zeros((n,5));z=np.zeros(5)
    x=np.ones((n,2));y=x.copy();mask=np.ones((n,5),bool);mask[0,0]=False
    args=(g,v,0.,w,z,z,np.full(5,.2),x,y,np.zeros(2))
    a,diag=ar.decode(*args,mask,np.ones(n),True,True,True)
    np.testing.assert_array_equal(a,[1,0,0,0,0]);assert diag[0]==diag[1]==0
    # Large range without subtractive cancellation, empty/singleton destinations.
    x=np.array([[1e12,1.],[1.,1e-12],[1e6,1e-6]]);y=1/x;lam=np.ones(2)
    for a in (np.array([1,1,1,0,0]),np.array([1,2,0,0,0]),np.zeros(5,np.int64)):
        value=ar.radio_prefix(x,y,a,lam)
        dense=sum((x[i,d-1]*y[j,d-1]+x[j,d-1]*y[i,d-1])/3 for d in (1,2) for i in range(3) for j in range(i+1,3) if a[i]==d and a[j]==d)
        np.testing.assert_allclose(value,dense,rtol=1e-12,atol=1e-12)
        assert value>=0
    return counts


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    plan=dict(kind='AR24_SYNTHETIC_ENGINEERING_NOT_RESEARCH',model_seeds=SEEDS,observation_seed=906198101,
        algebra_seed=906197101,order_seed=906199101,states=32,rounds=3,warmups=16,
        stress_cases=['tied_geometry_cycles','coincident_geometry','zero_battery','full_battery','task_extremes'],
        fp64_rtol=1e-12,fp64_atol=1e-12,near_tie_discrepancy_blocks_timing=True,
        timed_paths=['ar24',*REFS],raw_readings=1440,
        acceptance=dict(storage_fraction_compact64_max=.5,median_each_wide_ratio_max=1.,p95_ms_max=1.),
        research_task_seeds_used=False,new_simulator_steps=0,new_optimizer_updates=0,no_favorable_retiming=True)
    write('PLAN.json',plan);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    source_names=('ar24_prototype.py','test_ar24_engineering_2026_11_19.py','AGGREGATE_RADIO_DDQN_PROSPECTIVE_DESIGN_2026-11-19.txt')
    hashes={name:sha(ROOT/name) for name in source_names};write('SOURCE_FREEZE.json',hashes)
    x=synthetic_source.synthetic(plan['observation_seed']);extra=[]
    for kind in plan['stress_cases']:
        s=x[0].copy();tasks=s[:240].reshape(30,8)
        if kind=='tied_geometry_cycles':tasks[:,:2]=.5;tasks[:,6]=.8
        if kind=='coincident_geometry':tasks[:,:2]=s[526:528];s[529:531]=s[526:528]
        if kind=='zero_battery':s[528]=s[531]=0
        if kind=='full_battery':s[528]=s[531]=1
        if kind=='task_extremes':tasks[:,5]=np.tile([np.float32(1/3),1],15);tasks[:,7]=np.tile([np.float32(.25/.3),1],15);tasks[:,6]=np.tile([.6,1],15)
        extra.append(s)
    full=np.concatenate([x,np.stack(extra)])
    np.save(OUT/'synthetic_states.npy',x);np.save(OUT/'synthetic_stress_states.npy',np.stack(extra))
    counts=dict(eager_export_states=0,direct_decoder_states=0,autograd_cases=0,terminal_checks=0)
    runtimes={};costs={};checks=[];setup_start=time.perf_counter_ns()
    try:
        tests=algebra();print('algebra passed',flush=True)
        for fit,seed in enumerate(SEEDS):
            for arm in ar.ARMS:
                model=ar.Critic(arm,seed).eval();before={k:v.clone() for k,v in model.state_dict().items()}
                assert sum(p.numel() for p in model.parameters())==17116
                rt=ar.Runtime(model,OUT/f'{arm}_{fit}.onnx');maxdiff=0.
                for state in full:
                    s=torch.from_numpy(state)[None]
                    with torch.no_grad():a,q=model.greedy(s)
                    da,dq=rt.values(state);np.testing.assert_array_equal(a.numpy(),da)
                    np.testing.assert_allclose(q.numpy(),dq,rtol=1e-5,atol=2e-4)
                    maxdiff=max(maxdiff,float(abs(q.item()-dq[0])));counts['eager_export_states']+=1
                    # Direct scalar recomputation tests cached decisions on the EAGER heads.
                    # Its FP64 scalar is independent of ONNX FP32 rounding above.
                    with torch.no_grad():raw=model.net(s)[0].numpy()
                    g,v,b,beta,eta,lam=ar.heads(raw)
                    radio,deadline,homogeneous,cpu,sweep=model.flags
                    xx,yy,w,qq=ar.features(state,rt.scale,deadline,homogeneous);mask,h=ar.prior.validate_mask(state)
                    args=(g,v,b,w,qq,beta,eta,xx,yy,lam)
                    if sweep:
                        direct=direct_sweep(args,mask,state[:240].reshape(30,8)[:,6],radio,cpu)
                        np.testing.assert_array_equal(a.numpy()[0],direct);counts['direct_decoder_states']+=1
                    np.testing.assert_allclose(q.item(),-h*ar.prior.softplus64(scalar_numpy(*args,a.numpy()[0],radio,cpu)),rtol=1e-12,atol=1e-12)
                    assert mask[np.arange(30),da[0,:30]].all()
                s=torch.from_numpy(x[:2]);act=model.greedy(s)[0];ns=s.clone();ns[:,-1]+=1/40
                target=copy.deepcopy(model).requires_grad_(False)
                terminal=s.clone();terminal[:,-1]=39/40
                loss,parts=ar.prior.adp.objective(model,target,(s,act,torch.full((2,),-.4),ns,torch.zeros(2,dtype=torch.bool)),
                    (terminal,act,torch.full((2,),-.4)))
                loss.backward();assert torch.isfinite(loss)
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                assert all(p.grad is None for p in target.parameters())
                assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items());counts['autograd_cases']+=1
                boundary=s.clone();boundary[:,-1]=1
                assert torch.equal(model(boundary,act),torch.zeros(2,dtype=torch.float64))
                y=ar.prior.adp.targets(model,target,torch.tensor([-.4,-.2],dtype=torch.float64),boundary,torch.ones(2,dtype=torch.bool))
                np.testing.assert_array_equal(y.numpy(),[-.4,-.2]);counts['terminal_checks']+=2
                held=rt.values(x[0]);holdcopy=(held[0].copy(),held[1].copy())
                for state in (x[-1],full[-1],x[0]):rt.values(state)
                np.testing.assert_array_equal(held[0],holdcopy[0]);np.testing.assert_array_equal(held[1],holdcopy[1])
                for field,value in ((0,np.nan),(0,-.1),(5,2.),(7,0.),(270,1000.),(-1,1.),(-1,.0123)):
                    bad=x[0].copy();bad[field]=value
                    try:rt.values(bad)
                    except ValueError:pass
                    else:raise AssertionError(('invalid accepted',field))
                for bad in (x[0].astype(np.float64),np.zeros(532,np.float32)):
                    try:rt.values(bad)
                    except ValueError:pass
                    else:raise AssertionError('bad shape/dtype')
                checks.append(dict(arm=arm,fit=fit,states=len(full),action_disagreements=0,max_eager_export_Q_difference=maxdiff))
                if arm=='ar24':runtimes[(arm,fit)]=rt
                print(f'correctness passed: {arm} fit {fit}',flush=True)
            for arm in REFS:
                agent=old.base.build(arm,dict(model_seed=seed,cfg=copy.deepcopy(old.base.protocol()['training'])))
                runtimes[(arm,fit)]=old.base.references.Runtime(agent['online'].eval(),OUT/f'{arm}_{fit}.onnx')
                costs[(arm,fit)]=sum(p.numel()*p.element_size() for p in agent['online'].parameters())
            costs[('ar24',fit)]=68464
    except Exception as e:
        write('correctness_failure.json',dict(error=str(e),counts=counts,timing_started=False,thresholds_unchanged=True))
        raise
    write('correctness.json',dict(passed=True,counts=counts,algebra=tests,checks=checks,setup_ns=time.perf_counter_ns()-setup_start,
        numerical_scope='FP64 same-raw/delta rtol=atol1e-12; exported FP32 neural Q rtol1e-5/atol2e-4 engineering-only; actions exact'))
    rng=np.random.default_rng(plan['order_seed']);orders=[]
    for rnd in range(3):order=list(runtimes);rng.shuffle(order);orders.append(order)
    graphs={p.name:sha(p) for p in OUT.glob('*.onnx')}
    write('PRE_TIMING_FREEZE.json',dict(sources=hashes,graphs=graphs,orders=orders,plan_sha256=sha(OUT/'PLAN.json'),states_sha256=sha(OUT/'synthetic_states.npy')))
    rows=[]
    for rnd,order in enumerate(orders):
        for arm,fit in order:
            rt=runtimes[(arm,fit)]
            for state in x[:16]:rt.values(state)[0][0].copy()
            for index,state in enumerate(x):
                start=time.perf_counter_ns();rt.values(state)[0][0].copy();elapsed=time.perf_counter_ns()-start
                rows.append(dict(round=rnd,arm=arm,fit=fit,state=index,ns=elapsed))
    write('timings.json',rows);summaries={};screens=[]
    for fit in range(3):
        per={}
        for arm in plan['timed_paths']:
            ns=[r['ns'] for r in rows if r['arm']==arm and r['fit']==fit]
            per[arm]=dict(median_ms=float(np.median(ns)/1e6),p95_ms=float(np.quantile(ns,.95)/1e6),parameter_bytes=costs[(arm,fit)])
        summaries[str(fit)]=per;c=per['ar24']
        screens.append(dict(storage=c['parameter_bytes']<=.5*per['unmasked_compact64']['parameter_bytes'],
            median=all(c['median_ms']<=per[a]['median_ms'] for a in ('screened_wide128','unmasked_wide128')),p95=c['p95_ms']<=1.))
    assert all(sha(ROOT/p)==h for p,h in hashes.items());assert all(sha(OUT/p)==h for p,h in graphs.items())
    result=dict(kind=plan['kind'],correctness_passed=True,cost_gate_passed=all(all(s.values()) for s in screens),
        cost_screens=screens,costs=summaries,raw_readings=len(rows),parameter_bytes=68464,buffer_bytes=20,
        serialized_primary_graph_bytes=[(OUT/f'ar24_{fit}.onnx').stat().st_size for fit in range(3)],
        research_release=False,quality_evidence=False,new_simulator_steps=0,new_optimizer_updates=0,word_thesis_changed=False)
    write('verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
