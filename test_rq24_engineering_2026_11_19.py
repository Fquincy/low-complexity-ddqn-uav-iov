"""Isolated synthetic RQ24 algebra, autograd, export and matched-cost gate."""
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import rq24_prototype as rq
import run_assignment_price_research_bound as old

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation/rq24_engineering_2026-11-19_01/fixture'
SEEDS=(906190001,906190011,906190021)
REFS=('screened_plain24','screened_wide128','unmasked_wide128','unmasked_compact64')


def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def synthetic(seed,count=32):
    rng=np.random.default_rng(seed);x=np.zeros((count,533),np.float32)
    for j,s in enumerate(x):
        tasks=s[:240].reshape(30,8);tasks[:,:2]=rng.uniform(0,1,(30,2))
        tasks[:,5]=rng.uniform(1/3,1,30);tasks[:,6]=rng.uniform(.6,1,30);tasks[:,7]=rng.uniform(.25/.3,1,30)
        s[240:270]=np.log1p(rng.uniform(0,.2,30)/.3)
        s[270:274]=np.log1p(rng.uniform(0,.1,4)/.3)
        s[274:514]=np.log1p(rng.uniform(0,.1,240))
        s[526:532]=[.1,.1,1,.9,.9,1];s[-1]=(j%40)/40
    return x


def direct_decoder(g,v,b,w,q,beta,eta,mask,cycles,ei,ej,ed,co,lam):
    a,_=rq.decode(g,v,b,w,q,beta,eta,mask,cycles,ei,ej,ed,co,lam,False)
    for i in np.argsort(-cycles,kind='stable'):
        previous=a[i];best=previous;value=rq.scalar(g,v,b,w,q,beta,eta,a,ei,ej,ed,co,lam)
        for d in range(g.shape[1]):
            if mask[i,d]:
                trial=a.copy();trial[i]=d;z=rq.scalar(g,v,b,w,q,beta,eta,trial,ei,ej,ed,co,lam)
                if z<value-1e-12:best=d;value=z
        a[i]=best
    return a


def algebra():
    rng=np.random.default_rng(906191001);cases=0
    for n in (3,7,30):
        gain=rng.uniform(1e-12,1e-8,(n,2))
        gain[0]=gain[1] # deterministic ties
        for k in (0,4,n-1):
            ei,ej,ed=rq.edge_pairs(gain,k);actual=set(zip(ei.tolist(),ej.tolist(),ed.tolist()));expected=set()
            for d in range(2):
                for i in range(n):
                    donors=sorted((j for j in range(n) if j!=i),key=lambda j:(-gain[j,d],j))[:k]
                    expected.update((min(i,j),max(i,j),d+1) for j in donors)
            assert actual==expected and len(actual)==len(ei) and len(ei)<=2*n*min(k,n-1)
            if k==n-1:assert len(ei)==n*(n-1)
            cases+=1
    for case in range(12):
        n=3;g=rng.normal(size=(n,5));v=rng.normal(size=(2,5));b=.2
        w=rng.uniform(.01,.2,(n,5));w[:,0]=0;q=rng.uniform(0,1,5);beta=rng.uniform(0,1,5);beta[0]=0
        eta=rng.dirichlet(np.ones(5));mask=np.ones((n,5),bool);mask[0,0]=False
        cycles=rng.uniform(0,1,n);ei,ej,ed=rq.edge_pairs(rng.uniform(0,1,(n,2)),2);co=rng.uniform(0,2,len(ei));lam=.7
        a,diag=rq.decode(g,v,b,w,q,beta,eta,mask,cycles,ei,ej,ed,co,lam,True)
        np.testing.assert_array_equal(a,direct_decoder(g,v,b,w,q,beta,eta,mask,cycles,ei,ej,ed,co,lam))
        assert diag[1]<=diag[0]+1e-10
        initial,_=rq.decode(g,v,b,w,q,beta,eta,mask,cycles,ei,ej,ed,co,lam,False)
        W=np.array([w[np.arange(n),initial[:n]]*(initial[:n]==d) for d in range(5)]).sum(1)
        baseline=rq.scalar(g,v,b,w,q,beta,eta,initial,ei,ej,ed,co,lam)
        for i in range(n):
            without=W.copy();old_d=initial[i];without[old_d]-=w[i,old_d]
            def insertion(d):
                radio=sum(lam*co[e]/n for e in range(len(ei)) if ed[e]==d and
                          ((ei[e]==i and initial[ej[e]]==d) or (ej[e]==i and initial[ei[e]]==d)))
                return g[i,d]+beta[d]*(2*(q[d]+eta[d]*without[d])*w[i,d]+w[i,d]**2)/(4*n)+radio
            for d in range(5):
                trial=initial.copy();trial[i]=d
                delta=rq.scalar(g,v,b,w,q,beta,eta,trial,ei,ej,ed,co,lam)-baseline
                assert np.isclose(delta,insertion(d)-insertion(old_d),rtol=1e-10,atol=1e-10)
        cases+=1
    # Full edge retention equals an independently enumerated dense physical proxy.
    x=synthetic(906191007,1)[0];c=rq.physical.constants()
    unary,ei,ej,ed,co=rq.features(x,c,29,True)
    bits,cycles,tau,local,remote,backlog,gain,battery=rq.physical.prepare(x,c)
    independent={}
    for d in (1,2):
        u=d-1
        for i in range(30):
            for j in range(i+1,30):
                def base_T(z):
                    wait=sum(backlog[z,k]/(c[7]*np.log1p(c[8]*gain[z,k]/c[9])/np.log(2) if k<2 else c[16]) for k in range(4))
                    return wait+rq.tx(bits[z],gain[z,u],0.,c)+remote[d]+cycles[z]/c[2+d]
                ti,tj=base_T(i),base_T(j)
                ki=rq.phi(ti+rq.tx(bits[i],gain[i,u],c[8]*gain[j,u],c)-rq.tx(bits[i],gain[i,u],0.,c),tau[i])-rq.phi(ti,tau[i])
                kj=rq.phi(tj+rq.tx(bits[j],gain[j,u],c[8]*gain[i,u],c)-rq.tx(bits[j],gain[j,u],0.,c),tau[j])-rq.phi(tj,tau[j])
                independent[(i,j,d)]=ki+kj
    assert len(co)==870
    for i,j,d,val in zip(ei,ej,ed,co):assert np.isclose(val,independent[(int(i),int(j),int(d))],rtol=1e-12,atol=1e-12)
    return dict(graph_cases=9,coordinate_cases=12,exact_delta_checks=180,dense_coefficients=870)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    plan=dict(kind='SYNTHETIC_ENGINEERING_NOT_POLICY_EVIDENCE',model_seeds=SEEDS,
        observation_seed=906192001,order_seed=906193001,states=32,rounds=3,warmups=16,
        research_task_seeds_used=False,new_simulator_steps=0,new_optimizer_updates=0,
        criteria=dict(storage_max_fraction_compact64=.5,median_max_ratio_each_wide=1.,p95_max_ms=1.),
        no_retiming=True,full_workflow=['algebra','autograd','export_equivalence','freeze','matched_timing','verification'],
        limitations='Untrained synthetic models/states; no policy quality, research acceptance or hardware-general speedup claim')
    write('PLAN.json',plan);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
             ('rq24_prototype.py','test_rq24_engineering_2026_11_19.py','RADIO_QUEUE_COMPACT_DDQN_DESIGN_BRIEF_2026-11-19.txt')}
    write('SOURCE_FREEZE.json',sources)
    tests=algebra();print('algebra passed',flush=True)
    x=synthetic(plan['observation_seed']);np.save(OUT/'synthetic_states.npy',x)
    runtimes={};costs={};equivalence=[];models={};cfg=copy.deepcopy(old.base.protocol()['training'])
    for fit,seed in enumerate(SEEDS):
        for arm in rq.ARMS+REFS:
            model=rq.Critic(arm,seed).eval() if arm in rq.ARMS else old.base.build(arm,dict(model_seed=seed,cfg=cfg))['online'].eval()
            path=OUT/f'{arm}_{fit}.onnx'
            runtime=rq.Runtime(model,path) if arm in rq.ARMS else old.base.references.Runtime(model,path)
            # Compare singleton eager and exported routes on every synthetic input.
            maxdiff=0.;disagreements=0
            for state in x:
                with torch.inference_mode():
                    eager=model.greedy(torch.from_numpy(state)[None]);ea,eq=eager[:2]
                da,dq=runtime.values(state)
                disagreements+=int(not np.array_equal(ea.numpy(),da))
                maxdiff=max(maxdiff,float(np.max(abs(eq.numpy()-dq))))
                np.testing.assert_allclose(eq.numpy(),dq,rtol=1e-5,atol=2e-4)
            assert disagreements==0,(arm,disagreements)
            models[(arm,fit)]=model
            equivalence.append(dict(arm=arm,fit=fit,states=len(x),action_disagreements=disagreements,max_Q_difference=maxdiff))
            if arm in ('rq24',)+REFS:runtimes[(arm,fit)]=runtime
            if arm in rq.ARMS:
                assert sum(p.numel() for p in model.parameters())==17091
                assert sum(p.numel()*p.element_size() for p in model.buffers())==20
                state=torch.from_numpy(x[:1]);act=model.greedy(state)[0];ns=state.clone();ns[:,-1]=state[:,-1]+1/40
                target=copy.deepcopy(model).requires_grad_(False)
                ts=state.clone();ts[:,-1]=39/40
                loss,parts=rq.adp.objective(model,target,(state,act,torch.tensor([-.4]),ns,torch.tensor([False])),(ts,act,torch.tensor([-.4])))
                loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                assert all(p.grad is None for p in target.parameters())
                terminal=state.clone();terminal[:,-1]=1
                assert model(terminal,act).item()==0
                try:runtime.values(terminal.numpy()[0])
                except ValueError:pass
                else:raise AssertionError('terminal decoder must reject')
            print(f'export/autograd passed: {arm} fit {fit}',flush=True)
    # Invalid/extreme observed-state checks precede timing.
    rt=runtimes[('rq24',0)]
    for bad in (x[0].astype(np.float64),np.zeros(532,np.float32)):
        try:rt.values(bad)
        except ValueError:pass
        else:raise AssertionError('invalid shape/dtype accepted')
    for field,value in ((0,np.nan),(7,0.),(270,1000.),(-1,.0123)):
        bad=x[0].copy();bad[field]=value
        try:rt.values(bad)
        except ValueError:pass
        else:raise AssertionError(('invalid input accepted',field))
    for battery in (0.,1.):
        edge=x[0].copy();edge[528]=battery;edge[531]=battery
        action,q=rt.values(edge);assert np.isfinite(q).all()
    graphs={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.onnx')}
    write('PRE_TIMING_FREEZE.json',dict(graphs=graphs,synthetic_states_sha256=hashlib.sha256((OUT/'synthetic_states.npy').read_bytes()).hexdigest(),sources=sources))
    rng=np.random.default_rng(plan['order_seed']);rows=[]
    for rnd in range(plan['rounds']):
        jobs=list(runtimes);rng.shuffle(jobs)
        for arm,fit in jobs:
            runtime=runtimes[(arm,fit)]
            for state in x[:plan['warmups']]:runtime.values(state)[0][0].copy()
            for index,state in enumerate(x):
                begin=time.perf_counter_ns();runtime.values(state)[0][0].copy();elapsed=time.perf_counter_ns()-begin
                rows.append(dict(round=rnd,arm=arm,fit=fit,state=index,ns=elapsed))
    write('timings.json',rows)
    screens=[]
    for fit in range(len(SEEDS)):
        per={}
        for arm in ('rq24',)+REFS:
            ns=np.array([r['ns'] for r in rows if r['arm']==arm and r['fit']==fit])
            model=models[(arm,fit)]
            per[arm]=dict(parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                median_ms=float(np.median(ns)/1e6),p95_ms=float(np.quantile(ns,.95)/1e6))
        costs[str(fit)]=per
        screens.append(dict(storage=per['rq24']['parameter_bytes']<=.5*per['unmasked_compact64']['parameter_bytes'],
            median=all(per['rq24']['median_ms']<=per[r]['median_ms'] for r in ('screened_wide128','unmasked_wide128')),
            p95=per['rq24']['p95_ms']<=1.))
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in sources.items())
    assert all(hashlib.sha256((OUT/name).read_bytes()).hexdigest()==sha for name,sha in graphs.items())
    result=dict(kind=plan['kind'],algebra=tests,autograd_arms=12,equivalence=equivalence,
        raw_timing_rows=len(rows),costs=costs,cost_screens=screens,
        engineering_correctness_passed=True,engineering_cost_passed=all(all(s.values()) for s in screens),
        research_released=False,quality_evidence=False,novelty_established=False,
        new_simulator_steps=0,new_optimizer_updates=0,research_task_seeds_used=False,word_thesis_changed=False)
    write('verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('equivalence','costs')}),flush=True)


if __name__=='__main__':main()
