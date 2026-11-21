"""Saved-weight fast-route equivalence then one fixed timing experiment."""
import json,hashlib,shutil,time
from pathlib import Path
import numpy as np,torch
import deadline_screen_prototype as s
import deadline_screen_fast_runtime as fast
R=Path(__file__).resolve().parent;PRE=R/'validation/deadline_screen_engineering_2026-11-11_01';O=R/'validation/deadline_screen_fast_runtime_2026-11-11_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def check_screen():
    rng=np.random.default_rng(997310021);n=3334
    states=np.zeros((n,533),np.float32);task=states[:,:240].reshape(n,30,8)
    exact_c=rng.uniform(300e6,500e6,(n,30));exact_d=rng.uniform(.250,.300,(n,30))
    task[:,:,6]=exact_c/500e6;task[:,:,7]=exact_d/.3
    reference=s.allowed(states);actual=np.stack([~fast.validate_and_screen(x)[0] for x in states])
    np.testing.assert_array_equal(reference[:,:30,0],actual)
    assert np.all(exact_c[~actual]/1.5e9>exact_d[~actual]+1e-10)
    cases=0
    for d in np.linspace(.250,.300,1000):
        center=(d+1e-10)*1.5e9
        for c in [np.nextafter(center,-np.inf),center,np.nextafter(center,np.inf)]:
            x=states[0].copy();x[6]=np.float32(c/500e6);x[7]=np.float32(d/.3)
            a,h=fast.validate_and_screen(x);np.testing.assert_array_equal(~a,s.allowed(x[None])[0,:30,0]);assert not a[0];cases+=1
    bad=[]
    for field,val in [(0,np.nan),(532,1.),(532,.12345),(6,1.1),(7,-.1)]:
        x=states[0].copy();x[field]=val;bad.append(x)
    for x in bad:
        try:fast.validate_and_screen(x)
        except ValueError:pass
        else:raise AssertionError('invalid input admitted')
    return dict(random_task_cases=n*30,boundary_cases=cases,invalid_cases=len(bad),mask_exact=True)
def main():
    O.mkdir(exist_ok=False);s.existing.configure();s.existing.fused.numba.set_num_threads(1)
    for name,digest in read(PRE/'ARTIFACT_MANIFEST.json')['files'].items():assert sha(PRE/name)==digest,name
    freeze=read(PRE/'PRE_BENCHMARK_FREEZE.json');states=np.load(PRE/'observations.npz')['states'];models={};paths={}
    sources=['deadline_screen_fast_runtime.py','check_deadline_screen_fast_runtime.py','DEADLINE_SCREEN_RUNTIME_OPTIMIZATION_PROTOCOL_2026-11-11.json',*freeze['source_sha256']]
    for name in sources:
        dest=O/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/name,dest)
    shutil.copy2(PRE/'observations.npz',O/'observations.npz')
    (O/'models').mkdir()
    for key,meta in freeze['metadata'].items():
        model=s.Critic(meta['arm'],meta['seed']).eval();model.load_state_dict(torch.load(PRE/'models'/(key+'.pt'),weights_only=True,map_location='cpu'))
        shutil.copy2(PRE/'models'/(key+'.pt'),O/'models'/(key+'.pt'));models[key]=model
    write(O/'PRE_CHECK_FREEZE.json',dict(prior_freeze_sha256=sha(PRE/'PRE_BENCHMARK_FREEZE.json'),source_sha256={n:sha(R/n) for n in sources},
        observations_sha256=sha(O/'observations.npz'),metadata=freeze['metadata'],orders=freeze['orders'],model_sha256={key:sha(O/'models'/(key+'.pt')) for key in models}))
    boundary=check_screen();checks=0;failures=[];predictions={};runtimes={}
    for key,model in models.items():
        runtime=fast.Runtime(model);runtimes[key]=runtime;aa=[];qq=[]
        for index,state in enumerate(states):
            a,q=runtime.values(state);b,v=model.runtime(state)
            with torch.inference_mode():ea,eq=model.greedy(torch.from_numpy(state)[None])
            aa.append(a[0].tolist());qq.append(float(q[0]));checks+=1
            if not np.array_equal(a,b) or not np.array_equal(a,ea.numpy()) or not np.allclose(q,eq.numpy(),rtol=1e-5,atol=1e-6):
                failures.append(dict(model=key,observation=index,actual=a.tolist(),reference=ea.tolist(),q=q.tolist(),reference_q=eq.tolist()))
        predictions[key]=dict(actions=aa,q=qq)
    # Exact tie regression, zero core weights; original and native must skip only local.
    tie=s.Critic('no_price_both');
    for p in tie.parameters():p.data.zero_()
    x=states[0].copy();x[:240].reshape(30,8)[:,6]=1.;x[:240].reshape(30,8)[:,7]=5/6
    a,q=fast.Runtime(tie).values(x);ea,eq=tie.greedy(torch.from_numpy(x)[None]);np.testing.assert_array_equal(a,ea.numpy())
    assert np.all(a[0,:30]==1) and np.all(a[0,30:]==0)
    write(O/'equivalence.json',dict(checks=checks,failures=failures,screen=boundary,tie_regression=True))
    write(O/'predictions.json',predictions)
    if failures:
        write(O/'verification.json',dict(state='equivalence_failed_no_timing',failed=len(failures),research_launched=False));print(json.dumps(dict(failures=failures[:2],count=len(failures))));return
    payloads={str(p.relative_to(O)).replace('\\','/'):sha(p) for p in O.rglob('*') if p.is_file()}
    write(O/'PRE_TIMING_FREEZE.json',dict(files=payloads,orders=freeze['orders']))
    raw=[]
    for order in freeze['orders']:
        for arm in order['arms']:
            runtime=runtimes[f"{arm}_fit_{order['fit']:02d}"]
            for state in states[:32]:runtime.values(state)[0][0].copy()
            for index,state in enumerate(states):
                start=time.perf_counter_ns();action=runtime.values(state)[0][0].copy();ns=time.perf_counter_ns()-start
                assert action.shape==(32,)
                raw.append(dict(fit=order['fit'],round=order['round'],arm=arm,index=index,nanoseconds=ns))
    metrics={}
    for key,meta in freeze['metadata'].items():
        ns=np.array([row['nanoseconds'] for row in raw if row['fit']==meta['fit'] and row['arm']==meta['arm']],np.float64)
        assert len(ns)==640;metrics[key]=dict(**meta,median_ms=float(np.median(ns)/1e6),p95_ms=float(np.quantile(ns,.95)/1e6))
    screens=[]
    for fit in range(2):
        p=metrics[f'price_both_fit_{fit:02d}'];c=metrics[f'compact_both_fit_{fit:02d}'];w=metrics[f'additive128_unbounded_fit_{fit:02d}']
        screens.append(dict(fit=fit,storage=p['parameter_bytes']<=.5*c['parameter_bytes'],median=p['median_ms']<=w['median_ms'],p95=p['p95_ms']<=1.))
    for name,digest in payloads.items():assert sha(O/name)==digest
    write(O/'raw_timings.json',raw);write(O/'cost.json',dict(metrics=metrics,screens=screens))
    write(O/'verification.json',dict(state='equivalence_passed_timing_completed',checks=checks,timing_rows=len(raw),screens=screens,
        new_simulator_slots=0,new_optimizer_updates=0,research_launched=False))
    write(O/'COMPLETION_MANIFEST.json',dict(files={str(p.relative_to(O)).replace('\\','/'):sha(p) for p in O.rglob('*') if p.is_file()}))
    print(json.dumps(dict(checks=checks,screens=screens,primary=[metrics[f'price_both_fit_{i:02d}'] for i in range(2)])))
if __name__=='__main__':main()
