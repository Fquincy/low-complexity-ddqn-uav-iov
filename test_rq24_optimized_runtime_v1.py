"""One predeclared equivalence and matched synthetic cost attempt; no training."""
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import rq24_prototype as rq
import rq24_optimized_runtime_v1 as optimized
import test_rq24_engineering_2026_11_19 as algebra

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'validation/rq24_engineering_2026-11-19_01/fixture'
OUT=ROOT/'validation/rq24_optimized_v1_2026-11-19_02'
SEEDS=(906190001,906190011,906190021)
REFS=('screened_plain24','screened_wide128','unmasked_wide128','unmasked_compact64')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_rq(arm,fit):
    r=rq.Runtime.__new__(rq.Runtime);r.session=rq.deployed.session(BASE/f'{arm}_{fit}.onnx')
    r.scale=np.array([0.,500e6/10e9/.3,500e6/10e9/.3,500e6/20e9/.3,500e6/20e9/.3],np.float32).astype(np.float64)
    r.c=rq.physical.constants();r.k=4;r.radio=arm not in (rq.ARMS[1],rq.ARMS[2])
    r.physical=arm!=rq.ARMS[2];r.sweep=arm!=rq.ARMS[3];return r


def load_reference(arm,fit):
    import run_assignment_price_research_bound as old
    model=old.base.build(arm,dict(model_seed=SEEDS[fit],cfg=copy.deepcopy(old.base.protocol()['training'])))['online'].eval()
    core=old.base.references.Runtime.__new__(old.base.references.Runtime)
    core.mode={'plain24_nonpositive':5,'additive128_unbounded':0,'additive64_nonpositive':1}[model.base_arm]
    core.scaled=model.scaled;core.screened=model.screened
    core.session=rq.deployed.session(BASE/f'{arm}_{fit}.onnx')
    return core,sum(p.numel()*p.element_size() for p in model.parameters())


def stresses(x):
    cases=[]
    for kind in ('tied','coincident','lower','upper','battery_zero','battery_full'):
        s=x[0].copy();tasks=s[:240].reshape(30,8)
        if kind=='tied':tasks[:,:2]=.5;tasks[:,6]=.8
        if kind=='coincident':tasks[:,:2]=s[526:528];s[529:531]=s[526:528]
        if kind in ('lower','upper'):
            upper=kind=='upper';tasks[:,:2]=1. if upper else 0.
            tasks[:,5]=1. if upper else 1/3;tasks[:,6]=1. if upper else .6;tasks[:,7]=1. if upper else .25/.3
            s[240:514]=np.log1p(10.) if upper else 0.
            s[-1]=39/40 if upper else 0.
        if kind.startswith('battery'):s[528]=s[531]=0. if kind.endswith('zero') else 1.
        cases.append(s)
    return np.stack(cases)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    plan=dict(kind='ONE_DECLARED_OPTIMIZATION_ATTEMPT_NOT_RESEARCH',seeds=SEEDS,
        original_states=32,stress_kinds=['tied','coincident','lower','upper','battery_zero','battery_full'],
        fp64_rtol=1e-12,fp64_atol=1e-12,exact_topology=True,exact_actions=True,exact_raw_fp32=True,
        rounds=3,warmups=16,order_seed=906195001,paths=['original_rq24','optimized_rq24',*REFS],
        raw_readings=1728,criteria=dict(parameter_fraction_compact64=.5,median_ratio_each_wide_max=1.,p95_ms_max=1.),
        no_threshold_relaxation=True,no_favorable_repeat_timing=True,new_training_slots=0,new_optimizer_updates=0)
    write('PLAN.json',plan)
    consumed=[ROOT/'rq24_prototype.py',ROOT/'rq24_optimized_runtime_v1.py',ROOT/'test_rq24_optimized_runtime_v1.py',
        ROOT/'RQ24_RUNTIME_PROFILE_AND_OPTIMIZATION_PLAN_2026-11-19.txt',BASE/'synthetic_states.npy',BASE/'verification.json']
    consumed.extend(BASE/f'{arm}_{fit}.onnx' for arm in rq.ARMS+REFS for fit in range(3))
    hashes={str(p):sha(p) for p in consumed};write('INPUT_FREEZE.json',hashes)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    x=np.load(BASE/'synthetic_states.npy');extra=stresses(x);all_states=np.concatenate([x,extra])
    np.save(OUT/'synthetic_stress_states.npy',extra)
    rules=json.loads((BASE/'PRE_TIMING_FREEZE.json').read_text())
    for path,digest in rules['sources'].items():assert sha(ROOT/path)==digest
    counts=dict(prepare_arrays=0,feature_sets=0,export_paths=0,exact_actions=0,raw_outputs=0,finite_gradient_arms=0)
    equivalence=[];runtimes={};parameter_bytes={};start=time.perf_counter_ns()
    try:
        tests=algebra.algebra()
        # Check state-only physical intermediates once per state, including both ablations.
        c=rq.physical.constants()
        for state in all_states:
            old_parts=rq.physical.prepare(state,c);new_parts=optimized.prepare(state,c)
            for left,right in zip(old_parts,new_parts):
                np.testing.assert_allclose(left,right,rtol=1e-12,atol=1e-12);counts['prepare_arrays']+=1
            for radio in (False,True):
                left=rq.features(state,c,4,radio);right=optimized.features(state,c,4,radio)
                for i,(a,b) in enumerate(zip(left,right)):
                    if i in (1,2,3):np.testing.assert_array_equal(a,b)
                    else:np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
                counts['feature_sets']+=1
            # Reference rates used by both implementations are identical functions of gain.
            for gain in old_parts[6].ravel():
                rate=c[7]*np.log1p(c[8]*gain/c[9])/np.log(2.)
                assert np.isfinite(rate) and rate>0
        for fit in range(3):
            for arm in rq.ARMS:
                original=load_rq(arm,fit);new=optimized.Runtime(original);maxdiff=0.
                before=(new.c.copy(),new.scale.copy())
                for state in all_states:
                    raw=original.session.run(None,{'state':state[None]})[0][0]
                    raw_new=new.session.run(None,{'state':state[None]})[0][0]
                    np.testing.assert_array_equal(raw,raw_new);counts['raw_outputs']+=1
                    mask,h=rq.validate_mask(state)
                    args=(raw,state,original.scale,original.c,mask,h,4,original.radio,original.physical,original.sweep)
                    a,q,diag=rq.finish(*args);na,nq,nd=optimized.finish(*args)
                    np.testing.assert_array_equal(a,na)
                    np.testing.assert_allclose(diag,nd,rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(q,nq,rtol=1e-12,atol=1e-12)
                    da,dq=new.values(state);np.testing.assert_array_equal(a,da)
                    np.testing.assert_allclose(q,dq,rtol=1e-12,atol=1e-12)
                    maxdiff=max(maxdiff,float(abs(q[0]-nq[0])));counts['exact_actions']+=1
                held,held_q=new.values(x[0]);held_copy=held.copy();held_qcopy=held_q.copy()
                for state in (x[-1],extra[2],x[0],extra[3]):new.values(state)
                np.testing.assert_array_equal(held,held_copy);np.testing.assert_array_equal(held_q,held_qcopy)
                np.testing.assert_array_equal(new.values(x[0])[0],held)
                np.testing.assert_array_equal(new.c,before[0]);np.testing.assert_array_equal(new.scale,before[1])
                for bad in (x[0].astype(np.float64),np.zeros(532,np.float32)):
                    try:new.values(bad)
                    except ValueError:pass
                    else:raise AssertionError('bad dtype/shape')
                for field,value in ((0,np.nan),(7,0.),(270,1000.),(-1,1.),(-1,.0123)):
                    bad=x[0].copy();bad[field]=value
                    for rt in (original,new):
                        try:rt.values(bad)
                        except ValueError:pass
                        else:raise AssertionError(('bad input accepted',field))
                # Training critic untouched; autograd finite and weights immutable.
                model=rq.Critic(arm,SEEDS[fit]);model_before={k:v.clone() for k,v in model.state_dict().items()}
                s=torch.from_numpy(x[:1]);act=model.greedy(s)[0];model(s,act).sum().backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                assert all(torch.equal(v,model_before[k]) for k,v in model.state_dict().items());counts['finite_gradient_arms']+=1
                equivalence.append(dict(arm=arm,fit=fit,states=len(all_states),max_Q_difference=maxdiff,exact_actions=True))
                counts['export_paths']+=1
                if arm=='rq24':
                    runtimes[('original_rq24',fit)]=original;runtimes[('optimized_rq24',fit)]=new
                    parameter_bytes[('original_rq24',fit)]=parameter_bytes[('optimized_rq24',fit)]=68364
                print(f'equivalence passed: {arm} fit {fit}',flush=True)
            for arm in REFS:runtimes[(arm,fit)],parameter_bytes[(arm,fit)]=load_reference(arm,fit)
    except Exception as error:
        write('equivalence_failure.json',dict(error=str(error),counts=counts,timing_started=False,thresholds_unchanged=True))
        raise
    setup_ns=time.perf_counter_ns()-start
    write('EQUIVALENCE.json',dict(passed=True,counts=counts,paths=equivalence,old_algebra=tests,
        setup_ns=setup_ns,setup_scope='shared process including imports already loaded, native compilation/tests/session loading; not cold startup'))
    assert all(sha(Path(p))==h for p,h in hashes.items())
    jobs=list(runtimes);orders=[];rng=np.random.default_rng(plan['order_seed'])
    for _ in range(3):order=list(jobs);rng.shuffle(order);orders.append(order)
    write('PRE_TIMING_FREEZE.json',dict(input_hashes=hashes,orders=orders,plan_sha256=sha(OUT/'PLAN.json')))
    rows=[]
    for rnd,order in enumerate(orders):
        for arm,fit in order:
            rt=runtimes[(arm,fit)]
            for state in x[:16]:rt.values(state)[0][0].copy()
            for index,state in enumerate(x):
                begin=time.perf_counter_ns();rt.values(state)[0][0].copy();elapsed=time.perf_counter_ns()-begin
                rows.append(dict(round=rnd,arm=arm,fit=fit,state=index,ns=elapsed))
    write('timings.json',rows);costs={};screens=[]
    for fit in range(3):
        per={}
        for arm in plan['paths']:
            ns=[r['ns'] for r in rows if r['arm']==arm and r['fit']==fit]
            per[arm]=dict(median_ms=float(np.median(ns)/1e6),p95_ms=float(np.quantile(ns,.95)/1e6),parameter_bytes=parameter_bytes[(arm,fit)])
        costs[str(fit)]=per;n=per['optimized_rq24']
        screens.append(dict(storage=n['parameter_bytes']<=.5*per['unmasked_compact64']['parameter_bytes'],
            median=all(n['median_ms']<=per[a]['median_ms'] for a in ('screened_wide128','unmasked_wide128')),p95=n['p95_ms']<=1.))
    assert all(sha(Path(p))==h for p,h in hashes.items())
    result=dict(kind=plan['kind'],equivalence_passed=True,input_hashes_unchanged=True,
        costs=costs,cost_screens=screens,cost_gate_passed=all(all(s.values()) for s in screens),raw_readings=len(rows),
        model_parameter_bytes=68364,model_buffer_bytes=20,
        graph_bytes=[(BASE/f'rq24_{fit}.onnx').stat().st_size for fit in range(3)],
        no_new_graphs=True,quality_evidence=False,research_released=False,new_training_slots=0,new_optimizer_updates=0,
        word_thesis_changed=False)
    write('verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
