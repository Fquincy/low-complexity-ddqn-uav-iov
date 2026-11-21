"""Diagnostic only: preserved AR24 component costs, no changed controller."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import ar24_prototype as ar

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'validation/ar24_engineering_2026-11-19_01'
OUT=ROOT/'validation/ar24_runtime_profile_2026-11-19_01'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def bound_case(rt,state):
    mask,h=ar.prior.validate_mask(state);raw=rt.session.run(None,{'state':state[None]})[0][0]
    x,y,w,q=ar.features(state,rt.scale,True,False);g,v,b,beta,eta,lam=ar.heads(raw)
    args=(g,v,b,w,q,beta,eta,x,y,lam,mask,state[:240].reshape(30,8)[:,6],True,True,True)
    a,diag=ar.decode(*args);fa,fq=rt.values(state)
    np.testing.assert_array_equal(a,fa[0])
    np.testing.assert_allclose(-h*ar.prior.softplus64(diag[1]),fq[0],rtol=1e-12,atol=1e-12)
    return dict(validation=lambda:ar.prior.validate_mask(state),
        network=lambda:rt.session.run(None,{'state':state[None]})[0][0],
        features=lambda:ar.features(state,rt.scale,True,False),
        heads=lambda:ar.heads(raw),
        native_finish=lambda:ar.finish(raw,state,rt.scale,mask,h,*rt.flags),
        decode=lambda:ar.decode(*args),full_path=lambda:rt.values(state)[0][0].copy())


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    plan=dict(kind='AR24_FIXED_PATH_DIAGNOSTIC_NOT_ACCEPTANCE_RETIMING',fits=[0,1,2],states=32,
        rounds=5,warmups=8,order_seed=906200101,
        stages=['validation','network','features','heads','native_finish','decode','full_path'],
        component_overlap=True,prepared_inputs_only_for_components=True,
        full_path_recomputes_everything=True,references_retimed=False,
        no_new_graphs=True,new_training_slots=0,new_optimizer_updates=0,new_simulator_steps=0)
    write('PLAN.json',plan);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    paths=[ROOT/'ar24_prototype.py',ROOT/'profile_ar24_fixed_path_2026_11_19.py',BASE/'verification.json',
           BASE/'synthetic_states.npy',BASE/'PRE_TIMING_FREEZE.json']+[BASE/f'ar24_{i}.onnx' for i in range(3)]
    hashes={str(p):sha(p) for p in paths};write('INPUT_FREEZE.json',hashes)
    original=json.loads((BASE/'PRE_TIMING_FREEZE.json').read_text())
    for name,h in original['sources'].items():assert sha(ROOT/name)==h
    x=np.load(BASE/'synthetic_states.npy');contexts={}
    for fit in range(3):
        rt=ar.Runtime.__new__(ar.Runtime);rt.session=ar.prior.deployed.session(BASE/f'ar24_{fit}.onnx')
        rt.scale=np.array([0.,500e6/10e9/.3,500e6/10e9/.3,500e6/20e9/.3,500e6/20e9/.3],np.float32).astype(np.float64)
        rt.flags=ar.flags('ar24');contexts[fit]=[bound_case(rt,state) for state in x]
    for cases in contexts.values():
        for stage in plan['stages']:
            for case in cases[:8]:case[stage]()
    rng=np.random.default_rng(plan['order_seed']);rows=[];orders=[]
    for rnd in range(5):
        jobs=[(fit,stage) for fit in contexts for stage in plan['stages']];rng.shuffle(jobs);orders.append(jobs)
        for fit,stage in jobs:
            for i,case in enumerate(contexts[fit]):
                start=time.perf_counter_ns();case[stage]();ns=time.perf_counter_ns()-start
                rows.append(dict(round=rnd,fit=fit,stage=stage,state=i,ns=ns))
    summaries={}
    for fit in contexts:
        summaries[str(fit)]={}
        for stage in plan['stages']:
            ns=np.array([r['ns'] for r in rows if r['fit']==fit and r['stage']==stage],np.float64)/1e6
            summaries[str(fit)][stage]=dict(count=len(ns),median_ms=float(np.median(ns)),p95_ms=float(np.quantile(ns,.95)))
    assert all(sha(Path(p))==h for p,h in hashes.items())
    write('raw_profile.json',rows)
    report=dict(kind=plan['kind'],summaries=summaries,orders=orders,readings=len(rows),
        decomposition_action_checks=96,inputs_unchanged=True,research_release=False,
        original_cost_gate_passed=False,new_simulator_steps=0,new_optimizer_updates=0)
    write('profile.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
