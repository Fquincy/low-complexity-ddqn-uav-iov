"""Diagnostic profiling of preserved graphs; no optimization or acceptance retiming."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import rq24_prototype as rq

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'validation/rq24_engineering_2026-11-19_01/fixture'
OUT=ROOT/'validation/rq24_runtime_profile_2026-11-19_01'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def summarize(values):
    x=np.asarray(values,np.float64)/1e6
    return dict(count=len(x),median_ms=float(np.median(x)),p95_ms=float(np.quantile(x,.95)),mean_ms=float(x.mean()))


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    plan=dict(kind='DIAGNOSTIC_COMPONENT_PROFILE_NOT_ACCEPTANCE_RETIMING',
              source=str(SOURCE),fits=[0,1,2],states=32,rounds=5,warmups_per_component=8,
              order_seed=906194001,parameters_or_definitions_changed=False,
              compiled_backend='Preserved ONNX graphs and unchanged native functions, one thread',
              stages=['validation','network','features','heads','decode','full_path',
                      'feature_prepare','edge_topology','features_no_pairs'],
              attribution_limit='Components use prepared preceding inputs and separate Python/native dispatch. Medians do not add exactly; subprobes overlap. Not a replacement cost gate.',
              no_new_simulator_steps=True,no_optimizer_steps=True,no_new_graphs=True,
              references_retimed=False,original_failed_benchmark_preserved=True)
    write('PLAN.json',plan)
    paths=[SOURCE/'synthetic_states.npy',SOURCE/'verification.json',SOURCE/'PRE_TIMING_FREEZE.json',
           ROOT/'rq24_prototype.py',ROOT/'profile_rq24_fixed_path_2026_11_19.py',
           ROOT/'assignment_price_load_radio_heuristic.py',ROOT/'assignment_price_fast_runtime.py']
    paths.extend(SOURCE/f'rq24_{i}.onnx' for i in range(3))
    hashes={str(path):sha(path) for path in paths};write('INPUT_FREEZE.json',hashes)
    preserved=json.loads((SOURCE/'PRE_TIMING_FREEZE.json').read_text())
    for path,digest in preserved['sources'].items():assert sha(ROOT/path)==digest
    x=np.load(SOURCE/'synthetic_states.npy');assert x.shape==(32,533)
    contexts={};checks=[]
    for fit in range(3):
        rt=rq.Runtime.__new__(rq.Runtime)
        rt.session=rq.deployed.session(SOURCE/f'rq24_{fit}.onnx')
        rt.scale=np.array([0.,500e6/10e9/.3,500e6/10e9/.3,500e6/20e9/.3,500e6/20e9/.3],np.float32).astype(np.float64)
        rt.c=rq.physical.constants();rt.k=4;rt.radio=True;rt.physical=True;rt.sweep=True
        cases=[]
        for state in x:
            mask,h=rq.validate_mask(state)
            raw=rt.session.run(None,{'state':state[None]})[0][0]
            unary,ei,ej,ed,coeff=rq.features(state,rt.c,4,True)
            g,v,b,beta,eta,lam=rq.heads(raw);g+=unary/30
            w=state[:240].astype(np.float64).reshape(30,8)[:,6:7]*rt.scale
            q=np.r_[0.,np.expm1(state[270:274].astype(np.float64))]
            cycles=state[:240].reshape(30,8)[:,6]
            args=(g,v,b,w,q,beta,eta,mask,cycles,ei,ej,ed,coeff,lam,True)
            a,diag=rq.decode(*args)
            final_a,final_q=rt.values(state)
            np.testing.assert_array_equal(a,final_a[0])
            np.testing.assert_allclose(-h*rq.softplus64(diag[1]),final_q[0],rtol=1e-12,atol=1e-12)
            parts=rq.physical.prepare(state,rt.c)
            # Bound functions retain every prepared input; no cross-state cache in full_path.
            cases.append(dict(validation=lambda s=state:rq.validate_mask(s),
                network=lambda s=state:rt.session.run(None,{'state':s[None]})[0][0],
                features=lambda s=state:rq.features(s,rt.c,4,True),
                heads=lambda r=raw:rq.heads(r),decode=lambda a=args:rq.decode(*a),
                full_path=lambda s=state:rt.values(s)[0][0].copy(),
                feature_prepare=lambda s=state:rq.physical.prepare(s,rt.c),
                edge_topology=lambda gain=parts[6]:rq.edge_pairs(gain,4),
                features_no_pairs=lambda s=state:rq.features(s,rt.c,4,False)))
            checks.append(dict(fit=fit,edges=len(ei),changed_assignments=int(diag[2])))
        # Bind rt via per-fit local closure factory; avoid late binding across fits.
        def bind(runtime,case,stage):
            if stage=='network':return lambda:runtime.session.run(None,{'state':case[None]})[0][0]
            if stage=='features':return lambda:rq.features(case,runtime.c,4,True)
            if stage=='full_path':return lambda:runtime.values(case)[0][0].copy()
            if stage=='feature_prepare':return lambda:rq.physical.prepare(case,runtime.c)
            if stage=='features_no_pairs':return lambda:rq.features(case,runtime.c,4,False)
        for i,case in enumerate(cases):
            for stage in ('network','features','full_path','feature_prepare','features_no_pairs'):
                case[stage]=bind(rt,x[i],stage)
        contexts[fit]=cases
    # All compilation and setup complete before diagnostic readings.
    for cases in contexts.values():
        for stage in plan['stages']:
            for case in cases[:plan['warmups_per_component']]:case[stage]()
    rng=np.random.default_rng(plan['order_seed']);rows=[]
    for rnd in range(plan['rounds']):
        jobs=[(fit,stage) for fit in contexts for stage in plan['stages']];rng.shuffle(jobs)
        for fit,stage in jobs:
            for i,case in enumerate(contexts[fit]):
                start=time.perf_counter_ns();case[stage]();elapsed=time.perf_counter_ns()-start
                rows.append(dict(round=rnd,fit=fit,stage=stage,state=i,ns=elapsed))
    summaries={str(fit):{stage:summarize([r['ns'] for r in rows if r['fit']==fit and r['stage']==stage])
                        for stage in plan['stages']} for fit in contexts}
    assert all(sha(Path(path))==digest for path,digest in hashes.items())
    write('raw_profile.json',rows)
    result=dict(kind=plan['kind'],summaries=summaries,readings=len(rows),input_files_unchanged=True,
                exact_component_action_checks=len(checks),edge_count_min=min(c['edges'] for c in checks),
                edge_count_max=max(c['edges'] for c in checks),
                original_engineering_cost_passed=False,research_released=False,
                new_simulator_steps=0,new_optimizer_updates=0,word_thesis_changed=False)
    write('profile.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
