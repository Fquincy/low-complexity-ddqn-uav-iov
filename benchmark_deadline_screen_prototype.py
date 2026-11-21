"""Fixed engineering benchmark for screened random models; no simulator or fit."""
import hashlib,json,shutil,time
from pathlib import Path
import numpy as np,torch
import deadline_screen_prototype as s
from test_deadline_screen_prototype import observations
R=Path(__file__).resolve().parent;O=R/'validation/deadline_screen_engineering_2026-11-11_01'
SOURCES=['deadline_screen_prototype.py','test_deadline_screen_prototype.py','benchmark_deadline_screen_prototype.py','DEADLINE_SCREEN_ENGINEERING_PROTOCOL_2026-11-11.json',*s.existing.SOURCES]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def main():
    assert json.loads((O/'tests.json').read_text())['passed'];assert not (O/'PRE_BENCHMARK_FREEZE.json').exists()
    s.existing.configure();p=json.loads((R/'DEADLINE_SCREEN_ENGINEERING_PROTOCOL_2026-11-11.json').read_text())
    states=observations(128).numpy();np.savez_compressed(O/'observations.npz',states=states)
    models={};metadata={};payloads={};predictions={};(O/'models').mkdir()
    for fit,seed in enumerate(p['model_seeds']):
        for arm in p['arms']:
            key=f'{arm}_fit_{fit:02d}';m=s.Critic(arm,seed).eval();models[key]=m
            path=O/'models'/(key+'.pt');torch.save(m.state_dict(),path);payloads[str(path.relative_to(O))]=sha(path)
            metadata[key]=dict(arm=arm,fit=fit,seed=seed,parameter_bytes=s.existing.base.parameter_bytes(m),buffer_bytes=sum(b.numel()*b.element_size() for b in m.buffers()))
    for name in SOURCES:
        dest=O/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/name,dest)
    rng=np.random.default_rng(p['order_seed']);orders=[]
    for fit in range(2):
        for round_id in range(5):
            arms=list(p['arms']);rng.shuffle(arms);orders.append(dict(fit=fit,round=round_id,arms=arms))
    write(O/'PRE_BENCHMARK_FREEZE.json',dict(kind='SYNTHETIC_NOT_RESEARCH',model_payloads=payloads,metadata=metadata,
        source_sha256={n:sha(R/n) for n in SOURCES},observations_sha256=sha(O/'observations.npz'),orders=orders))
    checks=0
    for key,m in models.items():
        actions=[];values=[]
        with torch.inference_mode():
            for state in states:
                ea,eq=m.greedy(torch.from_numpy(state)[None]);a,q=m.runtime(state)
                np.testing.assert_array_equal(ea.numpy(),a);np.testing.assert_allclose(eq.numpy(),q,atol=1e-6,rtol=1e-5)
                assert np.take_along_axis(s.allowed(state[None]),a[:,:,None],2).all()
                actions.append(a[0].tolist());values.append(float(q[0]));checks+=1
        predictions[key]=dict(actions=actions,q=values)
    write(O/'predictions.json',predictions)
    raw=[]
    for order in orders:
        for arm in order['arms']:
            m=models[f"{arm}_fit_{order['fit']:02d}"]
            for state in states[:32]:m.runtime(state)[0][0].copy()
            for index,state in enumerate(states):
                start=time.perf_counter_ns();action=m.runtime(state)[0][0].copy();elapsed=time.perf_counter_ns()-start
                assert action.shape==(32,)
                raw.append(dict(fit=order['fit'],round=order['round'],arm=arm,index=index,nanoseconds=elapsed))
    write(O/'raw_timings.json',raw)
    metrics={}
    for key,meta in metadata.items():
        ns=np.array([r['nanoseconds'] for r in raw if r['fit']==meta['fit'] and r['arm']==meta['arm']],np.float64)
        assert len(ns)==640
        metrics[key]=dict(**meta,median_ms=float(np.median(ns)/1e6),p95_ms=float(np.quantile(ns,.95)/1e6))
    screens=[]
    for fit in range(2):
        primary=metrics[f'price_both_fit_{fit:02d}'];compact=metrics[f'compact_both_fit_{fit:02d}'];wide=metrics[f'additive128_unbounded_fit_{fit:02d}']
        screens.append(dict(fit=fit,parameter_storage=primary['parameter_bytes']<=.5*compact['parameter_bytes'],median=primary['median_ms']<=wide['median_ms'],p95=primary['p95_ms']<=1.))
    for name,digest in payloads.items():assert sha(O/name)==digest
    write(O/'benchmark.json',dict(kind='synthetic_random_weight_engineering_only',exact_actions_checked=checks,timing_rows=len(raw),metrics=metrics,screens=screens,
        no_policy_quality_claim=True,new_simulator_slots=0,new_optimizer_updates=0,research_launched=False))
    write(O/'ARTIFACT_MANIFEST.json',dict(files={str(path.relative_to(O)).replace('\\','/'):sha(path) for path in O.rglob('*') if path.is_file()}))
    print(json.dumps(dict(checks=checks,timing_rows=len(raw),screens=screens,primary=[metrics[f'price_both_fit_{fit:02d}'] for fit in range(2)])))
if __name__=='__main__':main()
