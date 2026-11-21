"""Replay frozen synthetic screened models and recompute raw timing statistics."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
import deadline_screen_prototype as s
R=Path(__file__).resolve().parent;O=R/'validation/deadline_screen_engineering_2026-11-11_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    s.existing.configure();manifest=read(O/'ARTIFACT_MANIFEST.json')
    for name,digest in manifest['files'].items():assert sha(O/name)==digest,name
    freeze=read(O/'PRE_BENCHMARK_FREEZE.json');report=read(O/'benchmark.json');spec=read(R/'DEADLINE_SCREEN_ENGINEERING_PROTOCOL_2026-11-11.json')
    for name,digest in freeze['source_sha256'].items():assert sha(R/name)==sha(O/'source_snapshot'/name)==digest
    assert sha(O/'observations.npz')==freeze['observations_sha256']
    states=np.load(O/'observations.npz')['states'];pred=read(O/'predictions.json');checks=0
    for key,meta in freeze['metadata'].items():
        model=s.Critic(meta['arm'],meta['seed']).eval();path=O/'models'/(key+'.pt')
        # The pre-benchmark model map preserves native Windows separators.
        payloads={name.replace('\\','/'):digest for name,digest in freeze['model_payloads'].items()}
        assert sha(path)==payloads[str(path.relative_to(O)).replace('\\','/')]
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        assert s.existing.base.parameter_bytes(model)==meta['parameter_bytes']
        assert sum(b.numel()*b.element_size() for b in model.buffers())==meta['buffer_bytes']
        with torch.inference_mode():
            for i,state in enumerate(states):
                a,q=model.greedy(torch.from_numpy(state)[None]);b,v=model.runtime(state)
                np.testing.assert_array_equal(a.numpy(),b);np.testing.assert_array_equal(b[0],pred[key]['actions'][i])
                np.testing.assert_allclose(q.numpy(),v,atol=1e-6,rtol=1e-5);assert float(v[0])==pred[key]['q'][i]
                assert np.take_along_axis(s.allowed(state[None]),b[:,:,None],2).all();checks+=1
    raw=read(O/'raw_timings.json');rng=np.random.default_rng(spec['order_seed']);orders=[]
    for fit in range(2):
        for round_id in range(5):
            arms=list(spec['arms']);rng.shuffle(arms);orders.append(dict(fit=fit,round=round_id,arms=arms))
    assert orders==freeze['orders']
    expected=[(r['fit'],r['round'],a,i) for r in orders for a in r['arms'] for i in range(128)]
    assert [(r['fit'],r['round'],r['arm'],r['index']) for r in raw]==expected
    for key,meta in freeze['metadata'].items():
        ns=np.array([r['nanoseconds'] for r in raw if r['fit']==meta['fit'] and r['arm']==meta['arm']],np.float64)
        assert len(ns)==640 and np.all(ns>0)
        assert report['metrics'][key]==dict(**meta,median_ms=float(np.median(ns)/1e6),p95_ms=float(np.quantile(ns,.95)/1e6))
    screens=[]
    for fit in range(2):
        p=report['metrics'][f'price_both_fit_{fit:02d}'];c=report['metrics'][f'compact_both_fit_{fit:02d}'];w=report['metrics'][f'additive128_unbounded_fit_{fit:02d}']
        screens.append(dict(fit=fit,parameter_storage=p['parameter_bytes']<=.5*c['parameter_bytes'],median=p['median_ms']<=w['median_ms'],p95=p['p95_ms']<=1.))
    assert report['screens']==screens
    assert report['exact_actions_checked']==checks==1792 and report['timing_rows']==len(raw)==8960
    result=dict(state='passed_synthetic_saved_artifact_audit',models=14,actions_replayed=checks,timing_rows_recomputed=len(raw),screens=screens,
        new_simulator_slots=0,new_optimizer_updates=0,research_launched=False)
    (O/'saved_artifact_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
