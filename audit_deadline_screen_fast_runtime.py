"""Independent saved-artifact replay/statistics audit; never measures latency."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
import deadline_screen_prototype as original
import deadline_screen_fast_runtime as fast
R=Path(__file__).resolve().parent;O=R/'validation/deadline_screen_fast_runtime_2026-11-11_01';OLD=R/'validation/deadline_screen_engineering_2026-11-11_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit():
    original.existing.configure();original.existing.fused.numba.set_num_threads(1)
    for name,digest in read(O/'COMPLETION_MANIFEST.json')['files'].items():assert sha(O/name)==digest,name
    freeze=read(O/'PRE_CHECK_FREEZE.json');prior=read(OLD/'PRE_BENCHMARK_FREEZE.json')
    assert sha(OLD/'PRE_BENCHMARK_FREEZE.json')==freeze['prior_freeze_sha256']
    assert freeze['metadata']==prior['metadata'] and freeze['orders']==prior['orders']
    for name,digest in freeze['source_sha256'].items():assert sha(R/name)==sha(O/'source_snapshot'/name)==digest,name
    assert sha(O/'observations.npz')==sha(OLD/'observations.npz')==freeze['observations_sha256']
    states=np.load(O/'observations.npz')['states'];pred=read(O/'predictions.json');equiv=read(O/'equivalence.json')
    assert not equiv['failures'] and equiv['tie_regression']
    pre=read(O/'PRE_TIMING_FREEZE.json')
    for name,digest in pre['files'].items():assert sha(O/name)==digest,name
    checks=0
    for key,meta in freeze['metadata'].items():
        model=original.Critic(meta['arm'],meta['seed']).eval();path=O/'models'/(key+'.pt')
        assert sha(path)==sha(OLD/'models'/(key+'.pt'))==freeze['model_sha256'][key]
        model.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'));runtime=fast.Runtime(model)
        assert original.existing.base.parameter_bytes(model)==meta['parameter_bytes']
        with torch.inference_mode():
            for index,state in enumerate(states):
                a,q=runtime.values(state);ea,eq=model.greedy(torch.from_numpy(state)[None])
                np.testing.assert_array_equal(a,ea.numpy());np.testing.assert_array_equal(a[0],pred[key]['actions'][index])
                np.testing.assert_allclose(q,eq.numpy(),atol=1e-6,rtol=1e-5);assert float(q[0])==pred[key]['q'][index]
                excluded,_=fast.validate_and_screen(state)
                np.testing.assert_array_equal(~excluded,original.allowed(state[None])[0,:30,0]);checks+=1
    # Reproduce random/boundary/invalid-state tests without new timing.
    from check_deadline_screen_fast_runtime import check_screen
    assert check_screen()==equiv['screen']
    raw=read(O/'raw_timings.json');cost=read(O/'cost.json')
    expected=[(order['fit'],order['round'],arm,i) for order in freeze['orders'] for arm in order['arms'] for i in range(128)]
    assert [(r['fit'],r['round'],r['arm'],r['index']) for r in raw]==expected and len(raw)==8960
    for key,meta in freeze['metadata'].items():
        ns=np.array([r['nanoseconds'] for r in raw if r['fit']==meta['fit'] and r['arm']==meta['arm']],np.float64)
        assert len(ns)==640 and np.all(ns>0)
        assert cost['metrics'][key]==dict(**meta,median_ms=float(np.median(ns)/1e6),p95_ms=float(np.quantile(ns,.95)/1e6))
    screens=[]
    for fit in range(2):
        p=cost['metrics'][f'price_both_fit_{fit:02d}'];c=cost['metrics'][f'compact_both_fit_{fit:02d}'];w=cost['metrics'][f'additive128_unbounded_fit_{fit:02d}']
        screens.append(dict(fit=fit,storage=p['parameter_bytes']<=.5*c['parameter_bytes'],median=p['median_ms']<=w['median_ms'],p95=p['p95_ms']<=1.))
    v=read(O/'verification.json');assert cost['screens']==v['screens']==screens and v['checks']==checks==1792
    assert v['new_simulator_slots']==v['new_optimizer_updates']==0
    result=dict(state='passed_same_weight_runtime_audit',models=14,actions_replayed=checks,timing_rows_recomputed=8960,
        boundary_checks_reproduced=True,screens=screens,new_timing_measurements=0,research_launched=False)
    (O/'saved_artifact_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
if __name__=='__main__':print(json.dumps(audit()))
