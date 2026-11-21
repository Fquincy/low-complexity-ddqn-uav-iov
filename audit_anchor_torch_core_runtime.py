"""Independent replay and integrity audit of the failed numerical candidate."""
import sys
from pathlib import Path
import numpy as np,torch,onnxruntime as ort
import run_anchor_development_v4 as r
import anchor_original_torch_core_runtime as runtime_adapter

ROOT=Path(__file__).resolve().parent

def audit(folder):
    folder=Path(folder);r.configure();r.patch_training_core();r.fused.numba.set_num_threads(1)
    manifest=r.read(folder/'COMPLETION_MANIFEST.json')
    for rel,digest in manifest['files'].items():assert r.base.old.sha(folder/rel)==digest,rel
    freeze=r.read(folder/'PRE_CHECK_FREEZE.json');rows=r.read(folder/'model_checks.json')
    spec=r.read(folder/'source_snapshot/ANCHOR_TORCH_CORE_RUNTIME_AMENDMENT_2026-11-11.json')
    assert r.base.old.sha(ROOT/'ANCHOR_TORCH_CORE_RUNTIME_AMENDMENT_2026-11-11.json')==freeze['protocol_sha256']
    run=ROOT/'development_runs/terminal_anchor_development_2026-11-06_01'
    assert r.base.old.sha(run/'ALL_FINAL_CHECKPOINTS_FREEZE.json')==spec['checkpoint_freeze_sha256']
    for name,digest in freeze['source_sha256'].items():assert r.base.old.sha(ROOT/name)==r.base.old.sha(folder/'source_snapshot'/name)==digest
    states=np.load(folder/'observations.npz')['observations']
    assert r.base.old.sha(folder/'observations.npz')==spec['observations_sha256']
    refs=r.read(folder/'reference_predictions.json')
    expected={f'{a}_fit_{fit:02d}' for fit in range(5) for a in r.ARMS}
    assert {x['key'] for x in rows}==set(refs)==expected and len(rows)==85
    total=0;failed=0;bad_observations=0
    for row in rows:
        arm=row['arm'];fit=row['fit'];key=row['key']
        ckpath=run/arm/f'fit_{fit:02d}'/'final.pt'
        assert r.base.old.sha(ckpath)==row['checkpoint_sha256']==freeze['checkpoint_sha256'][key]
        ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        model=r.make_model(arm,row['seed']);model.load_state_dict(ck['online']);model.eval()
        assert r.base.old.digest(model.state_dict())==ck['online_parameter_hash']
        graph=folder/'exports'/f'{key}.onnx'
        assert r.base.old.sha(graph)==row['graph_sha256']
        extra_path=graph.with_suffix('.extra.npy');assert r.base.old.sha(extra_path)==row['extra_sha256']
        session=runtime_adapter.session(graph)
        extra=np.load(extra_path);bad=[];maxerr=0.
        with torch.inference_mode():
            for index,state in enumerate(states):
                ea,eq,_=r.greedy(model,torch.from_numpy(state)[None],arm)
                np.testing.assert_array_equal(ea.numpy()[0],refs[key]['actions'][index])
                assert float(eq[0])==refs[key]['q'][index]
                if arm in r.base.SHARED_ARMS:a,q=r.compiled.compiled_values(session,state)
                else:a,q=r.runtime_values((session,extra),arm,state)
                branches=np.flatnonzero(a[0]!=ea.numpy()[0]);close=bool(np.allclose(q,eq.numpy(),atol=1e-6,rtol=1e-5))
                maxerr=max(maxerr,float(np.max(np.abs(q-eq.numpy()))))
                if len(branches) or not close:
                    bad.append(dict(observation=index,branches=branches.tolist(),reference_actions=ea.numpy()[0,branches].tolist(),
                        actual_actions=a[0,branches].tolist(),q_close=close,reference_q=float(eq[0]),actual_q=float(q[0])))
                total+=1
        assert bad==row['mismatches']['candidate'],key
        assert row['candidate_pass']==(not bad) and row['max_candidate_absolute_q_error']==maxerr
        failed+=bool(bad);bad_observations+=len(bad)
        del session,model,ck
    result=r.read(folder/'verification.json')
    assert total==10880 and failed==result['candidate_failed_models']==0
    assert result['all_candidate_actions_and_values_pass'] and result['cost_measured']
    assert result['near_tie_candidate_pass'] and result['near_tie_baseline_regression_reproduced']
    assert not result['heldout_evaluation_started'] and result['new_simulator_slots']==result['optimizer_updates']==0
    cost=audit_cost(folder,rows)
    summary=dict(state='passed_saved_weight_torch_core_runtime_audit',models_replayed=85,observations_replayed=total,
        failed_candidate_models=failed,failed_candidate_observations=bad_observations,
        reference_predictions_reverified=True,manifest_files_verified=len(manifest['files']),
        cost_audit=cost,heldout_evaluation_started=False)
    r.write(folder/'saved_artifact_audit.json',summary);return summary

def audit_cost(folder,rows):
    pre=r.read(folder/'PRE_TIMING_FREEZE.json')
    for name,digest in pre['payloads'].items():assert r.base.old.sha(folder/name)==digest
    assert pre['source_sha256']==r.read(folder/'PRE_CHECK_FREEZE.json')['source_sha256']
    raw=r.read(folder/'raw_timings.json');cost=r.read(folder/'cost.json');orders=r.read(folder/'TIMING_ORDERS.json')
    assert len(raw)==54400
    rng=np.random.default_rng(623600003);expected_orders=[]
    for fit in range(5):
        for round_id in range(5):
            order=list(r.ARMS);rng.shuffle(order);expected_orders.append(dict(fit=fit,round=round_id,order=order))
    assert orders==expected_orders
    expected_keys=[(x['fit'],x['round'],a,i) for x in orders for a in x['order'] for i in range(128)]
    assert [(x['fit'],x['round'],x['arm'],x['observation']) for x in raw]==expected_keys
    assert all(isinstance(x['nanoseconds'],int) and x['nanoseconds']>0 for x in raw)
    screens=[]
    for arm in r.ARMS:
        for fit in range(5):
            ns=np.array([x['nanoseconds'] for x in raw if x['arm']==arm and x['fit']==fit],np.float64)
            assert len(ns)==640
            got=cost['metrics'][arm][fit];meta=next(x for x in rows if x['arm']==arm and x['fit']==fit)
            assert got==dict(median_ms=float(np.median(ns)/1e6),p95_ms=float(np.quantile(ns,.95)/1e6),parameter_bytes=meta['parameter_bytes'])
    for fit in range(5):
        p=cost['metrics']['price_both'][fit];c=cost['metrics']['additive64_nonpositive'][fit];w=cost['metrics']['additive128_unbounded'][fit]
        screens.append(dict(fit=fit,storage=p['parameter_bytes']<=c['parameter_bytes']*.5,median=p['median_ms']<=w['median_ms'],p95=p['p95_ms']<=1.))
    assert screens==cost['screens']
    passed=all(all(x[k] for k in ['storage','median','p95']) for x in screens)
    assert passed==cost['all_fit_cost_screens_pass']
    return dict(raw_rows_recomputed=54400,all_fit_cost_screens_pass=passed)

if __name__=='__main__':
    import json
    print(json.dumps(audit(Path(sys.argv[1]))))
