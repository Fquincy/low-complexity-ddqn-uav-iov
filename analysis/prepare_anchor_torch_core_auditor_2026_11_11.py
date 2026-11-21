"""Create the independent replay auditor for the original-PyTorch-core route."""
from pathlib import Path
R=Path(__file__).resolve().parents[1];p=R/'audit_anchor_torch_core_runtime.py';assert not p.exists()
s=(R/'audit_anchor_numerical_correction.py').read_text()
s=s.replace('import run_anchor_development_v4 as r','import run_anchor_development_v4 as r\nimport anchor_original_torch_core_runtime as runtime_adapter')
s=s.replace('ANCHOR_RUNTIME_NUMERICAL_CORRECTION_PROTOCOL_2026-11-11.json','ANCHOR_TORCH_CORE_RUNTIME_AMENDMENT_2026-11-11.json')
a=s.index('        opts=ort.SessionOptions()');b=s.index('        extra=np.load(extra_path);',a)
s=s[:a]+'''        session=runtime_adapter.session(graph)
'''+s[b:]
a=s.index('    assert total==10880');b=s.index("    r.write(folder/'saved_artifact_audit.json'",a)
s=s[:a]+'''    assert total==10880 and failed==result['candidate_failed_models']==0
    assert result['all_candidate_actions_and_values_pass'] and result['cost_measured']
    assert result['near_tie_candidate_pass'] and result['near_tie_baseline_regression_reproduced']
    assert not result['heldout_evaluation_started'] and result['new_simulator_slots']==result['optimizer_updates']==0
    cost=audit_cost(folder,rows)
    summary=dict(state='passed_saved_weight_torch_core_runtime_audit',models_replayed=85,observations_replayed=total,
        failed_candidate_models=failed,failed_candidate_observations=bad_observations,
        reference_predictions_reverified=True,manifest_files_verified=len(manifest['files']),
        cost_audit=cost,heldout_evaluation_started=False)
'''+s[b:]
pos=s.index("if __name__=='__main__':")
s=s[:pos]+'''def audit_cost(folder,rows):
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

'''+s[pos:]
p.write_text(s,encoding='utf-8');print(p.name)
