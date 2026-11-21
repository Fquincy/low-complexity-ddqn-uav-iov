"""Independent replay, route/graph integrity and timing recalculation."""
import argparse,json
from pathlib import Path
import numpy as np,torch,onnxruntime as ort
import controller_v3_runtime_compatibility as c
import run_controller_v3_compatibility_gate as r
from audit_controller_redesign_engineering import exclusion_check
g=r.g

def check_manifest(folder):
    files=g.read(folder/'run_manifest.json')['files']
    for name,h in files.items():assert g.r.old.sha(folder/name)==h,name
    actual={str(x.relative_to(folder)).replace('\\','/') for x in folder.rglob('*') if x.is_file()}
    assert actual-set(files)<={'run_manifest.json','saved_artifact_audit.json','audited_vectors.npz'}
    return files

def audit(folder):
    g.r.configure();c.native.numba.set_num_threads(1);folder=Path(folder);files=check_manifest(folder)
    pre=g.read(folder/'PRE_EXPORT_FREEZE.json');timing=g.read(folder/'PRE_TIMING_FREEZE.json');p=g.read(r.PROTOCOL)
    assert pre['protocol_sha256']==timing['protocol_sha256']==g.r.old.sha(r.PROTOCOL)
    assert not pre['research_training_authorized']
    expected_routes={a:('validated_complete_ORT_policy' if a in p['routes']['validated_complete_ORT_policy'] else 'native_core_and_decoder') for a in g.ARMS}
    assert pre['routes']==timing['routes']==expected_routes
    for name,h in pre['source_sha256'].items():assert g.r.old.sha(g.ROOT/name)==g.r.old.sha(folder/'source_snapshot'/name)==h
    for name,h in pre['prior_artifacts'].items():
        path=folder/'PROJECT_STATUS_BEFORE.json' if name=='LOW_COMPLEXITY_PROJECT_STATUS.json' else g.ROOT/name
        assert g.r.old.sha(path)==h,name
    assert r.prior.dependency_info()==pre['dependencies']==timing['dependencies']
    eq=g.read(folder/'equivalence.json');meta=g.read(folder/'metadata.json');report=g.read(folder/'verification.json')
    assert eq['passed'] and len(eq['models'])==33 and eq['complete_actions']==6336
    assert timing['equivalence_sha256']==g.r.old.sha(folder/'equivalence.json')
    assert timing['regression_sha256']==g.r.old.sha(folder/'regression.json')
    states=np.load(folder/'observations.npz')['observations'];assert states.shape==(192,533)
    np.testing.assert_array_equal(states,np.load(r.prior.ORIGINAL/'observations.npz')['observations'])
    arrays=[];native=[];qs=[];values=[];keys=[];regression_checked=False
    for seed in [99021,99023,99029]:
        for arm in g.ARMS:
            key=f'{arm}_{seed}';item=meta[key];path=folder/'exports'/f'{key}.onnx';model=g.build(arm,seed)
            assert item['route']==expected_routes[arm] and item['model_seed']==seed
            assert g.r.old.sha(path)==item['graph_sha256']==timing['graphs'][key]
            assert path.stat().st_size==item['graph_bytes'] and g.r.parameter_bytes(model)==item['parameter_bytes']
            assert g.r.old.digest(model.state_dict())==item['parameter_hash']
            if expected_routes[arm]=='validated_complete_ORT_policy':
                assert g.r.old.sha(path)==g.r.old.sha(c.ORIGINAL/'exports'/f'{key}.onnx')
                assert item['reused_validated_full_graph']
            extra=np.load(folder/'exports'/f'{key}_extra.npy')
            if arm=='emc_q_nonpositive':np.testing.assert_array_equal(extra,model.movement_embeddings.detach().numpy())
            else:np.testing.assert_array_equal(extra,np.zeros((2,5,2),np.float32))
            session=g.c.session(path);options=session.get_session_options()
            assert session.get_providers()==['CPUExecutionProvider'] and options.intra_op_num_threads==options.inter_op_num_threads==1
            assert options.execution_mode==ort.ExecutionMode.ORT_SEQUENTIAL and options.graph_optimization_level==ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            aa=[];qq=[];bb=[];vv=[]
            with torch.inference_mode():
                for state in states:
                    a,q,_=g.greedy(model,arm,torch.from_numpy(state)[None]);b,v=c.values((session,extra),arm,state)
                    aa.append(a[0].numpy());qq.append(float(q[0]));bb.append(b[0]);vv.append(float(v[0]))
            assert np.isfinite(qq).all() and np.isfinite(vv).all()
            np.testing.assert_array_equal(aa,bb);np.testing.assert_allclose(qq,vv,rtol=1e-5,atol=1e-6)
            if arm in g.NEW_ARMS or g.r.bounded(arm):assert max(vv)<=0
            if arm=='shared_congestion_cost' and seed==99023:
                diag=g.read(r.BASE/'discrepancy_diagnosis.json')
                np.testing.assert_array_equal(bb[187],diag['eager_action']);assert int(bb[187][2])==2
                recorded=g.read(folder/'regression.json');assert recorded['passed'] and recorded['fixture_sha256']==g.r.old.sha(r.BASE/'discrepancy_fixture.npz')
                np.testing.assert_array_equal(recorded['complete_action'],bb[187]);regression_checked=True
            keys.append(key);arrays.append(aa);native.append(bb);qs.append(qq);values.append(vv)
    raw=g.read(folder/'raw_timings.json');assert len(raw)==14080
    order_rng=np.random.default_rng(99048);orders=[];expected=[]
    for round_ in range(5):
        order=[(b,a) for b in ['legacy','compatibility'] for a in g.ARMS];order_rng.shuffle(order);orders.append([list(x) for x in order])
        for backend,arm in order:expected.extend((round_,backend,arm,i) for i in range(128))
    assert report['orders']==orders and [(z['round'],z['backend'],z['arm'],z['observation']) for z in raw]==expected
    for backend in ['legacy','compatibility']:
        for arm in g.ARMS:
            ns=np.array([z['nanoseconds'] for z in raw if z['backend']==backend and z['arm']==arm],np.float64)
            assert len(ns)==640 and (ns>0).all();item=report['metrics'][backend][arm]
            assert item['median_ms']==float(np.median(ns/1e6)) and item['p95_ms']==float(np.quantile(ns/1e6,.95))
            assert item['parameter_bytes']==meta[f'{arm}_99021']['parameter_bytes']
            if backend=='compatibility':assert item['route']==expected_routes[arm]
    metrics=report['metrics']['compatibility'];a=metrics['amortized_workload_price24'];small=metrics['additive64_nonpositive'];wide=metrics['additive128_unbounded']
    assert all(expected_routes[x]=='native_core_and_decoder' for x in ['amortized_workload_price24','additive64_nonpositive','additive128_unbounded'])
    screens=dict(online_storage=a['parameter_bytes']<=small['parameter_bytes']*.5,complete_action_median=a['median_ms']<=wide['median_ms'],complete_action_p95=a['p95_ms']<=1.)
    assert report['screens']==screens and report['state']==('compatibility_cost_passed' if all(screens.values()) else 'compatibility_cost_failed')
    assert report['new_simulator_steps']==report['optimizer_updates']==0
    assert not report['research_training_started'] and not report['word_thesis_changed'] and not report['policy_quality_established']
    assert not report['cross_route_architecture_speedup_claim']
    for fn in [c.native.softplus,c.native.flat_decode,c.native.v3_decode,c.native.emc_decode,c.native.bdq_decode]:
        assert fn.nopython_signatures and not fn.targetoptions['fastmath'] and not fn.targetoptions['parallel'] and fn.targetoptions['boundscheck']
    vectors=folder/'audited_vectors.npz'
    np.savez_compressed(vectors,keys=np.array(keys),eager_actions=np.array(arrays),compatibility_actions=np.array(native),eager_Q=np.array(qs),compatibility_Q=np.array(values))
    result=dict(state='passed_compatibility_artifact_audit',cost_gate=report['state'],complete_actions_replayed=6336,
        timing_records_recomputed=14080,regression_reproduced=regression_checked,fixed_routes_verified=True,
        principal_references_same_native_treatment=True,historical_v2_graphs_unchanged=12,
        manifest_entries=len(files),prior_artifacts_preserved=len(pre['prior_artifacts']),vectors_sha256=g.r.old.sha(vectors),
        new_simulator_steps=0,optimizer_updates=0,research_training_started=False,scope=exclusion_check())
    g.r.write(folder/'saved_artifact_audit.json',result);print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();audit(a.folder)
