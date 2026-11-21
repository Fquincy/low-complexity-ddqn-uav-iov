"""Read-only independent replay/recalculation of saved native-fusion evidence."""
import argparse,json
from pathlib import Path
import numpy as np,torch,onnxruntime as ort
import run_controller_v3_fused_gate as runner
import controller_v3_fused_runtime as f
from audit_controller_redesign_engineering import exclusion_check

g=runner.g
def manifest_check(folder):
    files=g.read(folder/'run_manifest.json')['files']
    for name,h in files.items():assert g.r.old.sha(folder/name)==h,name
    actual={str(p.relative_to(folder)).replace('\\','/') for p in folder.rglob('*') if p.is_file()}
    assert actual-set(files)<={'run_manifest.json','saved_artifact_audit.json','audited_vectors.npz'}
    return files

def audit(folder):
    g.r.configure();f.numba.set_num_threads(1);folder=Path(folder);files=manifest_check(folder)
    before=g.read(folder/'PRE_EXPORT_FREEZE.json');timing=g.read(folder/'PRE_TIMING_FREEZE.json')
    assert before['protocol_sha256']==timing['protocol_sha256']==g.r.old.sha(runner.PROTOCOL)
    assert not before['research_training_authorized']
    for name,h in before['source_sha256'].items():assert g.r.old.sha(runner.ROOT/name)==g.r.old.sha(folder/'source_snapshot'/name)==h
    for name,h in before['original_hashes'].items():assert g.r.old.sha(runner.ROOT/name)==h
    dep=runner.dependency_info();assert dep==before['dependencies']==timing['dependencies']
    eq=g.read(folder/'equivalence.json');meta=g.read(folder/'metadata.json');report=g.read(folder/'verification.json')
    assert eq['passed'] and eq['complete_actions']==6336 and len(eq['models'])==33
    assert timing['equivalence_sha256']==g.r.old.sha(folder/'equivalence.json')
    states=np.load(folder/'observations.npz')['observations']
    np.testing.assert_array_equal(states,np.load(runner.ORIGINAL/'observations.npz')['observations'])
    vectors=[];native=[];qs=[];values=[];keys=[]
    for seed in [99021,99023,99029]:
        for arm in g.ARMS:
            key=f'{arm}_{seed}';model=g.build(arm,seed);item=meta[key];path=folder/'exports'/f'{key}.onnx'
            assert g.r.old.digest(model.state_dict())==item['parameter_hash']
            assert g.r.old.sha(path)==item['graph_sha256']==timing['graphs'][key]
            assert path.stat().st_size==item['graph_bytes'] and g.r.parameter_bytes(model)==item['parameter_bytes']
            extra=np.load(folder/'exports'/f'{key}_extra.npy')
            expected=model.movement_embeddings.detach().numpy() if arm=='emc_q_nonpositive' else np.zeros((2,5,2),np.float32)
            np.testing.assert_array_equal(extra,expected)
            sess=g.c.session(path);options=sess.get_session_options()
            assert sess.get_providers()==['CPUExecutionProvider'] and options.intra_op_num_threads==options.inter_op_num_threads==1
            assert options.execution_mode==ort.ExecutionMode.ORT_SEQUENTIAL and options.graph_optimization_level==ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            aa=[];qq=[];bb=[];vv=[]
            with torch.inference_mode():
                for state in states:
                    a,q,_=g.greedy(model,arm,torch.from_numpy(state)[None]);b,v=f.values((sess,extra),arm,state)
                    aa.append(a[0].numpy());qq.append(float(q[0]));bb.append(b[0]);vv.append(float(v[0]))
            assert np.isfinite(qq).all() and np.isfinite(vv).all()
            np.testing.assert_array_equal(aa,bb);np.testing.assert_allclose(qq,vv,rtol=1e-5,atol=1e-6)
            if arm in g.NEW_ARMS or g.r.bounded(arm):assert max(vv)<=0
            keys.append(key);vectors.append(aa);native.append(bb);qs.append(qq);values.append(vv)
    raw=g.read(folder/'raw_timings.json');assert len(raw)==14080
    rng=np.random.default_rng(99038);expected=[];orders=[]
    for round_ in range(5):
        order=[(b,a) for b in ['legacy','hybrid'] for a in g.ARMS];rng.shuffle(order);orders.append([list(x) for x in order])
        for backend,arm in order:expected.extend((round_,backend,arm,i) for i in range(128))
    assert [(z['round'],z['backend'],z['arm'],z['observation']) for z in raw]==expected
    assert report['orders']==orders
    for backend in ['legacy','hybrid']:
        for arm in g.ARMS:
            ns=np.array([z['nanoseconds'] for z in raw if z['backend']==backend and z['arm']==arm],np.float64)
            assert len(ns)==640 and np.all(ns>0)
            item=report['metrics'][backend][arm]
            assert item['median_ms']==float(np.median(ns/1e6)) and item['p95_ms']==float(np.quantile(ns/1e6,.95))
            assert item['parameter_bytes']==meta[f'{arm}_99021']['parameter_bytes']
    metrics=report['metrics']['hybrid'];a=metrics[g.NEW_ARMS[0]];e=metrics['additive64_nonpositive'];wide=metrics['additive128_unbounded']
    screens=dict(online_storage=a['parameter_bytes']<=e['parameter_bytes']*.5,
        complete_action_median=a['median_ms']<=wide['median_ms'],complete_action_p95=a['p95_ms']<=1.)
    assert report['screens']==screens and report['state']==('fused_cost_passed' if all(screens.values()) else 'fused_cost_failed')
    assert report['new_simulator_steps']==report['optimizer_updates']==0 and not report['research_training_started']
    assert not report['word_thesis_changed'] and not report['policy_quality_established']
    for func in [f.softplus,f.flat_decode,f.v3_decode,f.bdq_decode,f.emc_decode,f.shared_decode]:
        assert func.nopython_signatures and not func.targetoptions['fastmath'] and not func.targetoptions['parallel'] and func.targetoptions['boundscheck']
    vector_path=folder/'audited_vectors.npz'
    np.savez_compressed(vector_path,keys=np.array(keys),eager_actions=np.array(vectors),native_actions=np.array(native),eager_Q=np.array(qs),native_Q=np.array(values))
    result=dict(state='passed_fused_runtime_artifact_audit',cost_gate=report['state'],complete_actions_replayed=6336,
        timing_records_recomputed=14080,manifest_entries=len(files),original_artifacts_unchanged=len(before['original_hashes']),
        scope=exclusion_check(),native_modes_verified=True,dependencies_verified=True,
        vectors_sha256=g.r.old.sha(vector_path),new_simulator_steps=0,optimizer_updates=0,research_training_started=False)
    g.r.write(folder/'saved_artifact_audit.json',result);print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();audit(a.folder)
