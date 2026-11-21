"""Read-only independent replay audit of the compiled-inference engineering gate."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
import compiled_controller_inference as c
import run_compiled_inference_gate as gate
import run_controller_redesign_engineering as r
from audit_controller_redesign_engineering import exclusion_check

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def verify_manifest(folder):
    manifest=read(folder/'run_manifest.json')['files']
    for name,h in manifest.items():assert r.old.sha(folder/name)==h,name
    actual={str(p.relative_to(folder)).replace('\\','/') for p in folder.rglob('*') if p.is_file()}
    assert actual-set(manifest)<={'run_manifest.json','saved_artifact_audit.json','audited_equivalence_vectors.npz'}
    return manifest

def assert_equivalent(a,q,b,v):
    assert np.isfinite(q).all() and np.isfinite(v).all(),'nonfinite Q'
    np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(q,v,rtol=1e-5,atol=1e-6)

def audit(folder):
    r.configure();folder=Path(folder);manifest=verify_manifest(folder)
    before=read(folder/'PRE_EXPORT_FREEZE.json');timing_freeze=read(folder/'PRE_TIMING_FREEZE.json')
    p=read(gate.PROTOCOL)
    assert before['protocol_sha256']==timing_freeze['protocol_sha256']==r.old.sha(gate.PROTOCOL)
    for name,h in before['source_sha256'].items():
        assert r.old.sha(r.ROOT/name)==h,name
        assert r.old.sha(folder/'source_snapshot'/name)==h,name
    eq=read(folder/'equivalence.json');metadata=read(folder/'export_metadata.json');report=read(folder/'verification.json')
    assert eq['state']=='passed' and eq['total_complete_actions']==4608 and len(eq['models'])==24
    assert r.old.sha(folder/'equivalence.json')==timing_freeze['equivalence_sha256']
    x=np.load(folder/'equivalence_observations.npz')['observations'];assert x.shape==(192,533)
    common=np.load(r.ROOT/p['observations'])['observations'];np.testing.assert_array_equal(x[:128],common)
    np.testing.assert_array_equal(x[128:],np.random.default_rng(99005).uniform(0.,1.,(64,533)).astype(np.float32))
    vectors=[];qs=[];native_vectors=[];native_qs=[];keys=[]
    for seed in [99001,99006,99007]:
        for arm in r.ARMS:
            model=r.build(arm,seed,r.fixture_cfg())['online'].eval();item=metadata[f'{arm}_{seed}']
            assert r.old.digest(model.state_dict())==item['source_parameter_hash']
            path=folder/'exports'/f'{arm}_seed{seed}.onnx'
            assert r.old.sha(path)==item['onnx_sha256']==timing_freeze['all_exports_sha256'][f'{arm}_{seed}']
            assert path.stat().st_size==item['serialized_onnx_bytes']
            session=c.session(path);options=session.get_session_options()
            assert session.get_providers()==['CPUExecutionProvider']
            assert options.intra_op_num_threads==options.inter_op_num_threads==1
            assert options.execution_mode==ort.ExecutionMode.ORT_SEQUENTIAL
            assert options.graph_optimization_level==ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            aa=[];qq=[];bb=[];vv=[]
            with torch.inference_mode():
                for state in x:
                    actions,value,_=r.greedy(model,torch.from_numpy(state)[None],arm)
                    native_a,native_q=c.compiled_values(session,state)
                    aa.append(actions[0].numpy());qq.append(float(value[0]));bb.append(native_a[0]);vv.append(float(native_q[0]))
            assert_equivalent(aa,qq,bb,vv)
            if r.bounded(arm):assert max(vv)<=0
            vectors.append(aa);qs.append(qq);native_vectors.append(bb);native_qs.append(vv);keys.append(f'{arm}_{seed}')
    raw=read(folder/'raw_timings.json');assert len(raw)==10240
    for backend in ['eager','compiled']:
        for arm in r.ARMS:
            rows=[z for z in raw if z['backend']==backend and z['arm']==arm]
            assert len(rows)==640
            assert {(z['round'],z['observation']) for z in rows}=={(i,j) for i in range(5) for j in range(128)}
            ns=np.array([z['nanoseconds'] for z in rows],dtype=np.float64);assert (ns>0).all()
            item=report['metrics'][backend][arm]
            assert item['median_ms']==float(np.median(ns/1e6))
            assert item['p95_ms']==float(np.quantile(ns/1e6,.95))
            assert item['parameter_bytes']==metadata[f'{arm}_99001']['parameter_bytes']
    metrics=report['metrics']['compiled'];primary=metrics['shared_congestion_cost'];small=metrics['additive64_nonpositive'];wide=metrics['additive128_unbounded']
    screens=dict(online_storage=primary['parameter_bytes']/small['parameter_bytes']<=.5,
        complete_action_median=primary['median_ms']/wide['median_ms']<=1.,complete_action_p95=primary['p95_ms']<=1.)
    assert report['screens']==screens
    assert report['state']==('compiled_cost_gate_passed' if all(screens.values()) else 'compiled_cost_gate_failed')
    assert report['new_simulator_steps']==report['optimizer_updates']==0 and not report['research_training_started']
    assert not before['research_training_authorized']
    # Preserve replayed vectors separately from the original pre-audit manifest.
    path=folder/'audited_equivalence_vectors.npz'
    np.savez_compressed(path,keys=np.array(keys),eager_actions=np.array(vectors),compiled_actions=np.array(native_vectors),
        eager_Q=np.array(qs),compiled_Q=np.array(native_qs))
    result=dict(state='passed_compiled_artifact_audit',compiled_cost_gate=report['state'],
        complete_actions_independently_replayed=4608,timing_rows_recomputed=10240,manifest_entries_checked=len(manifest),
        equivalence_vectors_sha256=r.old.sha(path),all_frozen_source_hashes_unchanged=True,
        simulator_steps_added=0,optimizer_updates_added=0,research_training_started=False,scope=exclusion_check())
    r.write(folder/'saved_artifact_audit.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();print(json.dumps(audit(a.folder),indent=2))
