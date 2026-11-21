"""Independent saved-artifact replay. No simulator steps or optimizer updates."""
import argparse,ast,json
from pathlib import Path
import numpy as np
import torch,onnxruntime as ort
import run_controller_v3_engineering as g
import run_controller_redesign_engineering as r
import compiled_controller_inference as c
from audit_controller_redesign_engineering import exclusion_check

def manifest_check(folder):
    files=g.read(folder/'run_manifest.json')['files']
    for name,h in files.items():assert r.old.sha(folder/name)==h,name
    actual={str(p.relative_to(folder)).replace('\\','/') for p in folder.rglob('*') if p.is_file()}
    assert actual-set(files)<={'run_manifest.json','saved_artifact_audit.json','audited_vectors.npz'},'unexpected saved artifact'
    return files

def audit(folder):
    r.configure();folder=Path(folder);files=manifest_check(folder)
    freeze=g.read(folder/'PRE_EXPORT_FREEZE.json');timing=g.read(folder/'PRE_TIMING_FREEZE.json');p=g.read(g.PROTOCOL)
    assert freeze['protocol_sha256']==timing['protocol_sha256']==r.old.sha(g.PROTOCOL)
    assert not freeze['research_training_authorized']
    for name,h in freeze['source_sha256'].items():
        assert r.old.sha(r.ROOT/name)==r.old.sha(folder/'source_snapshot'/name)==h,name
    prior=g.read(folder/'prior_input_hashes.json')
    for name,h in prior.items():assert r.old.sha(r.ROOT/name)==h,name
    eq=g.read(folder/'equivalence.json');meta=g.read(folder/'metadata.json');report=g.read(folder/'verification.json')
    assert eq['passed'] and eq['complete_actions']==6336 and len(eq['models'])==33
    assert r.old.sha(folder/'equivalence.json')==timing['equivalence_sha256']
    x=np.load(folder/'observations.npz')['observations'];assert x.shape==(192,533)
    assert r.old.sha(r.ROOT/p['observations'])==p['observations_sha256']
    np.testing.assert_array_equal(x[:128],np.load(r.ROOT/p['observations'])['observations'])
    np.testing.assert_array_equal(x[128:],np.random.default_rng(99025).uniform(0,1,(64,533)).astype(np.float32))
    actions=[];qs=[];native=[];values=[];keys=[];distributions={}
    for seed in [99021,99023,99029]:
        for arm in g.ARMS:
            key=f'{arm}_{seed}';model=g.build(arm,seed);item=meta[key];path=folder/'exports'/f'{key}.onnx'
            assert r.old.digest(model.state_dict())==item['parameter_hash']
            assert r.old.sha(path)==item['graph_sha256']==timing['graphs'][key]
            assert path.stat().st_size==item['graph_bytes']
            assert r.parameter_bytes(model)==item['parameter_bytes']
            assert sum(v.numel() for v in model.parameters())==item['parameters']
            sess=c.session(path);opts=sess.get_session_options()
            assert sess.get_providers()==['CPUExecutionProvider']
            assert opts.intra_op_num_threads==opts.inter_op_num_threads==1
            assert opts.execution_mode==ort.ExecutionMode.ORT_SEQUENTIAL and opts.graph_optimization_level==ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            aa=[];qq=[];bb=[];vv=[]
            with torch.inference_mode():
                for row in x:
                    a,q,_=g.greedy(model,arm,torch.from_numpy(row)[None]);b,v=g.compiled_values(sess,row)
                    aa.append(a[0].numpy());qq.append(float(q[0]));bb.append(b[0]);vv.append(float(v[0]))
            assert np.isfinite(qq).all() and np.isfinite(vv).all()
            np.testing.assert_array_equal(aa,bb);np.testing.assert_allclose(qq,vv,rtol=1e-5,atol=1e-6)
            if arm in g.NEW_ARMS or r.bounded(arm):assert max(vv)<=0
            actions.append(aa);qs.append(qq);native.append(bb);values.append(vv);keys.append(key)
            distributions[key]=dict(min=float(min(vv)),median=float(np.median(vv)),p95=float(np.quantile(vv,.95)),max=float(max(vv)))
    raw=g.read(folder/'raw_timings.json');assert len(raw)==7040
    rng=np.random.default_rng(99028);orders=[];expected=[]
    for round_ in range(5):
        order=list(g.ARMS);rng.shuffle(order);orders.append(order)
        for arm in order:
            expected.extend((round_,arm,ix) for ix in range(128))
    assert [(z['round'],z['arm'],z['observation']) for z in raw]==expected
    assert report['orders']==orders
    metrics={}
    for arm in g.ARMS:
        ns=np.asarray([z['nanoseconds'] for z in raw if z['arm']==arm],dtype=np.float64)
        assert ns.shape==(640,) and (ns>0).all()
        median=float(np.median(ns/1e6));p95=float(np.quantile(ns/1e6,.95))
        item=report['metrics'][arm];metadata=meta[f'{arm}_99021']
        assert (item['median_ms'],item['p95_ms'])==(median,p95)
        assert item['parameter_bytes']==metadata['parameter_bytes'] and item['graph_bytes']==metadata['graph_bytes']
        metrics[arm]=dict(median_ms=median,p95_ms=p95,parameter_bytes=metadata['parameter_bytes'])
    a=metrics[g.NEW_ARMS[0]];small=metrics['additive64_nonpositive'];wide=metrics['additive128_unbounded']
    screens=dict(online_storage=a['parameter_bytes']<=.5*small['parameter_bytes'],
        complete_action_median=a['median_ms']<=wide['median_ms'],complete_action_p95=a['p95_ms']<=1.)
    assert report['screens']==screens and report['timing_rows']==7040
    assert report['state']==('engineering_cost_passed' if all(screens.values()) else 'engineering_cost_failed')
    assert report['optimizer_updates']==report['new_simulator_steps']==0
    assert not report['research_training_started'] and not report['policy_quality_established'] and not report['word_thesis_changed']
    scope=exclusion_check()
    tree=ast.parse((r.ROOT/'controller_v3_amortized.py').read_text())
    for node in ast.walk(tree):
        name=node.id if isinstance(node,ast.Name) else node.attr if isinstance(node,ast.Attribute) else ''
        assert 'cloud' not in name.lower() and 'charg' not in name.lower()
    vectors=folder/'audited_vectors.npz'
    np.savez_compressed(vectors,keys=np.array(keys),eager_actions=np.array(actions),compiled_actions=np.array(native),eager_Q=np.array(qs),compiled_Q=np.array(values))
    result=dict(state='passed_v3_engineering_artifact_audit',gate=report['state'],complete_actions_replayed=6336,
        timing_records_recomputed=7040,manifest_entries=len(files),prior_inputs_verified=len(prior),
        audited_vectors_sha256=r.old.sha(vectors),untrained_Q_distributions=distributions,
        new_simulator_steps=0,optimizer_updates=0,research_training_started=False,scope=scope)
    r.write(folder/'saved_artifact_audit.json',result);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);a=parser.parse_args();print(json.dumps(audit(a.folder),indent=2))
