"""Frozen family-route equivalence and fair cost gate; research modes disabled."""
import argparse,shutil,time,traceback,json
from pathlib import Path
import numpy as np,torch
import controller_v3_runtime_compatibility as c
import run_controller_v3_fused_gate as prior
g=prior.g;ROOT=g.ROOT
PROTOCOL=ROOT/'LOW_COMPLEXITY_V3_RUNTIME_COMPATIBILITY_PROTOCOL_v2_2026-11-05.json'
BASE=ROOT/'validation/controller_v3_runtime_optimization_2026-11-05_01/gate_01'
SOURCES=list(dict.fromkeys([*prior.SOURCES,'controller_v3_runtime_compatibility.py',
    'run_controller_v3_compatibility_gate.py','audit_controller_v3_compatibility_gate.py',
    'test_controller_v3_compatibility.py',PROTOCOL.name]))

def check_routes(p):
    assert p['routes']['native_core_and_decoder']==list(c.NATIVE_ARMS)
    assert p['routes']['validated_complete_ORT_policy']==list(c.WHOLE_POLICY_ARMS)
    for arm in ['amortized_workload_price24','plain24_nonpositive','additive64_nonpositive','additive128_unbounded','emc_q_nonpositive','bdq64_uniform_adapted']:
        assert c.route(arm)=='native_core_and_decoder'

def regression(runtime):
    fixture=np.load(BASE/'discrepancy_fixture.npz')['state'];diag=g.read(BASE/'discrepancy_diagnosis.json')
    a,q=c.values(runtime,'shared_congestion_cost',fixture)
    np.testing.assert_array_equal(a[0],diag['eager_action'])
    np.testing.assert_allclose(q,[diag['eager_Q']],rtol=1e-5,atol=1e-6)
    assert int(a[0,2])==2
    return dict(passed=True,model_seed=99023,observation_index=187,expected_vehicle_index=2,
        expected_destination=2,complete_action=a[0].tolist(),chosen_Q=float(q[0]),
        fixture_sha256=g.r.old.sha(BASE/'discrepancy_fixture.npz'),route=c.route('shared_congestion_cost'))

def artifact_sources():
    # Research code, frozen prior reports, artifacts, and mutable tracker snapshot.
    files=g.read(ROOT/'validation/controller_v3_runtime_optimization_2026-11-05_01_COMPLETION_MANIFEST.json')['files']
    for name,h in files.items():assert g.r.old.sha(ROOT/name)==h,name
    return files

def summarize(raw,metadata):
    metrics={b:{} for b in ['legacy','compatibility']}
    for backend in metrics:
        for arm in g.ARMS:
            values=[z for z in raw if z['backend']==backend and z['arm']==arm]
            assert len(values)==640 and len({(z['round'],z['observation']) for z in values})==640
            ns=np.array([z['nanoseconds'] for z in values],np.float64);assert (ns>0).all()
            metrics[backend][arm]=dict(median_ms=float(np.median(ns/1e6)),p95_ms=float(np.quantile(ns/1e6,.95)),
                calls=640,parameter_bytes=metadata[f'{arm}_99021']['parameter_bytes'],route='complete_ORT_policy' if backend=='legacy' else c.route(arm))
    return metrics

def run(output,tests):
    g.r.configure();c.native.numba.set_num_threads(1)
    p=g.read(PROTOCOL);check_routes(p);sources={n:g.r.old.sha(ROOT/n) for n in SOURCES};test=g.read(tests)
    assert test['passed'] and test['source_sha256']==sources
    prior_artifacts=artifact_sources();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    snapshot=out/'PROJECT_STATUS_BEFORE.json';shutil.copy2(ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json',snapshot)
    dependencies=prior.dependency_info()
    g.r.write(out/'PRE_EXPORT_FREEZE.json',dict(protocol_sha256=g.r.old.sha(PROTOCOL),source_sha256=sources,
        prior_artifacts=prior_artifacts,mutable_tracker_snapshot_sha256=g.r.old.sha(snapshot),dependencies=dependencies,
        routes={a:c.route(a) for a in g.ARMS},research_training_authorized=False))
    for name in SOURCES:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(tests,out/'mechanics_tests.json')
    inputs=np.load(prior.ORIGINAL/'observations.npz')['observations'];assert inputs.shape==(192,533)
    np.savez_compressed(out/'observations.npz',observations=inputs)
    export=out/'exports';export.mkdir();metadata={};eq=[];runtimes={};legacy={};reg=None
    try:
        for seed in [99021,99023,99029]:
            for arm in g.ARMS:
                print(f'Compatibility equivalence: {arm}, seed{seed}, route={c.route(arm)}',flush=True)
                model=g.build(arm,seed);runtime,item=c.export(model,arm,seed,export/f'{arm}_{seed}.onnx')
                np.save(export/f'{arm}_{seed}_extra.npy',runtime[1])
                start=time.perf_counter();c.values(runtime,arm,inputs[0]);item['first_dispatch_including_JIT_seconds']=time.perf_counter()-start
                aa=[];qq=[];bb=[];vv=[]
                with torch.inference_mode():
                    for state in inputs:
                        a,q,_=g.greedy(model,arm,torch.from_numpy(state)[None]);b,v=c.values(runtime,arm,state)
                        aa.append(a[0].numpy());qq.append(float(q[0]));bb.append(b[0]);vv.append(float(v[0]))
                g.equivalent(aa,qq,bb,vv)
                if arm in g.NEW_ARMS or g.r.bounded(arm):assert max(vv)<=0
                restored=(g.c.session(export/f'{arm}_{seed}.onnx'),np.load(export/f'{arm}_{seed}_extra.npy'))
                for index in [0,64,191]:
                    a,q=c.values(runtime,arm,inputs[index]);b,v=c.values(restored,arm,inputs[index])
                    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(q,v)
                assert g.r.old.digest(model.state_dict())==item['parameter_hash']
                metadata[f'{arm}_{seed}']=item;eq.append(dict(arm=arm,seed=seed,cases=192,exact_actions=True,route=c.route(arm),
                    max_Q_absolute_error=float(np.max(abs(np.array(qq)-vv))),reload_exact=True,weights_unchanged=True))
                if seed==99023 and arm=='shared_congestion_cost':reg=regression(runtime)
                if seed==99021:runtimes[arm]=runtime;legacy[arm]=g.c.session(prior.ORIGINAL/'exports'/f'{arm}_99021.onnx')
        assert reg and reg['passed']
        g.r.write(out/'regression.json',reg);g.r.write(out/'metadata.json',metadata)
        g.r.write(out/'equivalence.json',dict(passed=True,models=eq,complete_actions=6336,
            scope='finite_inputs_and_initializations_not_universal_equivalence_or_policy_quality'))
        for name,h in sources.items():assert g.r.old.sha(ROOT/name)==h
        g.r.write(out/'PRE_TIMING_FREEZE.json',dict(protocol_sha256=g.r.old.sha(PROTOCOL),source_sha256=sources,
            graphs={k:v['graph_sha256'] for k,v in metadata.items()},equivalence_sha256=g.r.old.sha(out/'equivalence.json'),
            regression_sha256=g.r.old.sha(out/'regression.json'),dependencies=dependencies,order_seed=99048,
            rounds=5,warmups=32,calls=128,paths=22,routes={a:c.route(a) for a in g.ARMS}))
        raw=[];orders=[];rng=np.random.default_rng(99048)
        for round_ in range(5):
            paths=[(b,a) for b in ['legacy','compatibility'] for a in g.ARMS];rng.shuffle(paths);orders.append(paths)
            for backend,arm in paths:
                fn=(lambda state,arm=arm:g.action(legacy[arm],state)) if backend=='legacy' else (lambda state,arm=arm:c.action(runtimes[arm],arm,state))
                for state in inputs[:32]:fn(state)
                for index,state in enumerate(inputs[:128]):
                    start=time.perf_counter_ns();a=fn(state);elapsed=time.perf_counter_ns()-start
                    assert a.shape==(32,) and np.all((a>=0)&(a<5))
                    raw.append(dict(round=round_,backend=backend,arm=arm,observation=index,nanoseconds=elapsed))
        metrics=summarize(raw,metadata);screens=g.screens(metrics['compatibility'])
        result=dict(state='compatibility_cost_passed' if all(screens.values()) else 'compatibility_cost_failed',screens=screens,
            metrics=metrics,orders=orders,timing_records=len(raw),exact_equivalence_passed=True,regression_passed=True,
            cross_route_architecture_speedup_claim=False,policy_quality_established=False,research_training_started=False,
            new_simulator_steps=0,optimizer_updates=0,word_thesis_changed=False,all_research_and_final_seeds_unconsumed=True)
        for name,h in prior_artifacts.items():assert g.r.old.sha(ROOT/name)==h
        g.r.write(out/'raw_timings.json',raw);g.r.write(out/'verification.json',result)
        g.r.write(out/'run_manifest.json',dict(files={str(x.relative_to(out)).replace('\\','/'):g.r.old.sha(x) for x in out.rglob('*') if x.is_file()}))
        print(json.dumps(result,indent=2));return result
    except Exception:
        g.r.write(out/'FAILED.json',dict(error=traceback.format_exc(),partial_equivalence=eq,metadata=metadata,regression=reg,
            timing_started=(out/'PRE_TIMING_FREEZE.json').exists(),new_simulator_steps=0,optimizer_updates=0,research_training_started=False))
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['engineering','development','final'],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--tests',type=Path)
    parser.add_argument('--confirm-development-run',action='store_true');parser.add_argument('--confirm-final-run',action='store_true')
    a=parser.parse_args()
    if a.mode!='engineering':
        if not getattr(a,'confirm_'+a.mode+'_run'):parser.error('--confirm-'+a.mode+'-run required; research disabled')
        parser.error('Research disabled even with confirmation')
    if not a.tests:parser.error('--tests evidence required')
    run(a.output,a.tests)
if __name__=='__main__':main()
