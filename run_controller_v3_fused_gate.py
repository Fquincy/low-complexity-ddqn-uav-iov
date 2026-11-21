"""One predeclared native-fusion attempt; no research modes/rollouts/updates."""
import argparse,json,shutil,time,traceback
from pathlib import Path
import numpy as np,torch
import controller_v3_fused_runtime as f
import run_controller_v3_engineering as g

ROOT=g.ROOT
PROTOCOL=ROOT/'LOW_COMPLEXITY_V3_FUSED_RUNTIME_PROTOCOL_2026-11-05.json'
ORIGINAL=ROOT/'validation/controller_v3_engineering_2026-11-05_01'
SOURCES=list(dict.fromkeys([*g.SOURCES,'controller_v3_fused_runtime.py','run_controller_v3_fused_gate.py',
    'audit_controller_v3_fused_gate.py','test_controller_v3_fused_runtime.py','profile_controller_v3_compiled.py',PROTOCOL.name]))

def dependency_info():
    import llvmlite.binding as binding
    assert f.numba.__version__=='0.65.1' and f.llvmlite.__version__=='0.47.0'
    binaries={str(p.relative_to(f.DEPS)):g.r.old.sha(p) for p in f.DEPS.rglob('*') if p.suffix.lower() in ('.pyd','.dll')}
    return dict(numba=f.numba.__version__,llvmlite=f.llvmlite.__version__,llvm=list(binding.llvm_version_info),
        dependency_path=str(f.DEPS),native_binary_hashes=binaries,numpy=np.__version__,torch=torch.__version__,
        onnxruntime=g.ort.__version__,numba_threads=f.numba.get_num_threads(),hardware=g.r.hardware())

def summarize(raw,metadata):
    result={backend:{} for backend in ['legacy','hybrid']}
    for backend in result:
        for arm in g.ARMS:
            rows=[z for z in raw if z['backend']==backend and z['arm']==arm]
            assert len(rows)==640 and len({(z['round'],z['observation']) for z in rows})==640
            ns=np.array([z['nanoseconds'] for z in rows],dtype=np.float64)
            assert np.all(ns>0)
            result[backend][arm]=dict(median_ms=float(np.median(ns/1e6)),p95_ms=float(np.quantile(ns/1e6,.95)),
                parameter_bytes=metadata[f'{arm}_99021']['parameter_bytes'],calls=len(rows))
    return result

def run(output,tests):
    g.r.configure();f.numba.set_num_threads(1);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    test=g.read(tests);source_hashes={n:g.r.old.sha(ROOT/n) for n in SOURCES}
    assert test['passed'] and test['source_sha256']==source_hashes
    p=g.read(PROTOCOL);deps=dependency_info()
    prior=g.read(ROOT/'validation/controller_v3_engineering_2026-11-05_01_COMPLETION_MANIFEST.json')['files']
    for name,h in prior.items():assert g.r.old.sha(ROOT/name)==h,name
    g.r.write(out/'PRE_EXPORT_FREEZE.json',dict(protocol_sha256=g.r.old.sha(PROTOCOL),source_sha256=source_hashes,
        original_hashes=prior,dependencies=deps,research_training_authorized=False))
    shutil.copy2(tests,out/'mechanics_tests.json')
    for name in SOURCES:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    states=np.load(ORIGINAL/'observations.npz')['observations'];assert states.shape==(192,533)
    np.savez_compressed(out/'observations.npz',observations=states)
    exports=out/'exports';exports.mkdir();metadata={};equivalence=[];runtimes={};legacies={}
    try:
        for seed in [99021,99023,99029]:
            for arm in g.ARMS:
                print(f'Core export/native equivalence: {arm}, seed{seed}',flush=True)
                model=g.build(arm,seed);runtime,item=f.export(model,arm,exports/f'{arm}_{seed}.onnx')
                np.save(exports/f'{arm}_{seed}_extra.npy',runtime[1])
                start=time.perf_counter();f.values(runtime,arm,states[0]);item['first_dispatch_including_JIT_seconds']=time.perf_counter()-start
                aa=[];qq=[];bb=[];vv=[]
                with torch.inference_mode():
                    for state in states:
                        a,q,_=g.greedy(model,arm,torch.from_numpy(state)[None]);b,v=f.values(runtime,arm,state)
                        aa.append(a[0].numpy());qq.append(float(q[0]));bb.append(b[0]);vv.append(float(v[0]))
                g.equivalent(aa,qq,bb,vv)
                if arm in g.NEW_ARMS or g.r.bounded(arm):assert max(vv)<=0
                reloaded=(g.c.session(exports/f'{arm}_{seed}.onnx'),np.load(exports/f'{arm}_{seed}_extra.npy'))
                for index in [0,64,191]:
                    a,q=f.values(runtime,arm,states[index]);b,v=f.values(reloaded,arm,states[index])
                    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(q,v)
                assert g.r.old.digest(model.state_dict())==item['parameter_hash']
                metadata[f'{arm}_{seed}']=item
                equivalence.append(dict(arm=arm,seed=seed,cases=192,exact_actions=True,
                    max_Q_absolute_error=float(np.max(abs(np.array(qq)-vv))),reload_exact=True,weights_unchanged=True))
                if seed==99021:
                    runtimes[arm]=runtime;legacies[arm]=g.c.session(ORIGINAL/'exports'/f'{arm}_99021.onnx')
        g.r.write(out/'metadata.json',metadata)
        g.r.write(out/'equivalence.json',dict(passed=True,complete_actions=6336,models=equivalence,
            scope='finite_saved_and_synthetic_cases_not_universal_equivalence_or_quality'))
        for name,h in source_hashes.items():assert g.r.old.sha(ROOT/name)==h
        g.r.write(out/'PRE_TIMING_FREEZE.json',dict(protocol_sha256=g.r.old.sha(PROTOCOL),
            equivalence_sha256=g.r.old.sha(out/'equivalence.json'),sources=source_hashes,
            graphs={k:v['graph_sha256'] for k,v in metadata.items()},dependencies=deps,
            timing_order_seed=99038,rounds=5,warmups=32,calls=128,paths=22))
        raw=[];orders=[];rng=np.random.default_rng(99038)
        for round_ in range(5):
            paths=[(b,a) for b in ['legacy','hybrid'] for a in g.ARMS];rng.shuffle(paths);orders.append(paths)
            for backend,arm in paths:
                fn=(lambda s,arm=arm:g.action(legacies[arm],s)) if backend=='legacy' else (lambda s,arm=arm:f.action(runtimes[arm],arm,s))
                for state in states[:32]:fn(state)
                for index,state in enumerate(states[:128]):
                    start=time.perf_counter_ns();a=fn(state);elapsed=time.perf_counter_ns()-start
                    assert a.shape==(32,) and np.all((a>=0)&(a<5))
                    raw.append(dict(round=round_,backend=backend,arm=arm,observation=index,nanoseconds=elapsed))
        metrics=summarize(raw,metadata);screens=g.screens(metrics['hybrid'])
        result=dict(state='fused_cost_passed' if all(screens.values()) else 'fused_cost_failed',screens=screens,
            metrics=metrics,orders=orders,timing_records=len(raw),equivalence_passed=True,
            new_simulator_steps=0,optimizer_updates=0,research_training_started=False,word_thesis_changed=False,
            policy_quality_established=False,final_seeds_unconsumed=True)
        for name,h in prior.items():assert g.r.old.sha(ROOT/name)==h,name
        g.r.write(out/'raw_timings.json',raw);g.r.write(out/'verification.json',result)
        g.r.write(out/'run_manifest.json',dict(files={str(x.relative_to(out)).replace('\\','/'):g.r.old.sha(x) for x in out.rglob('*') if x.is_file()}))
        print(json.dumps(result,indent=2));return result
    except Exception:
        g.r.write(out/'FAILED.json',dict(error=traceback.format_exc(),partial_equivalence=equivalence,metadata=metadata,
            timing_started=(out/'PRE_TIMING_FREEZE.json').exists(),optimizer_updates=0,new_simulator_steps=0,research_training_started=False))
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['engineering','development','final'],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--tests',type=Path)
    parser.add_argument('--confirm-development-run',action='store_true');parser.add_argument('--confirm-final-run',action='store_true')
    a=parser.parse_args()
    if a.mode!='engineering':
        if not getattr(a,'confirm_'+a.mode+'_run'):parser.error('--confirm-'+a.mode+'-run required; research disabled')
        parser.error('Research disabled even with confirmation')
    if not a.tests:parser.error('mechanics --tests evidence required')
    run(a.output,a.tests)
if __name__=='__main__':main()
