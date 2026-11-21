"""Fixed saved-weight backend correction test. No research/training path."""
import gc,json,shutil,sys,time,traceback
from pathlib import Path
import numpy as np,torch,onnxruntime as ort
import run_anchor_development_v4 as r

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/'ANCHOR_RUNTIME_NUMERICAL_CORRECTION_PROTOCOL_2026-11-11.json'
RUN=ROOT/'development_runs/terminal_anchor_development_2026-11-06_01'
SOURCES=['check_anchor_numerical_correction.py',PROTOCOL.name,*r.SOURCES]

def candidate_session(path):
    options=r.compiled.options()
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])

def invoke(runtime,arm,state):
    if arm in r.base.SHARED_ARMS:return r.compiled.compiled_values(runtime[0],state)
    return r.runtime_values(runtime,arm,state)

def core_decode(raw,arm,state):
    base_arm,scaled,_=r.anchor.ARMS[arm] if arm in r.CORRECTED else (arm,False,False)
    if base_arm=='plain24_nonpositive':a,q=r.fused.flat_decode(raw,2)
    else:a,q=r.fused.v3_decode(raw,state,base_arm=='amortized_workload_price24')
    return a[None],np.array([np.float32(q)*np.float32(40-round(float(state[-1])*40) if scaled else 1.)],np.float32)

def compare(action,value,reference_action,reference_value,index):
    branches=np.flatnonzero(action[0]!=reference_action[0])
    q_ok=bool(np.allclose(value,reference_value,rtol=1e-5,atol=1e-6))
    if not len(branches) and q_ok:return None
    return dict(observation=index,branches=branches.tolist(),reference_actions=reference_action[0,branches].tolist(),
        actual_actions=action[0,branches].tolist(),q_close=q_ok,reference_q=float(reference_value[0]),actual_q=float(value[0]))

def main(output):
    output=Path(output);output.mkdir(exist_ok=False)
    r.configure();r.patch_training_core();r.fused.numba.set_num_threads(1)
    spec=r.read(PROTOCOL);spec_hash=r.base.old.sha(PROTOCOL)
    assert spec['models']==85 and spec['observations_per_model']==128
    assert r.base.old.sha(r.PROTOCOL)==spec['scientific_protocol_sha256']
    assert r.base.old.sha(RUN/'ALL_FINAL_CHECKPOINTS_FREEZE.json')==spec['checkpoint_freeze_sha256']
    frozen=r.read(RUN/'ALL_FINAL_CHECKPOINTS_FREEZE.json');plan=r.read(RUN/'runtime_plan.json')
    source=ROOT/r.protocol()['frozen_checkpoint_inference']['equivalence_observations_path']
    assert r.base.old.sha(source)==spec['observations_sha256']
    states=np.load(source)['observations'];assert states.shape==(128,533)
    source_hashes={n:r.base.old.sha(ROOT/n) for n in SOURCES}
    for name,digest in plan['source_sha256'].items():
        assert r.base.old.sha(ROOT/name)==r.base.old.sha(RUN/'source_snapshot'/name)==digest,name
    for name in SOURCES:
        dest=output/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(source,output/'observations.npz')
    r.write(output/'PRE_CHECK_FREEZE.json',dict(protocol_sha256=spec_hash,source_sha256=source_hashes,
        checkpoint_sha256=frozen['checkpoints'],observations_sha256=r.base.old.sha(source)))
    rows=[];runtime_paths={};reference_data={};events=[]
    for fit,seed in enumerate(plan['model_seeds']):
        for arm in r.ARMS:
            key=f'{arm}_fit_{fit:02d}';ckpath=RUN/arm/f'fit_{fit:02d}'/'final.pt'
            assert r.base.old.sha(ckpath)==frozen['checkpoints'][key]
            ck=torch.load(ckpath,map_location='cpu',weights_only=False)
            model=r.make_model(arm,seed);model.load_state_dict(ck['online']);model.eval()
            assert r.base.old.digest(model.state_dict())==ck['online_parameter_hash']
            graph=output/'exports'/f'{key}.onnx';graph.parent.mkdir(exist_ok=True)
            extra=np.zeros((2,5,2),np.float32)
            oldgraph=RUN/'frozen_checkpoint_inference/exports'/graph.name
            if oldgraph.exists():
                shutil.copy2(oldgraph,graph);extra=np.load(oldgraph.with_suffix('.extra.npy'))
                baseline=(r.compiled.session(graph),extra);origin='retained_original_graph'
            elif arm in r.base.SHARED_ARMS:
                session,_=r.compiled.export(model,arm,graph);baseline=(session,extra);origin='unchanged_exporter_frozen_weights'
            else:
                baseline,_=r.fused.export(model.model if arm in r.CORRECTED else model,
                    r.anchor.ARMS[arm][0] if arm in r.CORRECTED else arm,graph)
                extra=baseline[1];origin='unchanged_exporter_frozen_weights'
            np.save(graph.with_suffix('.extra.npy'),extra)
            candidate=(candidate_session(graph),extra)
            mismatches={name:[] for name in ['baseline','candidate','original_core_native_decoder']}
            count_core=0;max_q_error=0.;refs_a=[];refs_q=[]
            with torch.inference_mode():
                for index,state in enumerate(states):
                    tensor=torch.from_numpy(state)[None]
                    ea,eq,_=r.greedy(model,tensor,arm);ea=ea.numpy();eq=eq.numpy()
                    refs_a.append(ea[0]);refs_q.append(float(eq[0]))
                    for name,runtime in [('baseline',baseline),('candidate',candidate)]:
                        a,q=invoke(runtime,arm,state);bad=compare(a,q,ea,eq,index)
                        if bad is not None:mismatches[name].append(bad)
                        if name=='candidate':max_q_error=max(max_q_error,float(np.max(np.abs(q-eq))))
                    if arm in r.CORRECTED and r.anchor.ARMS[arm][0] in r.NEW_ARMS or arm in r.NEW_ARMS:
                        core=model.model if arm in r.CORRECTED else model
                        raw=core.net(tensor).numpy()[0];a,q=core_decode(raw,arm,state)
                        count_core+=1;bad=compare(a,q,ea,eq,index)
                        if bad is not None:mismatches['original_core_native_decoder'].append(bad)
            assert r.base.old.digest(model.state_dict())==ck['online_parameter_hash']
            row=dict(key=key,arm=arm,fit=fit,seed=seed,observations=len(states),graph_sha256=r.base.old.sha(graph),
                graph_origin=origin,extra_sha256=r.base.old.sha(graph.with_suffix('.extra.npy')),
                checkpoint_sha256=frozen['checkpoints'][key],parameter_bytes=r.base.parameter_bytes(model),
                max_candidate_absolute_q_error=max_q_error,core_diagnostic_observations=count_core,
                mismatches=mismatches,candidate_pass=not mismatches['candidate'])
            rows.append(row);runtime_paths[key]=(graph,arm,extra)
            reference_data[key]=dict(actions=np.asarray(refs_a).tolist(),q=refs_q)
            r.write(output/'progress.json',dict(models_checked=len(rows),total=85,last=key,
                candidate_failed_models=sum(not x['candidate_pass'] for x in rows)))
            r.write(output/'model_checks.json',rows)
            print(json.dumps(dict(model=key,candidate_bad=len(mismatches['candidate']),baseline_bad=len(mismatches['baseline']))),flush=True)
            del model,ck,baseline,candidate;gc.collect()
    assert len(rows)==85
    regression=next(x for x in rows if x['key']=='no_price_both_fit_00')
    observed=[x for x in regression['mismatches']['baseline'] if x['observation']==110 and 18 in x['branches']]
    assert len(observed)==1
    r.write(output/'reference_predictions.json',reference_data)
    all_pass=all(x['candidate_pass'] for x in rows)
    if all_pass:
        measure_cost(output,runtime_paths,spec)
    for name,digest in source_hashes.items():assert r.base.old.sha(ROOT/name)==digest,name
    for key,digest in frozen['checkpoints'].items():
        arm,fit=key.rsplit('_fit_',1);assert r.base.old.sha(RUN/arm/('fit_'+fit)/'final.pt')==digest
    r.write(output/'verification.json',dict(state='equivalence_passed_cost_measured' if all_pass else 'candidate_failed_exact_equivalence_no_timing',
        models_checked=85,observations_per_model=128,model_observation_checks=10880,
        candidate_failed_models=sum(not x['candidate_pass'] for x in rows),
        baseline_failed_models=sum(bool(x['mismatches']['baseline']) for x in rows),
        all_candidate_actions_and_values_pass=all_pass,near_tie_baseline_regression_reproduced=True,
        near_tie_candidate_pass=not any(x['observation']==110 for x in regression['mismatches']['candidate']),
        cost_measured=all_pass,new_simulator_slots=0,optimizer_updates=0,heldout_evaluation_started=False,
        original_run_modified=False,research_launched=False,word_thesis_changed=False))
    files={str(p.relative_to(output)).replace('\\','/'):r.base.old.sha(p) for p in output.rglob('*') if p.is_file()}
    r.write(output/'COMPLETION_MANIFEST.json',dict(files=files))
    print(json.dumps(r.read(output/'verification.json')),flush=True)

def measure_cost(output,paths,spec):
    """One fixed benchmark, only reachable after all models pass equivalence."""
    states=np.load(output/'observations.npz')['observations'];raw=[];orders=[]
    rng=np.random.default_rng(623600003);runtimes={}
    for key,(graph,arm,extra) in paths.items():runtimes[key]=(candidate_session(graph),extra)
    r.write(output/'PRE_TIMING_FREEZE.json',dict(payloads={str(p.relative_to(output)):r.base.old.sha(p) for p in (output/'exports').rglob('*') if p.is_file()},
        protocol_sha256=r.base.old.sha(PROTOCOL),source_sha256=r.read(output/'PRE_CHECK_FREEZE.json')['source_sha256']))
    for fit in range(5):
        for round_id in range(5):
            order=list(r.ARMS);rng.shuffle(order);orders.append(dict(fit=fit,round=round_id,order=order))
    r.write(output/'TIMING_ORDERS.json',orders)
    for item in orders:
        for arm in item['order']:
            runtime=runtimes[f"{arm}_fit_{item['fit']:02d}"]
            def action(state):return invoke(runtime,arm,state)[0][0].copy()
            for state in states[:32]:action(state)
            for index,state in enumerate(states):
                start=time.perf_counter_ns();a=action(state);ns=time.perf_counter_ns()-start
                assert a.shape==(32,)
                raw.append(dict(fit=item['fit'],round=item['round'],arm=arm,observation=index,nanoseconds=ns))
    r.write(output/'raw_timings.json',raw)
    metrics={};rows=r.read(output/'model_checks.json')
    for arm in r.ARMS:
        metrics[arm]=[]
        for fit in range(5):
            ns=np.array([x['nanoseconds'] for x in raw if x['arm']==arm and x['fit']==fit],np.float64)
            meta=next(x for x in rows if x['arm']==arm and x['fit']==fit)
            metrics[arm].append(dict(median_ms=float(np.median(ns)/1e6),p95_ms=float(np.quantile(ns,.95)/1e6),parameter_bytes=meta['parameter_bytes']))
    screens=[]
    for fit in range(5):
        p=metrics['price_both'][fit];c=metrics['additive64_nonpositive'][fit];w=metrics['additive128_unbounded'][fit]
        screens.append(dict(fit=fit,storage=p['parameter_bytes']<=c['parameter_bytes']*.5,median=p['median_ms']<=w['median_ms'],p95=p['p95_ms']<=1.))
    r.write(output/'cost.json',dict(metrics=metrics,screens=screens,all_fit_cost_screens_pass=all(all(x[k] for k in ['storage','median','p95']) for x in screens)))

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Explicit new engineering output directory required')
    main(sys.argv[1])
