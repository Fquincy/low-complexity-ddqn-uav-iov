"""Complete saved-checkpoint inference gate for the declared original-core route."""
from run_anchor_development_v4 import *
from anchor_original_torch_core_runtime import TorchCoreSession
CONTINUATION_SOURCES=['run_anchor_posttraining_continuation.py','audit_anchor_posttraining_continuation.py',
 'anchor_continuation_inference.py','anchor_original_torch_core_runtime.py',
 'ANCHOR_TORCH_CORE_RUNTIME_AMENDMENT_2026-11-11.json']
def export_original_core(model,arm,path):
    (_,extra),meta=fused.export(model,arm,path)
    return (TorchCoreSession(model,arm),extra),meta

def frozen_inference_gate(out,fits):
    spec=protocol()['frozen_checkpoint_inference'];source=ROOT/spec['equivalence_observations_path']
    if base.old.sha(source)!=spec['equivalence_observations_sha256']:raise ValueError('frozen inference observation hash mismatch')
    states=np.load(source)['observations'];assert states.shape==(128,533)
    folder=out/'frozen_checkpoint_inference';folder.mkdir()
    np.savez_compressed(folder/'observations.npz',observations=states)
    results={arm:[] for arm in ARMS};raw=[];exports={};order_rng=np.random.default_rng(spec['order_seed'])
    fused.numba.set_num_threads(1)
    frozen=read(out/'ALL_FINAL_CHECKPOINTS_FREEZE.json')
    write(folder/'PRE_INFERENCE_FREEZE.json',dict(protocol_sha256=base.old.sha(PROTOCOL),
        runtime_amendment_sha256=base.old.sha(ROOT/'ANCHOR_TORCH_CORE_RUNTIME_AMENDMENT_2026-11-11.json'),
        continuation_source_sha256={n:base.old.sha(ROOT/n) for n in CONTINUATION_SOURCES},
        frozen_checkpoint_sha256=frozen['checkpoints'],observations_sha256=base.old.sha(folder/'observations.npz'),
        observations_source_sha256=base.old.sha(source),
        routes={arm:('validated_complete_ORT_policy' if arm in base.SHARED_ARMS else 'original_torch_core_native_decoder') for arm in ARMS},
        backend='Original PyTorch FP32 cores one-thread plus native decoder; four coupled families full-policy ORT CPU FP32'))
    orders=[];all_sessions={}
    for fit_id,(seed,_,_) in enumerate(fits):
        sessions={};models={};fit_exports={}
        for arm in ARMS:
            checkpoint=out/arm/f'fit_{fit_id:02d}'/'final.pt'
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model=make_model(arm,seed);model.load_state_dict(ck['online']);model.eval()
            if base.old.digest(model.state_dict())!=ck['online_parameter_hash']:raise ValueError('frozen inference weight hash mismatch')
            path=folder/'exports'/f'{arm}_fit_{fit_id:02d}.onnx';path.parent.mkdir(exist_ok=True)
            if arm in base.SHARED_ARMS:
                session,meta=compiled.export(model,arm,path);runtime=('complete',session)
            else:
                runtime,meta=export_original_core(model.model if arm in CORRECTED else model,anchor.ARMS[arm][0] if arm in CORRECTED else arm,path)
                runtime=('native',runtime)
                np.save(path.with_suffix('.extra.npy'),runtime[1][1])
            actions=[];values=[]
            with torch.inference_mode():
                for state in states:
                    eager_a,eager_q,_=greedy(model,torch.from_numpy(state)[None],arm)
                    if runtime[0]=='native':native_a,native_q=runtime_values(runtime[1],arm,state)
                    else:native_a,native_q=compiled.compiled_values(runtime[1],state)
                    base.np.testing.assert_array_equal(eager_a[0].numpy(),native_a[0])
                    base.np.testing.assert_allclose(eager_q.numpy(),native_q,rtol=1e-5,atol=1e-6)
                    actions.append(native_a[0]);values.append(float(native_q[0]))
            if bounded(arm):assert max(values)<=0
            sessions[arm]=runtime;models[arm]=model
            fit_exports[arm]=dict(graph_sha256=base.old.sha(path),graph_bytes=path.stat().st_size,
                parameter_sha256=base.old.digest(model.state_dict()),parameters=sum(x.numel() for x in model.parameters()),
                parameter_bytes=base.parameter_bytes(model),route='validated_complete_ORT_policy' if arm in base.SHARED_ARMS else 'original_torch_core_native_decoder',
                graph=path.name,export_seconds=meta.get('export_seconds',0.),session_startup_seconds=meta.get('session_startup_seconds',meta.get('startup_seconds',0.)))
        exports[f'fit_{fit_id:02d}']=fit_exports
        all_sessions[fit_id]=sessions
        del models
    for fit_id in range(len(fits)):
        for round_id in range(spec['timing_rounds']):
            order=list(ARMS);order_rng.shuffle(order)
            orders.append(dict(fit=fit_id,round=round_id,order=order))
    graph_hashes={fit:{arm:data['graph_sha256'] for arm,data in items.items()} for fit,items in exports.items()}
    payload_hashes={str(path.relative_to(folder)).replace('\\','/'):base.old.sha(path)
        for path in (folder/'exports').rglob('*') if path.is_file()}
    write(folder/'PRE_TIMING_FREEZE.json',dict(protocol_sha256=base.old.sha(PROTOCOL),
        inference_freeze_sha256=base.old.sha(folder/'PRE_INFERENCE_FREEZE.json'),export_hashes=graph_hashes,
        payload_hashes=payload_hashes,orders=orders,backend=read(folder/'PRE_INFERENCE_FREEZE.json')['backend'],
        timing_order_seed=spec['order_seed'],rounds=spec['timing_rounds'],warmups=spec['warmups'],
        calls=spec['calls_per_round'],fits=len(fits),paths=len(ARMS),frozen_at_ns=time.perf_counter_ns()))
    timing_freeze_sha=base.old.sha(folder/'PRE_TIMING_FREEZE.json')
    first_call_ns=None
    for item in orders:
        fit_id=item['fit'];round_id=item['round']
        for arm in item['order']:
            runtime=all_sessions[fit_id][arm]
            def infer(state):
                return runtime_values(runtime[1],arm,state)[0][0].copy() if runtime[0]=='native' else compiled.compiled_action(runtime[1],state)
            for state in states[:spec['warmups']]:infer(state)
            for index,state in enumerate(states):
                start=time.perf_counter_ns()
                if first_call_ns is None:first_call_ns=start
                action=infer(state);elapsed=time.perf_counter_ns()-start
                if action.shape!=(32,) or not np.all((action>=0)&(action<5)):raise AssertionError('invalid frozen action')
                raw.append(dict(fit=fit_id,round=round_id,arm=arm,observation=index,nanoseconds=elapsed))
    assert base.old.sha(folder/'PRE_TIMING_FREEZE.json')==timing_freeze_sha
    for path,digest in payload_hashes.items():assert base.old.sha(folder/path)==digest
    write(folder/'TIMING_STAGE_ORDER.json',dict(pre_timing_freeze_sha256=timing_freeze_sha,
        first_timed_call_ns=first_call_ns,completed_at_ns=time.perf_counter_ns()))
    del all_sessions
    metrics={}
    for arm in ARMS:
        metrics[arm]=[]
        for fit_id in range(len(fits)):
            ns=np.asarray([x['nanoseconds'] for x in raw if x['arm']==arm and x['fit']==fit_id],np.float64)
            assert len(ns)==spec['timing_rounds']*spec['calls_per_round'] and np.all(ns>0)
            metadata=exports[f'fit_{fit_id:02d}'][arm]
            metrics[arm].append(dict(fit=fit_id,median_ms=float(np.median(ns/1e6)),p95_ms=float(np.quantile(ns/1e6,.95)),
                parameter_bytes=metadata['parameter_bytes'],calls=len(ns),route=metadata['route']))
    screens={}
    for fit_id in range(len(fits)):
        primary=metrics['price_both'][fit_id];compact=metrics['additive64_nonpositive'][fit_id];wide=metrics['additive128_unbounded'][fit_id]
        screens[str(fit_id)]=dict(storage=primary['parameter_bytes']<=.5*compact['parameter_bytes'],
            median=primary['median_ms']<=wide['median_ms'],p95=primary['p95_ms']<=1.)
    report=dict(protocol=spec,observation_sha256=base.old.sha(source),exports=exports,metrics=metrics,orders=orders,
        raw_rows=len(raw),screens_per_fit=screens,all_fit_cost_screens_pass=all(all(v.values()) for v in screens.values()),
        equivalence_models=len(fits)*len(ARMS),equivalence_actions=len(fits)*len(ARMS)*len(states),
        research_training_launched=False,optimizer_updates_added=0,new_simulator_steps_added=0)
    write(folder/'verification.json',report);write(folder/'raw_timings.json',raw)
    return report
