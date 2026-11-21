"""One predeclared compilation/equivalence/timing engineering gate; no learner."""
from __future__ import annotations
import argparse,json,shutil,time,traceback
from pathlib import Path
import importlib.metadata
import numpy as np
import onnx
import onnxruntime as ort
import torch
import controller_redesign_v2 as models
import compiled_controller_inference as c
import run_controller_redesign_engineering as r

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/'LOW_COMPLEXITY_INFERENCE_OPTIMIZATION_PROTOCOL_v1_1_2026-11-05.json'

class DecoderFixture(torch.nn.Module):
    def forward(self,u,w,L,beta,motion,base):
        a,q,_=models.decode_cost_tables(u,w,L,beta,motion,base,enabled=True)
        return a,q

class Expm1Fixture(torch.nn.Module):
    def forward(self,x):return torch.expm1(x)

def check_small_fixtures(folder):
    """Exercise ties, oscillation repair and both sides of the fixed tolerance."""
    u=torch.tensor([[[10.,0.,0.,20.,20.],[10.,0.,0.,20.,20.]]])
    w=torch.ones_like(u);w[:,:,0]=0
    L=torch.zeros(1,5);beta=torch.tensor([[0.,1.,1.,1.,1.]])
    motion=torch.zeros(1,2,5);base=torch.zeros(1)
    fixture=DecoderFixture();path=folder/'decoder_fixtures.onnx'
    torch.onnx.export(fixture,(u,w,L,beta,motion,base),str(path),opset_version=17,dynamo=False,
        input_names=['u','w','L','beta','motion','base'],output_names=['actions','cost'])
    onnx.checker.check_model(onnx.load(str(path)));session=c.session(path)
    cases=[('symmetric',u,w,L,beta,motion,base),
           ('all_ties',torch.zeros_like(u),w,L,torch.zeros_like(beta),motion,base)]
    for delta in [.5e-6,.9e-6,1.e-6,1.1e-6,2.e-6]:
        unary=u.clone();unary[:,:,2]=.375-delta
        b=beta.clone();b[:,2]=0
        cases.append((f'repair_threshold_{delta}',unary,w,L,b,motion,base))
    results=[]
    for name,*args in cases:
        with torch.inference_mode():a,q=fixture(*args)
        native=session.run(None,{key:value.numpy() for key,value in zip(['u','w','L','beta','motion','base'],args)})
        np.testing.assert_array_equal(a.numpy(),native[0])
        np.testing.assert_allclose(q.numpy(),native[1],rtol=1e-5,atol=1e-6)
        results.append(dict(name=name,action=a.tolist(),reference_cost=q.tolist(),compiled_cost=native[1].tolist()))
    x=torch.tensor([[0.,1e-12,1e-8,1e-6,1e-4,.1,1.]],dtype=torch.float32)
    exp_path=folder/'expm1_lowering_fixture.onnx'
    torch.onnx.export(Expm1Fixture(),x,str(exp_path),opset_version=17,dynamo=False,input_names=['x'],output_names=['y'])
    native=c.session(exp_path).run(None,{'x':x.numpy()})[0]
    np.testing.assert_allclose(torch.expm1(x).numpy(),native,rtol=1e-5,atol=1e-6)
    return dict(decoder_cases=results,expm1_max_absolute_error=float(abs(native-torch.expm1(x).numpy()).max()),
        input_contract_and_roundtrip='checked_for_each_complete_model',kind='synthetic_mechanics_only')

def run(output):
    r.configure();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    p=json.loads(PROTOCOL.read_text());input_path=ROOT/p['observations']
    assert r.old.sha(input_path)==p['observations_sha256']
    assert ort.__version__=='1.31.0' and onnx.__version__=='1.23.2'
    assert importlib.metadata.version('ml_dtypes')=='0.5.4'
    names=[*r.SOURCES,'compiled_controller_inference.py','run_compiled_inference_gate.py','profile_controller_inference_2026_11_05.py',str(PROTOCOL.relative_to(ROOT))]
    frozen=dict(protocol_sha256=r.old.sha(PROTOCOL),input_sha256=r.old.sha(input_path),source_sha256={n:r.old.sha(ROOT/n) for n in names},
        hardware=r.hardware(),backend=p['backend'],research_training_authorized=False,export_started_at_ns=time.perf_counter_ns())
    r.write(out/'PRE_EXPORT_FREEZE.json',frozen)
    for name in names:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    common=np.load(input_path)['observations'];rng=np.random.default_rng(99005)
    synthetic=rng.uniform(0.,1.,(64,533)).astype(np.float32)
    cases=np.concatenate((common,synthetic));np.savez_compressed(out/'equivalence_observations.npz',observations=cases)
    export_folder=out/'exports';export_folder.mkdir()
    metadata={};equivalence=[];timed_models={};sessions={}
    try:
        for seed in [99001,99006,99007]:
            for arm in r.ARMS:
                print(f'Export/equivalence {arm}, synthetic initialization {seed}',flush=True)
                model=r.build(arm,seed,r.fixture_cfg())['online'].eval();before=r.old.digest(model.state_dict())
                path=export_folder/f'{arm}_seed{seed}.onnx'
                session,meta=c.export(model,arm,path);meta['seed']=seed;metadata[f'{arm}_{seed}']=meta
                eager_actions=[];eager_values=[];native_actions=[];native_values=[]
                with torch.inference_mode():
                    for state in cases:
                        a,q,_=r.greedy(model,torch.from_numpy(state)[None],arm)
                        na,nq=c.compiled_values(session,state)
                        eager_actions.append(a.numpy()[0]);eager_values.append(float(q[0]))
                        native_actions.append(na[0]);native_values.append(float(nq[0]))
                eager_actions=np.asarray(eager_actions);native_actions=np.asarray(native_actions)
                np.testing.assert_array_equal(eager_actions,native_actions,err_msg=f'action mismatch: {arm} seed{seed}')
                np.testing.assert_allclose(eager_values,native_values,rtol=1e-5,atol=1e-6,err_msg=f'Q mismatch: {arm} seed{seed}')
                assert np.isfinite(native_values).all()
                if r.bounded(arm):assert max(native_values)<=0
                restored=c.session(path)
                for index in [0,64,191]:
                    first=c.compiled_values(session,cases[index]);second=c.compiled_values(restored,cases[index])
                    np.testing.assert_array_equal(first[0],second[0]);np.testing.assert_array_equal(first[1],second[1])
                for bad in [np.zeros(532,np.float32),np.zeros((2,533),np.float32)]:
                    try:c.compiled_values(session,bad)
                    except ValueError:pass
                    else:raise AssertionError('compiled input contract not enforced')
                assert r.old.digest(model.state_dict())==before
                equivalence.append(dict(arm=arm,model_seed=seed,cases=len(cases),all_actions_equal=True,
                    max_Q_absolute_error=float(np.max(abs(np.array(eager_values)-native_values))),
                    weights_unchanged=True,roundtrip_and_repeat_exact=True))
                if seed==99001:timed_models[arm]=model;sessions[arm]=session
        fixtures=check_small_fixtures(export_folder)
        r.write(out/'equivalence.json',dict(state='passed',models=equivalence,total_complete_actions=24*len(cases),
            fixtures=fixtures,scope='finite_inputs_and_weights_not_universal_bitwise_equivalence',research_training=False))
        r.write(out/'export_metadata.json',metadata)
        # Equivalence is a hard prerequisite; write the timing freeze only now.
        for name,h in frozen['source_sha256'].items():assert r.old.sha(ROOT/name)==h
        r.write(out/'PRE_TIMING_FREEZE.json',dict(source_sha256=frozen['source_sha256'],protocol_sha256=r.old.sha(PROTOCOL),
            equivalence_sha256=r.old.sha(out/'equivalence.json'),all_exports_sha256={k:v['onnx_sha256'] for k,v in metadata.items()},
            paths=[f'{backend}/{arm}' for backend in ['eager','compiled'] for arm in r.ARMS],rounds=5,warmups=32,calls=128,
            hardware=r.hardware(),compiled_session_options=p['backend']))
        timing=[];orders=[];order_rng=np.random.default_rng(99008)
        for round_ in range(5):
            paths=[(backend,arm) for backend in ['eager','compiled'] for arm in r.ARMS];order_rng.shuffle(paths)
            orders.append([f'{b}/{a}' for b,a in paths])
            for backend,arm in paths:
                fn=(lambda state,arm=arm:c.compiled_action(sessions[arm],state)) if backend=='compiled' else (
                    lambda state,arm=arm:c.eager_action(timed_models[arm],arm,state))
                for state in common[:32]:fn(state)
                for index,state in enumerate(common):
                    start=time.perf_counter_ns();a=fn(state);elapsed=time.perf_counter_ns()-start
                    assert a.shape==(32,) and np.all((a>=0)&(a<5))
                    timing.append(dict(round=round_,backend=backend,arm=arm,observation=index,nanoseconds=elapsed))
        metrics={backend:{} for backend in ['eager','compiled']}
        for backend in metrics:
            for arm in r.ARMS:
                raw=np.array([x['nanoseconds'] for x in timing if x['backend']==backend and x['arm']==arm],dtype=np.float64)/1e6
                model=timed_models[arm]
                assert r.old.digest(model.state_dict())==metadata[f'{arm}_99001']['source_parameter_hash']
                metrics[backend][arm]=dict(median_ms=float(np.median(raw)),p95_ms=float(np.quantile(raw,.95)),calls=len(raw),
                    parameter_bytes=r.parameter_bytes(model),parameters=sum(p.numel() for p in model.parameters()),
                    onnx_serialized_bytes=metadata[f'{arm}_99001']['serialized_onnx_bytes'] if backend=='compiled' else None)
        screens=r.cost_screens(metrics['compiled'],r.spec())
        result=dict(state='compiled_cost_gate_passed' if all(screens.values()) else 'compiled_cost_gate_failed',screens=screens,
            metrics=metrics,orders=orders,total_timing_records=len(timing),equivalence_passed=True,
            compilation_and_session_startup_excluded_from_timing=True,optimizer_updates=0,new_simulator_steps=0,
            research_training_started=False,all_research_training_and_gate_seeds_unconsumed=True,
            frozen_model_and_scope_unchanged=True,backend='ONNX Runtime CPU FP32 1thread for all8 arms')
        r.write(out/'raw_timings.json',timing);r.write(out/'verification.json',result)
        r.write(out/'run_manifest.json',dict(files={str(x.relative_to(out)).replace('\\','/'):r.old.sha(x) for x in out.rglob('*') if x.is_file()}))
        print(json.dumps(result,indent=2))
        return result
    except Exception:
        r.write(out/'FAILED.json',dict(error=traceback.format_exc(),partial_equivalence=equivalence,export_metadata=metadata,
            original_model_changed=False,research_training_started=False,optimized_timing_started=(out/'PRE_TIMING_FREEZE.json').exists()))
        raise

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',type=Path,required=True);args=a.parse_args();run(args.output)
