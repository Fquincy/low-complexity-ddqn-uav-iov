"""One prospective, paired engineering benchmark of a versioned runtime."""
import json
from pathlib import Path
import shutil
import time
import warnings
import numpy as np
import torch
import numba
import assignment_price_prototype as adp
import assignment_price_fast_runtime as scripted
import assignment_price_deployment_runtime as fused
import assignment_price_reference_deployment as references
import run_horizon_loss_research as r

BASE=Path('validation/assignment_price_matched_deployment_2026-11-15_01')
OLD=Path('validation/assignment_price_prototype_2026-11-15_01/cost_gate')

def main():
    assert r.read(BASE/'tests.json')['passed']
    out=BASE/'cost_gate';out.mkdir(exist_ok=False)
    r.configure();p=r.protocol()
    originals=['screened_price24','unmasked_compact64','screened_wide128','unmasked_wide128']
    names=['fused_'+arm for arm in adp.ARMS]+originals
    sources=['assignment_price_prototype.py','assignment_price_fast_runtime.py','assignment_price_deployment_runtime.py','assignment_price_reference_deployment.py',
        'test_assignment_price_reference_deployment.py','benchmark_assignment_price_matched_deployment.py',
        'controller_v3_fused_runtime.py','controller_v3_amortized.py','deadline_screen_prototype.py',
        'terminal_anchor_prototype.py','run_horizon_loss_research.py','compiled_controller_inference.py']
    observation=r.ROOT/'validation/controller_redesign_engineering_2026-11-05_01/cost_gate/common_observations.npz'
    plan=dict(kind='ENGINEERING_MATCHED_CPU_GRAPH_RUNTIME_NOT_RESEARCH',arms=names,
        checkpoint_source=str(OLD),checkpoint_freeze_sha256=r.sha(OLD/'CHECKPOINT_FREEZE.json'),
        model_seeds=[99861,99869],new_model_seeds=False,order_seed=99907,
        rounds=5,warmups=32,states=128,calls_per_path=640,readings=10240,
        input_sha256=r.sha(observation),thresholds=r.read(OLD/'PRE_BENCHMARK_PLAN.json')['thresholds'],
        sources={n:r.sha(r.ROOT/n) for n in sources},test_report_sha256=r.sha(BASE/'tests.json'),
        precision='Unchanged FP32 heads and FP64 ADP decoder; frozen original control routes',
        measurement='Runtime.values(state)[0][0].copy(): includes validation, neural pass, mask, decoder, chosen Q and output copy',
        policy_quality_evidence=False,new_simulator_steps=0,new_optimizer_updates=0,
        timing_attempts=1,previous_failed_cost_gate_preserved=True,control_scope='All candidate/reference paths: ONNX CPU heads plus existing native decoder, identical session optimization and threads',
        environment=dict(python=__import__('sys').version,executable=__import__('sys').executable,
            numpy=np.__version__,torch=torch.__version__,numba=numba.__version__,numba_threads=numba.get_num_threads(),
            torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
            onnx=fused.onnx.__version__,onnxruntime=fused.deployment.ort.__version__,
            graph_threads=1,graph_provider='CPUExecutionProvider'))
    r.write(out/'PRE_BENCHMARK_PLAN.json',plan)
    states=np.load(observation)['observations'];defs={d['id']:d for d in p['learned_arms']}
    agents={};payloads={};startup=[];warnings_seen=[];equivalence=0
    for name in names:
        source=name.removeprefix('fused_')
        for fit,seed in enumerate(plan['model_seeds']):
            checkpoint=torch.load(OLD/f'{source}_{fit}.pt',weights_only=True)
            assert checkpoint['seed']==seed and checkpoint['engineering_only']
            model=adp.Critic(source,seed) if source in adp.ARMS else r.models.Critic(defs[source],seed)
            model.load_state_dict(checkpoint['online']);model.eval()
            started=time.perf_counter_ns()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                runtime=fused.Runtime(model,out/f'{name}_{fit}.onnx') if name.startswith('fused_') else references.Runtime(model,out/f'{name}_{fit}.onnx')
                runtime.values(states[0])
            warnings_seen.extend(str(w.message) for w in caught)
            startup.append(dict(arm=name,fit=fit,construction_first_action_ns=time.perf_counter_ns()-started))
            agents[name,fit]=runtime
            payloads[name,fit]=dict(parameter_bytes=sum(t.numel()*t.element_size() for t in model.parameters()),buffer_bytes=sum(t.numel()*t.element_size() for t in model.buffers()))
            graph=out/f'{name}_{fit}.onnx'
            payloads[name,fit]['serialized_head_bytes']=graph.stat().st_size
            warnings_seen.extend(runtime.warnings)
            for state in states:
                with torch.inference_mode():a,q=model.greedy(torch.from_numpy(state)[None])
                actual,value=runtime.values(state)
                np.testing.assert_array_equal(actual,a.numpy());np.testing.assert_allclose(value,q.numpy(),rtol=1e-5,atol=1e-6)
                equivalence+=1
    orders=[];rng=np.random.default_rng(plan['order_seed']);jobs=list(agents)
    for rnd in range(plan['rounds']):
        order=list(jobs);rng.shuffle(order);orders.append(order)
    freeze={x.name:r.sha(x) for x in out.glob('*.onnx')}
    r.write(out/'PRE_TIMING_FREEZE.json',dict(plan_sha256=r.sha(out/'PRE_BENCHMARK_PLAN.json'),orders=orders,graphs=freeze,
        checkpoints=r.read(OLD/'CHECKPOINT_FREEZE.json'),equivalence_checks=equivalence))
    rows=[]
    for rnd,order in enumerate(orders):
        for name,fit in order:
            runtime=agents[name,fit]
            for state in states[:plan['warmups']]:runtime.values(state)[0][0].copy()
            for index,state in enumerate(states):
                begin=time.perf_counter_ns();runtime.values(state)[0][0].copy();elapsed=time.perf_counter_ns()-begin
                rows.append(dict(arm=name,fit=fit,round=rnd,index=index,ns=elapsed))
    r.write(out/'raw_timings.json',rows)
    costs={}
    for name in names:
        costs[name]=[]
        for fit in range(2):
            ns=[x['ns'] for x in rows if x['arm']==name and x['fit']==fit]
            assert len(ns)==640
            costs[name].append(dict(**payloads[name,fit],median_ms=float(np.median(ns)/1e6),p95_ms=float(np.quantile(ns,.95)/1e6)))
    gates=[]
    for fit in range(2):
        x=costs['fused_coupled_sweep'][fit];t=plan['thresholds']
        gates.append(dict(storage=x['parameter_bytes']<=t['parameter_bytes_max_fraction_of_unmasked_compact64']*costs['unmasked_compact64'][fit]['parameter_bytes'],
            median=all(x['median_ms']<=t['median_ratio_max']*costs[name][fit]['median_ms'] for name in t['median_ms_max_fraction_of_each']),
            p95=x['p95_ms']<=t['p95_ms_max']))
    r.write(out/'cost_summary.json',dict(costs=costs,primary_screens=gates,all_primary_cost_screens_pass=all(all(x.values()) for x in gates),
        equivalence_checks=equivalence,raw_readings=len(rows),quality_evidence=False,new_simulator_steps=0,new_optimizer_updates=0,
        startup_scope='Shared process/cache; not fresh-process deployment cold start',
        storage_scope='FP32 model tensor payload plus buffers; serialized heads separately reported; shared runtime libraries and peak memory excluded'))
    r.write(out/'construction_startup.json',startup);r.write(out/'warnings.json',sorted(set(warnings_seen)))
    for name,h in plan['sources'].items():
        assert r.sha(r.ROOT/name)==h;shutil.copy2(r.ROOT/name,out/name)
    for name,h in freeze.items():assert r.sha(out/name)==h
    for name,h in r.read(OLD/'CHECKPOINT_FREEZE.json').items():assert r.sha(OLD/name)==h
    r.write(out/'manifest.json',{x.relative_to(out).as_posix():r.sha(x) for x in out.rglob('*') if x.is_file()})
    print(json.dumps(dict(primary=costs['fused_coupled_sweep'],compact=costs['unmasked_compact64'],wide=costs['unmasked_wide128'],screens=gates),indent=2))

if __name__=='__main__':main()
