"""Correctness checks, stage profiling and matched full-action timings."""
from pathlib import Path
import json,csv,time,itertools
import numpy as np
import torch
from emc_q import EMCQ,decode_tables,chosen_tables
from emc_decoder_variants import decode_matmul,decode_rank2
from benchmark_emc_q import FactorizedQNetwork,sha,weights

ROOT=Path(__file__).resolve().parent
DECODERS={'original_einsum':decode_tables,'matmul':decode_matmul,'rank2_broadcast':decode_rank2}


@torch.inference_mode()
def verify(model,observations):
    cases=0;max_error=0.
    for seed in range(20):
        torch.manual_seed(seed)
        u=torch.randn(3,2,2);m=torch.randn(3,2,2);f=torch.randn(3,2,2,2);g=torch.randn(3,2);e=torch.randn(2,2,2)
        original=decode_tables(u,m,f,g,e)
        for name,decoder in DECODERS.items():
            a,q=decoder(u,m,f,g,e)
            torch.testing.assert_close(a,original[0],rtol=0,atol=0)
            torch.testing.assert_close(q,original[1],rtol=1e-5,atol=1e-5)
            torch.testing.assert_close(chosen_tables(u,m,f,g,e,a),q)
        cases+=3
    for n in (1,7,128):
        tables=model.tables(torch.tensor(observations[:n]));reference=decode_tables(*tables,model.movement_embeddings)
        for name,decoder in DECODERS.items():
            a,q=decoder(*tables,model.movement_embeddings)
            torch.testing.assert_close(a,reference[0],rtol=0,atol=0)
            torch.testing.assert_close(q,reference[1],rtol=1e-5,atol=1e-5)
            max_error=max(max_error,float((q-reference[1]).abs().max()))
    # Fully tied and deliberately movement-coupled examples.
    u=torch.zeros(1,2,2);m=torch.zeros(1,2,2);f=torch.zeros(1,2,2,2);g=torch.ones(1,2);e=torch.zeros(2,2,2)
    for decoder in DECODERS.values():
        a,q=decoder(u,m,f,g,e);assert not a.any()
    f[0,0,1,0]=2;e[0,1,0]=1
    for decoder in DECODERS.values():
        a,q=decoder(u,m,f,g,e);assert a[0,0]==1 and a[0,-2]==1
    return dict(reduced_batch_cases=cases,full_model_batch_sizes=[1,7,128],actions_identical=True,
                max_value_difference=max_error,tie_and_interaction_checks=True,
                floating_point_near_ties_may_differ_outside_checked_cases=True)


@torch.inference_mode()
def full_action(model,decoder,state):
    x=torch.tensor(state,dtype=torch.float32)[None]
    action=decoder(*model.tables(x),model.movement_embeddings)[0][0].numpy().copy()
    return dict(destinations=action[:30].copy(),uav_movement=action[30:].copy())


@torch.inference_mode()
def main():
    out=ROOT/'validation/emc_decoder_profile_2026-11-04_01';out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    gate=json.loads((ROOT/'validation/emc_mechanics_2026-11-04_01/verification.json').read_text())
    for name,h in gate['source_sha256'].items():assert sha(ROOT/name)==h
    with np.load(ROOT/'observations.npz') as z:obs=z['observations'].copy()
    torch.manual_seed(20261104);model=EMCQ().eval()
    evidence=verify(model,obs)
    models={'emc':model}
    for width in (64,128):
        torch.manual_seed(20261104);models[f'additive{width}']=FactorizedQNetwork(533,32,5,(width,width)).eval()
    hashes={k:weights(v) for k,v in models.items()}
    paths={}
    for name,decoder in DECODERS.items():
        paths[name]=lambda state,decoder=decoder:full_action(model,decoder,state)
    for width in (64,128):
        def additive(state,width=width):
            a=models[f'additive{width}'](torch.tensor(state,dtype=torch.float32)[None])[0].argmax(1).numpy().copy()
            return dict(destinations=a[:30].copy(),uav_movement=a[30:].copy())
        paths[f'additive{width}']=additive
    refs=[paths['original_einsum'](state) for state in obs]
    rows=[];orders=[];rng=np.random.default_rng(20261106)
    for round_id in range(5):
        order=list(paths);rng.shuffle(order);orders.append(order)
        for name in order:
            for i in range(32):paths[name](obs[i])
            for i,state in enumerate(obs):
                start=time.perf_counter_ns();a=paths[name](state);elapsed=time.perf_counter_ns()-start
                if name in DECODERS:
                    for k in a:np.testing.assert_array_equal(a[k],refs[i][k])
                rows.append(dict(round=round_id,path=name,observation=i,nanoseconds=elapsed))
    # Isolated component timing is diagnostic: medians are not additive.
    x=torch.tensor(obs[0])[None];encoded=model.encoder(x);tables=model.tables(x)
    components={'encoder_only':lambda:model.encoder(x),'head_only':lambda:model.head(encoded)}
    for name,decoder in DECODERS.items():components[f'decoder_only_{name}']=lambda decoder=decoder:decoder(*tables,model.movement_embeddings)
    component_rows=[]
    for name,fn in components.items():
        for i in range(32):fn()
        for i in range(256):
            start=time.perf_counter_ns();fn();elapsed=time.perf_counter_ns()-start
            component_rows.append(dict(path=name,call=i,nanoseconds=elapsed))
    assert hashes=={k:weights(v) for k,v in models.items()}
    for filename,data in [('full_action_timings.csv',rows),('component_timings.csv',component_rows)]:
        with (out/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    def summarize(data):
        answer={}
        for name in sorted({r['path'] for r in data}):
            durations=np.array([r['nanoseconds'] for r in data if r['path']==name])/1e6
            answer[name]=dict(calls=len(durations),median_ms=float(np.median(durations)),p95_ms=float(np.quantile(durations,.95)))
        return answer
    result=dict(state='passed_decoder_equivalence_and_profile',date='2026-11-04',equivalence=evidence,
        full_action=summarize(rows),isolated_components=summarize(component_rows),round_order=orders,
        untrained=True,weights_unchanged=True,optimizer_updates=0,simulator_steps=0,
        include_state_conversion_and_action_dictionary=True,component_medians_not_additive=True,
        full_action_rows=len(rows),component_rows=len(component_rows),
        source_sha256={name:sha(ROOT/name) for name in ('emc_q.py','emc_decoder_variants.py','profile_emc_decoder.py','Source/train_chapter3_ddqn.py')},
        input_observations_sha256=sha(ROOT/'observations.npz'))
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(full_action=result['full_action'],components=result['isolated_components'],equivalence=evidence),indent=2))

if __name__=='__main__':main()
