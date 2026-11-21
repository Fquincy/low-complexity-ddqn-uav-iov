"""Full physical-action selection microbenchmark, including native decode."""
from pathlib import Path
import sys,json,time,hashlib,platform,csv
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent/'Source'))
from train_chapter3_ddqn import FactorizedQNetwork
from spi_ddqn import SPIDDQN,select
from emc_q import EMCQ

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def weights(model):return hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()

@torch.inference_mode()
def select_full(model,name,state):
    if name=='spi_historical':action,_=select(model,state)
    else:
        x=torch.tensor(state,dtype=torch.float32)[None]
        action=(model.greedy(x)[0][0] if name=='emc_q' else model(x)[0].argmax(1)).numpy().copy()
    # Common physical action layout; includes dictionary decode in every path.
    return {'destinations':action[:30].copy(),'uav_movement':action[30:].copy()}


def main(out):
    tests=json.loads((ROOT/'validation/emc_mechanics_2026-11-04_01/verification.json').read_text())
    assert tests['state']=='passed'
    for p,h in tests['source_sha256'].items():assert sha(ROOT/p)==h
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    with np.load(ROOT/'observations.npz') as z:obs=z['observations'].copy()
    assert obs.shape==(128,533)
    models={}
    for name in ('additive128','additive64','emc_q','spi_historical'):
        torch.manual_seed(20261104)
        models[name]= EMCQ().eval() if name=='emc_q' else SPIDDQN().eval() if name=='spi_historical' else FactorizedQNetwork(533,32,5,(64,64) if name=='additive64' else (128,128)).eval()
    initial={n:weights(m) for n,m in models.items()}
    refs={n:[select_full(m,n,s) for s in obs] for n,m in models.items()}
    rows=[];orders=[];rng=np.random.default_rng(20261105)
    for round_id in range(3):
        order=list(models);rng.shuffle(order);orders.append(order)
        for name in order:
            model=models[name]
            for i in range(32):select_full(model,name,obs[i])
            for i,state in enumerate(obs):
                start=time.perf_counter_ns();a=select_full(model,name,state);elapsed=time.perf_counter_ns()-start
                for k in a:np.testing.assert_array_equal(a[k],refs[name][i][k])
                rows.append(dict(round=round_id,path=name,observation=i,nanoseconds=elapsed))
    assert initial=={n:weights(m) for n,m in models.items()}
    with (out/'timings.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summaries={}
    for n,m in models.items():
        durations=np.array([r['nanoseconds'] for r in rows if r['path']==n])/1e6
        summaries[n]=dict(median_ms=float(np.median(durations)),p95_ms=float(np.quantile(durations,.95)),calls=len(durations),
                         parameters=sum(p.numel() for p in m.parameters()),parameter_bytes=sum(p.numel()*p.element_size() for p in m.parameters()))
    # Exact dimensions independently reconcile native storage arithmetic.
    assert summaries['emc_q']['parameters']==69361
    v=dict(state='complete_verified_cost_gate',date='2026-11-04',summaries=summaries,round_order=orders,
        python=platform.python_version(),torch=str(torch.__version__),platform=platform.platform(),
        cpu_threads=1,interop_threads=1,weights_trained=False,weights_unchanged=True,all_actions_match_reference=True,
        include_state_tensor_conversion_and_physical_action_dictionary=True,exclude_environment_step_and_file_io=True,
        raw_timing_rows=len(rows),observation_sha256=sha(ROOT/'observations.npz'),timing_csv_sha256=sha(out/'timings.csv'),
        source_sha256={p:sha(ROOT/p) for p in ('emc_q.py','benchmark_emc_q.py','Source/train_chapter3_ddqn.py','spi_ddqn.py')},
        quality_claim=False,optimizer_updates=0,simulator_steps=0)
    (out/'verification.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main(ROOT/'validation/emc_cost_gate_2026-11-04_01')
