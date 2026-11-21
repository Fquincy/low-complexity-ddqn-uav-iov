"""Engineering timing gate; untrained weights, no environment steps."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'Source'))
import csv,json,time,hashlib
import numpy as np
import torch
from spi_ddqn import SPIDDQN,select
from train_chapter3_ddqn import FactorizedQNetwork,to_action_dict
from chapter3_sequential_conditional import ConditionalQNetwork
from benchmark_chapter3_sequential_selection import SequentialSelection


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(out):
    out.mkdir(parents=True,exist_ok=False)
    with np.load('observations.npz') as z:obs=z['observations'][:128].copy()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261103)
    additive=FactorizedQNetwork(533,32,5,(128,128)).eval()
    conditional=ConditionalQNetwork().eval();spi=SPIDDQN().eval()
    paths={}
    for name in ('additive','sequential_cached','spi'):
        durations=[]
        if name=='sequential_cached':selector=SequentialSelection(conditional,True)
        for r in range(2):
            for i in range(32):
                if name=='additive':
                    with torch.inference_mode():additive(torch.tensor(obs[i])[None])
                elif name=='sequential_cached':selector.select(obs[i])
                else:select(spi,obs[i])
            for i in range(128):
                start=time.perf_counter_ns()
                if name=='additive':
                    with torch.inference_mode(): action=additive(torch.tensor(obs[i])[None])[0].argmax(1).numpy()
                elif name=='sequential_cached':
                    selected=selector.select(obs[i]); action=np.concatenate((selected['destinations'],selected['uav_movement']))
                else: action,_=select(spi,obs[i])
                elapsed=time.perf_counter_ns()-start;assert np.all((np.asarray(action)>=0)&(np.asarray(action)<5));durations.append(elapsed)
        paths[name]=dict(median_ms=float(np.median(durations)/1e6),p95_ms=float(np.quantile(durations,.95)/1e6),calls=len(durations))
    result=dict(state='complete',paths=paths,observations=128,rounds=2,warmup_slots=32,timed_slots=128,
      parameter_bytes={k:sum(p.numel()*p.element_size() for p in m.parameters()) for k,m in {'additive':additive,'conditional':conditional,'spi':spi}.items()},
      source_sha256={n:sha(Path(n) if Path(n).exists() else Path('Source')/n) for n in ('spi_ddqn.py','benchmark_spi_cost.py','chapter3_sequential_conditional.py','train_chapter3_ddqn.py')},
      env_step_in_timing=False,weights_trained=False,low_complexity_claim=False)
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main(Path('validation/spi_cost_gate_2026-11-01_02'))
