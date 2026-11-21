"""Original PyTorch core arithmetic, unchanged native decoder; saved weights only."""
from pathlib import Path
import numpy as np,torch
import run_anchor_development_v4 as r

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'development_runs/terminal_anchor_development_2026-11-06_01'

class TorchCoreSession:
    """Minimal core adapter retaining the original FP32 Linear/ReLU operations."""
    def __init__(self,model,arm):
        self.core=r.fused.Core(model,arm).eval().requires_grad_(False)
    def run(self,names,feeds):
        x=feeds['states']
        with torch.inference_mode():
            raw=self.core(torch.from_numpy(x))
        return [v.numpy() for v in raw] if isinstance(raw,tuple) else [raw.numpy()]

def session(graph):
    graph=Path(graph);arm,fit=graph.stem.rsplit('_fit_',1)
    if arm in r.base.SHARED_ARMS:return r.compiled.session(graph)
    path=RUN/arm/f'fit_{fit}'/'final.pt'
    frozen=r.read(RUN/'ALL_FINAL_CHECKPOINTS_FREEZE.json')
    if r.base.old.sha(path)!=frozen['checkpoints'][graph.stem]:raise ValueError('checkpoint freeze changed')
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=r.make_model(arm,ck['seed'] if 'seed' in ck else r.read(RUN/'runtime_plan.json')['model_seeds'][int(fit)])
    model.load_state_dict(ck['online']);model.eval()
    if r.base.old.digest(model.state_dict())!=ck['online_parameter_hash']:raise ValueError('parameter digest changed')
    base_arm=r.anchor.ARMS[arm][0] if arm in r.CORRECTED else arm
    return TorchCoreSession(model.model if arm in r.CORRECTED else model,base_arm)
