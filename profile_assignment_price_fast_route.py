"""Diagnostic profiling only: not a replacement for the preserved cost gate."""
import cProfile
import io
import json
from pathlib import Path
import pstats
import torch
import numpy as np
import assignment_price_prototype as a
import assignment_price_fast_runtime as fast
import run_horizon_loss_research as r

def main():
    out=Path('validation/assignment_price_fused_2026-11-15_01')
    out.mkdir(exist_ok=False)
    r.configure()
    states=np.load(r.ROOT/'validation/controller_redesign_engineering_2026-11-05_01/cost_gate/common_observations.npz')['observations']
    saved=torch.load(r.ROOT/'validation/assignment_price_prototype_2026-11-15_01/cost_gate/coupled_sweep_0.pt',weights_only=True)
    model=a.Critic();model.load_state_dict(saved['online']);runtime=fast.Runtime(model)
    for state in states[:32]:runtime.values(state)
    profiler=cProfile.Profile();profiler.enable()
    for state in states:runtime.values(state)
    profiler.disable();profiler.dump_stats(str(out/'optimized_runtime.prof'))
    stream=io.StringIO();pstats.Stats(profiler,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(45)
    (out/'profile.txt').write_text(stream.getvalue(),encoding='utf-8')
    (out/'profile_scope.json').write_text(json.dumps(dict(kind='DIAGNOSTIC_PROFILE_NOT_COST_ACCEPTANCE',calls=128,model='saved engineering coupled_sweep_0',new_simulator_steps=0,new_optimizer_updates=0,retiming_for_acceptance=False),indent=2),encoding='utf-8')
    print(stream.getvalue())
if __name__=='__main__':main()
