"""Paired diagnostic profiles; never replaces the saved acceptance benchmark."""
import cProfile
import io
import pstats
from pathlib import Path
import time
import numpy as np
import torch
import assignment_price_prototype as adp
import assignment_price_deployment_runtime as candidate
import assignment_price_reference_deployment as reference
import run_horizon_loss_research as r

BASE=Path('validation/assignment_price_matched_profile_2026-11-15_01')
OLD=Path('validation/assignment_price_prototype_2026-11-15_01/cost_gate')
INPUT='validation/controller_redesign_engineering_2026-11-05_01/cost_gate/common_observations.npz'

def main():
    BASE.mkdir(exist_ok=False);r.configure()
    states=np.load(r.ROOT/INPUT)['observations'];defs={d['id']:d for d in r.protocol()['learned_arms']}
    plan=dict(kind='DIAGNOSTIC_PROFILING_NOT_ACCEPTANCE',arms=['coupled_sweep','unmasked_wide128'],
        fits=2,states=128,rounds=5,warmups=32,order_seed=99917,input_sha256=r.sha(r.ROOT/INPUT),
        source_hashes={n:r.sha(r.ROOT/n) for n in ['profile_assignment_price_matched_paths.py','assignment_price_deployment_runtime.py','assignment_price_reference_deployment.py']},
        new_simulator_steps=0,new_optimizer_updates=0,acceptance_retiming=False)
    r.write(BASE/'PLAN.json',plan);agents={}
    for fit,seed in enumerate([99861,99869]):
        for arm in plan['arms']:
            saved=torch.load(OLD/f'{arm}_{fit}.pt',weights_only=True)
            model=adp.Critic(arm,seed) if arm=='coupled_sweep' else r.models.Critic(defs[arm],seed)
            model.load_state_dict(saved['online']);model.eval()
            runtime=candidate.Runtime(model,BASE/f'{arm}_{fit}.onnx') if arm=='coupled_sweep' else reference.Runtime(model,BASE/f'{arm}_{fit}.onnx')
            agents[arm,fit]=runtime
            for x in states[:32]:runtime.values(x)
            profile=cProfile.Profile();profile.enable()
            for x in states:runtime.values(x)
            profile.disable();profile.dump_stats(str(BASE/f'{arm}_{fit}.prof'))
            stream=io.StringIO();pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(25)
            (BASE/f'{arm}_{fit}_profile.txt').write_text(stream.getvalue(),encoding='utf-8')
    rows=[];rng=np.random.default_rng(plan['order_seed']);orders=[]
    for rnd in range(5):
        order=list(agents);rng.shuffle(order);orders.append(order)
        for arm,fit in order:
            runtime=agents[arm,fit]
            for index,x in enumerate(states):
                start=time.perf_counter_ns()
                if arm=='coupled_sweep':mask,h=candidate.validate_mask(x)
                else:mask,h=reference.original.validate_unmasked(x)
                validated=time.perf_counter_ns()
                outputs=runtime.session.run(None,{'state':x[None]})
                neural=time.perf_counter_ns()
                if arm=='coupled_sweep':
                    g,v,b,beta,eta=outputs
                    result=candidate.finish(x,runtime.scale,g,v,b[0],beta,eta,mask,h,runtime.sweep,runtime.separable)
                else:result=reference.finish(outputs[0],mask,h,0,runtime.scaled)
                finished=time.perf_counter_ns()
                result[0][0].copy();copied=time.perf_counter_ns()
                rows.append(dict(arm=arm,fit=fit,round=rnd,index=index,validation_ns=validated-start,
                    graph_ns=neural-validated,native_finish_ns=finished-neural,copy_ns=copied-finished))
    r.write(BASE/'component_rows.json',rows);r.write(BASE/'orders.json',orders)
    summary={}
    for arm,fit in agents:
        group=[x for x in rows if x['arm']==arm and x['fit']==fit]
        summary[f'{arm}_{fit}']={k:dict(median_us=float(np.median([x[k] for x in group])/1e3),
            p95_us=float(np.quantile([x[k] for x in group],.95)/1e3)) for k in ['validation_ns','graph_ns','native_finish_ns','copy_ns']}
    r.write(BASE/'component_summary.json',dict(scope='Instrumented diagnostic segments; do not add quantiles or substitute for cost acceptance',profiles=summary))
    r.write(BASE/'manifest.json',{p.relative_to(BASE).as_posix():r.sha(p) for p in BASE.rglob('*') if p.is_file()})
    print(__import__('json').dumps(summary,indent=2))

if __name__=='__main__':main()
