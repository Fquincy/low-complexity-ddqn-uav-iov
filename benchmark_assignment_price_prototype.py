"""One declared engineering cost measurement, not a policy-quality experiment."""
import copy
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import assignment_price_prototype as adp
import run_horizon_loss_research as r

BASE=r.ROOT/'validation/assignment_price_prototype_2026-11-15_01'
OUT=BASE/'cost_gate'

def main():
    assert r.read(BASE/'tests.json')['passed']
    OUT.mkdir(parents=True,exist_ok=False);r.configure();p=r.protocol()
    source='validation/controller_redesign_engineering_2026-11-05_01/cost_gate/common_observations.npz'
    assert r.sha(r.ROOT/source)==p['action_cost']['common_inputs_sha256']
    states=np.load(r.ROOT/source)['observations'];assert states.shape==(128,533)
    seeds=[99861,99869]
    controls=['screened_price24','unmasked_compact64','screened_wide128','unmasked_wide128']
    names=list(adp.ARMS)+controls
    plan=dict(kind='ENGINEERING_COST_ONLY_NOT_RESEARCH',model_seeds=seeds,arms=names,
        observations=128,observation_sha256=r.sha(r.ROOT/source),rounds=5,warmups_per_path_round=32,
        timed_calls_per_path_round=128,readings=16*5*128,order_seed=99877,
        thresholds=p['action_cost']['primary_each_fit'],source_hashes={n:r.sha(r.ROOT/n) for n in [
            'assignment_price_prototype.py','benchmark_assignment_price_prototype.py','test_assignment_price_prototype.py',
            'ASSIGNMENT_DEPENDENT_PRICE_DESIGN_BRIEF_2026-11-15.txt']},
        precision='ADP: FP32 network with FP64 scalar/decoder; original controls: frozen FP32/native routes',
        quality_evidence=False,new_simulator_steps=0,research_seeds_used=False)
    r.write(OUT/'PRE_BENCHMARK_PLAN.json',plan)
    defs={d['id']:d for d in p['learned_arms']};agents={};payloads={};startup=[]
    for name in names:
        for fit,seed in enumerate(seeds):
            started=time.perf_counter_ns()
            model=adp.Critic(name,seed) if name in adp.ARMS else r.models.Critic(defs[name],seed)
            model=model.eval();runtime=adp.Runtime(model) if name in adp.ARMS else r.models.Runtime(model)
            runtime.values(states[0]);startup.append(dict(arm=name,fit=fit,construction_first_action_ns=time.perf_counter_ns()-started))
            agents[name,fit]=(model,runtime)
            payloads[name,fit]=dict(parameter_bytes=sum(t.numel()*t.element_size() for t in model.parameters()),
                buffer_bytes=sum(t.numel()*t.element_size() for t in model.buffers()))
            torch.save(dict(online=model.state_dict(),seed=seed,engineering_only=True,arm=name),OUT/f'{name}_{fit}.pt')
            for state in states:
                with torch.inference_mode():a,q=model.greedy(torch.from_numpy(state)[None])
                na,nq=runtime.values(state)
                np.testing.assert_array_equal(a.numpy(),na);np.testing.assert_allclose(q.numpy(),nq,atol=1e-6,rtol=1e-5)
    frozen={x.name:r.sha(x) for x in OUT.glob('*.pt')};r.write(OUT/'CHECKPOINT_FREEZE.json',frozen)
    jobs=list(agents);rng=np.random.default_rng(plan['order_seed']);orders=[]
    for rnd in range(5):order=list(jobs);rng.shuffle(order);orders.append(order)
    r.write(OUT/'PRE_TIMING_FREEZE.json',dict(plan_sha256=r.sha(OUT/'PRE_BENCHMARK_PLAN.json'),orders=orders,checkpoints=frozen))
    rows=[]
    for rnd,order in enumerate(orders):
        for name,fit in order:
            runtime=agents[name,fit][1]
            for state in states[:32]:runtime.values(state)[0][0].copy()
            for i,state in enumerate(states):
                started=time.perf_counter_ns();runtime.values(state)[0][0].copy();elapsed=time.perf_counter_ns()-started
                rows.append(dict(arm=name,fit=fit,round=rnd,index=i,ns=elapsed))
    r.write(OUT/'raw_timings.json',rows);r.write(OUT/'construction_startup.json',startup)
    costs={}
    for name in names:
        costs[name]=[]
        for fit in range(2):
            values=np.array([x['ns'] for x in rows if x['arm']==name and x['fit']==fit])
            assert len(values)==640 and (values>0).all()
            costs[name].append(dict(**payloads[name,fit],median_ms=float(np.median(values)/1e6),
                p95_ms=float(np.quantile(values,.95)/1e6)))
    gates=[]
    for fit in range(2):
        x=costs['coupled_sweep'][fit]
        gates.append(dict(storage=x['parameter_bytes']<=.5*costs['unmasked_compact64'][fit]['parameter_bytes'],
            median=x['median_ms']<=min(costs[n][fit]['median_ms'] for n in ['screened_wide128','unmasked_wide128']),
            p95=x['p95_ms']<=1.))
    r.write(OUT/'cost_summary.json',dict(costs=costs,primary_screens=gates,all_primary_cost_screens_pass=all(all(x.values()) for x in gates),
        engineering_equivalence_checks=2048,raw_readings=len(rows),quality_evidence=False,
        workspace_scope='FP64 heads, masks, N/D assignment work arrays and framework temporaries excluded from parameter bytes. No peak-process-memory claim.',
        constructor_startup_scope='Same process/import caches; one first-action startup record per model; not fresh-process deployment startup.',
        mathematical_workspace_bound='O(ND+N+D) scalars plus sorting workspace; actual process peak unmeasured',
        new_simulator_steps=0,new_optimizer_updates=0,research_launched=False,retiming=False))
    for name,h in frozen.items():assert r.sha(OUT/name)==h
    for name,h in plan['source_hashes'].items():assert r.sha(r.ROOT/name)==h
    for n in plan['source_hashes']:shutil.copy2(r.ROOT/n,OUT/Path(n).name)
    r.write(OUT/'manifest.json',{x.relative_to(OUT).as_posix():r.sha(x) for x in OUT.rglob('*') if x.is_file()})
    print(json.dumps(dict(primary=costs['coupled_sweep'],screened_wide=costs['screened_wide128'],unmasked_wide=costs['unmasked_wide128'],screens=gates),indent=2))

if __name__=='__main__':main()
