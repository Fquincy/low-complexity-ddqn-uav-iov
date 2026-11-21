"""Complete fixture-only summaries and limited instrumentation; no claims."""
import copy
import gzip
import json
from pathlib import Path
import time
import numpy as np
import torch
import run_assignment_price_full_fixture as w

def read_episode(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)

def analysis_protocol(p):
    result=copy.deepcopy(p)
    result['fixture']=dict(engineering_seeds=dict(bootstrap=867254007))
    return result

def report(output,p):
    out=Path(output);pl=w.plan();states=np.load(out/'coupled_sweep/evaluation.npz')['states'][:16]
    agents={};costs={};storage={};evaluations={};heuristics={}
    # Rebuild every exported final snapshot; never select checkpoints.
    for arm in w.ARMS:
        folder=out/arm;ck=torch.load(folder/'final.pt',weights_only=True,map_location='cpu');a=w.build(arm,pl)
        a['online'].load_state_dict(ck['online']);model=a['online'].eval()
        path=folder/'reporting_snapshot.onnx'
        runtime=w.deployed.Runtime(model,path) if arm in w.adp.ARMS else w.references.Runtime(model,path)
        agents[arm]=runtime
        parameters=sum(x.numel()*x.element_size() for x in model.parameters());buffers=sum(x.numel()*x.element_size() for x in model.buffers())
        storage[arm]=dict(parameter_bytes=parameters,buffer_bytes=buffers,final_checkpoint_bytes=(folder/'final.pt').stat().st_size,
            serialized_deployment_bytes=(folder/'final.onnx').stat().st_size,optimizer_tensor_bytes=sum(v.numel()*v.element_size() for item in ck['optimizer']['state'].values() for v in item.values() if isinstance(v,torch.Tensor)),
            scope='Actual serialized files and tensors; not process peak or full library memory')
        evaluations[arm]=[[w.r.compact(read_episode(folder/'evaluation.json.gz'))]]
    for policy in w.HEURISTICS:heuristics[policy]=[w.r.compact(read_episode(out/f'heuristic_{policy}.json.gz'))]
    order=list(agents);np.random.default_rng(867254003).shuffle(order)
    freeze=dict(kind=w.KIND,cost_acceptance=False,states=16,warmups=4,rounds=1,order=order,
        checkpoints=w.r.read(out/'FINAL_CHECKPOINT_FREEZE.json'),graphs={a:w.r.sha(out/a/'reporting_snapshot.onnx') for a in agents},
        scope='Tiny workflow timing fixture only; not protocol acceptance benchmark or favorable retiming')
    w.r.write(out/'FIXTURE_TIMING_FREEZE.json',freeze);rows=[]
    for arm in order:
        runtime=agents[arm]
        for state in states[:4]:runtime.values(state)[0][0].copy()
        for index,state in enumerate(states):
            begin=time.perf_counter_ns();runtime.values(state)[0][0].copy();elapsed=time.perf_counter_ns()-begin
            rows.append(dict(arm=arm,index=index,ns=elapsed))
        ns=[row['ns'] for row in rows if row['arm']==arm]
        costs[arm]=[dict(parameter_bytes=storage[arm]['parameter_bytes'],median_ms=float(np.median(ns)/1e6),p95_ms=float(np.quantile(ns,.95)/1e6))]
    w.r.write(out/'fixture_timings.json',rows);w.r.write(out/'deployment_storage.json',storage)
    ap=analysis_protocol(p)
    summary=w.r.analysis.summarize(ap,evaluations,heuristics,costs,fixture=True)
    w.r.write(out/'fixture_analysis.json',summary)
    diagnostics={}
    for arm in w.ARMS:
        logs=[json.loads(line) for line in (out/arm/'optimizer.jsonl').read_text().splitlines()]
        diagnostics[arm]={name:w.r.analysis.distribution([v for row in logs for v in (row[name] if isinstance(row[name],list) else [row[name]])]) for name in
            ['chosen_Q','bellman_targets','TD_residuals','grad_norm_before_clip','loss','uniform_loss','anchor_loss']}
        diagnostics[arm]['clipping_fraction']=float(np.mean([row['clipped'] for row in logs]))
    w.r.write(out/'optimizer_distributions.json',dict(kind=w.KIND,quality_evidence=False,arms=diagnostics))
