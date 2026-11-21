"""Prospective multi-fit analysis and diagnostic reports, without selection."""
import copy
import json
from pathlib import Path
import numpy as np
import run_assignment_price_research_bound as w
from audit_value_bound_development import read as episode_read

def report(out,p,pl):
    out=Path(out);fits=len(pl['model_seeds']);evaluations={};heuristics={};costs={}
    timing=w.r.read(out/'timings.json');storage=w.r.read(out/'storage.json');diagnostics={}
    for arm in w.base.ARMS:
        evaluations[arm]=[];costs[arm]=[];diagnostics[arm]=[]
        for fit in range(fits):
            folder=out/arm/f'fit_{fit:02d}'
            evaluations[arm].append([w.r.compact(episode_read(folder/f'eval_{ep:03d}.json.gz')) for ep in range(len(pl['evaluation_seeds']))])
            ns=[row['ns'] for row in timing if row['arm']==arm and row['fit']==fit]
            costs[arm].append(dict(parameter_bytes=storage[f'{arm}/{fit:02d}']['parameter_bytes'],median_ms=float(np.median(ns)/1e6),p95_ms=float(np.quantile(ns,.95)/1e6)))
            fields=['chosen_Q','bellman_targets','TD_residuals','grad_norm_before_clip','loss','uniform_loss','anchor_loss']
            values={field:[] for field in fields};clip=0;updates=0;optimizer_ns=0;anchors=[]
            with (folder/'optimizer.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    row=json.loads(line);updates+=1;clip+=row['clipped'];optimizer_ns+=row['optimizer_ns'];anchors.extend(row['anchor_Q'])
                    for field in fields:values[field].extend(row[field] if isinstance(row[field],list) else [row[field]])
            diagnostics[arm].append(dict(distributions={k:w.r.analysis.distribution(v) for k,v in values.items()},
                anchor_Q=None if not anchors else w.r.analysis.distribution(anchors),updates=updates,clipping_fraction=clip/updates,optimizer_ns=optimizer_ns))
    for policy in w.base.HEURISTICS:
        heuristics[policy]=[w.r.compact(episode_read(out/'heuristics'/policy/f'eval_{ep:03d}.json.gz')) for ep in range(len(pl['evaluation_seeds']))]
    ap=copy.deepcopy(p);ap['fixture']=dict(engineering_seeds=dict(bootstrap=pl['bootstrap_seed']))
    result=w.r.analysis.summarize(ap,evaluations,heuristics,costs,fixture=pl['mode']=='fixture')
    w.r.write(out/'analysis.json',result);w.r.write(out/'costs.json',costs)
    w.r.write(out/'optimizer_distributions.json',dict(kind=pl['kind'],quality_evidence=pl['quality_evidence'],arms=diagnostics))
    counts=w.r.read(out/'counters.json')
    w.r.write(out/'verification.json',dict(kind=pl['kind'],protocol_sha256=w.base.SPEC_SHA,
        fits=fits,controllers=len(w.base.ARMS),final_checkpoints=len(counts),
        training_slots=sum(v['slots'] for v in counts.values()),evaluation_slots=(len(counts)+len(w.base.HEURISTICS))*len(pl['evaluation_seeds'])*40,
        optimizer_updates=sum(v['updates'] for v in counts.values()),target_copies=sum(v['copies'] for v in counts.values()),
        raw_timing_rows=len(timing),quality_evidence=pl['quality_evidence'],research_launched=pl['mode']=='development',
        final_phase_launched=False,word_thesis_changed=False))
