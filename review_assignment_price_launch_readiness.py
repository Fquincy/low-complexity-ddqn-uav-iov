"""Read-only launch/resource audit and one prospective engineering heuristic timing."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import sys
import time
import numpy as np
import torch
import run_assignment_price_research_bound as w

ROOT=w.r.ROOT
OUT=ROOT/'validation/assignment_price_launch_review_2026-11-15_01'
BIND=ROOT/'validation/assignment_price_research_binding_2026-11-15_01'

def walk_numbers(value,path=()):
    if isinstance(value,dict):
        if isinstance(value.get('start'),int) and isinstance(value.get('count'),int):
            yield ('interval',value['start'],value['start']+value['count'],path)
        for key,item in value.items():yield from walk_numbers(item,path+(key,))
    elif isinstance(value,list):
        for index,item in enumerate(value):yield from walk_numbers(item,path+(str(index),))
    elif isinstance(value,int) and not isinstance(value,bool):yield ('point',value,value,path)

def main():
    OUT.mkdir(exist_ok=False);w.r.configure();p=w.base.protocol();pl=w.plan('development')
    binding=w.r.read(BIND/'ENGINEERING_BINDING_MANIFEST.json')
    assert all(w.r.sha(ROOT/n)==h for n,h in binding['source_hashes'].items())
    assert all(w.r.sha(BIND/n)==h for n,h in w.r.read(BIND/'COMPLETION_MANIFEST.json').items())
    assert not w.DEVELOPMENT_RELEASED
    fixture=BIND/'fixture_evidence';fpl=w.r.read(fixture/'runtime_plan.json')
    old_environment=w.r.read(BIND/'environment.json')
    environment=dict(executable=sys.executable,python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,
        onnx=w.base.deployed.onnx.__version__,onnxruntime=w.base.deployed.deployment.ort.__version__,
        torch_path=torch.__file__,numpy_path=np.__file__,torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
    for key in ['executable','python','platform','torch','numpy']:assert environment[key]==old_environment[key],key
    w.r.write(OUT/'source_environment_verification.json',dict(passed=True,tested_binding_manifest_sha256=w.r.sha(BIND/'ENGINEERING_BINDING_MANIFEST.json'),
        protocol_sha256=w.r.sha(w.base.SPEC),sources=binding['source_hashes'],environment=environment,
        environment_scope='Matches the version/path fields captured at binding; no OS scheduler or unchanged installed-binary proof',development_released=False))
    # Fresh collision pass over declarations and actual saved run plans, not logs.
    points=pl['model_seeds']+[s+o for s in pl['model_seeds'] for o in [100000,200000,300000]]+[pl['order_seed'],pl['timing_seed'],pl['bootstrap_seed']]
    intervals=[(x['start'],x['start']+x['count']) for x in pl['train_blocks']]+[(min(pl['evaluation_seeds']),max(pl['evaluation_seeds'])+1)]
    files=list(ROOT.glob('*PROTOCOL*.json'))+list(ROOT.glob('*SPEC*.json'))+list(ROOT.glob('*AMENDMENT*.json'))+list(ROOT.glob('*RELEASE*.json'))
    patterns={'runtime_plan.json','plan.json','runtime_protocol.json','settings.json','launch_record.json','launch.json'}
    for directory in [ROOT/'development_runs',ROOT/'validation']:
        for parent,dirs,names in os.walk(directory):
            dirs[:]=[d for d in dirs if d not in ['source_snapshot','release_source_snapshot','completion','__pycache__','cost_gate','literature','tests_first_pass','attempt_01']]
            files.extend(Path(parent)/name for name in names if name in patterns)
    collisions=[];checked={};skipped=[]
    for path in sorted(set(files)):
        # These are the current reservation itself and documented copies, not consumption.
        if path.name==w.base.SPEC.name:skipped.append(str(path.relative_to(ROOT)));continue
        value=json.loads(path.read_text(encoding='utf-8-sig'))
        # Prior protocols with their own seed reservation may refer to this protocol;
        # only the exact current reservation is excluded by the filename above.
        for kind,a,b,where in walk_numbers(value):
            hit=any(a<=x<b for x in points) or any(max(a,c)<min(b,d) for c,d in intervals) if kind=='interval' else a in points or any(c<=a<d for c,d in intervals)
            if hit:collisions.append(dict(file=str(path.relative_to(ROOT)),kind=kind,start=a,end=b,path=list(where)))
        checked[str(path.relative_to(ROOT))]=w.r.sha(path)
    w.r.write(OUT/'seed_consumption_refresh.json',dict(passed=not collisions,scope='Root protocol/spec/amendment/release declarations plus existing saved runtime plans/settings/launch files; not unlabelled seeds in arbitrary logs',
        reservation_sha256=w.r.sha(w.base.SPEC),points=points,intervals=intervals,checked_documents=checked,
        excluded_current_reservation_copies=skipped,collisions=collisions,new_research_seeds_consumed=False))
    assert not collisions,collisions[:3]
    # Full-run disk/wall extrapolation: retain categories, assumptions and headroom.
    cfg=pl['cfg'];d=p['development'];counts=w.r.read(fixture/'counters.json');stats=[]
    for arm in w.base.ARMS:
        per=[]
        for fit in range(len(fpl['model_seeds'])):
            folder=fixture/arm/f'fit_{fit:02d}'
            log=folder/'optimizer.jsonl';per.append(dict(log_bytes_per_update=log.stat().st_size/counts[f'{arm}/{fit:02d}']['updates'],
                training_ns_per_slot=counts[f'{arm}/{fit:02d}']['training_wall_ns']/counts[f'{arm}/{fit:02d}']['slots'],
                train_episode_bytes=statistics.mean(sum(x.stat().st_size for x in folder.glob(f'train_{ep:03d}.*')) for ep in range(2)),
                eval_episode_bytes=statistics.mean(sum(x.stat().st_size for x in folder.glob(f'eval_{ep:03d}.*')) for ep in range(2))))
        stats.append(dict(arm=arm,mean_fixture_log_bytes_per_update=statistics.mean(x['log_bytes_per_update'] for x in per),
            mean_fixture_training_ns_per_slot=statistics.mean(x['training_ns_per_slot'] for x in per),
            mean_train_episode_bytes=statistics.mean(x['train_episode_bytes'] for x in per),mean_eval_episode_bytes=statistics.mean(x['eval_episode_bytes'] for x in per)))
    batch_ratio=cfg['batch_size']/fpl['cfg']['batch_size'];anchor_ratio=cfg['anchor']['batch']/fpl['cfg']['anchor']['batch']
    record_bytes=2*533*4+32+8;replay_capacity_bytes=cfg['replay_capacity']*record_bytes;replay_populated_bytes=cfg['slots_per_fit']*record_bytes
    update_count=cfg['optimizer_updates_per_fit'];fits=len(pl['model_seeds'])
    log_upper=sum(x['mean_fixture_log_bytes_per_update'] for x in stats)*max(batch_ratio,anchor_ratio)*update_count*fits
    train_bytes=sum(x['mean_train_episode_bytes'] for x in stats)*cfg['episodes_per_fit']*fits
    eval_bytes=sum(x['mean_eval_episode_bytes'] for x in stats)*len(pl['evaluation_seeds'])*fits
    heuristic_bytes=sum(statistics.mean(sum(x.stat().st_size for x in (fixture/'heuristics'/policy).glob(f'eval_{ep:03d}.*')) for ep in range(2)) for policy in w.base.HEURISTICS)*len(pl['evaluation_seeds'])
    # Use raw populated replay arrays as a conservative no-compression storage bound.
    replay_saved_bytes=replay_populated_bytes*d['final_checkpoints'];checkpoints=sum(x.stat().st_size for x in fixture.rglob('final.pt'))/30*75
    total=log_upper+train_bytes+eval_bytes+heuristic_bytes+replay_saved_bytes+checkpoints
    disk=shutil.disk_usage(ROOT)
    training_hours_lower=sum(x['mean_fixture_training_ns_per_slot'] for x in stats)*cfg['slots_per_fit']*fits/3.6e12
    w.r.write(OUT/'resource_projection.json',dict(scope='Engineering extrapolation, not guaranteed disk/runtime bound or profiled full-batch performance',
        per_arm_fixture_basis=stats,free_disk_bytes=disk.free,projected_optimizer_log_bytes_scaled_by_batch=log_upper,
        projected_training_trace_bytes=train_bytes,projected_learned_evaluation_trace_bytes=eval_bytes,projected_heuristic_trace_bytes=heuristic_bytes,
        raw_populated_replay_file_bound_bytes=replay_saved_bytes,projected_checkpoint_bytes=checkpoints,
        projected_total_bytes=total,recommended_2x_space_bytes=total*2,disk_headroom_2x_pass=disk.free>=total*2,
        replay_allocated_bytes_per_training_agent=replay_capacity_bytes,replay_populated_bytes_per_fit=replay_populated_bytes,
        fixture_per_slot_training_extrapolation_hours=training_hours_lower,
        runtime_limitations='Batch128 versus fixture4, terminal anchor32 versus4, growing queues, export/startup/evaluation/audit/JSON parsing not estimated by training-only extrapolation; do not quote as ETA',
        memory_limitations='Allocator/process peak unmeasured; replay216MB per active agent plus models/optimizer, logs and summary arrays; source review needed for cumulative fit retention'))
    # One standalone heuristic timing schedule, declared before timing. No simulator.
    source='validation/controller_redesign_engineering_2026-11-05_01/cost_gate/common_observations.npz'
    from assignment_price_load_radio_heuristic import Runtime
    hp=dict(kind='SUPPLEMENTARY_ENGINEERING_HEURISTIC_COST_NOT_POLICY_EVIDENCE',policy='observed_load_radio',
        input=source,input_sha256=w.r.sha(ROOT/source),states=128,rounds=5,warmups=32,readings=640,
        source_sha256=w.r.sha(ROOT/'assignment_price_load_radio_heuristic.py'),measure='Complete reused Runtime.values(state)[0][0].copy(); includes validation/preparation/decode/output copy',
        standalone=True,acceptance_screens=False,timing_attempts=1,changes_to_research_thresholds=False,new_simulator_steps=0,new_optimizer_updates=0)
    w.r.write(OUT/'PRE_HEURISTIC_TIMING_PLAN.json',hp)
    states=np.load(ROOT/source)['observations'];runtime=Runtime();first=time.perf_counter_ns();runtime.values(states[0]);first=time.perf_counter_ns()-first
    rows=[]
    for rnd in range(5):
        for state in states[:32]:runtime.values(state)[0][0].copy()
        for i,state in enumerate(states):
            begin=time.perf_counter_ns();runtime.values(state)[0][0].copy();elapsed=time.perf_counter_ns()-begin
            rows.append(dict(round=rnd,index=i,ns=elapsed))
    w.r.write(OUT/'heuristic_timings.json',rows)
    w.r.write(OUT/'heuristic_cost_summary.json',dict(readings=640,median_ms=float(np.median([x['ns'] for x in rows])/1e6),p95_ms=float(np.quantile([x['ns'] for x in rows],.95)/1e6),
        first_call_including_native_compile_ms=first/1e6,first_call_scope='Existing process/import caches; not fresh-process startup',
        scientific_screens_applied=False,source_unchanged=w.r.sha(ROOT/'assignment_price_load_radio_heuristic.py')==hp['source_sha256']))
    print(json.dumps(dict(seed_documents=len(checked),source_environment_verified=True,resources=w.r.read(OUT/'resource_projection.json'),heuristic=w.r.read(OUT/'heuristic_cost_summary.json')),indent=2))

if __name__=='__main__':main()
