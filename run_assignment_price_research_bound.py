"""Protocol-bound multi-fit workflow. Development requires a separate release."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import run_assignment_price_full_fixture as base

r=base.r
CONFIRM=base.CONFIRM
DEVELOPMENT_RELEASED=False
FILES=['run_assignment_price_research_bound.py','assignment_price_bound_reports.py','audit_assignment_price_research_bound.py',
       'test_assignment_price_research_bound.py','assignment_price_fresh_startup.py',
       'run_assignment_price_full_fixture.py','assignment_price_full_fixture_analysis.py','audit_assignment_price_full_fixture.py',
       'assignment_price_prototype.py','assignment_price_native_heads_runtime.py','assignment_price_all_reference_deployment.py',
       'assignment_price_load_radio_heuristic.py','assignment_price_fast_runtime.py','assignment_price_deployment_runtime.py',
       'assignment_price_reference_deployment.py','audit_assignment_price_fixture_v2.py']

def plan(mode):
    p=base.protocol();cfg=copy.deepcopy(p['training'])
    if mode=='fixture':
        cfg.update(episodes_per_fit=2,slots_per_fit=80,warmup=8,batch_size=4,replay_capacity=32,
                   target_interval_updates=8,epsilon_decay=80,optimizer_updates_per_fit=73,target_copies_per_fit=9)
        cfg['anchor']['batch']=4
        return dict(mode=mode,kind=base.KIND,model_seeds=[867350001,867350011],
            train_blocks=[dict(start=867351000,count=2),dict(start=867351100,count=2)],
            evaluation_seeds=[867352000,867352001],order_seed=867353000,timing_seed=867354003,
            bootstrap_seed=867354007,cfg=cfg,timing=dict(states=8,rounds=1,warmups=2),
            quality_evidence=False,research_seed_use=False)
    if mode!='development':raise RuntimeError('Final/unknown mode disabled')
    d=p['development'];cfg['epsilon_decay']=cfg['slots_per_fit']
    return dict(mode=mode,kind='RESEARCH_DEVELOPMENT',model_seeds=d['model_seeds'],train_blocks=d['train_blocks'],
        evaluation_seeds=list(range(d['evaluation_seeds']['start'],d['evaluation_seeds']['start']+d['evaluation_seeds']['count'])),
        order_seed=p['rng']['arm_order_seed'],timing_seed=p['rng']['timing_order_seed'],bootstrap_seed=p['rng']['bootstrap_seed'],
        cfg=cfg,timing=dict(states=128,rounds=5,warmups=32),quality_evidence=True,research_seed_use=True)

def fit_plan(pl,fit):return dict(model_seed=pl['model_seeds'][fit],cfg=pl['cfg'])
def build(arm,pl,fit):return base.build(arm,fit_plan(pl,fit))
def load_agent(out,arm,fit,pl):
    folder=Path(out)/arm/f'fit_{fit:02d}';ck=torch.load(folder/'final.pt',weights_only=True,map_location='cpu')
    a=build(arm,pl,fit);a['online'].load_state_dict(ck['online']);a['target'].load_state_dict(ck['target'])
    a['opt'].load_state_dict(ck['optimizer']);a['online'].eval();a['target'].eval()
    return a,ck
def export(a,path,arm):return base.deployed.Runtime(a['online'],path) if arm in base.adp.ARMS else base.references.Runtime(a['online'],path)
def from_graph(a,path,arm):
    if arm in base.adp.ARMS:
        runtime=base.deployed.Runtime.__new__(base.deployed.Runtime)
        runtime.arm=arm;runtime.scale=a['online'].core.cycle_to_work.detach().numpy().copy()
        runtime.sweep=arm not in ('coupled_no_sweep','separable64');runtime.separable=arm=='separable64';runtime.fixed=arm=='fixed_price_plain_sweep'
        runtime.session=base.deployed.deployment.session(path)
    else:
        runtime=base.references.Runtime.__new__(base.references.Runtime)
        runtime.mode={'amortized_workload_price24':2,'amortized_no_price24':3,'plain24_nonpositive':5,
            'additive64_nonpositive':1,'additive128_unbounded':0,'bdq64_uniform_adapted':4}[a['online'].base_arm]
        runtime.scaled=a['online'].scaled;runtime.screened=a['online'].screened
        runtime.session=base.references.base.deployment.session(path)
    return runtime

def run(output,mode='fixture',confirmation=None):
    # Must precede plan() and output/model/task construction.
    if mode=='development':
        if confirmation!=CONFIRM:raise RuntimeError('Explicit development confirmation required')
        if not DEVELOPMENT_RELEASED:raise RuntimeError('Development disabled pending separate launch review')
    if mode not in ('fixture','development'):raise RuntimeError('Final/unknown mode disabled')
    p=base.protocol();pl=plan(mode);r.configure();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    r.write(out/'runtime_plan.json',pl)
    files=set(r.source_files(r.protocol()))|set(FILES)|{base.SPEC.name}
    for name in files:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(r.ROOT/name,dest)
    r.write(out/'PRE_RUN_FREEZE.json',dict(protocol_sha256=base.SPEC_SHA,plan_sha256=r.sha(out/'runtime_plan.json'),sources={n:r.sha(r.ROOT/n) for n in sorted(files)}))
    jobs=[(arm,fit) for arm in base.ARMS for fit in range(len(pl['model_seeds']))]
    np.random.default_rng(pl['order_seed']).shuffle(jobs);r.write(out/'job_order.json',jobs)
    events=[]
    def event(stage,**kw):
        events.append(dict(sequence=len(events),stage=stage,**kw));r.write(out/'events.json',events);r.write(out/'progress.json',events[-1])
    checkpoints={};counts={};storage={}
    for arm,fit in jobs:
        event('training',arm=arm,fit=fit);folder=out/arm/f'fit_{fit:02d}';folder.mkdir(parents=True)
        a=build(arm,pl,fit);block=pl['train_blocks'][fit];begin=time.perf_counter_ns()
        with (folder/'optimizer.jsonl').open('w',encoding='utf-8') as stream:
            for ep in range(block['count']):
                row=base.episode(a,block['start']+ep,True,stream=stream);row['run_kind']=pl['kind'];r.save_episode(folder/f'train_{ep:03d}.json.gz',row)
        key=f'{arm}/{fit:02d}';counts[key]={k:a[k] for k in ('slots','updates','copies')}
        counts[key]['training_wall_ns']=time.perf_counter_ns()-begin
        checkpoint=dict(kind=pl['kind'],arm=arm,fit=fit,model_seed=pl['model_seeds'][fit],plan_sha256=r.sha(out/'runtime_plan.json'),
            online=copy.deepcopy(a['online'].state_dict()),target=copy.deepcopy(a['target'].state_dict()),optimizer=a['opt'].state_dict(),
            slots=a['slots'],updates=a['updates'],copies=a['copies'],actor_history=a['actor_history'])
        torch.save(checkpoint,folder/'final.pt');checkpoints[key]=r.sha(folder/'final.pt')
        replay=a['replay'];size=replay.size
        np.savez_compressed(folder/'replay.npz',**{name:getattr(replay,name)[:size] for name in ['states','next_states','actions','rewards','dones']},
            size=size,position=replay.position,terminal=np.array(sorted(replay.terminal),np.int64))
        storage[key]=dict(parameter_bytes=sum(t.numel()*t.element_size() for t in a['online'].parameters()),
            buffer_bytes=sum(t.numel()*t.element_size() for t in a['online'].buffers()),
            target_tensor_bytes=sum(t.numel()*t.element_size() for t in a['target'].state_dict().values()),
            optimizer_tensor_bytes=sum(v.numel()*v.element_size() for item in checkpoint['optimizer']['state'].values() for v in item.values() if isinstance(v,torch.Tensor)),
            replay_allocated_tensor_bytes=sum(getattr(replay,n).nbytes for n in ['states','next_states','actions','rewards','dones']),
            replay_populated_tensor_bytes=sum(getattr(replay,n)[:size].nbytes for n in ['states','next_states','actions','rewards','dones']),
            checkpoint_file_bytes=(folder/'final.pt').stat().st_size,replay_file_bytes=(folder/'replay.npz').stat().st_size)
        del a,checkpoint
    r.write(out/'ALL_CHECKPOINT_FREEZE.json',checkpoints);event('all_checkpoints_frozen',count=len(checkpoints))
    common=np.load(r.ROOT/p['action_cost']['common_inputs'])['observations'][:pl['timing']['states']]
    runtimes={};graphs={};equivalence=0;startup=[]
    for arm,fit in jobs:
        event('export',arm=arm,fit=fit);folder=out/arm/f'fit_{fit:02d}';a,ck=load_agent(out,arm,fit,pl)
        begin=time.perf_counter_ns();runtime=export(a,folder/'final.onnx',arm)
        construction=time.perf_counter_ns()-begin
        for state in common:
            with torch.inference_mode():actions,values=a['online'].greedy(torch.from_numpy(state)[None])
            actual,q=runtime.values(state);np.testing.assert_array_equal(actual,actions.numpy());np.testing.assert_allclose(q,values.numpy(),rtol=1e-5,atol=1e-6);equivalence+=1
        key=f'{arm}/{fit:02d}';runtimes[arm,fit]=runtime;graphs[key]=r.sha(folder/'final.onnx')
        storage[key]['serialized_graph_bytes']=(folder/'final.onnx').stat().st_size
        startup.append(dict(arm=arm,fit=fit,export_session_ns=construction,scope='shared-process import caches; fresh-process record separately'))
        r.write(folder/'export_warnings.json',runtime.warnings)
        assert r.sha(folder/'final.pt')==checkpoints[key]
        del a,ck
    r.write(out/'ALL_EXPORT_FREEZE.json',dict(checkpoints=checkpoints,graphs=graphs,equivalence_checks=equivalence));event('all_exports_frozen',count=len(graphs))
    # Each helper is a distinct fresh Python process. Two-at-a-time limits memory.
    np.save(out/'startup_state.npy',common[0]);startup_jobs=[]
    def cold(job):
        arm,fit=job;path=out/arm/f'fit_{fit:02d}'/'fresh_startup.json'
        command=[sys.executable,str(r.ROOT/'assignment_price_fresh_startup.py'),'--run',str(out),'--arm',arm,'--fit',str(fit),'--output',str(path)]
        begin=time.perf_counter_ns();completed=subprocess.run(command,capture_output=True,text=True,cwd=r.ROOT)
        if completed.returncode:raise RuntimeError(completed.stderr)
        record=r.read(path);record['parent_process_wall_ns']=time.perf_counter_ns()-begin
        r.write(path,record);return dict(arm=arm,fit=fit,**record)
    event('fresh_process_startup',paths=len(jobs))
    with ThreadPoolExecutor(max_workers=2) as pool:startup_jobs=list(pool.map(cold,jobs))
    r.write(out/'startup.json',dict(shared_process=startup,fresh_process=startup_jobs,parallelism=2,scope='Fresh processes include imports/load/native compilation; concurrent helpers affect elapsed time; no hardware-independent claim'))
    orders=[];rng=np.random.default_rng(pl['timing_seed'])
    for rnd in range(pl['timing']['rounds']):order=list(jobs);rng.shuffle(order);orders.append(order)
    r.write(out/'PRE_TIMING_FREEZE.json',dict(plan_sha256=r.sha(out/'runtime_plan.json'),checkpoints=checkpoints,graphs=graphs,orders=orders))
    event('timing',paths=len(jobs));timings=[]
    for rnd,order in enumerate(orders):
        for arm,fit in order:
            runtime=runtimes[arm,fit]
            for state in common[:pl['timing']['warmups']]:runtime.values(state)[0][0].copy()
            for index,state in enumerate(common):
                begin=time.perf_counter_ns();runtime.values(state)[0][0].copy();duration=time.perf_counter_ns()-begin
                timings.append(dict(arm=arm,fit=fit,round=rnd,index=index,ns=duration))
    r.write(out/'timings.json',timings);event('timing_complete',rows=len(timings))
    for arm,fit in jobs:
        event('evaluation',arm=arm,fit=fit);folder=out/arm/f'fit_{fit:02d}';a,ck=load_agent(out,arm,fit,pl)
        a['runtime']=runtimes[arm,fit];before=r.models.original.existing.base.old.digest(a['online'].state_dict())
        for ep,seed in enumerate(pl['evaluation_seeds']):
            row=base.episode(a,seed);row['run_kind']=pl['kind'];r.save_episode(folder/f'eval_{ep:03d}.json.gz',row)
        assert before==r.models.original.existing.base.old.digest(a['online'].state_dict())
        assert r.sha(folder/'final.pt')==checkpoints[f'{arm}/{fit:02d}']
        del a,ck
    for policy in base.HEURISTICS:
        event('heuristic_evaluation',policy=policy);folder=out/'heuristics'/policy;folder.mkdir(parents=True)
        for ep,seed in enumerate(pl['evaluation_seeds']):
            row=base.episode(None,seed,policy=policy);row['run_kind']=pl['kind'];r.save_episode(folder/f'eval_{ep:03d}.json.gz',row)
    r.write(out/'counters.json',counts);r.write(out/'storage.json',storage)
    import assignment_price_bound_reports as reports
    reports.report(out,p,pl)
    event('complete',research_launched=mode=='development')
    r.write(out/'manifest.json',{path.relative_to(out).as_posix():r.sha(path) for path in out.rglob('*') if path.is_file()})
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--mode',default='fixture');parser.add_argument('--confirm-development')
    args=parser.parse_args();run(args.output,args.mode,args.confirm_development)
