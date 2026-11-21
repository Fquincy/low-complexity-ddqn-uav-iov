"""Frozen comparison orchestration. Fixture authorized; development explicitly gated."""
import argparse,ast,copy,gzip,hashlib,importlib.metadata,json,platform,shutil,sys,time,traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import deadline_screen_research_models as models
import deadline_screen_research_analysis as analysis
from chapter3_iov_env_v2 import Chapter3Config,Chapter3IoVEnv
from train_chapter3_ddqn import to_action_dict
from validate_chapter3_model import select_destinations

ROOT=Path(__file__).resolve().parent
SPEC=ROOT/'DEADLINE_SCREEN_RESEARCH_PROTOCOL_2026-11-12.json'
SPEC_SHA='23b1b292e80a4dadba99ee99b1f7151a80a37db4f6c8b5d03ac198729acf84de'
CONFIRM='LAUNCH_FROZEN_DEADLINE_SCREEN_DEVELOPMENT'
DEVELOPMENT_RELEASED=False  # Separate launch review must release a source/versioned authorization.
NEW_FILES=['deadline_screen_research_models.py','deadline_screen_research_analysis.py','run_deadline_screen_research.py',
           'audit_deadline_screen_research.py','test_deadline_screen_research.py']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)
def protocol():
    require(sha(SPEC)==SPEC_SHA,'protocol hash changed')
    p=read(SPEC)
    require(sha(ROOT/p['comparison_file'])==p['comparison_sha256'],'comparison hash changed')
    for name,h in p['definition_sha256'].items():require(sha(ROOT/name)==h,'frozen source changed: '+name)
    require(asdict(Chapter3Config())=={k:tuple(v) if isinstance(v,list) else v for k,v in p['simulator']['defaults'].items()},'simulator configuration changed')
    return p

def configure():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    models.fast.decoder.numba.set_num_threads(1)

def plan(p,mode):
    x=copy.deepcopy(p['fixture'] if mode=='integration-fixture' else p['development'])
    cfg=copy.deepcopy(p['training'])
    if mode=='integration-fixture':
        for k in ('warmup','batch_size','replay_capacity','target_interval_updates','episodes_per_fit'):cfg[k]=x[k]
        cfg['epsilon_decay']=x['epsilon_decay']
    else:cfg['epsilon_decay']=cfg['slots_per_fit']
    return dict(mode=mode,kind='SYNTHETIC_NOT_RESEARCH' if mode=='integration-fixture' else 'RESEARCH_DEVELOPMENT',
        model_seeds=x['model_seeds'],train_blocks=x['train_blocks'],evaluation_seeds=list(range(x['evaluation_seeds']['start'],x['evaluation_seeds']['start']+x['evaluation_seeds']['count'])),
        cfg=cfg,order_seed=x['engineering_seeds']['order'] if mode=='integration-fixture' else p['rng']['arm_order_seed'],
        timing_seed=x['engineering_seeds']['timing'] if mode=='integration-fixture' else p['rng']['timing_order_seed'])

def build(definition,seed,cfg):
    online=models.Critic(definition,seed)
    require(sum(x.numel() for x in online.parameters())==definition['expected_online_parameter_count'],'parameter count')
    require(sum(x.numel()*x.element_size() for x in online.buffers())==definition['expected_buffer_bytes'],'buffer count')
    target=copy.deepcopy(online).eval().requires_grad_(False)
    return dict(online=online,target=target,definition=definition,cfg=cfg,seed=seed,slots=0,updates=0,copies=0,
        replay=models.original.existing.TerminalBuffer(cfg['replay_capacity']),
        opt=torch.optim.Adam(online.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay']),
        action_rng=np.random.default_rng(seed+100000),replay_rng=np.random.default_rng(seed+200000),anchor_rng=np.random.default_rng(seed+300000),actor_history=[])

def update(a):
    cfg=a['cfg']
    if a['slots']<cfg['warmup']:return None
    start=time.perf_counter_ns()
    raw=a['replay'].sample(cfg['batch_size'],a['replay_rng']);s,act,r,ns,d=map(torch.tensor,raw[:5])
    anchor=a['replay'].terminal_sample(cfg['anchor']['batch'],a['anchor_rng']) if a['online'].anchored else None
    next_actions=np.full((len(d),32),-1,np.int64);next_values=np.zeros(len(d),np.float32)
    with torch.no_grad():
        active=~d.bool()
        if active.any():
            na=a['online'].greedy(ns[active])[0];next_actions[active.numpy()]=na.numpy()
            next_values[active.numpy()]=a['target'](ns[active],na).numpy()
        y=r+torch.from_numpy(next_values)
    q=a['online'](s,act)
    branch=None
    if a['definition']['model_key']=='bdq64_uniform_adapted':
        branch=a['online'].model(s).gather(2,act[:,:,None]).squeeze(2)
        uniform=F.smooth_l1_loss(branch,y[:,None].expand_as(branch),beta=cfg['huber_beta'])
    else:uniform=F.smooth_l1_loss(q,y,beta=cfg['huber_beta'])
    anchor_loss=uniform*0
    anchor_q=[];anchor_r=[]
    if anchor is not None:
        aq=a['online'](anchor[0],anchor[1]);anchor_q=aq.detach().tolist();anchor_r=anchor[2].tolist()
        anchor_loss=F.smooth_l1_loss(aq,anchor[2],beta=cfg['huber_beta'])
    loss=uniform+cfg['anchor']['weight']*anchor_loss
    require(torch.isfinite(y).all() and torch.isfinite(q).all() and torch.isfinite(loss),'nonfinite value/loss')
    a['opt'].zero_grad(set_to_none=True);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(a['online'].parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
    require(all(x.grad is None and not x.requires_grad for x in a['target'].parameters()),'target gradient')
    a['opt'].step();a['updates']+=1
    require(all(torch.isfinite(x).all() for x in a['online'].parameters()),'nonfinite parameters')
    copied=a['updates']%cfg['target_interval_updates']==0
    if copied:a['target'].load_state_dict(a['online'].state_dict());a['copies']+=1
    return dict(step=a['updates'],physical_slots=a['slots'],target_copy=copied,loss=float(loss.detach()),uniform_loss=float(uniform.detach()),
        anchor_loss=float(anchor_loss.detach()),anchor_Q=anchor_q,anchor_rewards=anchor_r,empty_anchor=bool(a['online'].anchored and anchor is None),
        anchor_indices=[] if anchor is None else anchor[-1].tolist(),terminal_pool=sorted(a['replay'].terminal),
        replay_indices=raw[-1].tolist(),sample_rewards=r.tolist(),sample_dones=d.tolist(),next_actions=next_actions.tolist(),
        next_target_values=next_values.tolist(),bellman_targets=y.tolist(),chosen_Q=q.detach().tolist(),
        branch_values=[] if branch is None else branch.detach().tolist(),TD_residuals=(q.detach()-y).tolist(),
        grad_norm_before_clip=float(norm),clipped=bool(norm>cfg['gradient_clip']),optimizer_ns=time.perf_counter_ns()-start)

def episode(a,seed,kind,training=False,policy=None,episode_id=0,log_stream=None):
    env=Chapter3IoVEnv(seed=seed);states=[];following=[];slots=[];qs=[];tqs=[]
    for t in range(40):
        state=env._observation().copy();states.append(state)
        tasks=models.original.existing.base.old.workload(env);before=env.uav_positions.copy();vehicle_positions=env.vehicle_positions.copy()
        mask=models.original.allowed(state[None])[0];excluded=~mask[:30,0]
        # Exact task values are audit-only: never passed to the controller.
        for task in tasks:
            i=task['vehicle_id'];lo,hi=models.original.intervals(np.array([state[8*i+6]],np.float32),500e6)
            dlo,dhi=models.original.intervals(np.array([state[8*i+7]],np.float32),.3)
            require(lo[0]<=task['required_cycles']<=hi[0] and dlo[0]<=task['deadline_s']<=dhi[0],'interval containment')
            require(not excluded[i] or task['required_cycles']/1.5e9>task['deadline_s']+1e-10,'false exclusion')
        if policy:
            action=np.concatenate((select_destinations(env,policy),np.zeros(2,np.int64)))
        elif training:
            warm=a['slots']<a['cfg']['warmup'];eps=1. if warm else 1-.95*min(a['slots']/a['cfg']['epsilon_decay'],1.)
            u=float(a['action_rng'].random());uniforms=a['action_rng'].random(32);explore=bool(warm or u<eps)
            if explore:action=models.original.explore(state[None],uniforms[None],a['definition']['screened'])[0]
            else:
                with torch.inference_mode():action=a['online'].greedy(torch.from_numpy(state)[None])[0][0].numpy().copy()
            a['actor_history'].append(dict(warmup=warm,epsilon=eps,explore=explore,epsilon_uniform=u,branch_uniforms=uniforms.tolist()))
        else:
            with torch.inference_mode():
                act,q=a['online'].greedy(torch.from_numpy(state)[None]);tq=a['target'](torch.from_numpy(state)[None],act)
            action=act[0].numpy().copy();qs.append(float(q[0]));tqs.append(float(tq[0]))
            native,v=a['runtime'].values(state);np.testing.assert_array_equal(action,native[0]);np.testing.assert_allclose(q.numpy(),v,rtol=1e-5,atol=1e-6)
        require(action.shape==(32,) and np.all((action>=0)&(action<5)),'illegal action')
        if a is not None and a['definition']['screened']:require(mask[np.arange(32),action].all(),'out of screened set')
        ns,reward,done,info=env.step(to_action_dict(action,30));following.append(ns.copy())
        require(np.isfinite(reward) and np.isfinite(ns).all() and done==(t==39),'invalid transition')
        if training:
            a['replay'].add(state,action,reward,ns,done);a['slots']+=1
            item=update(a)
            if item is not None:log_stream.write(json.dumps(item,allow_nan=False)+'\n')
        after=env.uav_positions.copy()
        slots.append(dict(slot=t,action=action.tolist(),reward=reward,done=done,reward_components=info['reward_components'],energy_j=info['energy_j'],
            tasks_arrived=tasks,vehicle_positions=vehicle_positions.tolist(),uav_before=before.tolist(),uav_after=after.tolist(),actual_distance_m=np.linalg.norm(after-before,axis=1).tolist(),excluded_local=excluded.tolist()))
    summary=env.episode_summary();rewards=[x['reward'] for x in slots]
    return dict(seed=seed,episode=episode_id,run_kind=kind,return_sum=sum(rewards),deadline_success=summary['deadline_success_rate'],
        tasks_generated=summary['tasks_generated'],tasks_on_time=summary['tasks_on_time'],slot_rewards=rewards,summary=summary,slots=slots,
        tasks=env.export_task_records(),chosen_q=qs,target_q=tqs,realized_return_to_go=np.cumsum(rewards[::-1])[::-1].tolist(),
        workload_sha256=models.original.existing.base.old.digest([x['tasks_arrived'] for x in slots]),observations=np.stack(states),next_observations=np.stack(following))

def save_episode(path,row):
    row=dict(row);np.savez_compressed(path.with_suffix('').with_suffix('.npz'),states=row.pop('observations'),next_states=row.pop('next_observations'))
    with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(row,f,allow_nan=False)

def compact(row):return {k:v for k,v in row.items() if k not in ('observations','next_observations')}

def source_files(p):
    pending=[ROOT/n for n in [*NEW_FILES,*p['definition_sha256']]];seen=set()
    while pending:
        file=pending.pop()
        if file in seen:continue
        seen.add(file)
        if file.suffix!='.py':continue
        for node in ast.walk(ast.parse(file.read_text(encoding='utf-8-sig'))):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                for base in (ROOT,ROOT/'Source'):
                    candidate=base/Path(*name.split('.')).with_suffix('.py')
                    if candidate.is_file() and candidate not in seen:pending.append(candidate)
    return sorted({str(x.relative_to(ROOT)).replace('\\','/') for x in seen}|{SPEC.name,p['comparison_file']})

def run(output,mode='integration-fixture',confirmation=None):
    # Guard before imports/configuration/output creation and all model/task seeds.
    if mode=='final':raise RuntimeError('Final execution disabled')
    if mode not in ('integration-fixture','development'):raise ValueError('unknown mode')
    if mode=='development':
        if confirmation!=CONFIRM:raise RuntimeError('Explicit development confirmation required')
        if not DEVELOPMENT_RELEASED:raise RuntimeError('Development disabled pending separate launch review')
    p=protocol();pl=plan(p,mode);configure();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    sources=source_files(p)
    for name in sources:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    write(out/'runtime_plan.json',pl)
    versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    write(out/'environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),processor=platform.processor(),
        torch=torch.__version__,numpy=np.__version__,torch_path=torch.__file__,numba_path=models.fast.decoder.numba.__file__,packages=versions,threads=torch.get_num_threads()))
    write(out/'PRE_RUN_FREEZE.json',dict(protocol_sha256=sha(SPEC),source_sha256={n:sha(ROOT/n) for n in sources},kind=pl['kind'],plan_sha256=sha(out/'runtime_plan.json')))
    defs={d['id']:d for d in p['learned_arms']};fits=len(pl['model_seeds']);jobs=[(n,i) for n in defs for i in range(fits)]
    np.random.default_rng(pl['order_seed']).shuffle(jobs);write(out/'arm_order.json',jobs)
    events=[]
    def event(stage,**kw):
        events.append(dict(sequence=len(events),stage=stage,**kw));write(out/'events.json',events);write(out/'progress.json',events[-1])
    frozen={};counts={}
    for arm,fit in jobs:
        event('training',arm=arm,fit=fit);folder=out/arm/f'fit_{fit:02d}';folder.mkdir(parents=True)
        a=build(defs[arm],pl['model_seeds'][fit],pl['cfg']);start=time.perf_counter_ns()
        with (folder/'optimizer.jsonl').open('w',encoding='utf-8') as stream:
            for i in range(pl['cfg']['episodes_per_fit']):
                row=episode(a,pl['train_blocks'][fit]['start']+i,pl['kind'],True,episode_id=i,log_stream=stream)
                row.update(model_seed=a['seed'],arm=arm);save_episode(folder/f'train_{i:03d}.json.gz',row)
        ck={k:copy.deepcopy(a[k].state_dict()) for k in ('online','target','opt')}
        ck.update(arm=arm,model_seed=a['seed'],definition=defs[arm],cfg=pl['cfg'],slots=a['slots'],updates=a['updates'],copies=a['copies'],
            actor_history=a['actor_history'],action_rng_state=a['action_rng'].bit_generator.state,replay_rng_state=a['replay_rng'].bit_generator.state,
            anchor_rng_state=a['anchor_rng'].bit_generator.state,training_ns=time.perf_counter_ns()-start,replay_storage_bytes=a['replay'].storage_bytes)
        torch.save(ck,folder/'final.pt');key=f'{arm}/fit_{fit:02d}/final.pt';frozen[key]=sha(folder/'final.pt')
        counts[key]={k:a[k] for k in ('slots','updates','copies')};del a,ck
    write(out/'ALL_FINAL_CHECKPOINTS_FREEZE.json',frozen);event('all_checkpoints_frozen')
    obs=ROOT/p['action_cost']['common_inputs'];require(sha(obs)==p['action_cost']['common_inputs_sha256'],'engineering input changed')
    shutil.copy2(obs,out/'engineering_observations.npz');z=np.load(obs);states=z['observations'];require(states.shape==(128,533),'engineering observations')
    agents={};predictions={}
    for arm,fit in jobs:
        folder=out/arm/f'fit_{fit:02d}';ck=torch.load(folder/'final.pt',weights_only=False,map_location='cpu')
        online=models.Critic(defs[arm],pl['model_seeds'][fit]).eval();online.load_state_dict(ck['online'])
        target=copy.deepcopy(online).eval().requires_grad_(False);target.load_state_dict(ck['target'])
        a=dict(online=online,target=target,definition=defs[arm],runtime=models.Runtime(online));agents[(arm,fit)]=a
        predictions[f'{arm}/{fit}']=[]
        for state in states:
            with torch.inference_mode():act,q=online.greedy(torch.from_numpy(state)[None])
            b,v=a['runtime'].values(state);np.testing.assert_array_equal(act.numpy(),b);np.testing.assert_allclose(q.numpy(),v,atol=1e-6,rtol=1e-5)
            predictions[f'{arm}/{fit}'].append(dict(action=b[0].tolist(),Q=float(v[0])))
    write(out/'predictions.json',predictions);event('all_runtime_equivalence_passed')
    rng=np.random.default_rng(pl['timing_seed']);orders=[]
    for rnd in range(5):
        order=list(jobs);rng.shuffle(order);orders.append(order)
    write(out/'PRE_TIMING_FREEZE.json',dict(checkpoints=frozen,orders=orders,observations_sha256=sha(out/'engineering_observations.npz'),
        source_sha256={n:sha(ROOT/n) for n in sources}))
    raw=[]
    for rnd,order in enumerate(orders):
        for arm,fit in order:
            runtime=agents[(arm,fit)]['runtime']
            for state in states[:32]:runtime.values(state)[0][0].copy()
            for i,state in enumerate(states):
                start=time.perf_counter_ns();runtime.values(state)[0][0].copy();ns=time.perf_counter_ns()-start
                raw.append(dict(arm=arm,fit=fit,round=rnd,index=i,nanoseconds=ns))
    costs={n:[] for n in defs}
    for arm in defs:
        for fit in range(fits):
            ns=np.array([x['nanoseconds'] for x in raw if x['arm']==arm and x['fit']==fit],np.float64)
            model=agents[(arm,fit)]['online']
            costs[arm].append(dict(median_ms=float(np.median(ns)/1e6),p95_ms=float(np.quantile(ns,.95)/1e6),
                parameter_bytes=sum(x.numel()*x.element_size() for x in model.parameters()),buffer_bytes=sum(x.numel()*x.element_size() for x in model.buffers()),
                checkpoint_bytes=(out/arm/f'fit_{fit:02d}'/'final.pt').stat().st_size))
    write(out/'raw_timings.json',raw);write(out/'costs.json',costs);event('all_timing_completed')
    evaluations={n:[[] for _ in range(fits)] for n in defs};invariance=[]
    for arm,fit in jobs:
        a=agents[(arm,fit)];folder=out/arm/f'fit_{fit:02d}'
        before=models.original.existing.base.old.digest((a['online'].state_dict(),a['target'].state_dict(),a['online'].training,a['target'].training))
        for seed in pl['evaluation_seeds']:
            row=episode(a,seed,pl['kind'],episode_id='evaluation');row.update(model_seed=pl['model_seeds'][fit],arm=arm)
            save_episode(folder/f'evaluation_{seed}.json.gz',row);evaluations[arm][fit].append(compact(row))
        after=models.original.existing.base.old.digest((a['online'].state_dict(),a['target'].state_dict(),a['online'].training,a['target'].training))
        require(before==after,'evaluation mutated model');invariance.append(dict(arm=arm,fit=fit,before=before,after=after))
        require(sha(folder/'final.pt')==frozen[f'{arm}/fit_{fit:02d}/final.pt'],'checkpoint mutation')
    heuristics={x['id']:[] for x in p['heuristics']}
    for name in heuristics:
        for seed in pl['evaluation_seeds']:
            row=episode(None,seed,pl['kind'],policy=name,episode_id='evaluation');row.update(arm=name,model_seed=None)
            save_episode(out/f'heuristic_{name}_{seed}.json.gz',row);heuristics[name].append(compact(row))
    write(out/'evaluation_invariance.json',invariance);event('all_evaluation_completed')
    summary=analysis.summarize(p,evaluations,heuristics,costs,mode=='integration-fixture')
    write(out/'analysis.json',summary)
    verification=dict(kind=pl['kind'],checkpoints=len(frozen),training_slots=sum(x['slots'] for x in counts.values()),
        optimizer_updates=sum(x['updates'] for x in counts.values()),target_copies=sum(x['copies'] for x in counts.values()),
        evaluation_slots=(len(jobs)+len(heuristics))*len(pl['evaluation_seeds'])*40,raw_timing_rows=len(raw),
        quality_evidence=mode!='integration-fixture',research_launched=mode=='development',counts=counts)
    write(out/'verification.json',verification);event('summaries_completed')
    write(out/'run_manifest.json',dict(files={x.relative_to(out).as_posix():sha(x) for x in out.rglob('*') if x.is_file()}))
    return verification

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=['integration-fixture','development','final']);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--confirm-development',choices=[CONFIRM]);args=parser.parse_args()
    if args.mode=='final':parser.error('Final execution disabled')
    if args.mode=='development' and args.confirm_development!=CONFIRM:parser.error('Explicit development confirmation required')
    try:print(json.dumps(run(args.output,args.mode,args.confirm_development)))
    except Exception:traceback.print_exc();return 1
    return 0

if __name__=='__main__':sys.exit(main())
