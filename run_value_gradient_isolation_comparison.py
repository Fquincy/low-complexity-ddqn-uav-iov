"""Frozen gradient-isolation comparison. Only the declared engineering fixture is released."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

import return_calibrated_handoff as h
import value_gradient_isolation as vgi
BOUND_STEP = vgi.ddqn_retention_step
BOUND_TD = vgi.Critic.td_prediction
import heuristic_initialized_ddqn_baseline as baseline
from heuristic_pilot_teacher_adapter import build_teacher
from run_centered_linear_comparison import array_batch, read, sha, write

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'Source'))
from chapter3_iov_env_v2 import Chapter3Config, Chapter3IoVEnv

SPEC=ROOT/'VALUE_GRADIENT_ISOLATION_COMPARISON_PROTOCOL_2026-11-20.json'
SPEC_SHA='2a3a4cd58f3f04eabf1d54958e058af7294fbb73a2f41000c3242999bcc4f683'
CONFIRM='LAUNCH_FROZEN_VALUE_GRADIENT_ISOLATION_COMPARISON'
DEVELOPMENT_RELEASED=False
IMPLEMENTATION=['run_value_gradient_isolation_comparison.py','audit_value_gradient_isolation_comparison.py']


def protocol():
    if sha(SPEC)!=SPEC_SHA:raise ValueError('Frozen handoff protocol changed')
    spec=read(SPEC)
    for name,digest in spec['definition_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen definition changed: '+name)
    if json.loads(json.dumps(asdict(Chapter3Config())))!=spec['simulator_defaults']:
        raise ValueError('Simulator defaults changed')
    return spec


def plan(spec,mode):
    if mode=='development':
        return dict(mode=mode,kind='EXPLORATORY_PAIRED_DEVELOPMENT',fits=3,seeds=copy.deepcopy(spec['seeds']),
            demo_episodes=10,online_episodes=50,eval_episodes=16,bc_steps=200,batch=64,
            warmup=128,replay_capacity=4096,target_interval=200,timing_states=128,timing_rounds=3,timing_warmups=32)
    if mode!='fixture':raise ValueError('Final/unknown modes disabled')
    fixture=spec['fixture']
    keys=['fits','seeds','demo_episodes','online_episodes','eval_episodes','bc_steps','batch',
          'warmup','replay_capacity','target_interval','timing_states','timing_rounds','timing_warmups']
    return dict(mode=mode,kind=fixture['kind'],**{k:copy.deepcopy(fixture[k]) for k in keys})


def summarize(per_fit,values):
    P,R,A,B,C,D,E,F,G=list(vgi.ARMS)
    policy=all(row['contrasts'][arm][metric]>=0 for row in per_fit.values() for arm in [E,F,A,R] for metric in ['return_value','deadline'])
    policy=policy and all(np.mean([row['contrasts'][arm][metric] for row in per_fit.values()])>0 for arm in [E,F,A,R] for metric in ['return_value','deadline'])
    value=all(values[f'{P}/{i}']['nrmse']<values[f'{D}/{i}']['nrmse'] and abs(values[f'{P}/{i}']['bias'])<abs(values[f'{D}/{i}']['bias']) for i in per_fit)
    interaction={}
    for fit,row in per_fit.items():
        interaction[fit]={metric:sum(sign*row['absolute'][arm][metric] for arm,sign in [(A,1),(B,-1),(C,-1),(D,1)]) for metric in ['return_value','deadline']}
        interaction[fit].update({metric:sum(sign*(abs(values[f'{arm}/{fit}']['bias']) if metric=='abs_bias' else values[f'{arm}/{fit}']['nrmse']) for arm,sign in [(A,1),(B,-1),(C,-1),(D,1)]) for metric in ['nrmse','abs_bias']})
    means={arm:{metric:float(np.mean([row['absolute'][arm][metric] for row in per_fit.values()])) for metric in ['return_value','deadline']} for arm in vgi.ARMS}
    mean_contrasts={arm:{metric:float(np.mean([row['contrasts'][arm][metric] for row in per_fit.values()])) for metric in ['return_value','deadline']} for arm in vgi.ARMS if arm!=P}
    value_contrasts={fit:{arm:dict(nrmse=values[f'{P}/{fit}']['nrmse']-values[f'{arm}/{fit}']['nrmse'],
        abs_bias=abs(values[f'{P}/{fit}']['bias'])-abs(values[f'{arm}/{fit}']['bias'])) for arm in vgi.ARMS if arm!=P} for fit in per_fit}
    mechanism={fit:{label:{**{m:row['absolute'][left][m]-row['absolute'][right][m] for m in ['return_value','deadline']},
        'nrmse':values[f'{left}/{fit}']['nrmse']-values[f'{right}/{fit}']['nrmse'],
        'abs_bias':abs(values[f'{left}/{fit}']['bias'])-abs(values[f'{right}/{fit}']['bias'])}
        for label,left,right in [('P-A',P,A),('P-R',P,R),('R-A',R,A)]} for fit,row in per_fit.items()}
    return dict(policy_screen=bool(policy),value_screen=bool(value),promising=bool(policy and value),factorial_interaction=interaction,
        equal_fit_means=means,equal_fit_mean_contrasts=mean_contrasts,value_contrasts=value_contrasts,mechanism_contrasts=mechanism)


def synthetic_parity_states():
    x=((np.arange(128*533).reshape(128,533)%37)/80).astype(np.float32)
    task=x[:,:240].reshape(128,30,8);task[:,:,6]=np.where(np.arange(30)%2==0,.9,.6);task[:,:,7]=.9
    x[:,-1]=(np.arange(128)%40)/40
    return x


@torch.no_grad()
def select_action(model,state):
    x=np.asarray(state)
    if x.dtype!=np.float32 or x.shape!=(533,):raise ValueError('One FP32 observation required')
    actions,q,_=model.greedy(torch.from_numpy(x)[None])
    return actions[0].numpy().copy(),float(q[0])


@torch.no_grad()
def check_export_parity(training,deployed,states):
    for x in [torch.from_numpy(states)]+[torch.from_numpy(s)[None] for s in states]:
        a,q,_=training.greedy(x);b,r,_=deployed.greedy(x)
        if not torch.equal(a,b) or not torch.equal(q,r):raise ValueError('Same-boundary export parity failed')


def run(output,mode='fixture',confirmation=None):
    if mode=='development':
        if confirmation!=CONFIRM:raise RuntimeError('Explicit development confirmation required')
        if not DEVELOPMENT_RELEASED:raise RuntimeError('Development disabled pending separate readiness and launch release')
    if mode not in ['fixture','development']:raise RuntimeError('Final/unknown modes disabled')
    spec=protocol();pl=plan(spec,mode)
    if vgi.ddqn_retention_step is not BOUND_STEP or vgi.Critic.td_prediction is not BOUND_TD:
        raise ValueError('Source-bound TD entrypoint substituted')
    if mode=='fixture' and ('development_runs' in Path(output).resolve().parts or
            Path(output).resolve()==(ROOT/spec['artifact_retention']['reserved_output']).resolve()):
        raise ValueError('Engineering fixture must be isolated from research output')
    environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,platform=platform.platform(),threads=1)
    if environment!=spec['numerical_environment']:raise ValueError('Frozen numerical environment changed')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    out=Path(output);out.mkdir(parents=True,exist_ok=False);root=out/'run';root.mkdir()
    write(root/'runtime_plan.json',pl)
    sources=set(spec['definition_sha256'])|set(IMPLEMENTATION)|{SPEC.name}
    for name in sorted(sources):
        p=root/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    write(root/'PRE_RUN_FREEZE.json',dict(protocol_sha256=sha(SPEC),plan_sha256=sha(root/'runtime_plan.json'),
        source_sha256={n:sha(ROOT/n) for n in sorted(sources)},
        teacher_backend='Python reference: frozen function bodies, identity native decorators',
        environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,platform=platform.platform(),threads=1)))
    run_begin=time.perf_counter_ns();costs=dict(export_parity_ns=0)
    events=[]
    def event(stage,**fields):
        events.append(dict(sequence=len(events),stage=stage,**fields));write(root/'events.json',events);write(out/'progress.json',events[-1])
    teacher=build_teacher(ROOT,Chapter3Config)
    def teacher_action(s):return teacher.values(s)[0][0]
    def episode(path,seed,actor,callback=None,model=None):
        env=Chapter3IoVEnv(Chapter3Config(),seed=seed);s=env.reset(seed=seed)
        rows=[];qs=[];info_rows=[];begin=time.perf_counter_ns();actor_ns=0
        for t in range(40):
            tick=time.perf_counter_ns();a=np.asarray(actor(s),dtype=np.int64);actor_ns+=time.perf_counter_ns()-tick
            if a.shape!=(32,) or not np.all((a>=0)&(a<5)) or not np.take_along_axis(baseline.allowed(s[None]),a[None,:,None],2).all():raise ValueError('Illegal policy action')
            ns,reward,done,info=env.step(dict(destinations=a[:30],uav_movement=a[30:]))
            row=(s.copy(),a.copy(),float(reward),ns.copy(),bool(done));rows.append(row)
            if model is not None:
                with torch.no_grad():qs.append(float(model(torch.from_numpy(s)[None],torch.from_numpy(a)[None])[0]))
            info_rows.append({k:v for k,v in info.items() if k in ['reward_components','tasks_generated','tasks_completed','tasks_unfinished','deadline_success_rate']})
            if callback:callback(row,t)
            s=ns
            if done:break
        if len(rows)!=40 or not done:raise ValueError('Wrong episode horizon')
        b=array_batch(rows)
        np.savez_compressed(path.with_suffix('.npz'),states=b[0].numpy(),actions=b[1].numpy(),rewards=b[2].numpy(),next_states=b[3].numpy(),dones=b[4].numpy(),q=np.array(qs,np.float64))
        record=dict(seed=seed,steps=40,return_value=sum(r[2] for r in rows),summary=env.episode_summary(),records=env.export_task_records(),step_info=info_rows,wall_ns=time.perf_counter_ns()-begin,actor_ns=actor_ns)
        write(path.with_suffix('.json'),record)
        return rows,record
    demo_begin=time.perf_counter_ns()
    demonstrations={};episodes_by_fit={};initial={};initial_hashes={}
    for fit in range(pl['fits']):
        event('demonstrations',fit=fit);folder=root/f'demo_{fit}';folder.mkdir();allrows=[];eps=[]
        for ep in range(10):
            rows,_=episode(folder/f'episode_{ep:03d}',pl['seeds']['demonstration_seed_blocks'][fit]['start']+ep,teacher_action)
            b=array_batch(rows);eps.append(h.DemoEpisode(b[0],b[1],b[2],b[4]));allrows.extend(rows)
        demonstrations[fit]=allrows;episodes_by_fit[fit]=eps
        torch.manual_seed(pl['seeds']['model_seeds'][fit]);m=vgi.Critic(spec['primary'])
        initial[fit]=copy.deepcopy(m.state_dict());initial_hashes[str(fit)]=h.tensor_hash(m)
    write(root/'DEMONSTRATION_FREEZE.json',dict(files={p.relative_to(root).as_posix():sha(p) for p in sorted(root.glob('demo_*/*'))},initial_tensor_hashes=initial_hashes))
    costs['demo_generation_ns']=time.perf_counter_ns()-demo_begin
    event('demonstrations_frozen')
    jobs=[(a['id'],i) for a in spec['arms'] for i in range(pl['fits'])]
    np.random.default_rng(pl['seeds']['job_order_seed']).shuffle(jobs);write(root/'job_order.json',jobs)
    bc_hashes={};handoff_hashes={};alphas={};checkpoints={};deployments={}
    parity_states=synthetic_parity_states();np.save(root/'export_parity_states.npy',parity_states)
    for arm,fit in jobs:
        event('training',arm=arm,fit=fit);key=f'{arm}/{fit}';folder=root/arm/f'fit_{fit}';folder.mkdir(parents=True)
        model=vgi.Critic(arm);model.load_state_dict(initial[fit]);d=model.definition
        if h.tensor_hash(model)!=initial_hashes[str(fit)]:raise ValueError('Raw initialization mismatch')
        demo=demonstrations[fit] if d.bc else None
        ds=array_batch(demo) if d.bc else None
        bc_rng=np.random.default_rng(pl['seeds']['bc_sampler_seeds'][fit]);bc_log=[];probes={}
        if d.bc:probes['BEFORE_BC']=h.probe(model,ds[0],ds[1],1.)
        begin=time.perf_counter_ns()
        if d.bc:
            bc_opt=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
            for k in range(pl['bc_steps']):
                idx=bc_rng.choice(len(demo),pl['batch'],replace=False);bc_opt.zero_grad(set_to_none=True)
                loss=h.bc_loss(model,ds[0][idx],ds[1][idx]);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError('Nonfinite BC')
                bc_opt.step();bc_log.append(dict(step=k,indices=idx.tolist(),loss=float(loss.detach()),gradient_norm=float(norm)))
            del bc_opt
            probes['AFTER_BC']=h.probe(model,ds[0],ds[1],1.)
        bc_wall=time.perf_counter_ns()-begin;bc_hashes[key]=h.tensor_hash(model)
        write(folder/'bc.json',bc_log)
        begin=time.perf_counter_ns()
        result=h.handoff(model,episodes_by_fit[fit] if d.calibration else None)
        model=result.model;alpha=result.alpha;calibration=result.report if d.calibration else None
        if d.calibration:
            probes['AFTER_SCALE']=calibration['stages']['after_scale'];probes['AFTER_VALUE_FIT']=calibration['stages']['after_value_fit']
            withheld=calibration['residuals']['withheld_returns']
            calibration['withheld_fit_worsened_vs_after_bc']=withheld['after_value_fit']['rmse']>withheld['after_bc']['rmse']
        for stage in ['AFTER_BC','AFTER_SCALE','AFTER_VALUE_FIT']:
            probes.setdefault(stage,dict(skipped=True,reason='Stage not applicable to this arm'))
        calibration_wall=time.perf_counter_ns()-begin;write(folder/'calibration.json',calibration)
        handoff_hashes[key]=h.tensor_hash(model);alphas[key]=alpha
        target,opt=vgi.td_setup(model) if d.ddqn else (None,None)
        if opt is not None and opt.state:raise ValueError('TD optimizer did not reset')
        replay=list(demo) if d.bc and d.ddqn else [];ids=[f'demo:{i}' for i in range(len(replay))]
        choice_rng=np.random.default_rng(pl['seeds']['epsilon_branch_seeds'][fit]);replay_rng=np.random.default_rng(pl['seeds']['replay_sampler_seeds'][fit])
        keep_rng=np.random.default_rng(pl['seeds']['retention_sampler_seeds'][fit]) if d.retention else None
        slots=0;updates=0;copies=0;log=[];drawlog=[];own_probe_rows=[];begin=time.perf_counter_ns()
        def actor(s):
            u=float(choice_rng.random());v=choice_rng.random(32)
            eps=1. if slots<pl['warmup'] else 1-.95*min(slots/(pl['online_episodes']*40),1)
            if u<eps:
                mask=baseline.allowed(s[None])[0];a=np.array([np.flatnonzero(mask[b])[int(v[b]*mask[b].sum())] for b in range(32)],np.int64)
            else:a=baseline.deploy_actions(model,s)
            drawlog.append(dict(slot=slots,epsilon=eps,epsilon_uniform=u,branch_uniforms=v.tolist(),action=a.tolist()))
            return a
        def update(row,t):
            nonlocal slots,updates,copies
            if not d.bc and slots<pl['warmup']:own_probe_rows.append(row)
            replay.append(row);ids.append(f'online:{slots}')
            if len(replay)>pl['replay_capacity']:replay.pop(0);ids.pop(0)
            if slots>=pl['warmup']:
                idx,keep=h.draw_indices(model,len(replay),replay_rng,keep_rng,len(demo) if d.retention else None,batch=pl['batch'])
                sampled=array_batch([replay[i] for i in idx]);first=None
                if updates==0:
                    ss,aa,rr,ns,done=sampled
                    with torch.no_grad():
                        pred=model(ss,aa);current=model.greedy(ss)[1];nq=model.greedy(ns[~done])[1]
                    first=dict(nonterminal_indices=torch.nonzero(~done).flatten().tolist(),
                        reward=rr[~done].tolist(),same_state_gap=(current[~done]-pred[~done]).tolist(),
                        successor_minus_current=(nq-current[~done]).tolist(),
                        residual=(rr[~done]+nq-pred[~done]).tolist())
                captured={};hooks=[]
                if updates==0 and arm in [vgi.PRIMARY,vgi.RESCALE,vgi.BASE]:
                    for name,param in model.named_parameters():
                        hooks.append(param.register_hook(lambda grad,name=name: captured.__setitem__(name,grad.detach().clone())))
                try:
                    detail=vgi.ddqn_retention_step(model,target,opt,sampled,alpha,
                        (ds[0][keep],ds[1][keep]) if d.retention else None)
                finally:
                    for hook in hooks:hook.remove()
                if detail['gradient_rule']!=vgi.TD_RULES.get(arm,'ordinary'):
                    raise ValueError('Wrong TD gradient rule executed')
                if captured:
                    np.savez_compressed(folder/'first_update_gradients.npz',**{n:g.numpy() for n,g in captured.items()})
                updates+=1;copied=updates%pl['target_interval']==0
                if copied:target.load_state_dict(model.state_dict());copies+=1
                detail.update(update=updates,online_slot=slots,indices=idx.tolist(),transition_ids=[ids[i] for i in idx],retention_indices=keep.tolist() if keep is not None else None,target_copy=copied,first_update_decomposition=first)
                log.append(detail)
            slots+=1
        for ep in range(pl['online_episodes'] if d.ddqn else 0):
            episode(folder/f'train_{ep:03d}',pl['seeds']['online_seed_blocks'][fit]['start']+ep,actor,update)
        if not d.bc:
            own=array_batch(own_probe_rows);raw=vgi.Critic(arm);raw.load_state_dict(initial[fit])
            probes['BEFORE_BC']=h.probe(raw,own[0],own[1],1.)
            probe_states,probe_actions=own[0],own[1]
            probe_source=dict(kind='own_online_warmup',transition_ids=[f'online:{i}' for i in range(len(own_probe_rows))])
        else:
            probe_states,probe_actions=ds[0],ds[1]
            probe_source=dict(kind='demonstrations',transition_ids=[f'demo:{i}' for i in range(len(demo))])
        probes['FINAL']=h.probe(model,probe_states,probe_actions,alpha)
        write(folder/'probes.json',dict(source=probe_source,stages=probes))
        count=dict(slots=slots,updates=updates,target_copies=copies,initial_target_copy=d.ddqn,bc_updates=len(bc_log),
            retention_batches=sum(v['retention_indices'] is not None for v in log),ridge_solves=int(d.calibration),
            initial_tensor_hash=initial_hashes[str(fit)],bc_end_tensor_hash=bc_hashes[key],handoff_tensor_hash=handoff_hashes[key],
            alpha=alpha,preloaded_demo_hash=sha(root/f'demo_{fit}/episode_000.npz') if d.bc and d.ddqn else None,
            parameter_bytes=baseline.parameter_bytes(model),buffer_bytes=sum(v.numel()*v.element_size() for v in model.buffers()),
            bc_wall_ns=bc_wall,calibration_wall_ns=calibration_wall,online_wall_ns=time.perf_counter_ns()-begin)
        write(folder/'counts.json',count);write(folder/'optimizer.json',log);write(folder/'action_draws.json',drawlog)
        if replay:
            b=array_batch(replay);np.savez_compressed(folder/'final_replay.npz',states=b[0].numpy(),actions=b[1].numpy(),rewards=b[2].numpy(),next_states=b[3].numpy(),dones=b[4].numpy())
        write(folder/'final_rng.json',dict(choices=choice_rng.bit_generator.state,replay=replay_rng.bit_generator.state,
            retention=keep_rng.bit_generator.state if keep_rng is not None else None,bc=bc_rng.bit_generator.state,transition_ids=ids))
        torch.save(dict(online=model.state_dict(),target=target.state_dict() if target else None,optimizer=opt.state_dict() if opt else None,counts=count),folder/'final.pt')
        checkpoints[key]=sha(folder/'final.pt')
        tick=time.perf_counter_ns();exported=vgi.deployment_model(model)
        if type(exported) is not h.Critic or hasattr(exported,'td_prediction'):raise ValueError('Training class leaked into deployment')
        if h.tensor_hash(exported)!=h.tensor_hash(model):raise ValueError('Export tensor hash mismatch')
        for name,value in model.state_dict().items():
            if value.untyped_storage().data_ptr()==exported.state_dict()[name].untyped_storage().data_ptr():
                raise ValueError('Export aliases training state')
        torch.save(exported.state_dict(),folder/'deployment_weights.pt')
        deployed=h.Critic(vgi.BASE if arm in vgi.TD_RULES else arm).eval().requires_grad_(False)
        deployed.load_state_dict(torch.load(folder/'deployment_weights.pt',weights_only=True))
        check_export_parity(model,deployed,parity_states)
        write(folder/'export_parity.json',dict(arm=arm,training_tensor_hash=h.tensor_hash(model),
            deployment_tensor_hash=h.tensor_hash(deployed),original_class=type(deployed).__module__+'.'+type(deployed).__name__,
            single_states=128,batch_states=128,exact_actions=True,exact_Q=True))
        deployments[key]=sha(folder/'deployment_weights.pt');costs['export_parity_ns']+=time.perf_counter_ns()-tick
    for fit in range(pl['fits']):
        if len({bc_hashes[f'{a}/{fit}'] for a,d in vgi.ARMS.items() if d.bc})!=1:raise ValueError('BC-end parity failed')
        if len({handoff_hashes[f'{a}/{fit}'] for a,d in vgi.ARMS.items() if d.calibration})!=1:raise ValueError('A/B/E handoff parity failed')
        if len({alphas[f'{a}/{fit}'] for a,d in vgi.ARMS.items() if d.calibration})!=1:raise ValueError('A/B/E alpha parity failed')
    write(root/'CHECKPOINT_FREEZE.json',dict(checkpoints=checkpoints,deployments=deployments,bc_end_hashes=bc_hashes,handoff_hashes=handoff_hashes,alphas=alphas))
    event('checkpoints_frozen')
    models={}
    for arm,fit in jobs:
        m=h.Critic(vgi.BASE if arm in vgi.TD_RULES else arm).eval().requires_grad_(False)
        m.load_state_dict(torch.load(root/arm/f'fit_{fit}/deployment_weights.pt',weights_only=True));models[f'{arm}/{fit}']=m
    evaluation_begin=time.perf_counter_ns()
    evaluations={};values={}
    for key,m in list(models.items())+[('teacher',None)]:
        event('evaluation',policy=key);folder=root/'evaluation'/key;folder.mkdir(parents=True);records=[];errors=[];gall=[]
        for ep in range(pl['eval_episodes']):
            rows,record=episode(folder/f'episode_{ep:03d}',pl['seeds']['evaluation_seed_block']['start']+ep,
                teacher_action if m is None else lambda s,m=m:baseline.deploy_actions(m,s),model=m)
            records.append(record)
            if m is not None:
                saved=dict(np.load((folder/f'episode_{ep:03d}').with_suffix('.npz')));g=h.episode_returns(saved['rewards']);errors.extend(saved['q']-g);gall.extend(g)
        evaluations[key]=dict(return_value=float(np.mean([r['return_value'] for r in records])),deadline=float(np.mean([r['summary']['deadline_success_rate'] for r in records])))
        if m is not None:values[key]=dict(bias=float(np.mean(errors)),nrmse=float(np.sqrt(np.mean(np.square(errors)))/max(np.sqrt(np.mean(np.square(gall))),1e-12)))
    costs['evaluation_ns']=time.perf_counter_ns()-evaluation_begin
    per_fit={}
    for fit in range(pl['fits']):
        absolute={a:evaluations[f'{a}/{fit}'] for a in vgi.ARMS};primary=absolute[spec['primary']];teacher_mean=evaluations['teacher']
        per_fit[str(fit)]=dict(absolute=absolute,teacher=teacher_mean,
            contrasts={a:{k:primary[k]-r[k] for k in primary} for a,r in absolute.items() if a!=spec['primary']},
            minus_teacher={a:{k:r[k]-teacher_mean[k] for k in r} for a,r in absolute.items()})
    write(root/'analysis.json',dict(per_fit=per_fit,value_metrics=values,**summarize(per_fit,values),kind=pl['kind'],quality_claim=False))
    states=[]
    for p in sorted((root/'evaluation').rglob('*.npz')):states.extend(np.load(p)['states'])
    rng=np.random.default_rng(pl['seeds']['timing_order_seed']);chosen=rng.permutation(len(states))[:pl['timing_states']]
    ss=np.stack([states[i] for i in chosen]);np.save(root/'timing_states.npy',ss)
    order=[(r,k) for r in range(pl['timing_rounds']) for k in list(models)+['teacher']];rng.shuffle(order)
    write(root/'TIMING_FREEZE.json',dict(states_sha256=sha(root/'timing_states.npy'),state_indices=chosen.tolist(),order=order))
    timing=[];warmups=[]
    for r,key in order:
        action=teacher_action if key=='teacher' else lambda s,m=models[key]:select_action(m,s)[0]
        for j in range(pl['timing_warmups']):
            a=action(ss[j%len(ss)]);warmups.append(dict(round=r,policy=key,state=j%len(ss),action=a.tolist()))
        for i,s in enumerate(ss):
            begin=time.perf_counter_ns();a=action(s);elapsed=time.perf_counter_ns()-begin
            timing.append(dict(round=r,policy=key,state=i,ns=elapsed,action=a.tolist()))
    write(root/'timing.json',timing);write(root/'timing_warmups.json',warmups)
    costs['preaudit_wall_ns']=time.perf_counter_ns()-run_begin
    write(root/'stage_costs.json',costs);event('artifacts_complete')
    write(root/'manifest.json',{p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()})
    from audit_value_gradient_isolation_comparison import audit
    audit_begin=time.perf_counter_ns();receipt=audit(root);receipt['audit_wall_ns']=time.perf_counter_ns()-audit_begin
    write(out/'saved_artifact_audit.json',receipt)
    write(out/'completion.json',dict(status='ENGINEERING_AUDITED_COMPLETE' if mode=='fixture' else 'DEVELOPMENT_AUDITED_COMPLETE',
        kind=pl['kind'],audit_passed=receipt['passed'],manifest_sha256=sha(root/'manifest.json'),research_launch_authorized=mode=='development'))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fixture','development','final'],default='fixture');p.add_argument('--output',required=True);p.add_argument('--confirmation')
    args=p.parse_args();print(json.dumps(run(args.output,args.mode,args.confirmation)))

