"""Eight-arm engineering harness. Research modes are deliberately disabled.

Only marked synthetic integration fixtures and a pre-training CPU cost gate run.
Nothing in this module changes the old frozen experiment or thesis.
"""
from __future__ import annotations
import argparse,copy,gzip,json,platform,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import run_value_bound_tradeoff as old
from controller_redesign_v2 import SHARED_ARMS,SharedCostNetwork,AdaptedBDQ,parameter_bytes
from chapter3_iov_env_v2 import Chapter3Config,Chapter3IoVEnv
from train_chapter3_ddqn import FactorizedQNetwork,to_action_dict
from emc_q import EMCQ

ROOT=Path(__file__).resolve().parent
SPEC=ROOT/'analysis/controller_redesign_2026-11-05/prospective_comparison_v2.json'
SPEC_SHA='35a41a18bc642a79be4f4f3dfbdf0a7b7a18ecd4a351700b0e6e7f1e2b1b99b5'
ARMS=(*SHARED_ARMS,'additive64_nonpositive','additive128_unbounded','emc_q_nonpositive','bdq64_uniform_adapted')
SOURCES=['controller_redesign_v2.py','run_controller_redesign_engineering.py','audit_controller_redesign_engineering.py',
 'test_controller_redesign_engineering.py','run_value_bound_tradeoff.py','emc_q.py','audit_value_bound_development.py',
 'Source/chapter3_iov_env.py','Source/chapter3_iov_env_v2.py','Source/train_chapter3_ddqn.py',
 'Source/validate_chapter3_model.py',str(SPEC.relative_to(ROOT))]

def spec():
    if old.sha(SPEC)!=SPEC_SHA:raise ValueError('prospective version-2 design hash changed')
    s=json.loads(SPEC.read_text());assert tuple(s['arms'])==ARMS and not s['security']
    return s

def configure():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def bounded(arm):return arm in SHARED_ARMS or arm in ('additive64_nonpositive','emc_q_nonpositive')

def greedy(model,states,arm):
    if arm in SHARED_ARMS:return model.greedy(states)
    if arm=='bdq64_uniform_adapted':
        branches=model(states);actions=branches.argmax(2)
        return actions,branches.gather(2,actions[:,:,None]).squeeze(2).mean(1),{}
    action,q,_=old.greedy(model,states,arm)
    return action,q,{}

def chosen(model,states,actions,arm):
    if arm in SHARED_ARMS:return model(states,actions)
    if arm=='bdq64_uniform_adapted':return model(states).gather(2,actions[:,:,None]).squeeze(2).mean(1)
    return old.chosen_value(model,states,actions,arm)[0]

def build(arm,seed,cfg):
    if arm not in ARMS:raise ValueError('unknown arm')
    torch.manual_seed(seed)
    if arm in SHARED_ARMS:model=SharedCostNetwork(arm)
    elif arm=='bdq64_uniform_adapted':model=AdaptedBDQ()
    elif arm=='emc_q_nonpositive':model=EMCQ()
    else:model=FactorizedQNetwork(533,32,5,(128,128) if arm=='additive128_unbounded' else (64,64))
    target=copy.deepcopy(model).eval().requires_grad_(False)
    return dict(arm=arm,seed=seed,online=model,target=target,
        opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0),
        replay=old.Replay(cfg['capacity']),action_rng=np.random.default_rng(seed+100000),
        replay_rng=np.random.default_rng(seed+200000),slots=0,updates=0,copies=0,actor_history=[],cfg=cfg)

@torch.no_grad()
def targets(agent,rewards,next_states,dones):
    if not torch.all((dones==0)|(dones==1)):raise ValueError('binary done mask required')
    y=rewards.detach().clone();active=~dones.bool()
    if active.any():
        actions=greedy(agent['online'],next_states[active],agent['arm'])[0]
        y[active]+=agent['cfg']['gamma']*chosen(agent['target'],next_states[active],actions,agent['arm'])
    return y

def update(agent):
    cfg=agent['cfg']
    if agent['slots']<cfg['warmup']:return None
    s,a,r,ns,d,ix=agent['replay'].sample(cfg['batch'],agent['replay_rng'])
    s,a,r,ns,d=map(torch.tensor,(s,a,r,ns,d))
    y=targets(agent,r,ns,d)
    if agent['arm']=='bdq64_uniform_adapted':
        branches=agent['online'](s).gather(2,a[:,:,None]).squeeze(2)
        q=branches.mean(1);loss=F.smooth_l1_loss(branches,y[:,None].expand_as(branches),beta=1.)
    else:
        q=chosen(agent['online'],s,a,agent['arm']);loss=F.smooth_l1_loss(q,y,beta=1.)
    if y.requires_grad or not torch.isfinite(q).all() or not torch.isfinite(y).all():raise FloatingPointError('Q/target invalid')
    agent['opt'].zero_grad(set_to_none=True);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(agent['online'].parameters(),cfg['clip'],error_if_nonfinite=True)
    if any(p.requires_grad or p.grad is not None for p in agent['target'].parameters()):raise AssertionError('target gradient')
    agent['opt'].step();agent['updates']+=1
    if any(not torch.isfinite(p).all() for p in agent['online'].parameters()):raise FloatingPointError('nonfinite parameters')
    copied=agent['updates']%cfg['target_interval']==0
    if copied:agent['target'].load_state_dict(agent['online'].state_dict());agent['copies']+=1
    return dict(step=agent['updates'],physical_slots=agent['slots'],loss=float(loss.detach()),
        grad_norm_before_clip=float(norm),gradient_clipped=bool(norm>cfg['clip']),
        chosen_q=old.distribution(q),bellman_target=old.distribution(y),td_residual=old.distribution(q.detach()-y),
        replay_indices=ix.tolist(),target_copy=copied,terminal_samples=int(d.sum()))

def episode(agent,seed,training=False,policy=None,episode_id=0):
    env=Chapter3IoVEnv(seed=seed);states=[];next_states=[];slots=[];q_values=[];target_values=[];logs=[]
    for t in range(40):
        state=env._observation().copy();states.append(state);work=old.workload(env);before=env.uav_positions.copy()
        details={}
        if policy:
            action=np.concatenate((old.select_destinations(env,policy),np.zeros(2,dtype=np.int64)))
        elif training:
            cfg=agent['cfg'];warm=agent['slots']<cfg['warmup']
            eps=1. if warm else 1-.95*min(agent['slots']/cfg['epsilon_decay'],1.)
            explore=bool(warm or agent['action_rng'].random()<eps)
            if explore:action=agent['action_rng'].integers(0,5,32,dtype=np.int64)
            else:
                with torch.inference_mode():action=greedy(agent['online'],torch.from_numpy(state)[None],agent['arm'])[0][0].numpy().copy()
            agent['actor_history'].append(dict(warmup=warm,epsilon=eps,explore=explore))
        else:
            with torch.inference_mode():
                tensor=torch.from_numpy(state)[None]
                at,q,details=greedy(agent['online'],tensor,agent['arm'])
                tq=chosen(agent['target'],tensor,at,agent['arm'])
                action=at[0].numpy().copy();q_values.append(float(q[0]));target_values.append(float(tq[0]))
                details={k:v[0].cpu().tolist() for k,v in details.items()}
        ns,reward,done,info=env.step(to_action_dict(action,30));next_states.append(ns.copy())
        if not np.isfinite(reward) or not np.isfinite(ns).all() or done!=(t==39):raise AssertionError('invalid transition')
        if training:
            agent['replay'].add(state,action,reward,ns,done);agent['slots']+=1
            item=update(agent)
            if item is not None:logs.append(item)
        after=env.uav_positions.copy()
        slots.append(dict(slot=t,action=action.tolist(),reward=reward,done=done,reward_components=info['reward_components'],
            energy_j=info['energy_j'],tasks_arrived=work,uav_before=before.tolist(),uav_after=after.tolist(),
            actual_distance_m=np.linalg.norm(after-before,axis=1).tolist(),decoder=details))
    summary=env.episode_summary();rewards=[s['reward'] for s in slots]
    result=dict(seed=seed,episode=episode_id,run_kind='SYNTHETIC_NOT_RESEARCH',return_sum=sum(rewards),
        deadline_success=summary['deadline_success_rate'],tasks_generated=summary['tasks_generated'],
        tasks_on_time=summary['tasks_on_time'],slot_rewards=rewards,terminal_value=q_values[-1] if q_values else None,
        workload_sha256=old.digest([s['tasks_arrived'] for s in slots]),summary=summary,slots=slots,tasks=env.export_task_records(),
        chosen_q=q_values,target_q=target_values,realized_return_to_go=np.cumsum(rewards[::-1])[::-1].tolist(),
        observations=np.stack(states),next_observations=np.stack(next_states))
    return result,logs

def save_episode(path,row):
    row=dict(row)
    np.savez_compressed(Path(path).with_suffix('').with_suffix('.npz'),states=row.pop('observations'),next_states=row.pop('next_observations'))
    old.save_episode(path,row)

def values_summary(rows):
    q=np.asarray([v for r in rows for v in r['chosen_q']],dtype=np.float64)
    target=np.asarray([v for r in rows for v in r['target_q']],dtype=np.float64)
    g=np.asarray([v for r in rows for v in r['realized_return_to_go']],dtype=np.float64)
    if not (np.isfinite(q).all() and np.isfinite(target).all() and np.isfinite(g).all()):raise FloatingPointError('evaluation values nonfinite')
    residual=q-g
    return dict(chosen_q=old.distribution(torch.tensor(q)),target_q=old.distribution(torch.tensor(target)),
        return_residual=old.distribution(torch.tensor(residual)),normalized_rmse=float(np.sqrt(np.mean(residual**2))/max(1,np.sqrt(np.mean(g**2)))),
        absolute_bias=float(abs(np.mean(residual))),near_zero_large_return_fraction=float(((abs(q)<=1e-6)&(abs(g)>=1)).mean()),
        terminal_abs_error_p95=float(np.quantile([abs(r['chosen_q'][-1]-r['slot_rewards'][-1]) for r in rows],.95)))

def comparisons(fresh,heuristics):
    primary=fresh['shared_congestion_cost'];result={}
    for key,rows in {**fresh,**heuristics}.items():
        if key=='shared_congestion_cost':continue
        if [(r['seed'],r['workload_sha256']) for r in rows]!=[(r['seed'],r['workload_sha256']) for r in primary]:raise AssertionError('unpaired comparison')
        result[key]=dict(return_mean_difference=float(np.mean([a['return_sum']-b['return_sum'] for a,b in zip(primary,rows)])),
            deadline_mean_difference=float(np.mean([a['deadline_success']-b['deadline_success'] for a,b in zip(primary,rows)])))
    return result

def fixture_cfg():
    return dict(episodes=1,warmup=8,batch=4,capacity=64,learning_rate=1e-4,gamma=1.,clip=10.,target_interval=8,epsilon_decay=40)

def checkpoint(agent):
    return dict(arm=agent['arm'],model_seed=agent['seed'],online=agent['online'].state_dict(),target=agent['target'].state_dict(),
        optimizer=agent['opt'].state_dict(),slots=agent['slots'],updates=agent['updates'],copies=agent['copies'],
        actor_history=agent['actor_history'],cfg=agent['cfg'],action_rng=agent['action_rng'].bit_generator.state,
        replay_rng=agent['replay_rng'].bit_generator.state,torch_rng=torch.get_rng_state(),replay_position=agent['replay'].position,
        replay={k:getattr(agent['replay'],k)[:agent['replay'].size].copy() for k in ['states','next_states','actions','rewards','dones']})

def run_fixture(output):
    """Exercise every arm's fit/freeze/eval/report path on one NONRESEARCH plan."""
    configure();s=spec();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    runtime=dict(run_kind='SYNTHETIC_NOT_RESEARCH',model_seed=s['fixture']['model_seed'],
        train_seeds=s['fixture']['train'],evaluation_seeds=s['fixture']['evaluation'],cfg=fixture_cfg(),
        source_sha256={name:old.sha(ROOT/name) for name in SOURCES},design_sha256=SPEC_SHA,
        fixture_evaluation_reused_for_validation=True,research_launch=False)
    write(out/'runtime_plan.json',runtime)
    for name in SOURCES:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    events=[];agents={};invariance=[];fresh={};heuristics={};counts={}
    def event(phase,**extra):
        events.append(dict(sequence=len(events),phase=phase,**extra));write(out/'events.json',events)
    try:
        for arm in ARMS:
            a=build(arm,runtime['model_seed'],runtime['cfg']);agents[arm]=a;folder=out/arm;folder.mkdir()
            for stage in [0,1]:
                if stage==1:
                    event('synthetic_training',arm=arm)
                    row,logs=episode(a,runtime['train_seeds'][0],True,episode_id=1)
                    save_episode(folder/'train_ep1.json.gz',row)
                    (folder/'optimizer.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in logs),encoding='utf-8')
                    counts[arm]=dict(slots=a['slots'],updates=a['updates'],target_copies=a['copies'])
                    assert counts[arm]==dict(slots=40,updates=33,target_copies=4)
                before=old.learner_snapshot(a)
                row,_=episode(a,runtime['evaluation_seeds'][0],episode_id=f'validation_{stage}')
                assert old.learner_snapshot(a)==before
                invariance.append(dict(arm=arm,phase=f'validation_{stage}',before=before,after=before))
                save_episode(folder/f'validation_{stage}.json.gz',row)
            torch.save(checkpoint(a),folder/'final.pt')
        assert len({old.digest(a['actor_history']) for a in agents.values()})==1
        assert len({old.digest(a['action_rng'].bit_generator.state) for a in agents.values()})==1
        frozen={arm:old.sha(out/arm/'final.pt') for arm in ARMS}
        write(out/'FRESH_GATE_FREEZE.json',frozen);event('all_checkpoints_frozen',checkpoint_count=len(frozen))
        for policy in ['all_local','nearest_rsu','uav_selective']:
            row,_=episode(None,runtime['evaluation_seeds'][0],policy=policy,episode_id='fixture_evaluation')
            save_episode(out/f'heuristic_{policy}.json.gz',row);heuristics[policy]=[row]
        for arm,a in agents.items():
            event('synthetic_post_freeze_evaluation',arm=arm)
            before=old.learner_snapshot(a);row,_=episode(a,runtime['evaluation_seeds'][0],episode_id='fixture_evaluation')
            after=old.learner_snapshot(a);assert before==after
            invariance.append(dict(arm=arm,phase='post_freeze',before=before,after=after))
            save_episode(out/arm/'evaluation.json.gz',row);fresh[arm]=[row]
        analysis=dict(run_kind='SYNTHETIC_NOT_RESEARCH',comparisons=comparisons(fresh,heuristics),
            value_stability={arm:values_summary(rows) for arm,rows in fresh.items()},quality_claim=False)
        write(out/'comparisons_and_values.json',analysis);write(out/'evaluation_invariance.json',invariance)
        for arm in ARMS:assert old.sha(out/arm/'final.pt')==frozen[arm]
        event('complete',training_slots=320,optimizer_updates=264)
        write(out/'verification.json',dict(state='completed_synthetic_fixture_not_research',research_launch=False,
            counts=counts,training_slots=320,optimizer_updates=264,checkpoints=frozen,quality_claim=False))
        files={str(p.relative_to(out)).replace('\\','/'):old.sha(p) for p in out.rglob('*') if p.is_file()}
        write(out/'run_manifest.json',dict(files=files))
        return out
    except Exception:
        write(out/'FAILED.json',dict(error=traceback.format_exc(),run_kind='SYNTHETIC_NOT_RESEARCH'))
        raise

def hardware():
    return dict(platform=platform.platform(),processor=platform.processor(),python=sys.version,
        torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        backend='PyTorch eager CPU FP32, fixed for all arms before measurement',cuda_used=False)

def select_complete(model,arm,state):
    x=np.asarray(state,dtype=np.float32)
    with torch.inference_mode():
        action=greedy(model,torch.from_numpy(x)[None],arm)[0][0].numpy().copy()
    return action

def cost_screens(metrics,s):
    p=metrics['shared_congestion_cost'];small=metrics['additive64_nonpositive'];wide=metrics['additive128_unbounded'];c=s['cost_gate']
    return dict(online_storage=p['parameter_bytes']/small['parameter_bytes']<=c['online_bytes_ratio_vs_additive64_at_most'],
        complete_action_median=p['median_ms']/wide['median_ms']<=c['complete_action_median_ratio_vs_additive128_at_most'],
        complete_action_p95=p['p95_ms']<=c['complete_action_p95_ms_at_most'])

def benchmark(output):
    configure();s=spec();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    # Lock the backend/configuration and source hashes BEFORE any timing.
    write(out/'PRE_TIMING_FREEZE.json',dict(design_sha256=SPEC_SHA,hardware=hardware(),model_seed=s['fixture']['model_seed'],
        source_sha256={n:old.sha(ROOT/n) for n in SOURCES},rounds=5,warmups=32,calls=128,
        seed_role='declared_timing_probes_only_not_training_or_quality_gate',probe_seeds=s['development']['probes']))
    states=[]
    for i,seed in enumerate(s['development']['probes']):
        row,_=episode(None,seed,policy=['all_local','nearest_rsu','uav_selective'][i%3],episode_id='cost_probe')
        row['run_kind']='ENGINEERING_COST_PROBE_NOT_POLICY_EVIDENCE'
        states.extend(row['observations']);save_episode(out/f'probe_{seed}.json.gz',row)
    common=np.stack(states)[np.linspace(0,len(states)-1,128,dtype=int)]
    np.savez_compressed(out/'common_observations.npz',observations=common)
    models={arm:build(arm,s['fixture']['model_seed'],fixture_cfg())['online'].eval() for arm in ARMS}
    before={arm:old.digest(m.state_dict()) for arm,m in models.items()}
    order_rng=np.random.default_rng(s['rng']['timing_order']);timing=[];order=[]
    for round_ in range(5):
        seq=list(ARMS);order_rng.shuffle(seq);order.append(seq)
        for arm in seq:
            model=models[arm]
            for i in range(32):select_complete(model,arm,common[i])
            for i,state in enumerate(common):
                start=time.perf_counter_ns();action=select_complete(model,arm,state);elapsed=time.perf_counter_ns()-start
                if action.shape!=(32,) or not np.all((0<=action)&(action<5)):raise AssertionError('bad benchmark action')
                timing.append(dict(round=round_,arm=arm,observation=i,nanoseconds=elapsed))
    metrics={}
    for arm,m in models.items():
        raw=np.array([x['nanoseconds'] for x in timing if x['arm']==arm],dtype=np.float64)/1e6
        assert old.digest(m.state_dict())==before[arm]
        metrics[arm]=dict(parameters=sum(p.numel() for p in m.parameters()),parameter_bytes=parameter_bytes(m),
            buffer_bytes=sum(b.numel()*b.element_size() for b in m.buffers()),median_ms=float(np.median(raw)),
            p95_ms=float(np.quantile(raw,.95)),calls=len(raw))
    screens=cost_screens(metrics,s)
    result=dict(state='cost_gate_passed' if all(screens.values()) else 'cost_gate_failed',screens=screens,metrics=metrics,
        hardware=hardware(),research_training_started=False,training_slots=0,optimizer_updates=0,
        probe_slots=8*40,probe_seeds_consumed=s['development']['probes'],all_development_training_and_gate_seeds_unconsumed=True,
        all_final_seeds_unconsumed=True,quality_evidence=False,order=order)
    write(out/'raw_timings.json',timing);write(out/'verification.json',result)
    write(out/'run_manifest.json',dict(files={str(p.relative_to(out)).replace('\\','/'):old.sha(p) for p in out.rglob('*') if p.is_file()}))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['fixture','cost','development','final'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--confirm-development-run',action='store_true')
    parser.add_argument('--confirm-final-run',action='store_true')
    a=parser.parse_args()
    if a.mode in ('development','final'):
        if not getattr(a,'confirm_'+a.mode+'_run'):parser.error('--confirm-'+a.mode+'-run is required; research remains disabled in this engineering gate')
        parser.error('Research launch is disabled in this engineering-only harness, including with confirmation')
    if a.mode=='fixture':
        out=run_fixture(a.output)
        from audit_controller_redesign_engineering import audit_fixture
        print(json.dumps(audit_fixture(out),indent=2))
    else:print(json.dumps(benchmark(a.output),indent=2))

if __name__=='__main__':main()
