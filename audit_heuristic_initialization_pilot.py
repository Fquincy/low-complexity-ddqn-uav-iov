"""Read-only saved artifact audit; no simulation or optimizer steps.

Checks recorded update arithmetic, not an independent replay of every historical
gradient/weight update. Recomputes final greedy actions/values from checkpoints.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import heuristic_initialized_ddqn_baseline as baseline


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def close(a,b):return np.allclose(a,b,rtol=1e-5,atol=1e-6)


def audit(run_dir):
    root=Path(run_dir);manifest=read(root/'manifest.json')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p!=root/'manifest.json'}
    require(actual==set(manifest),'unlisted or missing artifact')
    for name,expected in manifest.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'unsafe manifest path')
        require(sha(root/name)==expected,'artifact hash mismatch: '+name)
    freeze=read(root/'PRE_RUN_FREEZE.json');plan=read(root/'runtime_plan.json')
    require(sha(root/'runtime_plan.json')==freeze['plan_sha256'],'plan freeze')
    spec_name='HEURISTIC_INITIALIZED_DDQN_PAIRED_PILOT_PROTOCOL_2026-11-19.json'
    spec=read(root/'source_snapshot'/spec_name)
    require(freeze['protocol_sha256']=='a73994fdfea9eda763232fa4c86d52e6a1317f421c40fd6f25ddb0f78c1745fc','unrecognized protocol')
    require(sha(root/'source_snapshot'/spec_name)==freeze['protocol_sha256'],'protocol freeze')
    for name,expected in freeze['source_sha256'].items():require(sha(root/'source_snapshot'/name)==expected,'source snapshot')
    for name,expected in spec['definition_sha256'].items():require(freeze['source_sha256'][name]==expected,'source binding')
    # Audit numeric predictions with current verified sources, not an unbound model.
    for name in ['heuristic_initialized_ddqn_baseline.py','controller_v3_amortized.py']:
        require(sha(Path(__file__).parent/name)==freeze['source_sha256'][name],'audit model source changed')
    require(plan['kind']=='ENGINEERING_FIXTURE_NOT_RESEARCH' if plan['mode']=='fixture' else plan['kind']=='EXPLORATORY_PAIRED_DEVELOPMENT','run kind')
    if plan['mode']=='fixture':
        require({k:plan[k] for k in ['fits','demo_episodes','online_episodes','eval_episodes','bc_steps','batch','warmup','replay_capacity','target_interval','timing_states','timing_rounds','timing_warmups']}==
                dict(fits=2,demo_episodes=1,online_episodes=1,eval_episodes=1,bc_steps=2,batch=4,warmup=8,replay_capacity=128,target_interval=8,timing_states=4,timing_rounds=1,timing_warmups=1),'fixture budgets')
        require(set(plan['seeds']['model_seeds']).isdisjoint(spec['seeds']['model_seeds']),'fixture uses pilot model seed')
        for name in ['demonstration_seed_blocks','online_seed_blocks']:
            for block in plan['seeds'][name]:
                for reserved in spec['seeds'][name]:
                    require(max(block['start'],reserved['start'])>min(block['start']+block['count']-1,reserved['start']+reserved['count']-1),'fixture/pilot seed overlap')
    else:
        require(plan['mode']=='pilot' and plan['seeds']==spec['seeds'],'pilot seed binding')
        require({k:plan[k] for k in ['fits','demo_episodes','online_episodes','eval_episodes','bc_steps','batch','warmup','replay_capacity','target_interval','timing_states','timing_rounds','timing_warmups']}==
                dict(fits=3,demo_episodes=10,online_episodes=50,eval_episodes=16,bc_steps=200,batch=64,warmup=128,replay_capacity=4096,target_interval=200,timing_states=128,timing_rounds=3,timing_warmups=32),'pilot budgets')
    events=read(root/'events.json');stages=[e['stage'] for e in events]
    require([e['sequence'] for e in events]==list(range(len(events))),'event sequence')
    require(stages.count('checkpoints_frozen')==1 and stages.index('checkpoints_frozen')<stages.index('evaluation'),'evaluation before checkpoint freeze')
    require(stages[-1]=='artifacts_complete','incomplete workflow')
    demo_freeze=read(root/'DEMONSTRATION_FREEZE.json')
    for name,expected in demo_freeze['files'].items():require(sha(root/name)==expected,'demo freeze')
    checkpoints=read(root/'CHECKPOINT_FREEZE.json')['checkpoints']
    require(len(checkpoints)==2*plan['fits'],'checkpoint count')
    require({p.relative_to(root).as_posix() for p in root.rglob('*.pt')}=={f'{arm}/fit_{fit}/final.pt' for arm in spec['arms'] for fit in range(plan['fits'])},'non-final/missing checkpoint')
    episode_cache={};sim_slots=0
    for p in sorted(root.rglob('*.npz')):
        if p.name=='final_replay.npz':continue
        a=dict(np.load(p,allow_pickle=False));row=read(p.with_suffix('.json'))
        require(len(a['states'])==40 and row['steps']==40,'episode length')
        require(a['states'].shape==(40,533) and a['actions'].shape==(40,32),'episode shapes')
        require(a['states'].dtype==np.float32 and a['actions'].dtype==np.int64,'episode dtypes')
        require(all(np.isfinite(v).all() for v in a.values()),'nonfinite episode')
        require(np.array_equal(a['dones'],np.arange(40)==39),'terminal positions')
        require(close(a['states'][1:],a['next_states'][:-1]),'state continuity')
        require(close(a['states'][:,-1],np.arange(40)/40) and a['next_states'][-1,-1]==1,'horizon')
        require(np.all((a['actions']>=0)&(a['actions']<5)),'action bounds')
        require(np.take_along_axis(baseline.allowed(a['states']),a['actions'][:,:,None],2).all(),'screened action')
        require(close(a['rewards'].sum(dtype=np.float64),row['return_value']),'episode return')
        records=row['records'];summary=row['summary']
        components=row['step_info'];require(len(components)==40,'step accounting count')
        cfg=spec['simulator_defaults']
        expected_rewards=[-cfg['delay_reward_weight']*i['reward_components']['normalized_delay_accrual']
                          -cfg['uav_energy_reward_weight']*i['reward_components']['normalized_uav_energy']
                          -cfg['deadline_miss_penalty']*i['reward_components']['new_deadline_misses_per_vehicle'] for i in components]
        require(close(a['rewards'],expected_rewards),'reward accounting')
        require(len(records)==1200 and summary['tasks_generated']==1200,'task generation')
        on_time=sum(bool(r['on_time']) for r in records)
        completed=sum(r['status']=='completed' for r in records)
        require(on_time==summary['tasks_on_time'] and completed==summary['tasks_completed'],'task summary')
        require(1200-completed==summary['tasks_unfinished'] and close(on_time/1200,summary['deadline_success_rate']),'unfinished/deadline accounting')
        episode_cache[p.relative_to(root).as_posix()]=(a,row);sim_slots+=40
    def episodes(folder):
        return [episode_cache[p.relative_to(root).as_posix()] for p in sorted(folder.glob('*.npz')) if p.name!='final_replay.npz']
    def rows(eps):
        return [(a['states'][i],a['actions'][i],a['rewards'][i],a['next_states'][i],a['dones'][i]) for a,r in eps for i in range(40)]
    update_count=0;bc_count=0;value_metrics={};policies={};storage={}
    teacher_eps=episodes(root/'evaluation'/'teacher')
    require(len(teacher_eps)==plan['eval_episodes'],'teacher evaluation count')
    exogenous_fields=['task_id','vehicle_id','arrival_s','input_bits','required_cycles','deadline_s']
    def workload(record):
        return [[task[k] for k in exogenous_fields] for task in record['records']]
    teacher_mean={m:float(np.mean([r['return_value'] if m=='return_value' else r['summary']['deadline_success_rate'] for a,r in teacher_eps])) for m in ('return_value','deadline')}
    for fit in range(plan['fits']):
        torch.manual_seed(plan['seeds']['model_seeds'][fit]);initial=baseline.TeacherInitializedCompactDDQN()
        digest=hashlib.sha256()
        for name,value in initial.state_dict().items():digest.update(name.encode());digest.update(value.detach().numpy().tobytes())
        require(digest.hexdigest()==demo_freeze['initial_tensor_hashes'][str(fit)],'seeded initialization')
        demos=episodes(root/f'demo_{fit}');require(len(demos)==plan['demo_episodes'],'demo count')
        for ep,(a,r) in enumerate(demos):require(r['seed']==plan['seeds']['demonstration_seed_blocks'][fit]['start']+ep,'demo seed')
        preload=rows(demos);preload_hash=hashlib.sha256(np.stack([r[0] for r in preload]).tobytes()+np.stack([r[1] for r in preload]).tobytes()).hexdigest()
        for arm in spec['arms']:
            folder=root/arm/f'fit_{fit}';count=read(folder/'counts.json');key=f'{arm}/{fit}'
            require(count['initial_tensor_hash']==demo_freeze['initial_tensor_hashes'][str(fit)],'initial hash pair')
            require(count['preloaded_demo_hash']==preload_hash,'replay parity')
            require(sha(folder/'final.pt')==checkpoints[key],'checkpoint freeze')
            ck=torch.load(folder/'final.pt',weights_only=True,map_location='cpu')
            require(ck['counts']==count,'checkpoint counts')
            model=baseline.TeacherInitializedCompactDDQN().eval();model.load_state_dict(ck['online'])
            require(baseline.parameter_bytes(model)==count['parameter_bytes']==67364 and count['buffer_bytes']==20,'storage')
            bc=read(folder/'bc.json');want=plan['bc_steps'] if arm=='teacher_bc_then_ddqn' else 0
            require(len(bc)==count['bc_updates']==want,'BC count')
            rng=np.random.default_rng(plan['seeds']['bc_sampler_seeds'][fit])
            for k,item in enumerate(bc):
                require(item['step']==k and item['indices']==rng.choice(len(preload),plan['batch'],replace=False).tolist(),'BC draw')
                require(np.isfinite(item['loss']) and np.isfinite(item['gradient_norm']),'BC finite')
            bc_count+=len(bc)
            train_eps=episodes(folder)
            require(len(train_eps)==plan['online_episodes'],'online episode count')
            for ep,(a,r) in enumerate(train_eps):require(r['seed']==plan['seeds']['online_seed_blocks'][fit]['start']+ep,'online seed')
            if arm==spec['arms'][1]:
                paired=episodes(root/spec['arms'][0]/f'fit_{fit}')
                require(all(workload(r)==workload(other) for (_,r),(_,other) in zip(train_eps,paired)),'paired training workload mismatch')
            trainrows=rows(train_eps);log=read(folder/'optimizer.json');draws=read(folder/'action_draws.json')
            n=len(trainrows);require(count['slots']==n and len(draws)==n,'slot/draw count')
            require(len(log)==count['updates']==n-plan['warmup'],'update count')
            require(count['target_copies']==len(log)//plan['target_interval'],'target copy count')
            require(all(float(v['step'])==len(log) for v in ck['optimizer']['state'].values()),'Adam update count/reset')
            sampling=np.random.default_rng(plan['seeds']['replay_sampler_seeds'][fit]);choices=np.random.default_rng(plan['seeds']['epsilon_branch_seeds'][fit])
            replay=list(preload);ids=[f'demo:{i}' for i in range(len(preload))]
            for t,row in enumerate(trainrows):
                draw=draws[t];u=float(choices.random());v=choices.random(32)
                eps=1. if t<plan['warmup'] else 1-.95*min(t/n,1)
                require(draw['slot']==t and draw['epsilon_uniform']==u and draw['branch_uniforms']==v.tolist() and draw['epsilon']==eps,'exploration RNG')
                require(draw['action']==row[1].tolist(),'action draw binding')
                if u<eps:
                    mask=baseline.allowed(row[0][None])[0]
                    aa=[int(np.flatnonzero(mask[i])[int(v[i]*mask[i].sum())]) for i in range(32)]
                    require(aa==draw['action'],'exploration action')
                replay.append(row);ids.append(f'online:{t}')
                if len(replay)>plan['replay_capacity']:replay.pop(0);ids.pop(0)
                if t<plan['warmup']:continue
                item=log[t-plan['warmup']];idx=sampling.integers(0,len(replay),plan['batch'])
                require(item['online_slot']==t and item['update']==t-plan['warmup']+1,'update position')
                require(item['indices']==idx.tolist() and item['transition_ids']==[ids[i] for i in idx],'replay draw')
                rr=np.array([replay[i][2] for i in idx],np.float32);dd=np.array([replay[i][4] for i in idx])
                require(close(item['rewards'],rr) and np.array_equal(item['dones'],dd),'batch binding')
                pred=np.array(item['prediction']);tq=np.array(item['target_q']);yy=np.array(item['target'])
                require(np.isfinite(pred).all() and np.isfinite(tq).all() and np.isfinite(item['gradient_norm']),'finite values')
                require(np.all(pred<=0) and np.all(tq<=0) and np.all(tq[dd]==0),'sign/terminal target')
                require(close(yy,rr+tq),'Bellman target arithmetic')
                error=np.abs(pred-yy);require(close(item['loss'],np.where(error<1,.5*error**2,error-.5).mean()),'Huber loss')
                aa=np.asarray(item['next_actions']);require(np.all(aa[dd]==-1) and np.all((aa[~dd]>=0)&(aa[~dd]<5)),'next selection/terminal bypass')
                if (~dd).any():
                    next_s=np.stack([replay[i][3] for i in idx])[~dd]
                    require(np.take_along_axis(baseline.allowed(next_s),aa[~dd,:,None],2).all(),'target next-state screen')
                require(item['target_copy']==(item['update']%plan['target_interval']==0),'copy schedule')
            update_count+=len(log)
            saved=dict(np.load(folder/'final_replay.npz'));saved_rng=read(folder/'final_rng.json')
            for j,name in enumerate(['states','actions','rewards','next_states','dones']):require(close(saved[name],np.stack([r[j] for r in replay])),'final replay content')
            require(saved_rng['transition_ids']==ids and saved_rng['choices']==choices.bit_generator.state and saved_rng['replay']==sampling.bit_generator.state and saved_rng['bc']==rng.bit_generator.state,'final RNG states')
            ee=episodes(root/'evaluation'/key);require(len(ee)==plan['eval_episodes'],'student evaluation count')
            errors=[];terminal=[];qall=[];gall=[];horizon_errors=[[] for _ in range(4)]
            for ep,(a,r) in enumerate(ee):
                require(r['seed']==plan['seeds']['evaluation_seed_block']['start']+ep,'evaluation seed')
                require(workload(r)==workload(teacher_eps[ep][1]),'paired evaluation workload mismatch')
                q=[]
                for i,state in enumerate(a['states']):
                    with torch.no_grad():aa,vv,_=model.greedy(torch.from_numpy(state)[None])
                    require(np.array_equal(aa.numpy()[0],a['actions'][i]),'saved policy mismatch')
                    q.append(float(vv[0]))
                require(close(q,a['q']),'saved value mismatch')
                g=np.cumsum(a['rewards'][::-1],dtype=np.float64)[::-1]
                error=np.array(q)-g;errors.extend(error);qall.extend(q);gall.extend(g);terminal.append(abs(error[-1]))
                for j in range(4):horizon_errors[j].extend(error[j*10:(j+1)*10])
            policies[key]={m:float(np.mean([r['return_value'] if m=='return_value' else r['summary']['deadline_success_rate'] for a,r in ee])) for m in ('return_value','deadline')}
            value_metrics[key]=dict(bias=float(np.mean(errors)),rmse=float(np.sqrt(np.mean(np.square(errors)))),
                nrmse=float(np.sqrt(np.mean(np.square(errors)))/max(np.sqrt(np.mean(np.square(gall))),1e-12)),
                terminal_p95=float(np.quantile(terminal,.95)),error_quantiles=np.quantile(errors,[0,.25,.5,.75,1]).tolist(),
                error_by_slot_decile=[dict(mean=float(np.mean(v)),quantiles=np.quantile(v,[0,.5,1]).tolist()) for v in horizon_errors],
                clip_fraction=float(np.mean([v['gradient_norm']>10 for v in log])))
            storage[key]=dict(parameter_bytes=67364,buffer_bytes=20,checkpoint_bytes=(folder/'final.pt').stat().st_size)
    analysis=read(root/'analysis.json');diffs=[]
    for fit in range(plan['fits']):
        bc=policies[f'teacher_bc_then_ddqn/{fit}'];control=policies[f'common_demo_no_bc/{fit}']
        diff={m:bc[m]-control[m] for m in bc};diffs.append(diff)
        expected=dict(bc=bc,control=control,teacher=teacher_mean,difference=diff,
            bc_minus_teacher={m:bc[m]-teacher_mean[m] for m in bc},control_minus_teacher={m:control[m]-teacher_mean[m] for m in bc})
        require(analysis['per_fit'][str(fit)]==expected,'paired comparison recomputation')
    promising=all(v[m]>=0 for v in diffs for m in ('return_value','deadline')) and all(np.mean([v[m] for v in diffs])>0 for m in ('return_value','deadline'))
    require(analysis['promising']==bool(promising) and analysis['quality_claim'] is False,'decision logic')
    states=np.load(root/'timing_states.npy');tf=read(root/'TIMING_FREEZE.json');timing=read(root/'timing.json')
    require(sha(root/'timing_states.npy')==tf['states_sha256'] and len(states)==plan['timing_states'],'timing freeze')
    expected_keys={(r,k,i) for r in range(plan['timing_rounds']) for k in list(policies)+['teacher'] for i in range(len(states))}
    require(len(timing)==len(expected_keys) and {(r['round'],r['policy'],r['state']) for r in timing}==expected_keys,'timing inventory')
    require([(timing[i]['round'],timing[i]['policy']) for i in range(0,len(timing),len(states))]==[tuple(k) for k in tf['order']],'timing order')
    timing_stats={}
    for key in list(policies)+['teacher']:
        vals=[r['ns'] for r in timing if r['policy']==key];require(all(v>0 for v in vals),'timing values')
        timing_stats[key]=dict(median_ns=float(np.median(vals)),p95_ns=float(np.quantile(vals,.95)))
    # Verify timed student actions against their final checkpoints; no teacher in this path.
    for key in policies:
        arm,fit=key.split('/');model=baseline.TeacherInitializedCompactDDQN().eval()
        ck=torch.load(root/arm/f'fit_{fit}'/'final.pt',weights_only=True);model.load_state_dict(ck['online'])
        for item in (v for v in timing if v['policy']==key):
            require(np.array_equal(baseline.deploy_actions(model,states[item['state']]),item['action']),'timed student policy')
    require(sim_slots==plan['fits']*plan['demo_episodes']*40+plan['fits']*2*plan['online_episodes']*40+(plan['fits']*2+1)*plan['eval_episodes']*40,'total slot count')
    return dict(passed=True,kind=plan['kind'],files_checked=len(manifest),simulation_slots_recorded=sim_slots,
                historical_ddqn_updates=update_count,historical_bc_updates=bc_count,value_distributions=value_metrics,
                storage=storage,timing=timing_stats,recorded_promising=bool(promising),
                new_simulator_steps=0,new_optimizer_updates=0,
                limitation='No replay of historical gradients/weights; recorded arithmetic and final policy recomputed',
                research_launch_authorized=False)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('run_dir');args=p.parse_args()
    print(json.dumps(audit(args.run_dir),indent=2))
