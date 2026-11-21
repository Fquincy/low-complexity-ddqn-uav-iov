"""Independent saved-artifact audit. No runner/helper fit/loss/decision calls."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

import return_calibrated_handoff as model_source
import heuristic_initialized_ddqn_baseline as baseline
from freeze_heuristic_initialization_pilot_plan_2026_11_19 import ranges

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'Source'))
from chapter3_iov_env_v2 import Chapter3Config
from heuristic_pilot_teacher_adapter import build_teacher
SPEC_NAME='RETURN_CALIBRATED_HANDOFF_COMPARISON_PROTOCOL_2026-11-20.json'
SPEC_SHA='1602eda36acad77475f0c734fde10448daa2f1b96243c00d3f6f257870a7bee1'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def close(a,b):return np.allclose(a,b,rtol=1e-5,atol=1e-6)
def tensor_hash(model):
    digest=hashlib.sha256()
    for name,v in model.state_dict().items():digest.update(name.encode());digest.update(v.detach().numpy().tobytes())
    return digest.hexdigest()
def returns(r):return np.cumsum(np.asarray(r,dtype=np.float64)[::-1])[::-1].copy()


def verify_probe(model,x,a,record,temperature,check_hash=True):
    with torch.no_grad():
        chosen,q,_=model.greedy(x);dq=model(x,a);p,m,v,_=model.tables(x)
        scores=torch.cat((p,m),1);mask=torch.from_numpy(baseline.allowed(x))
        rank=scores.masked_fill(~mask,torch.inf).sort(2).values
        ce=F.cross_entropy((-p/temperature).masked_fill(~mask[:,:30],-torch.inf).reshape(-1,5),a[:,:30].reshape(-1))
        ce+=F.cross_entropy((-m/temperature).reshape(-1,5),a[:,30:].reshape(-1))
    require(np.array_equal(chosen.numpy(),record['actions']),'probe actions')
    for name,actual in dict(greedy_q=q.numpy(),demonstration_q=dq.numpy(),value=v.numpy(),
            all_action_ranges=(scores.max(2).values-scores.min(2).values).sum(1).numpy(),
            margins=(rank[:,:,1]-rank[:,:,0]).numpy()).items():
        require(np.allclose(actual,record[name],rtol=2e-5,atol=2e-5),'probe '+name)
    require(close(float(ce),record['imitation_loss']),'probe loss')
    if check_hash:require(record['tensor_hash']==tensor_hash(model),'probe tensor hash')


def audit(run_dir):
    root=Path(run_dir);torch.set_num_threads(1)
    manifest=read(root/'manifest.json')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p!=root/'manifest.json'}
    require(actual==set(manifest),'unlisted/missing artifact')
    for name,digest in manifest.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'unsafe artifact path')
        require(sha(root/name)==digest,'artifact hash: '+name)
    freeze=read(root/'PRE_RUN_FREEZE.json');pl=read(root/'runtime_plan.json')
    spec=read(root/'source_snapshot'/SPEC_NAME)
    require(freeze['protocol_sha256']==SPEC_SHA==sha(root/'source_snapshot'/SPEC_NAME),'protocol binding')
    require(sha(root/'runtime_plan.json')==freeze['plan_sha256'],'runtime plan binding')
    require(freeze['environment']==spec['numerical_environment'],'numerical environment declaration')
    for name,digest in freeze['source_sha256'].items():require(sha(root/'source_snapshot'/name)==digest,'snapshot hash')
    for name,digest in spec['definition_sha256'].items():require(freeze['source_sha256'][name]==digest,'definition binding')
    for name in ['return_calibrated_handoff.py','centered_linear_comparison_models.py','centered_linear_ddqn_prototype.py','heuristic_initialized_ddqn_baseline.py','controller_v3_amortized.py','heuristic_pilot_teacher_adapter.py','assignment_price_load_radio_heuristic.py','assignment_price_fast_runtime.py']:
        require(sha(ROOT/name)==freeze['source_sha256'][name],'current model source binding')
    # Independently fixed fixture/development budget contracts.
    budget_keys=['fits','demo_episodes','online_episodes','eval_episodes','bc_steps','batch','warmup','replay_capacity','target_interval','timing_states','timing_rounds','timing_warmups']
    if pl['mode']=='fixture':
        expected=[2,10,1,1,2,4,8,416,8,4,1,1]
        require(pl['kind']=='ENGINEERING_FIXTURE_NOT_RESEARCH','fixture labeling')
        expected_seeds=dict(model_seeds=[982800001,982800011],
            demonstration_seed_blocks=[dict(start=982810000+100*i,count=10) for i in range(2)],
            online_seed_blocks=[dict(start=982820000+100*i,count=1) for i in range(2)],
            evaluation_seed_block=dict(start=982830000,count=1),bc_sampler_seeds=[982840001,982840011],
            epsilon_branch_seeds=[982850001,982850011],replay_sampler_seeds=[982860001,982860011],
            retention_sampler_seeds=[982870001,982870011],job_order_seed=982880001,timing_order_seed=982880011)
        require(pl['seeds']==expected_seeds,'fixture seed binding')
        for a,b,label in ranges(pl['seeds']):
            for c,d,old in ranges(spec['seeds']):require(max(a,c)>min(b,d),'fixture/research seed overlap')
    else:
        expected=[3,10,50,16,200,64,128,4096,200,128,3,32]
        require(pl['mode']=='development' and pl['kind']=='EXPLORATORY_PAIRED_DEVELOPMENT' and pl['seeds']==spec['seeds'],'development binding')
    require([pl[k] for k in budget_keys]==expected,'budget override')
    arms=[a['id'] for a in spec['arms']];defs={a['id']:a for a in spec['arms']};A,B,C,D,E,Farm,G=arms
    expected_files={'runtime_plan.json','PRE_RUN_FREEZE.json','DEMONSTRATION_FREEZE.json','job_order.json',
        'CHECKPOINT_FREEZE.json','events.json','analysis.json','timing_states.npy','TIMING_FREEZE.json','timing.json'}
    expected_files.update('source_snapshot/'+name for name in freeze['source_sha256'])
    for fit in range(pl['fits']):
        expected_files.update(f'demo_{fit}/episode_{ep:03d}.{suffix}' for ep in range(10) for suffix in ['npz','json'])
        for arm in arms:
            base=f'{arm}/fit_{fit}/'
            expected_files.update(base+name for name in ['bc.json','calibration.json','probes.json','counts.json','optimizer.json','action_draws.json','final_rng.json','final.pt'])
            if defs[arm]['ddqn']:
                expected_files.add(base+'final_replay.npz')
                expected_files.update(base+f'train_{ep:03d}.{suffix}' for ep in range(pl['online_episodes']) for suffix in ['npz','json'])
    expected_files.update(f'evaluation/{key}/episode_{ep:03d}.{suffix}' for key in [f'{a}/{i}' for a in arms for i in range(pl['fits'])]+['teacher'] for ep in range(pl['eval_episodes']) for suffix in ['npz','json'])
    require(actual==expected_files,'unexpected or missing declared artifact')
    events=read(root/'events.json');stages=[e['stage'] for e in events]
    require([e['sequence'] for e in events]==list(range(len(events))),'event order')
    require(stages.count('demonstrations_frozen')==stages.count('checkpoints_frozen')==1,'global freeze events')
    require(stages.index('demonstrations_frozen')<stages.index('training') and stages.index('checkpoints_frozen')<stages.index('evaluation'),'freeze before fitting/evaluation')
    require(stages[-1]=='artifacts_complete','unfinished workflow')
    jobs=[(arm,fit) for arm in arms for fit in range(pl['fits'])]
    np.random.default_rng(pl['seeds']['job_order_seed']).shuffle(jobs)
    require(read(root/'job_order.json')==[list(x) for x in jobs],'job RNG')
    require([(e['arm'],e['fit']) for e in events if e['stage']=='training']==jobs,'training order')
    demo_freeze=read(root/'DEMONSTRATION_FREEZE.json');ckfreeze=read(root/'CHECKPOINT_FREEZE.json')
    expected_checkpoints={f'{a}/fit_{fit}/final.pt' for a in arms for fit in range(pl['fits'])}
    require({p.relative_to(root).as_posix() for p in root.rglob('*.pt')}==expected_checkpoints,'nonfinal/missing checkpoint')
    require(set(ckfreeze['checkpoints'])=={f'{a}/{fit}' for a in arms for fit in range(pl['fits'])},'checkpoint freeze inventory')
    demo_paths={p.relative_to(root).as_posix() for p in root.glob('demo_*/*') if p.is_file()}
    require(set(demo_freeze['files'])==demo_paths,'demo freeze inventory')
    for name,digest in demo_freeze['files'].items():require(sha(root/name)==digest,'demo freeze hashes')
    cache={};slots=0
    for p in sorted(root.rglob('*.npz')):
        if p.name=='final_replay.npz':continue
        data=dict(np.load(p,allow_pickle=False));record=read(p.with_suffix('.json'))
        require(data['states'].shape==(40,533) and data['actions'].shape==(40,32),'episode shape')
        require(data['states'].dtype==np.float32 and data['actions'].dtype==np.int64,'episode dtype')
        require(all(np.isfinite(v).all() for v in data.values()),'nonfinite data')
        require(record['steps']==40 and np.array_equal(data['dones'],np.arange(40)==39),'episode terminal')
        require(close(data['states'][1:],data['next_states'][:-1]) and close(data['states'][:,-1],np.arange(40)/40) and data['next_states'][-1,-1]==1,'state timeline')
        require(np.all((data['actions']>=0)&(data['actions']<5)) and np.take_along_axis(baseline.allowed(data['states']),data['actions'][:,:,None],2).all(),'episode legality')
        require(close(data['rewards'].sum(dtype=np.float64),record['return_value']),'episode return')
        cfg=spec['simulator_defaults'];info=record['step_info'];tasks=record['records'];summary=record['summary']
        require(len(info)==40 and len(tasks)==1200 and summary['tasks_generated']==1200,'task/slot accounting')
        rr=[-cfg['delay_reward_weight']*v['reward_components']['normalized_delay_accrual']-cfg['uav_energy_reward_weight']*v['reward_components']['normalized_uav_energy']-cfg['deadline_miss_penalty']*v['reward_components']['new_deadline_misses_per_vehicle'] for v in info]
        require(close(data['rewards'],rr),'reward components')
        ontime=sum(bool(v['on_time']) for v in tasks);completed=sum(v['status']=='completed' for v in tasks)
        require(ontime==summary['tasks_on_time'] and completed==summary['tasks_completed'] and summary['tasks_unfinished']==1200-completed and close(ontime/1200,summary['deadline_success_rate']),'task outcomes/unfinished')
        cache[p.relative_to(root).as_posix()]=(data,record);slots+=40
    def episodes(folder):return [cache[p.relative_to(root).as_posix()] for p in sorted(folder.glob('*.npz')) if p.name!='final_replay.npz']
    def rows(eps):return [(x['states'][i],x['actions'][i],x['rewards'][i],x['next_states'][i],x['dones'][i]) for x,r in eps for i in range(40)]
    def workload(record):return [[v[k] for k in ['task_id','vehicle_id','arrival_s','input_bits','required_cycles','deadline_s']] for v in record['records']]
    teacher_eps=episodes(root/'evaluation/teacher');require(len(teacher_eps)==pl['eval_episodes'],'teacher evaluation count')
    for i,(x,r) in enumerate(teacher_eps):require(r['seed']==pl['seeds']['evaluation_seed_block']['start']+i,'teacher seed')
    teacher_mean=dict(return_value=float(np.mean([r['return_value'] for x,r in teacher_eps])),deadline=float(np.mean([r['summary']['deadline_success_rate'] for x,r in teacher_eps])))
    policies={};values={};distributions={};storage={};total_bc=0;total_td=0;total_keep=0;total_ridge=0;models={}
    for fit in range(pl['fits']):
        torch.manual_seed(pl['seeds']['model_seeds'][fit]);raw=model_source.Critic(A)
        require(tensor_hash(raw)==demo_freeze['initial_tensor_hashes'][str(fit)],'raw initial seed/hash')
        demos=episodes(root/f'demo_{fit}');require(len(demos)==10,'demo episodes')
        for i,(x,r) in enumerate(demos):require(r['seed']==pl['seeds']['demonstration_seed_blocks'][fit]['start']+i,'demo seeds')
        demo=rows(demos);sx=torch.from_numpy(np.stack([r[0] for r in demo]));sa=torch.from_numpy(np.stack([r[1] for r in demo]))
        demo_g=np.concatenate([returns(x['rewards']) for x,r in demos])
        # BC-only endpoint supplies unchanged calibrated features/branches.
        eck=torch.load(root/E/f'fit_{fit}/final.pt',weights_only=True);em=model_source.Critic(E).eval();em.load_state_dict(eck['online'])
        cal=read(root/E/f'fit_{fit}/calibration.json');alpha=cal['alpha']
        require(np.isfinite(alpha) and 0<alpha<=1,'finite alpha')
        with torch.no_grad():
            p,m,v,_=em.tables(sx);hidden=em.student.net[1](em.student.net[0](sx)).numpy().astype(np.float64)
            cost=((p.gather(2,sa[:,:30,None]).squeeze(2)-p.mean(2)).sum(1)+(m.gather(2,sa[:,30:,None]).squeeze(2)-m.mean(2)).sum(1)).numpy()
        # Independent augmented least-squares formulation (not helper normal solve).
        sr=max(1e-6,float(np.median(np.abs(np.stack([r[2] for r in demo[:320]]).astype(np.float64)))))
        scaled=torch.cat((p,m),1).numpy();ranges_scaled=np.ptp(scaled,axis=2).sum(1)
        rmax=float(np.max(ranges_scaled[:320]/alpha));want_alpha=sr/(sr+rmax)
        require(close(alpha,want_alpha) and close(sr,cal['reward_scale']) and close(rmax,cal['before_range_max']),'alpha/fit split')
        require(cal['fitting_episodes']==list(range(8)) and cal['withheld_return_episodes']==[8,9] and cal['withheld_action_labels_used_by_BC_and_retention'] is True,'calibration split declaration')
        hh=hidden[:320];mu=hh.mean(0);sig=np.maximum(1e-6,np.sqrt(np.mean((hh-mu)**2,0)));X=(hh-mu)/sig
        sg=max(1e-6,float(np.sqrt(np.mean(demo_g[:320]**2))));u=(demo_g[:320]+cost[:320])/sg
        aug=np.concatenate([X/np.sqrt(320),np.sqrt(.001)*np.eye(24)]);rhs=np.concatenate([(u-u.mean())/np.sqrt(320),np.zeros(24)])
        beta=np.linalg.lstsq(aug,rhs,rcond=None)[0];w=sg*beta/sig;b=sg*u.mean()-w@mu
        require(np.allclose(em.student.net[2].weight[160].detach().numpy(),w,rtol=2e-4,atol=2e-4) and np.allclose(float(em.student.net[2].bias[160].detach()),b,rtol=2e-4,atol=2e-4),'independent ridge/export coefficients')
        cr=cal['ridge']
        require(close(cr['feature_mean'],mu) and close(cr['feature_scale'],sig) and close(cr['return_scale'],sg) and close(cr['beta'],beta),'ridge metadata')
        require(cr['rho']==.001 and cr['rank']==int(np.linalg.matrix_rank(X)),'ridge rank/rho')
        gram=X.T@X/320+.001*np.eye(24)
        require(close(cr['regularized_condition'],np.linalg.cond(gram)),'ridge conditioning')
        require(np.isfinite(cr['max_fp32_export_error']) and cr['max_fp32_export_error']>=0 and np.isfinite(cr['max_equation_residual']),'ridge finite export metadata')
        # Reconstruct pre-calibration branch rows via positive inverse scaling;
        # FP32 inversion is approximate and is NOT an independent BC weight replay.
        bc_state=copy.deepcopy(eck['online'])
        bc_state['student.net.2.weight'][:160]/=alpha;bc_state['student.net.2.bias'][:160]/=alpha
        bc_state['student.net.2.weight'][160]=raw.state_dict()['student.net.2.weight'][160].clone()
        bc_state['student.net.2.bias'][160]=raw.state_dict()['student.net.2.bias'][160].clone()
        scale_state=copy.deepcopy(eck['online']);scale_state['student.net.2.weight'][160]=bc_state['student.net.2.weight'][160];scale_state['student.net.2.bias'][160]=bc_state['student.net.2.bias'][160]
        for stage,state,temp in [('after_bc',bc_state,1.),('after_scale',scale_state,alpha),('after_value_fit',eck['online'],alpha)]:
            stage_model=model_source.Critic(E).eval();stage_model.load_state_dict(state)
            verify_probe(stage_model,sx,sa,cal['stages'][stage],temp,check_hash=stage=='after_value_fit')
            for label,sl in [('fitting',slice(0,320)),('withheld_returns',slice(320,400))]:
                residual=np.asarray(cal['stages'][stage]['demonstration_q'])[sl]-demo_g[sl]
                rr=cal['residuals'][label][stage]
                require(close(rr['rmse'],np.sqrt(np.mean(residual**2))) and close(rr['error']['mean'],residual.mean()) and close(rr['error']['quantiles'],np.quantile(residual,[0,.05,.5,.95,1])) and rr['error']['count']==len(residual),'calibration residuals')
        warn=cal['residuals']['withheld_returns']['after_value_fit']['rmse']>cal['residuals']['withheld_returns']['after_bc']['rmse']
        require(cal['withheld_fit_worsened_vs_after_bc']==bool(warn),'withheld warning')
        require(cal['stages']['after_bc']['actions']==cal['stages']['after_scale']['actions']==cal['stages']['after_value_fit']['actions'],'handoff changed actions')
        require(max(cal['stages']['after_scale']['all_action_ranges'][:320])<=sr+1e-5+sr*1e-5,'gap bound')
        for arm in arms:
            d=defs[arm];folder=root/arm/f'fit_{fit}';key=f'{arm}/{fit}'
            count=read(folder/'counts.json');ck=torch.load(folder/'final.pt',weights_only=True)
            require(sha(folder/'final.pt')==ckfreeze['checkpoints'][key] and ck['counts']==count,'checkpoint/counts binding')
            model=model_source.Critic(arm).eval();model.load_state_dict(ck['online']);models[key]=model
            require(count['initial_tensor_hash']==demo_freeze['initial_tensor_hashes'][str(fit)],'paired initial hash')
            require(count['bc_end_tensor_hash']==ckfreeze['bc_end_hashes'][key] and count['handoff_tensor_hash']==ckfreeze['handoff_hashes'][key] and count['alpha']==ckfreeze['alphas'][key],'stage hash metadata')
            require(count['parameter_bytes']==67364 and count['buffer_bytes']==20,'storage metadata')
            require(sum(p.numel()*p.element_size() for p in model.parameters())==67364 and sum(v.numel()*v.element_size() for v in model.buffers())==20,'actual model storage')
            require(count['preloaded_demo_hash']==(sha(root/f'demo_{fit}/episode_000.npz') if d['demo_replay'] else None),'demonstration access/preload')
            recorded_cal=read(folder/'calibration.json')
            if d['calibration']:
                require(recorded_cal==cal,'A/B/E independent calibration parity')
                require(count['handoff_tensor_hash']==tensor_hash(em) and count['alpha']==alpha,'A/B/E handoff hash/alpha')
            else:require(recorded_cal is None and count['alpha']==1.,'unexpected calibration')
            require(count['ridge_solves']==int(d['calibration']),'ridge count');total_ridge+=count['ridge_solves']
            bc=read(folder/'bc.json');brng=np.random.default_rng(pl['seeds']['bc_sampler_seeds'][fit])
            require(len(bc)==count['bc_updates']==(pl['bc_steps'] if d['bc'] else 0),'BC update count')
            for i,item in enumerate(bc):
                require(item['step']==i and item['indices']==brng.choice(400,pl['batch'],replace=False).tolist(),'BC sampling')
                require(np.isfinite(item['loss']) and np.isfinite(item['gradient_norm']),'BC finite')
            total_bc+=len(bc)
            train=episodes(folder);require(len(train)==(pl['online_episodes'] if d['ddqn'] else 0),'training exposure')
            for i,(x,r) in enumerate(train):require(r['seed']==pl['seeds']['online_seed_blocks'][fit]['start']+i,'online seed')
            if d['ddqn']:
                require(all(workload(r)==workload(other) for (_,r),(_,other) in zip(train,episodes(root/A/f'fit_{fit}'))),'paired online workloads')
            allrows=rows(train);n=len(allrows);log=read(folder/'optimizer.json');draws=read(folder/'action_draws.json')
            require(count['slots']==n and len(draws)==n and len(log)==count['updates']==(n-pl['warmup'] if d['ddqn'] else 0),'slot/update counts')
            require(count['target_copies']==len(log)//pl['target_interval'] and count['initial_target_copy']==d['ddqn'],'target copy counts')
            if d['ddqn']:
                require(ck['target'] is not None and ck['optimizer'] is not None,'TD state missing')
                require(all(float(v['step'])==len(log) for v in ck['optimizer']['state'].values()),'optimizer reset/update count')
            else:require(ck['target'] is None and ck['optimizer'] is None,'BC-only TD state')
            replay=list(demo) if d['demo_replay'] else [];ids=[f'demo:{i}' for i in range(len(replay))]
            choice=np.random.default_rng(pl['seeds']['epsilon_branch_seeds'][fit]);sampling=np.random.default_rng(pl['seeds']['replay_sampler_seeds'][fit]);keep=np.random.default_rng(pl['seeds']['retention_sampler_seeds'][fit]) if d['retention'] else None
            for t,row in enumerate(allrows):
                draw=draws[t];uu=float(choice.random());vv=choice.random(32);eps=1. if t<pl['warmup'] else 1-.95*min(t/n,1)
                require(draw['slot']==t and draw['epsilon_uniform']==uu and draw['branch_uniforms']==vv.tolist() and draw['epsilon']==eps and draw['action']==row[1].tolist(),'action RNG binding')
                if uu<eps:
                    mask=baseline.allowed(row[0][None])[0];want=[int(np.flatnonzero(mask[j])[int(vv[j]*mask[j].sum())]) for j in range(32)]
                    require(want==draw['action'],'exploration actions')
                replay.append(row);ids.append(f'online:{t}')
                if len(replay)>pl['replay_capacity']:replay.pop(0);ids.pop(0)
                if t<pl['warmup']:continue
                item=log[t-pl['warmup']];idx=sampling.integers(0,len(replay),pl['batch'])
                require(item['update']==t-pl['warmup']+1 and item['online_slot']==t and item['indices']==idx.tolist() and item['transition_ids']==[ids[i] for i in idx],'TD replay draws')
                want_keep=keep.integers(0,400,pl['batch']).tolist() if keep is not None else None
                require(item['retention_indices']==want_keep,'retention RNG/access')
                require(item['alpha']==count['alpha'] and item['optimizer_steps']==1 and item['target_copy_performed'] is False,'TD update metadata')
                require(item['target_copy']==(item['update']%pl['target_interval']==0),'target schedule')
                reward=np.array([replay[i][2] for i in idx],np.float32);done=np.array([replay[i][4] for i in idx],bool)
                pred=np.array(item['prediction']);tq=np.array(item['target_q']);yy=np.array(item['target'])
                require(close(item['rewards'],reward) and np.array_equal(item['dones'],done),'TD batch rewards/dones')
                require(np.isfinite(pred).all() and np.isfinite(tq).all() and close(yy,reward+tq) and (tq[done]==0).all(),'target arithmetic/terminal')
                err=np.abs(pred-yy);td=float(np.where(err<1,.5*err**2,err-.5).mean())
                require(close(td,item['td_loss']),'TD Huber loss')
                require(np.isfinite(item['retention_loss']) and item['retention_loss']>=0 and close(item['total_loss'],item['td_loss']+item['retention_loss']),'combined loss')
                if not d['retention']:require(item['retention_loss']==0,'no-retention auxiliary loss')
                norms=item['gradient_norms'];require(all(np.isfinite(v) and v>=0 for v in norms.values()),'gradient finiteness')
                require(close(norms['total']**2,norms['shared_hidden']**2+norms['branch_output']**2+norms['value_output']**2) and close(item['clip_norm'],norms['total']) and item['clipped']==(item['clip_norm']>10),'gradient partition/clip arithmetic')
                nexta=np.array(item['next_actions']);require((nexta[done]==-1).all() and np.all((nexta[~done]>=0)&(nexta[~done]<5)),'successor terminal bypass')
                if (~done).any():
                    ns=np.stack([replay[i][3] for i in idx])[~done]
                    require(np.take_along_axis(baseline.allowed(ns),nexta[~done,:,None],2).all(),'successor screen')
                first=item['first_update_decomposition']
                if item['update']==1:
                    require(first is not None and first['nonterminal_indices']==np.flatnonzero(~done).tolist(),'first update decomposition')
                    current=np.array(item['current_greedy_q'])[~done]
                    require(close(first['reward'],reward[~done]) and close(first['same_state_gap'],current-pred[~done]) and close(first['successor_minus_current'],tq[~done]-current),'first update components')
                    require(close(first['residual'],yy[~done]-pred[~done]) and close(first['residual'],np.array(first['reward'])+np.array(first['same_state_gap'])+np.array(first['successor_minus_current'])),'first update identity')
                else:require(first is None,'unexpected first-update diagnostic')
            total_td+=len(log);total_keep+=sum(v['retention_indices'] is not None for v in log)
            require(count['retention_batches']==(len(log) if d['retention'] else 0),'retention count')
            saved_rng=read(folder/'final_rng.json')
            require(saved_rng==dict(choices=choice.bit_generator.state,replay=sampling.bit_generator.state,retention=keep.bit_generator.state if keep is not None else None,bc=brng.bit_generator.state,transition_ids=ids),'final RNG binding')
            if replay:
                stored=dict(np.load(folder/'final_replay.npz'))
                for j,name in enumerate(['states','actions','rewards','next_states','dones']):require(close(stored[name],np.stack([r[j] for r in replay])),'final replay')
            else:require(not (folder/'final_replay.npz').exists(),'unexpected replay')
            probes=read(folder/'probes.json');ps=probes['stages']
            if d['bc']:
                require(probes['source']==dict(kind='demonstrations',transition_ids=[f'demo:{i}' for i in range(400)]),'demo probe provenance')
                px,pa=sx,sa
                initial=model_source.Critic(arm);initial.load_state_dict(raw.state_dict());verify_probe(initial,px,pa,ps['BEFORE_BC'],1.)
                bcm=model_source.Critic(arm);bcm.load_state_dict(bc_state);verify_probe(bcm,px,pa,ps['AFTER_BC'],1.,check_hash=False)
                require(ps['AFTER_BC']['tensor_hash']==count['bc_end_tensor_hash'],'BC stage hash')
            else:
                own=allrows[:pl['warmup']];px=torch.from_numpy(np.stack([r[0] for r in own]));pa=torch.from_numpy(np.stack([r[1] for r in own]))
                require(probes['source']==dict(kind='own_online_warmup',transition_ids=[f'online:{i}' for i in range(pl['warmup'])]),'plain probe information')
                initial=model_source.Critic(arm);initial.load_state_dict(raw.state_dict());verify_probe(initial,px,pa,ps['BEFORE_BC'],1.)
                require(ps['AFTER_BC']['skipped'] is True,'plain BC probe')
            verify_probe(model,px,pa,ps['FINAL'],count['alpha'])
            if d['calibration']:
                require(ps['AFTER_SCALE']==cal['stages']['after_scale'] and ps['AFTER_VALUE_FIT']==cal['stages']['after_value_fit'],'calibration stage binding')
            else:require(ps['AFTER_SCALE']['skipped'] and ps['AFTER_VALUE_FIT']['skipped'],'noncalibrated stages')
            if not d['ddqn']:require(tensor_hash(model)==count['handoff_tensor_hash'] and ps['AFTER_VALUE_FIT']==ps['FINAL'],'BC-only drift')
            ee=episodes(root/'evaluation'/key);require(len(ee)==pl['eval_episodes'],'evaluation count')
            errors=[];qall=[];gall=[];terminal=[];quarters=[[] for _ in range(4)]
            for i,(x,r) in enumerate(ee):
                require(r['seed']==pl['seeds']['evaluation_seed_block']['start']+i and workload(r)==workload(teacher_eps[i][1]),'paired evaluation workload')
                decoded=[];selected_q=[]
                with torch.no_grad():
                    for state in x['states']:
                        aa,qq,_=model.greedy(torch.from_numpy(state)[None]);decoded.append(aa[0].tolist());selected_q.append(float(qq[0]))
                require(np.array_equal(decoded,x['actions']) and close(selected_q,x['q']),'final policy/value reproduction')
                g=returns(x['rewards']);error=np.array(x['q'])-g;errors.extend(error);qall.extend(x['q']);gall.extend(g);terminal.append(float(error[-1]))
                for j in range(4):quarters[j].extend(error[10*j:10*j+10])
            policies[key]=dict(return_value=float(np.mean([r['return_value'] for x,r in ee])),deadline=float(np.mean([r['summary']['deadline_success_rate'] for x,r in ee])))
            values[key]=dict(bias=float(np.mean(errors)),nrmse=float(np.sqrt(np.mean(np.square(errors)))/max(np.sqrt(np.mean(np.square(gall))),1e-12)))
            distributions[key]=dict(error_quantiles=np.quantile(errors,[0,.05,.5,.95,1]).tolist(),terminal_errors=terminal,
                error_by_slot_quarter=[dict(mean=float(np.mean(z)),quantiles=np.quantile(z,[0,.5,1]).tolist()) for z in quarters],
                positive_Q_fraction=float(np.mean(np.array(qall)>0)),nearzero_Q_fraction=float(np.mean(np.abs(qall)<1e-6)),
                nonterminal_error_quantiles=np.quantile(np.reshape(errors,(-1,40))[:,:39],[0,.05,.5,.95,1]).tolist(),
                gradient_norm_quantiles=np.quantile([r['clip_norm'] for r in log],[0,.5,.95,1]).tolist() if log else None,
                clip_fraction=float(np.mean([r['clipped'] for r in log])) if log else None,
                terminal_target_quantiles=np.quantile(np.array([r['target'] for r in log])[np.array([r['dones'] for r in log],bool)],[0,.05,.5,.95,1]).tolist() if log and np.array([r['dones'] for r in log]).any() else None,
                nonterminal_target_quantiles=np.quantile(np.array([r['target'] for r in log])[~np.array([r['dones'] for r in log],bool)],[0,.05,.5,.95,1]).tolist() if log else None,
                deadline_misses_by_destination={str(j):sum(v['destination']==j and not v['on_time'] for x,r in ee for v in r['records']) for j in range(5)})
            storage[key]=dict(parameter_bytes=67364,buffer_bytes=20,checkpoint_bytes=(folder/'final.pt').stat().st_size)
        require(len({ckfreeze['bc_end_hashes'][f'{arm}/{fit}'] for arm in arms if defs[arm]['bc']})==1,'BC paired weights')
    analysis=read(root/'analysis.json');per_fit={}
    for i in range(pl['fits']):
        absolute={a:policies[f'{a}/{i}'] for a in arms};primary=absolute[A]
        per_fit[str(i)]=dict(absolute=absolute,teacher=teacher_mean,contrasts={a:{k:primary[k]-v[k] for k in primary} for a,v in absolute.items() if a!=A},minus_teacher={a:{k:v[k]-teacher_mean[k] for k in v} for a,v in absolute.items()})
    require(per_fit==analysis['per_fit'],'all paired comparisons')
    for key in values:
        for metric in values[key]:require(close(values[key][metric],analysis['value_metrics'][key][metric]),'value metric')
    for arm in arms:
        for metric in ['return_value','deadline']:
            require(close(analysis['equal_fit_means'][arm][metric],np.mean([row['absolute'][arm][metric] for row in per_fit.values()])),'equal-fit absolute means')
            if arm!=A:require(close(analysis['equal_fit_mean_contrasts'][arm][metric],np.mean([row['contrasts'][arm][metric] for row in per_fit.values()])),'equal-fit comparison means')
    for fit in per_fit:
        for arm in arms:
            if arm==A:continue
            require(close(analysis['value_contrasts'][fit][arm]['nrmse'],values[f'{A}/{fit}']['nrmse']-values[f'{arm}/{fit}']['nrmse']) and close(analysis['value_contrasts'][fit][arm]['abs_bias'],abs(values[f'{A}/{fit}']['bias'])-abs(values[f'{arm}/{fit}']['bias'])),'all value contrasts')
    policy=all(row['contrasts'][arm][metric]>=0 for row in per_fit.values() for arm in [E,Farm] for metric in ['return_value','deadline'])
    policy=policy and all(np.mean([row['contrasts'][arm][metric] for row in per_fit.values()])>0 for arm in [E,Farm] for metric in ['return_value','deadline'])
    value=all(values[f'{A}/{i}']['nrmse']<values[f'{D}/{i}']['nrmse'] and abs(values[f'{A}/{i}']['bias'])<abs(values[f'{D}/{i}']['bias']) for i in range(pl['fits']))
    require(analysis['policy_screen']==bool(policy) and analysis['value_screen']==bool(value) and analysis['promising']==bool(policy and value) and analysis['quality_claim'] is False,'frozen decision logic')
    for fit,row in per_fit.items():
        for metric in ['return_value','deadline','nrmse','abs_bias']:
            def cell(arm):return row['absolute'][arm][metric] if metric in ['return_value','deadline'] else (values[f'{arm}/{fit}']['nrmse'] if metric=='nrmse' else abs(values[f'{arm}/{fit}']['bias']))
            require(close(analysis['factorial_interaction'][fit][metric],cell(A)-cell(B)-cell(C)+cell(D)),'factorial interaction')
    timing=read(root/'timing.json');tf=read(root/'TIMING_FREEZE.json');timing_states=np.load(root/'timing_states.npy')
    states=[]
    for p in sorted((root/'evaluation').rglob('*.npz')):states.extend(cache[p.relative_to(root).as_posix()][0]['states'])
    rng=np.random.default_rng(pl['seeds']['timing_order_seed']);chosen=rng.permutation(len(states))[:pl['timing_states']]
    require(tf['state_indices']==chosen.tolist() and np.array_equal(timing_states,np.stack([states[i] for i in chosen])) and sha(root/'timing_states.npy')==tf['states_sha256'],'timing states')
    paths=[f'{a}/{i}' for a,i in jobs]+['teacher'];order=[(r,k) for r in range(pl['timing_rounds']) for k in paths];rng.shuffle(order)
    require(tf['order']==[list(v) for v in order],'timing order')
    require(len(timing)==len(order)*pl['timing_states'],'timing count');timing_stats={}
    for block,(round_id,key) in enumerate(order):
        for j in range(pl['timing_states']):
            item=timing[block*pl['timing_states']+j]
            require((item['round'],item['policy'],item['state'])==(round_id,key,j) and item['ns']>0,'timing calls')
            if key!='teacher':
                with torch.no_grad():aa=models[key].greedy(torch.from_numpy(timing_states[j])[None])[0][0].tolist()
                require(aa==item['action'],'timed policy')
    for key in paths:
        ns=[v['ns'] for v in timing if v['policy']==key];timing_stats[key]=dict(median_ns=float(np.median(ns)),p95_ns=float(np.quantile(ns,.95)),
            per_round={str(r):dict(median_ns=float(np.median([v['ns'] for v in timing if v['policy']==key and v['round']==r])),p95_ns=float(np.quantile([v['ns'] for v in timing if v['policy']==key and v['round']==r],.95))) for r in range(pl['timing_rounds'])})
    # Reproduce the declared reference policy from saved observations only;
    # this makes reference inferences, never new simulator episodes.
    teacher=build_teacher(ROOT,Chapter3Config);teacher_calls=0
    for fit in range(pl['fits']):
        for x,record in episodes(root/f'demo_{fit}'):
            for state,action in zip(x['states'],x['actions']):
                chosen=teacher.values(state)[0][0];teacher_calls+=1
                require(np.array_equal(chosen,action),'demonstration teacher policy')
    for x,record in teacher_eps:
        for state,action in zip(x['states'],x['actions']):
            chosen=teacher.values(state)[0][0];teacher_calls+=1
            require(np.array_equal(chosen,action),'evaluation teacher policy')
    for item in timing:
        if item['policy']=='teacher':
            chosen=teacher.values(timing_states[item['state']])[0][0];teacher_calls+=1
            require(chosen.tolist()==item['action'],'timed teacher policy')
    require(slots==pl['fits']*400+6*pl['fits']*pl['online_episodes']*40+(7*pl['fits']+1)*pl['eval_episodes']*40,'total simulated slots')
    require(total_bc==6*pl['fits']*pl['bc_steps'] and total_td==6*pl['fits']*(pl['online_episodes']*40-pl['warmup']) and total_keep==2*pl['fits']*(pl['online_episodes']*40-pl['warmup']) and total_ridge==3*pl['fits'],'total learning exposures')
    return dict(passed=True,kind=pl['kind'],files_checked=len(manifest),simulation_slots_recorded=slots,
        historical_bc_updates=total_bc,historical_ddqn_updates=total_td,retention_batches=total_keep,ridge_solves=total_ridge,
        final_checkpoints=len(ckfreeze['checkpoints']),value_distributions=distributions,storage=storage,timing=timing_stats,
        recorded_promising=bool(policy and value),new_simulator_steps=0,new_optimizer_updates=0,teacher_reference_recomputations=teacher_calls,
        limitation='No historical gradient/weight replay; BC branches reconstructed approximately by inverse scale; independent augmented ridge and saved final/reference policies verified',
        research_launch_authorized=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_dir');args=p.parse_args();print(json.dumps(audit(args.run_dir),indent=2))
