"""Eleven-arm training orchestration; research modes stay disabled pending decision."""
from __future__ import annotations
import argparse,copy,gzip,json,shutil,time,traceback
from pathlib import Path
import numpy as np,torch
import run_controller_redesign_engineering as base
from controller_v3_amortized import NEW_ARMS,AmortizedPriceQ
import controller_v3_fused_runtime as fused
import compiled_controller_inference as compiled
import terminal_anchor_prototype as anchor
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/'TERMINAL_ANCHOR_RESEARCH_PROTOCOL_2026-11-06.json'
ENGINEERING_PROTOCOL=ROOT/'LOW_COMPLEXITY_CONTROLLER_REDESIGN_COMPARISON_PROTOCOL_v2_2026-11-05.txt'
COMPAT_PROTOCOL=ROOT/'LOW_COMPLEXITY_V3_RUNTIME_COMPATIBILITY_PROTOCOL_v2_2026-11-05.json'
BASE_ARMS=(*base.SHARED_ARMS,'additive64_nonpositive','additive128_unbounded','emc_q_nonpositive','bdq64_uniform_adapted')
CORRECTED=tuple(anchor.ARMS)
ARMS=(*CORRECTED,*NEW_ARMS,*BASE_ARMS)
_base_greedy=base.greedy
_base_chosen=base.chosen
_base_bounded=base.bounded
_base_update=base.update
RESEARCH_ENABLED=False
SOURCES=list(dict.fromkeys([*base.SOURCES,'controller_v3_amortized.py','controller_v3_fused_runtime.py',
    'controller_v3_runtime_compatibility.py','compiled_controller_inference.py','run_anchor_research_preparation.py','audit_anchor_research_preparation.py',
    'test_anchor_research_preparation.py','terminal_anchor_prototype.py',PROTOCOL.name,ENGINEERING_PROTOCOL.name,COMPAT_PROTOCOL.name]))
FIXTURE_CFG=dict(episodes=1,warmup=8,batch=4,capacity=64,learning_rate=1e-4,gamma=1.,clip=10.,target_interval=8,epsilon_decay=40)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def protocol():
    p=read(PROTOCOL);assert tuple(p['arms'])==ARMS and p['new_simulator_steps_this_gate']==p['optimizer_updates_this_gate']==0
    assert p['development']['training_slots']==17*5*200*40==680000
    assert p['development']['optimizer_updates']==17*5*(8000-1024+1)==593045
    assert p['final']['status']=='reserved_untouched_no_launch'
    return p

def configure():base.configure()
def bounded(arm):return _base_bounded(arm) or arm in NEW_ARMS or arm in CORRECTED

def build(arm,seed,cfg):
    if arm not in ARMS:raise ValueError('unknown research arm')
    model=make_model(arm,seed)
    target=copy.deepcopy(model).eval().requires_grad_(False)
    return dict(arm=arm,seed=seed,online=model,target=target,
        opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0),
        replay=TerminalBuffer(cfg['capacity']) if arm in CORRECTED else base.old.Replay(cfg['capacity']),
        anchor_rng=np.random.default_rng(seed+300000),action_rng=np.random.default_rng(seed+100000),
        replay_rng=np.random.default_rng(seed+200000),slots=0,updates=0,copies=0,actor_history=[],cfg=cfg)

def make_model(arm,seed):
    torch.manual_seed(seed)
    if arm in CORRECTED:return anchor.Critic(arm,seed)
    if arm in NEW_ARMS:return AmortizedPriceQ(arm)
    if arm in base.SHARED_ARMS:return base.SharedCostNetwork(arm)
    if arm=='bdq64_uniform_adapted':return base.AdaptedBDQ()
    if arm=='emc_q_nonpositive':return base.EMCQ()
    if arm in ('additive64_nonpositive','additive128_unbounded'):
        return base.FactorizedQNetwork(533,32,5,(128,128) if arm=='additive128_unbounded' else (64,64))
    raise ValueError('unknown model arm')

def greedy(model,states,arm):
    if arm in CORRECTED:return (*model.greedy(states),{})
    if arm in NEW_ARMS:return model.greedy(states)
    return _base_greedy(model,states,arm)

def chosen(model,states,actions,arm):
    if arm in CORRECTED:return model(states,actions)
    if arm in NEW_ARMS:return model(states,actions)
    return _base_chosen(model,states,actions,arm)

def patch_training_core():
    """Reuse validated env/episode/update/checkpoint mechanics, add v3 model APIs."""
    base.ARMS=ARMS;base.build=build;base.greedy=greedy;base.chosen=chosen;base.bounded=bounded;base.update=update

def value_summary_by_time(rows):
    full=base.values_summary(rows);bins=[]
    for start in range(0,40,10):
        selected=[]
        for row in rows:
            selected.append(dict(chosen_q=row['chosen_q'][start:start+10],target_q=row['target_q'][start:start+10],
                realized_return_to_go=row['realized_return_to_go'][start:start+10],slot_rewards=row['slot_rewards'][start:start+10]))
        q=np.asarray([v for row in selected for v in row['chosen_q']],np.float64)
        tq=np.asarray([v for row in selected for v in row['target_q']],np.float64)
        g=np.asarray([v for row in selected for v in row['realized_return_to_go']],np.float64)
        residual=q-g
        bins.append(dict(start_slot=start,end_slot=start+9,count=len(q),Q_mean=float(q.mean()),target_mean=float(tq.mean()),
            return_to_go_mean=float(g.mean()),residual_mean=float(residual.mean()),
            normalized_rmse=float(np.sqrt(np.mean(residual**2))/max(1.,np.sqrt(np.mean(g**2))))))
    full['by_time_bin']=bins
    return full

def compare_all(evaluations,heuristics):
    validate_comparison_rows(evaluations,heuristics)
    primary=evaluations['price_both'];out={}
    references=[a for a in ARMS if a!='price_both']
    primary_by_fit={}
    for row in primary:primary_by_fit.setdefault(row.get('model_seed'),{})[row['seed']]=row
    for arm in references:
        rows=evaluations[arm]
        other_by_fit={}
        for row in rows:other_by_fit.setdefault(row.get('model_seed'),{})[row['seed']]=row
        if set(other_by_fit)!=set(primary_by_fit):raise ValueError(f'unpaired initialization set:{arm}')
        differences=[]
        for fit_seed,primary_rows in primary_by_fit.items():
            other_rows=other_by_fit[fit_seed]
            if set(other_rows)!=set(primary_rows):raise ValueError(f'unpaired workload set:{arm}:{fit_seed}')
            returns=[];deadlines=[]
            for task_seed,p_row in primary_rows.items():
                o_row=other_rows[task_seed]
                if p_row['workload_sha256']!=o_row['workload_sha256']:raise ValueError(f'unpaired workload hash:{arm}:{task_seed}')
                returns.append(p_row['return_sum']-o_row['return_sum']);deadlines.append(p_row['deadline_success']-o_row['deadline_success'])
            differences.append(dict(seed=fit_seed,return_difference=float(np.mean(returns)),deadline_difference=float(np.mean(deadlines))))
        out[arm]=dict(per_fit=differences,mean_return_difference=float(np.mean([x['return_difference'] for x in differences])),
            mean_deadline_difference=float(np.mean([x['deadline_difference'] for x in differences])))
    for name,rows in heuristics.items():
        by_seed={x['seed']:x for x in rows}
        if set(by_seed)!=set(next(iter(primary_by_fit.values()))):raise ValueError(f'unpaired heuristic workload set:{name}')
        per_fit=[]
        for fit_seed,primary_rows in primary_by_fit.items():
            rd=[];dd=[]
            for task_seed,x in primary_rows.items():
                y=by_seed[task_seed]
                if x['workload_sha256']!=y['workload_sha256']:raise ValueError(f'unpaired heuristic hash:{name}:{task_seed}')
                rd.append(x['return_sum']-y['return_sum']);dd.append(x['deadline_success']-y['deadline_success'])
            per_fit.append(dict(seed=fit_seed,return_difference=float(np.mean(rd)),deadline_difference=float(np.mean(dd))))
        out[name]=dict(per_fit=per_fit,mean_return_difference=float(np.mean([x['return_difference'] for x in per_fit])),
            mean_deadline_difference=float(np.mean([x['deadline_difference'] for x in per_fit])))
    return out

def crossed_bootstrap(primary,reference,draws=5000,seed=520600002):
    validate_unique_rows(primary);validate_unique_rows(reference)
    """Paired crossed bootstrap over trained fits and common task episodes."""
    p_by_fit={}
    r_by_fit={}
    for row in primary:p_by_fit.setdefault(row['model_seed'],{})[row['seed']]=row
    for row in reference:r_by_fit.setdefault(row['model_seed'],{})[row['seed']]=row
    fits=sorted(p_by_fit)
    if len(fits)!=5 or sorted(r_by_fit)!=fits:raise ValueError('final bootstrap requires five paired trained fits')
    tasks=sorted(p_by_fit[fits[0]])
    if len(tasks)!=200:raise ValueError('final bootstrap requires 200 common workload episodes')
    matrices=[]
    for metric in ['return_sum','deadline_success']:
        matrix=np.empty((5,200),np.float64)
        for i,fit in enumerate(fits):
            if sorted(p_by_fit[fit])!=tasks or sorted(r_by_fit[fit])!=tasks:raise ValueError('unpaired final task set')
            for j,task in enumerate(tasks):
                a=p_by_fit[fit][task];b=r_by_fit[fit][task]
                if a['workload_sha256']!=b['workload_sha256']:raise ValueError('final workload mismatch')
                matrix[i,j]=a[metric]-b[metric]
        matrices.append(matrix)
    rng=np.random.default_rng(seed);sampled=[]
    for _ in range(draws):
        fi=rng.integers(0,5,5);ti=rng.integers(0,200,200)
        sampled.append([float(m[np.ix_(fi,ti)].mean()) for m in matrices])
    values=np.asarray(sampled)
    return dict(draws=draws,seed=seed,fit_resampling='paired_with_replacement_shared_across_arms',
        task_resampling='paired_with_replacement_shared_across_arms',
        return_difference_mean=float(matrices[0].mean()),return_difference_one_sided_97_5_lower=float(np.quantile(values[:,0],.025)),
        deadline_difference_mean=float(matrices[1].mean()),deadline_difference_one_sided_97_5_lower=float(np.quantile(values[:,1],.025)))

def quality_screens(comparisons,values,fit_count,bootstrap=None,phase=None):
    if fit_count!=5:return dict(evaluated=False,reason='all-five-fit-screen_not_applicable_to_synthetic_fixture')
    p=protocol();screens={};primary='price_both'
    required=set(p['comparisons']['primary_against_each_of']+p['comparisons']['heuristics']+['no_price_both']+p['comparisons']['correction_ablations'])
    if not required.issubset(comparisons) or any(len(comparisons[a]['per_fit'])!=5 for a in required):
        raise ValueError('incomplete comparison fit set')
    fit_sets=[{row['seed'] for row in comparisons[a]['per_fit']} for a in required]
    if any(len(seeds)!=5 or seeds!=fit_sets[0] for seeds in fit_sets):
        raise ValueError('duplicate or mismatched comparison fit identities')
    refs=p['comparisons']['primary_against_each_of']
    screens['references_each_fit']=all(all(r['return_difference']>=-2. and r['deadline_difference']>=-.01
        for r in comparisons[a]['per_fit']) for a in refs)
    screens['heuristics_each_fit']=all(all(r['return_difference']>=0. and r['deadline_difference']>=0.
        for r in comparisons[h]['per_fit']) for h in p['comparisons']['heuristics'])
    ab=comparisons['no_price_both']
    screens['mechanism_each_fit']=all(r['return_difference']>=0. and r['deadline_difference']>=0. for r in ab['per_fit'])
    screens['mechanism_mean']=ab['mean_return_difference']>=2. and ab['mean_deadline_difference']>=.01
    vd=p['value_diagnostics_per_fit_per_arm'];screens['value_each_fit']=all(
        len(values[a])==5 and all(x['normalized_rmse']<=vd['normalized_rmse_max'] and
        x['absolute_bias']<=vd['absolute_bias_max'] and
        x['near_zero_large_return_fraction']<=vd['near_zero_large_return_fraction_max'] and
        x['terminal_abs_error_p95']<=vd['terminal_abs_error_p95_max'] for x in values[a]) for a in p['value_screen_scope']['gated_arms'])
    screens['correction_ablations_each_fit']=all(all(r['return_difference']>=0 and r['deadline_difference']>=0 for r in comparisons[a]['per_fit']) for a in p['comparisons']['correction_ablations'])
    if phase=='final':
        if bootstrap is None:raise ValueError('locked final screen requires paired crossed bootstrap')
        screens['final_crossed_bootstrap']=(bootstrap['return_difference_one_sided_97_5_lower']>=-2. and
            bootstrap['deadline_difference_one_sided_97_5_lower']>=-.01)
    return screens

def checkpoint(agent):return base.checkpoint(agent)

def final_checkpoint(agent):
    return dict(arm=agent['arm'],model_seed=agent['seed'],online=copy.deepcopy(agent['online'].state_dict()),
        target=copy.deepcopy(agent['target'].state_dict()),slots=agent['slots'],updates=agent['updates'],
        copies=agent['copies'],cfg=agent['cfg'],online_parameter_hash=base.old.digest(agent['online'].state_dict()),
        target_parameter_hash=base.old.digest(agent['target'].state_dict()),
        action_rng_state=agent['action_rng'].bit_generator.state,replay_rng_state=agent['replay_rng'].bit_generator.state,
        actor_history=copy.deepcopy(agent['actor_history']))

def load_evaluator(path,arm,seed,cfg):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['arm']!=arm or ck['model_seed']!=seed:raise ValueError('checkpoint identity mismatch')
    online=make_model(arm,seed);target=copy.deepcopy(online)
    online.load_state_dict(ck['online']);target.load_state_dict(ck['target'])
    online.eval();target.eval()
    if base.old.digest(online.state_dict())!=ck['online_parameter_hash'] or base.old.digest(target.state_dict())!=ck['target_parameter_hash']:
        raise ValueError('checkpoint weight hash mismatch')
    return dict(arm=arm,online=online,target=target),ck

def frozen_inference_gate(out,fits):
    spec=protocol()['frozen_checkpoint_inference'];source=ROOT/spec['equivalence_observations_path']
    if base.old.sha(source)!=spec['equivalence_observations_sha256']:raise ValueError('frozen inference observation hash mismatch')
    states=np.load(source)['observations'];assert states.shape==(128,533)
    folder=out/'frozen_checkpoint_inference';folder.mkdir()
    np.savez_compressed(folder/'observations.npz',observations=states)
    results={arm:[] for arm in ARMS};raw=[];exports={};order_rng=np.random.default_rng(spec['order_seed'])
    fused.numba.set_num_threads(1)
    frozen=read(out/'ALL_FINAL_CHECKPOINTS_FREEZE.json')
    write(folder/'PRE_INFERENCE_FREEZE.json',dict(protocol_sha256=base.old.sha(PROTOCOL),
        frozen_checkpoint_sha256=frozen['checkpoints'],observations_sha256=base.old.sha(folder/'observations.npz'),
        observations_source_sha256=base.old.sha(source),
        routes={arm:('validated_complete_ORT_policy' if arm in base.SHARED_ARMS else 'native_core_and_decoder') for arm in ARMS},
        backend='ORT CPU FP32 sequential one-thread; native fastmath and parallel disabled'))
    orders=[];all_sessions={}
    for fit_id,(seed,_,_) in enumerate(fits):
        sessions={};models={};fit_exports={}
        for arm in ARMS:
            checkpoint=out/arm/f'fit_{fit_id:02d}'/'final.pt'
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model=make_model(arm,seed);model.load_state_dict(ck['online']);model.eval()
            if base.old.digest(model.state_dict())!=ck['online_parameter_hash']:raise ValueError('frozen inference weight hash mismatch')
            path=folder/'exports'/f'{arm}_fit_{fit_id:02d}.onnx';path.parent.mkdir(exist_ok=True)
            if arm in base.SHARED_ARMS:
                session,meta=compiled.export(model,arm,path);runtime=('complete',session)
            else:
                runtime,meta=fused.export(model.model if arm in CORRECTED else model,anchor.ARMS[arm][0] if arm in CORRECTED else arm,path)
                runtime=('native',runtime)
                np.save(path.with_suffix('.extra.npy'),runtime[1][1])
            actions=[];values=[]
            with torch.inference_mode():
                for state in states:
                    eager_a,eager_q,_=greedy(model,torch.from_numpy(state)[None],arm)
                    if runtime[0]=='native':native_a,native_q=runtime_values(runtime[1],arm,state)
                    else:native_a,native_q=compiled.compiled_values(runtime[1],state)
                    base.np.testing.assert_array_equal(eager_a[0].numpy(),native_a[0])
                    base.np.testing.assert_allclose(eager_q.numpy(),native_q,rtol=1e-5,atol=1e-6)
                    actions.append(native_a[0]);values.append(float(native_q[0]))
            if bounded(arm):assert max(values)<=0
            sessions[arm]=runtime;models[arm]=model
            fit_exports[arm]=dict(graph_sha256=base.old.sha(path),graph_bytes=path.stat().st_size,
                parameter_sha256=base.old.digest(model.state_dict()),parameters=sum(x.numel() for x in model.parameters()),
                parameter_bytes=base.parameter_bytes(model),route='validated_complete_ORT_policy' if arm in base.SHARED_ARMS else 'native_core_and_decoder',
                graph=path.name,export_seconds=meta.get('export_seconds',0.),session_startup_seconds=meta.get('session_startup_seconds',meta.get('startup_seconds',0.)))
        exports[f'fit_{fit_id:02d}']=fit_exports
        all_sessions[fit_id]=sessions
        del models
    for fit_id in range(len(fits)):
        for round_id in range(spec['timing_rounds']):
            order=list(ARMS);order_rng.shuffle(order)
            orders.append(dict(fit=fit_id,round=round_id,order=order))
    graph_hashes={fit:{arm:data['graph_sha256'] for arm,data in items.items()} for fit,items in exports.items()}
    payload_hashes={str(path.relative_to(folder)).replace('\\','/'):base.old.sha(path)
        for path in (folder/'exports').rglob('*') if path.is_file()}
    write(folder/'PRE_TIMING_FREEZE.json',dict(protocol_sha256=base.old.sha(PROTOCOL),
        inference_freeze_sha256=base.old.sha(folder/'PRE_INFERENCE_FREEZE.json'),export_hashes=graph_hashes,
        payload_hashes=payload_hashes,orders=orders,backend=read(folder/'PRE_INFERENCE_FREEZE.json')['backend'],
        timing_order_seed=spec['order_seed'],rounds=spec['timing_rounds'],warmups=spec['warmups'],
        calls=spec['calls_per_round'],fits=len(fits),paths=len(ARMS),frozen_at_ns=time.perf_counter_ns()))
    timing_freeze_sha=base.old.sha(folder/'PRE_TIMING_FREEZE.json')
    first_call_ns=None
    for item in orders:
        fit_id=item['fit'];round_id=item['round']
        for arm in item['order']:
            runtime=all_sessions[fit_id][arm]
            def infer(state):
                return runtime_values(runtime[1],arm,state)[0][0].copy() if runtime[0]=='native' else compiled.compiled_action(runtime[1],state)
            for state in states[:spec['warmups']]:infer(state)
            for index,state in enumerate(states):
                start=time.perf_counter_ns()
                if first_call_ns is None:first_call_ns=start
                action=infer(state);elapsed=time.perf_counter_ns()-start
                if action.shape!=(32,) or not np.all((action>=0)&(action<5)):raise AssertionError('invalid frozen action')
                raw.append(dict(fit=fit_id,round=round_id,arm=arm,observation=index,nanoseconds=elapsed))
    assert base.old.sha(folder/'PRE_TIMING_FREEZE.json')==timing_freeze_sha
    for path,digest in payload_hashes.items():assert base.old.sha(folder/path)==digest
    write(folder/'TIMING_STAGE_ORDER.json',dict(pre_timing_freeze_sha256=timing_freeze_sha,
        first_timed_call_ns=first_call_ns,completed_at_ns=time.perf_counter_ns()))
    del all_sessions
    metrics={}
    for arm in ARMS:
        metrics[arm]=[]
        for fit_id in range(len(fits)):
            ns=np.asarray([x['nanoseconds'] for x in raw if x['arm']==arm and x['fit']==fit_id],np.float64)
            assert len(ns)==spec['timing_rounds']*spec['calls_per_round'] and np.all(ns>0)
            metadata=exports[f'fit_{fit_id:02d}'][arm]
            metrics[arm].append(dict(fit=fit_id,median_ms=float(np.median(ns/1e6)),p95_ms=float(np.quantile(ns/1e6,.95)),
                parameter_bytes=metadata['parameter_bytes'],calls=len(ns),route=metadata['route']))
    screens={}
    for fit_id in range(len(fits)):
        primary=metrics['price_both'][fit_id];compact=metrics['additive64_nonpositive'][fit_id];wide=metrics['additive128_unbounded'][fit_id]
        screens[str(fit_id)]=dict(storage=primary['parameter_bytes']<=.5*compact['parameter_bytes'],
            median=primary['median_ms']<=wide['median_ms'],p95=primary['p95_ms']<=1.)
    report=dict(protocol=spec,observation_sha256=base.old.sha(source),exports=exports,metrics=metrics,orders=orders,
        raw_rows=len(raw),screens_per_fit=screens,all_fit_cost_screens_pass=all(all(v.values()) for v in screens.values()),
        equivalence_models=len(fits)*len(ARMS),equivalence_actions=len(fits)*len(ARMS)*len(states),
        research_training_launched=False,optimizer_updates_added=0,new_simulator_steps_added=0)
    write(folder/'verification.json',report);write(folder/'raw_timings.json',raw)
    return report

def eval_snapshot(agent):return base.old.digest((agent['online'].state_dict(),agent['target'].state_dict(),agent['online'].training,agent['target'].training))

def compact_eval(row):return {k:row[k] for k in ('seed','model_seed','workload_sha256','return_sum','deadline_success','chosen_q','target_q','realized_return_to_go','slot_rewards')}

def save_episode(folder,name,row):base.save_episode(Path(folder)/name,row)

def training_fixture(output):return run_plan(output,fixture=True)

def run_plan(output,fixture=False,phase=None,multi_fixture=False):
    if not fixture and not RESEARCH_ENABLED:raise RuntimeError('research execution disabled in implementation gate')
    configure();patch_training_core();p=protocol();out=Path(output)
    if fixture:
        plan=dict(kind='SYNTHETIC_NOT_RESEARCH',model_seeds=[99131],train_blocks=[dict(start=991310000,count=1)],
            evaluation_seeds=[991310100],cfg=FIXTURE_CFG,episodes=1,diagnostic_validation_seeds=[],
            quality_screen_authority=False,seed_namespace='synthetic only; excluded from research')
    else:
        if not RESEARCH_ENABLED:raise RuntimeError('research execution disabled in implementation gate')
        if phase!='development':raise ValueError('final execution disabled')
        pp=p[phase];plan=dict(kind='RESEARCH',model_seeds=pp['model_seeds'],train_blocks=pp['train_blocks'],
            evaluation_seeds=list(range(pp['evaluation_seeds']['start'],pp['evaluation_seeds']['start']+pp['evaluation_seeds']['count'])) if isinstance(pp['evaluation_seeds'],dict) else pp['evaluation_seeds'],
            cfg=dict(episodes=200,warmup=1024,batch=128,capacity=50000,learning_rate=1e-4,gamma=1.,clip=10.,target_interval=200,epsilon_decay=8000),
            episodes=200,diagnostic_validation_seeds=pp.get('diagnostic_validation_seeds',[]),quality_screen_authority=True,phase=phase)
    if multi_fixture:
        if not fixture:raise ValueError('synthetic override requires fixture mode')
        plan=synthetic_multifit_plan()
    plan.setdefault('validation_interval',50)
    validate_plan(plan)
    verify_prior_engineering()
    out.mkdir(parents=True,exist_ok=False)
    cost_audit=read(ROOT/'validation/controller_v3_runtime_compatibility_2026-11-05_01/saved_artifact_audit.json')
    cost_result=read(ROOT/'validation/controller_v3_runtime_compatibility_2026-11-05_01/verification.json')
    if cost_audit['state']!='passed_compatibility_artifact_audit' or cost_result['state']!='compatibility_cost_passed':
        raise ValueError('prior controller engineering gate is not audited/passing')
    plan.update(protocol_sha256=base.old.sha(PROTOCOL),source_sha256={n:base.old.sha(ROOT/n) for n in SOURCES},
        inherited_v2_protocol_sha256=base.old.sha(ENGINEERING_PROTOCOL),compatibility_protocol_sha256=base.old.sha(COMPAT_PROTOCOL),
        prior_engineering_completion_manifest_sha256=base.old.sha(ROOT/'validation/controller_v3_runtime_compatibility_2026-11-05_01_COMPLETION_MANIFEST.json'),
        prior_engineering_audit_sha256=base.old.sha(ROOT/'validation/controller_v3_runtime_compatibility_2026-11-05_01/saved_artifact_audit.json'),
        arm_order_seed=p['paired_randomization']['arm_order_rng_seed'],fixture=fixture,training_launched=not fixture and RESEARCH_ENABLED)
    write(out/'runtime_plan.json',plan)
    shutil.copy2(ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json',out/'PROJECT_STATUS_BEFORE.json')
    for name in SOURCES:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    events=[];counts={};order_rng=np.random.default_rng(plan['arm_order_seed'])
    def event(stage,**kwargs):
        events.append(dict(sequence=len(events),stage=stage,**kwargs));write(out/'events.json',events)
    fits=[(seed,block['start'],fit) for fit,(seed,block) in enumerate(zip(plan['model_seeds'],plan['train_blocks']))]
    try:
        # Initialization diagnostics are saved before training and never used to select weights.
        for fit_id,(model_seed,train_start,_) in enumerate(fits):
            arm_order=list(ARMS);order_rng.shuffle(arm_order)
            write(out/f'arm_order_fit_{fit_id:02d}.json',dict(order=arm_order,seed=plan['arm_order_seed']))
            for arm in arm_order:
                agent=build(arm,model_seed,plan['cfg']);folder=out/arm/f'fit_{fit_id:02d}';folder.mkdir(parents=True)
                if plan['diagnostic_validation_seeds']:
                    for validation_seed in plan['diagnostic_validation_seeds']:
                        row,_=base.episode(agent,validation_seed,episode_id='validation_initial')
                        row['model_seed']=model_seed;row['run_kind']=plan['kind'];save_episode(folder,f'validation_initial_{validation_seed}.json.gz',row)
                logs=[]
                for episode_index in range(plan['episodes']):
                    row,ep_logs=base.episode(agent,train_start+episode_index,training=True,episode_id=episode_index)
                    row['model_seed']=model_seed;row['run_kind']=plan['kind']
                    save_episode(folder,f'train_{episode_index:04d}.json.gz',row);logs.extend(ep_logs)
                    if (episode_index+1)%plan['validation_interval']==0 or episode_index+1==plan['episodes']:
                        event('fit_progress',arm=arm,fit=fit_id,episode=episode_index+1,
                            slots=agent['slots'],updates=agent['updates'])
                        if plan['diagnostic_validation_seeds'] and (episode_index+1)%plan['validation_interval']==0:
                            snap=base.old.learner_snapshot(agent)
                            for validation_seed in plan['diagnostic_validation_seeds']:
                                val,_=base.episode(agent,validation_seed,episode_id=f'validation_{episode_index+1}')
                                val['model_seed']=model_seed;val['run_kind']=plan['kind']
                                save_episode(folder,f'validation_{episode_index+1}_{validation_seed}.json.gz',val)
                                if base.old.learner_snapshot(agent)!=snap:raise AssertionError('validation mutated learner')
                if logs:
                    (folder/'optimizer.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in logs),encoding='utf-8')
                agent['online'].eval();agent['target'].eval()
                ck=final_checkpoint(agent);torch.save(ck,folder/'final.pt')
                counts[f'{arm}_fit_{fit_id:02d}']=dict(slots=agent['slots'],updates=agent['updates'],target_copies=agent['copies'],
                    checkpoints=1,optimizer_rows=len(logs),model_seed=model_seed,train_seed_start=train_start)
                del ck,agent,logs
        # Freeze every arm and initialization before opening any held-out gate episode.
        frozen={key:base.old.sha(out/key.rsplit('_fit_',1)[0]/('fit_'+key.rsplit('_fit_',1)[1])/'final.pt') for key in counts}
        write(out/'ALL_FINAL_CHECKPOINTS_FREEZE.json',dict(checkpoints=frozen,count=len(frozen),before_evaluation=True))
        event('all_final_checkpoints_frozen',count=len(frozen))
        event('frozen_checkpoint_inference_gate_started',checkpoint_count=len(frozen))
        inference_report=frozen_inference_gate(out,fits)
        write(out/'FROZEN_CHECKPOINT_INFERENCE.json',inference_report)
        event('frozen_checkpoint_inference_gate_complete',all_fit_cost_screens_pass=inference_report['all_fit_cost_screens_pass'])
        event('heldout_evaluation_started',seed_count=len(plan['evaluation_seeds']))
        heuristics={name:[] for name in p['comparisons']['heuristics']}
        for seed in plan['evaluation_seeds']:
            for name in heuristics:
                row,_=base.episode(None,seed,policy=name,episode_id='heldout_gate');row['model_seed']=None;row['run_kind']=plan['kind'];save_episode(out,f'heuristic_{name}_{seed}.json.gz',row);heuristics[name].append(compact_eval(row))
        evaluations={arm:[] for arm in ARMS};invariance=[]
        for fit_id,(model_seed,_,_) in enumerate(fits):
            for arm in ARMS:
                folder=out/arm/f'fit_{fit_id:02d}'
                agent,ck=load_evaluator(folder/'final.pt',arm,model_seed,plan['cfg']);before=eval_snapshot(agent)
                for seed in plan['evaluation_seeds']:
                    row,_=base.episode(agent,seed,episode_id='heldout_gate');row['model_seed']=model_seed;row['run_kind']=plan['kind'];save_episode(folder,f'evaluation_{seed}.json.gz',row);evaluations[arm].append(compact_eval(row))
                    if eval_snapshot(agent)!=before:raise AssertionError('held-out evaluation mutated weights')
                invariance.append(dict(arm=arm,fit=fit_id,before=before,after=eval_snapshot(agent)))
                del agent,ck
        comparisons=compare_all(evaluations,heuristics)
        values={arm:[value_summary_by_time([row for row in evaluations[arm] if row['model_seed']==seed])
            for seed in plan['model_seeds']] for arm in ARMS}
        bootstrap=(crossed_bootstrap(evaluations['price_both'],evaluations['additive128_unbounded'])
            if not fixture and phase=='final' else None)
        screens=quality_screens(comparisons,values,len(fits),bootstrap,phase) if not fixture else dict(evaluated=False,reason='synthetic_fixture_excluded')
        analysis=dict(run_kind=plan['kind'],comparisons=comparisons,value_stability=values,final_primary_bootstrap=bootstrap,
            quality_screens=screens,quality_claim=False,selection_performed=False,all_fits_retained=True,
            interpretation='diagnostics are empirical greedy-policy realized-return consistency, not Qstar proof')
        write(out/'comparisons_and_value_stability.json',analysis);write(out/'evaluation_invariance.json',invariance)
        for key,digest in frozen.items():
            fit=key.rsplit('_fit_',1)[1];arm=key.rsplit('_fit_',1)[0]
            path=out/arm/f'fit_{fit}'/'final.pt'
            assert base.old.sha(path)==digest
        state='completed_synthetic_full_fixture_not_research' if fixture else (phase+'_screen_passed' if aggregate_screens(screens,inference_report) else phase+'_screen_failed')
        slots=sum(x['slots'] for x in counts.values());updates=sum(x['updates'] for x in counts.values())
        result=dict(state=state,run_kind=plan['kind'],fixture=fixture,checkpoint_count=len(frozen),counts=counts,
            training_slots=slots,optimizer_updates=updates,target_copies=sum(x['target_copies'] for x in counts.values()),
            checkpoints_frozen_before_evaluation=True,heldout_evaluation_seeds=plan['evaluation_seeds'],
            frozen_checkpoint_inference_screens=inference_report['screens_per_fit'],
            frozen_checkpoint_inference_all_fit_screens_pass=inference_report['all_fit_cost_screens_pass'],
            research_training_launched=not fixture and RESEARCH_ENABLED,policy_quality_claim=False,
            thesis_changed=False,all_failed_fits_retained=True,
            empirical_protocol_screens_passed=False if fixture else aggregate_screens(screens,inference_report))
        event('complete',training_slots=slots,optimizer_updates=updates)
        write(out/'verification.json',result)
        manifest={str(x.relative_to(out)).replace('\\','/'):base.old.sha(x) for x in out.rglob('*') if x.is_file()}
        write(out/'run_manifest.json',dict(files=manifest))
        return result
    except Exception:
        write(out/'FAILED.json',dict(error=traceback.format_exc(),fixture=fixture,training_launched=False if fixture else RESEARCH_ENABLED))
        raise


PRIOR_STATUS_ARCHIVE=ROOT/'validation/controller_v3_research_fixture_2026-11-05_04/PROJECT_STATUS_BEFORE.json'

def verify_prior_engineering():
    manifest=read(ROOT/'validation/controller_v3_runtime_compatibility_2026-11-05_01_COMPLETION_MANIFEST.json')
    for name,digest in manifest['files'].items():
        path=PRIOR_STATUS_ARCHIVE if name=='LOW_COMPLEXITY_PROJECT_STATUS.json' else ROOT/name
        if base.old.sha(path)!=digest:raise ValueError('engineering evidence changed:'+name)
    return manifest

def synthetic_multifit_plan():
    return dict(kind='SYNTHETIC_NOT_RESEARCH',model_seeds=[99601,99607],
        train_blocks=[dict(start=996010000,count=2),dict(start=996070000,count=2)],
        evaluation_seeds=[996090100,996090101],cfg=dict(FIXTURE_CFG,episodes=2,epsilon_decay=80),
        episodes=2,diagnostic_validation_seeds=[996090200],validation_interval=1,
        quality_screen_authority=False,seed_namespace='synthetic engineering only; never research')

def validate_plan(plan):
    if not plan['model_seeds'] or len(set(plan['model_seeds']))!=len(plan['model_seeds']):raise ValueError('model seeds')
    if len(plan['train_blocks'])!=len(plan['model_seeds']):raise ValueError('fit block count')
    if any(b['count']!=plan['episodes'] for b in plan['train_blocks']):raise ValueError('episode counts')
    train=[s for b in plan['train_blocks'] for s in range(b['start'],b['start']+b['count'])]
    groups=[train,plan['evaluation_seeds'],plan['diagnostic_validation_seeds'],plan['model_seeds']]
    flat=[s for group in groups for s in group]
    if len(flat)!=len(set(flat)) or not plan['evaluation_seeds']:raise ValueError('overlapping or missing seeds')
    if plan['kind']=='SYNTHETIC_NOT_RESEARCH':
        if any(not (99000<=s<100000 or 990000000<=s<1000000000) for s in flat):raise ValueError('unsafe synthetic seed')
    if plan['episodes']<=0 or plan['validation_interval']<=0 or plan['cfg']['episodes']!=plan['episodes']:
        raise ValueError('invalid episode schedule')

def research_cfg():
    return dict(episodes=200,warmup=1024,batch=128,capacity=50000,learning_rate=1e-4,
        gamma=1.,clip=10.,target_interval=200,epsilon_decay=8000)

def validate_unique_rows(rows):
    keys=[(r.get('model_seed'),r['seed']) for r in rows]
    if not keys or len(keys)!=len(set(keys)):raise ValueError('empty or duplicate comparison rows')

def validate_comparison_rows(evaluations,heuristics):
    if set(evaluations)!=set(ARMS):raise ValueError('missing controller')
    for rows in [*evaluations.values(),*heuristics.values()]:validate_unique_rows(rows)
    keys=[set((r.get('model_seed'),r['seed']) for r in rows) for rows in evaluations.values()]
    if any(k!=keys[0] for k in keys):raise ValueError('unpaired comparison identities')

def aggregate_screens(quality,inference):
    return bool(quality and 'evaluated' not in quality and all(quality.values())
        and inference['all_fit_cost_screens_pass'])


class TerminalBuffer(base.old.Replay):
    """Preserve original array dtypes/uniform sampling, track terminal slots."""
    def __init__(self,capacity):super().__init__(capacity);self.terminal=set()
    def add(self,*row):
        index=self.position;self.terminal.discard(index);super().add(*row)
        if row[4]:self.terminal.add(index)
    def terminal_sample(self,n,rng):
        if not self.terminal:return None
        ix=rng.choice(sorted(self.terminal),size=n,replace=True)
        arrays=(self.states[ix],self.actions[ix].astype(np.int64),self.rewards[ix],self.next_states[ix],self.dones[ix])
        return (*map(torch.tensor,arrays),ix)

def update(agent):
    if agent['arm'] not in CORRECTED:return _base_update(agent)
    cfg=agent['cfg']
    if agent['slots']<cfg['warmup']:return None
    raw=agent['replay'].sample(cfg['batch'],agent['replay_rng'])
    batch=(*map(torch.tensor,raw[:5]),raw[-1])
    terminal=agent['replay'].terminal_sample(32,agent['anchor_rng']) if agent['online'].anchored else None
    start=time.perf_counter_ns()
    loss,detail=anchor.objective(agent['online'],agent['target'],batch,terminal)
    s,a,r,ns,d,indices=batch
    q=chosen(agent['online'],s,a,agent['arm']);y=torch.tensor(detail['target'],dtype=q.dtype)
    agent['opt'].zero_grad(set_to_none=True);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(agent['online'].parameters(),cfg['clip'],error_if_nonfinite=True)
    assert all(v.grad is None for v in agent['target'].parameters())
    agent['opt'].step();agent['updates']+=1
    assert all(torch.isfinite(v).all() for v in agent['online'].parameters())
    copied=agent['updates']%cfg['target_interval']==0
    if copied:agent['target'].load_state_dict(agent['online'].state_dict());agent['copies']+=1
    return dict(step=agent['updates'],physical_slots=agent['slots'],loss=float(loss.detach()),
        grad_norm_before_clip=float(norm),gradient_clipped=bool(norm>cfg['clip']),
        chosen_q=base.old.distribution(q),bellman_target=base.old.distribution(y),td_residual=base.old.distribution(q.detach()-y),
        replay_indices=indices.tolist(),target_copy=copied,terminal_samples=int(d.sum()),
        anchor_indices=[] if terminal is None else terminal[-1].tolist(),terminal_pool=sorted(agent['replay'].terminal),
        uniform_loss=detail['uniform'],anchor_loss=detail['anchor'],empty_anchor=detail['empty_anchor'],
        objective_update_ns=time.perf_counter_ns()-start)

def runtime_values(runtime,arm,state):
    if arm not in CORRECTED:return fused.values(runtime,arm,state)
    x=np.asarray(state,dtype=np.float32)
    if x.shape!=(533,) or not np.isfinite(x).all():raise ValueError('finite state required')
    t=float(x[-1])*40;rounded=round(t)
    if abs(t-rounded)>1e-4 or not 0<=rounded<40:raise ValueError('nonterminal grid time required')
    a,q=fused.values(runtime,anchor.ARMS[arm][0],x)
    return a,q*np.float32(40-rounded if anchor.ARMS[arm][1] else 1.)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['integration-fixture','development','final'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--confirm-development-run',action='store_true');p.add_argument('--confirm-final-run',action='store_true')
    a=p.parse_args()
    if a.mode not in ('fixture','integration-fixture'):
        if not getattr(a,'confirm_'+a.mode+'_run'):p.error('--confirm-'+a.mode+'-run is required; research remains disabled')
        p.error('Research modes are disabled in this preparation gate, including with confirmation')
    result=run_plan(a.output,fixture=True,multi_fixture=a.mode=='integration-fixture')
    from audit_anchor_research_preparation import audit_research
    print(json.dumps(dict(runner=result,audit=audit_research(a.output)),indent=2))

if __name__=='__main__':main()
