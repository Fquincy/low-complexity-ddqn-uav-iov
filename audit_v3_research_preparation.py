"""Independent saved-artifact audit for the eleven-arm synthetic fixture."""
import argparse,gzip,json
from pathlib import Path
import numpy as np,torch
import onnxruntime as ort
import run_v3_research_preparation as runner
import run_controller_redesign_engineering as base
import controller_v3_fused_runtime as fused
import compiled_controller_inference as compiled
from audit_controller_redesign_engineering import check_episode,verify_arrays,exclusion_check,verify_manifest

def audit_fixture(folder):
    base.configure();runner.patch_training_core();folder=Path(folder);manifest=verify_manifest(folder)
    plan=runner.read(folder/'runtime_plan.json');p=runner.protocol()
    assert plan['kind']=='SYNTHETIC_NOT_RESEARCH' and plan['fixture'] and not plan['training_launched']
    assert plan['protocol_sha256']==base.old.sha(runner.PROTOCOL)
    assert plan['model_seeds']==[99131] and plan['train_blocks']==[dict(start=991310000,count=1)]
    assert plan['evaluation_seeds']==[991310100] and plan['cfg']==runner.FIXTURE_CFG
    assert plan['inherited_v2_protocol_sha256']==base.old.sha(runner.ENGINEERING_PROTOCOL)
    assert plan['compatibility_protocol_sha256']==base.old.sha(runner.COMPAT_PROTOCOL)
    for name,h in plan['source_sha256'].items():
        assert base.old.sha(runner.ROOT/name)==base.old.sha(folder/'source_snapshot'/name)==h,name
    frozen=runner.read(folder/'ALL_FINAL_CHECKPOINTS_FREEZE.json')
    assert frozen['count']==11 and frozen['before_evaluation']
    events=runner.read(folder/'events.json');freeze_seq=next(x['sequence'] for x in events if x['stage']=='all_final_checkpoints_frozen')
    assert all(x['sequence']>freeze_seq for x in events if x['stage']=='complete')
    result=runner.read(folder/'verification.json')
    assert result['state']=='completed_synthetic_full_fixture_not_research'
    assert result['checkpoint_count']==11 and result['training_slots']==440 and result['optimizer_updates']==363
    assert result['target_copies']==44 and not result['research_training_launched'] and not result['policy_quality_claim']
    inf=audit_frozen_inference(folder,[(99131,991310000,0)])
    fingerprints={};actor_streams=[];episodes=0;optimizer_rows=0;fit_records={}
    for arm in runner.ARMS:
        fit=folder/arm/'fit_00';ckpt_path=fit/'final.pt'
        assert base.old.sha(ckpt_path)==frozen['checkpoints'][f'{arm}_fit_00']
        ck=torch.load(ckpt_path,map_location='cpu',weights_only=False)
        assert set(ck)=={'arm','model_seed','online','target','slots','updates','copies','cfg','online_parameter_hash','target_parameter_hash','action_rng_state','replay_rng_state','actor_history'}
        assert (ck['arm'],ck['model_seed'],ck['slots'],ck['updates'],ck['copies'])==(arm,99131,40,33,4)
        assert ck['cfg']==runner.FIXTURE_CFG
        assert base.old.digest(ck['online'])==ck['online_parameter_hash']
        assert base.old.digest(ck['target'])==ck['target_parameter_hash']
        actor_streams.append(base.old.digest(ck['actor_history']));assert len(ck['actor_history'])==40
        fit_records[arm]=[]
        for name in ['train_0000.json.gz','evaluation_991310100.json.gz']:
            with gzip.open(fit/name,'rt',encoding='utf-8') as stream:row=json.load(stream)
            assert row['model_seed']==99131 and row['run_kind']=='SYNTHETIC_NOT_RESEARCH'
            check_episode(row,base.Chapter3Config());states,next_states=verify_arrays(fit,name,row);episodes+=1
            fingerprints.setdefault(name,[]).append(row['workload_sha256'])
            actions=[x['action'] for x in row['slots']]
            assert all(len(x)==32 and all(0<=d<5 for d in x) for x in actions)
            assert row['slots'][-1]['done'] is True and all(not x['done'] for x in row['slots'][:-1])
            if not name.startswith('train_'):
                agent,loaded=runner.load_evaluator(ckpt_path,arm,99131,runner.FIXTURE_CFG)
                x=torch.tensor(states);act=torch.tensor(actions)
                with torch.inference_mode():
                    q=runner.chosen(agent['online'],x,act,arm).numpy()
                    tq=runner.chosen(agent['target'],x,act,arm).numpy()
                np.testing.assert_allclose(q,row['chosen_q'],rtol=1e-5,atol=1e-5)
                np.testing.assert_allclose(tq,row['target_q'],rtol=1e-5,atol=1e-5)
                np.testing.assert_allclose(row['realized_return_to_go'],np.cumsum(row['slot_rewards'][::-1])[::-1],rtol=0,atol=1e-9)
                if runner.bounded(arm):assert max(q)<=0 and max(tq)<=0
                fit_records[arm].append(runner.compact_eval(row))
        logs=[json.loads(x) for x in (fit/'optimizer.jsonl').read_text().splitlines()]
        assert len(logs)==33
        for i,x in enumerate(logs):
            assert x['step']==i+1 and x['physical_slots']==i+8 and x['target_copy']==((i+1)%8==0)
            assert len(x['replay_indices'])==4 and max(x['replay_indices'])<i+8
            assert np.isfinite([x['loss'],x['grad_norm_before_clip']]).all()
            assert x['terminal_samples']>=0
            for key in ['chosen_q','bellman_target','td_residual']:
                assert all(np.isfinite(list(x[key].values())))
        optimizer_rows+=len(logs)
        fit_summary=result['counts'][f'{arm}_fit_00']
        assert fit_summary['slots']==40 and fit_summary['updates']==33 and fit_summary['target_copies']==4
    assert len(set(actor_streams))==1
    assert all(len(set(rows))==1 for rows in fingerprints.values())
    heuristics={}
    for name in p['comparisons']['heuristics']:
        with gzip.open(folder/f'heuristic_{name}_991310100.json.gz','rt',encoding='utf-8') as stream:row=json.load(stream)
        check_episode(row,base.Chapter3Config());episodes+=1
        assert row['run_kind']=='SYNTHETIC_NOT_RESEARCH'
        assert row['workload_sha256']==fingerprints['evaluation_991310100.json.gz'][0]
        heuristics[name]=[runner.compact_eval(row)]
    analysis=runner.read(folder/'comparisons_and_value_stability.json')
    assert analysis['comparisons']==runner.compare_all({a:fit_records[a] for a in runner.ARMS},heuristics)
    assert analysis['quality_screens']['evaluated'] is False and analysis['quality_claim'] is False
    assert len(analysis['value_stability'])==11
    for arm in runner.ARMS:
        assert analysis['value_stability'][arm]==[runner.value_summary_by_time(fit_records[arm])]
    for item in runner.read(folder/'evaluation_invariance.json'):assert item['before']==item['after']
    scope=exclusion_check()
    result_audit=dict(state='passed_v3_synthetic_orchestration_audit',run_kind='SYNTHETIC_NOT_RESEARCH',
        checkpoints_checked=11,episodes_checked=episodes,optimizer_rows_checked=optimizer_rows,
        paired_task_sets_verified=2,workload_vectors_shared_across_arms=True,manifest_entries_checked=len(manifest),
        training_slots_checked=440,optimizer_updates_checked=363,all_final_weights_verified=True,
        frozen_inference_audit=inf,
        thesis_changed=False,research_quality_evidence=False,simulator_steps_added=0,optimizer_updates_added=0,scope=scope)
    base.write(folder/'saved_artifact_audit.json',result_audit);return result_audit

def audit_frozen_inference(folder,fits):
    folder=Path(folder)/'frozen_checkpoint_inference';spec=runner.protocol()['frozen_checkpoint_inference']
    pre=runner.read(folder/'PRE_INFERENCE_FREEZE.json');timing=runner.read(folder/'PRE_TIMING_FREEZE.json')
    report=runner.read(folder/'verification.json');states=np.load(folder/'observations.npz')['observations']
    source=runner.ROOT/spec['equivalence_observations_path']
    assert base.old.sha(source)==spec['equivalence_observations_sha256']==pre['observations_source_sha256']
    np.testing.assert_array_equal(states,np.load(source)['observations'])
    assert pre['protocol_sha256']==timing['protocol_sha256']==base.old.sha(runner.PROTOCOL)
    assert pre['observations_sha256']==base.old.sha(folder/'observations.npz')
    freeze=runner.read(folder.parent/'ALL_FINAL_CHECKPOINTS_FREEZE.json')
    assert pre['frozen_checkpoint_sha256']==freeze['checkpoints']
    exports=report['exports'];vectors=0
    raw=runner.read(folder/'raw_timings.json');expected=[];rng=np.random.default_rng(spec['order_seed']);orders=[]
    for fit_id,(seed,_,_) in enumerate(fits):
        sessions={};models={}
        for arm in runner.ARMS:
            path=folder/'exports'/f'{arm}_fit_{fit_id:02d}.onnx';meta=exports[f'fit_{fit_id:02d}'][arm]
            ck=torch.load(folder.parent/arm/f'fit_{fit_id:02d}'/'final.pt',map_location='cpu',weights_only=False)
            model=runner.make_model(arm,seed);model.load_state_dict(ck['online']);model.eval()
            assert base.old.digest(model.state_dict())==ck['online_parameter_hash']==meta['parameter_sha256']
            assert sum(x.numel() for x in model.parameters())==meta['parameters']
            assert base.parameter_bytes(model)==meta['parameter_bytes']
            assert base.old.sha(path)==meta['graph_sha256']==timing['export_hashes'][f'fit_{fit_id:02d}'][arm]
            assert path.stat().st_size==meta['graph_bytes']
            if arm in base.SHARED_ARMS:
                assert meta['route']=='validated_complete_ORT_policy';extra=np.zeros((2,5,2),np.float32)
                session=compiled.session(path);mode='complete'
            else:
                assert meta['route']=='native_core_and_decoder';extra=np.load(path.with_suffix('.extra.npy'))
                if arm=='emc_q_nonpositive':np.testing.assert_array_equal(extra,model.movement_embeddings.detach().numpy())
                else:np.testing.assert_array_equal(extra,np.zeros((2,5,2),np.float32))
                session=compiled.session(path);mode='native'
            opts=session.get_session_options()
            assert session.get_providers()==['CPUExecutionProvider']
            assert opts.intra_op_num_threads==opts.inter_op_num_threads==1
            assert opts.execution_mode==ort.ExecutionMode.ORT_SEQUENTIAL
            assert opts.graph_optimization_level==ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sessions[arm]=(mode,session,extra);models[arm]=model
            with torch.inference_mode():
                for state in states:
                    a,q,_=runner.greedy(model,torch.from_numpy(state)[None],arm)
                    if mode=='native':b,v=fused.values((session,extra),arm,state)
                    else:b,v=compiled.compiled_values(session,state)
                    np.testing.assert_array_equal(a[0].numpy(),b[0])
                    np.testing.assert_allclose(q.numpy(),v,rtol=1e-5,atol=1e-6)
                    if runner.bounded(arm):assert float(v[0])<=0
                    vectors+=1
        for round_id in range(spec['timing_rounds']):
            order=list(runner.ARMS);rng.shuffle(order);orders.append(dict(fit=fit_id,round=round_id,order=order))
            for arm in order:
                mode,session,extra=sessions[arm]
                for _ in range(spec['warmups']):
                    state=states[_]
                    if mode=='native':fused.action((session,extra),arm,state)
                    else:compiled.compiled_action(session,state)
                for index,state in enumerate(states):
                    start=runner.time.perf_counter_ns()
                    action=fused.action((session,extra),arm,state) if mode=='native' else compiled.compiled_action(session,state)
                    elapsed=runner.time.perf_counter_ns()-start
                    assert action.shape==(32,) and np.all((0<=action)&(action<5))
                    expected.append((fit_id,round_id,arm,index,elapsed))
    assert report['orders']==orders and len(raw)==len(expected)==len(fits)*spec['timing_rounds']*len(runner.ARMS)*128
    for actual,want in zip(raw,expected):
        fit_id,round_id,arm,index,elapsed=want
        assert (actual['fit'],actual['round'],actual['arm'],actual['observation'])==(fit_id,round_id,arm,index)
        assert actual['nanoseconds']>0
    computed={}
    for arm in runner.ARMS:
        computed[arm]=[]
        for fit_id in range(len(fits)):
            ns=np.asarray([x['nanoseconds'] for x in raw if x['fit']==fit_id and x['arm']==arm],np.float64)
            expected_row=dict(fit=fit_id,median_ms=float(np.median(ns/1e6)),p95_ms=float(np.quantile(ns/1e6,.95)),
                parameter_bytes=exports[f'fit_{fit_id:02d}'][arm]['parameter_bytes'],calls=len(ns),
                route=exports[f'fit_{fit_id:02d}'][arm]['route'])
            assert report['metrics'][arm][fit_id]==expected_row
            computed[arm].append(expected_row)
    screens={}
    for fit_id in range(len(fits)):
        primary=computed['amortized_workload_price24'][fit_id];compact=computed['additive64_nonpositive'][fit_id];wide=computed['additive128_unbounded'][fit_id]
        screens[str(fit_id)]=dict(storage=primary['parameter_bytes']<=.5*compact['parameter_bytes'],
            median=primary['median_ms']<=wide['median_ms'],p95=primary['p95_ms']<=1.)
    assert report['screens_per_fit']==screens
    assert report['all_fit_cost_screens_pass']==all(all(x.values()) for x in screens.values())
    return dict(complete_actions_replayed=vectors,timing_rows_recomputed=len(raw),
        frozen_fit_count=len(fits),all_weight_hashes_verified=True,all_screens_recomputed=True)

def audit_research(folder):
    """Audit a completed full development/final run. Never run from CLI in this gate."""
    base.configure();runner.patch_training_core();folder=Path(folder);manifest=verify_manifest(folder);plan=runner.read(folder/'runtime_plan.json')
    p=runner.protocol();phase='final' if plan.get('phase')=='final' else 'development'
    assert plan['kind']=='RESEARCH_NOT_LAUNCHED' and not plan['fixture']
    assert plan['protocol_sha256']==base.old.sha(runner.PROTOCOL)
    expected=p[phase];fits=[(seed,block['start'],i) for i,(seed,block) in enumerate(zip(expected['model_seeds'],expected['train_blocks']))]
    assert plan['model_seeds']==expected['model_seeds'] and plan['train_blocks']==expected['train_blocks']
    assert plan['evaluation_seeds']==(list(range(expected['evaluation_seeds']['start'],expected['evaluation_seeds']['start']+expected['evaluation_seeds']['count'])) if isinstance(expected['evaluation_seeds'],dict) else expected['evaluation_seeds'])
    for name,digest in plan['source_sha256'].items():assert base.old.sha(runner.ROOT/name)==base.old.sha(folder/'source_snapshot'/name)==digest
    assert plan['inherited_v2_protocol_sha256']==base.old.sha(runner.ENGINEERING_PROTOCOL)
    assert plan['compatibility_protocol_sha256']==base.old.sha(runner.COMPAT_PROTOCOL)
    for name,digest in runner.read(runner.ROOT/'validation/controller_v3_runtime_compatibility_2026-11-05_01_COMPLETION_MANIFEST.json')['files'].items():
        path=folder/'PROJECT_STATUS_BEFORE.json' if name=='LOW_COMPLEXITY_PROJECT_STATUS.json' else runner.ROOT/name
        assert base.old.sha(path)==digest,name
    frozen=runner.read(folder/'ALL_FINAL_CHECKPOINTS_FREEZE.json');assert frozen['count']==55 and frozen['before_evaluation']
    events=runner.read(folder/'events.json')
    freeze_event=next(x['sequence'] for x in events if x['stage']=='all_final_checkpoints_frozen')
    inference_event=next(x['sequence'] for x in events if x['stage']=='frozen_checkpoint_inference_gate_complete')
    heldout_event=next(x['sequence'] for x in events if x['stage']=='heldout_evaluation_started')
    assert freeze_event<inference_event<heldout_event
    fit_hashes={};training_workloads={};evaluation_workloads={};evaluations={a:[] for a in runner.ARMS};heuristics={}
    episode_count=0;update_count=0
    episodes=plan['episodes'];expected_slots=episodes*40;expected_updates=expected_slots-1024+1
    target_interval=200
    for arm in runner.ARMS:
        evaluations[arm]=[]
    for fit_id,(model_seed,train_start,_) in enumerate(fits):
        for arm in runner.ARMS:
            d=folder/arm/f'fit_{fit_id:02d}';checkpoint=d/'final.pt';digest=base.old.sha(checkpoint)
            key=f'{arm}_fit_{fit_id:02d}';assert frozen['checkpoints'][key]==digest
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert (ck['arm'],ck['model_seed'],ck['slots'],ck['updates'],ck['copies'])==(arm,model_seed,expected_slots,expected_updates,expected_updates//target_interval)
            assert base.old.digest(ck['online'])==ck['online_parameter_hash']
            assert base.old.digest(ck['target'])==ck['target_parameter_hash']
            fit_hashes[key]=digest
            log_path=d/'optimizer.jsonl';logs=[json.loads(x) for x in log_path.read_text().splitlines()]
            assert len(logs)==expected_updates
            for i,row in enumerate(logs):
                assert row['step']==i+1 and row['physical_slots']==i+1024
                assert row['target_copy']==((i+1)%target_interval==0)
                assert len(row['replay_indices'])==128 and max(row['replay_indices'])<i+1024
                assert np.isfinite([row['loss'],row['grad_norm_before_clip']]).all()
                update_count+=1
            for ep in range(episodes):
                name=f'train_{ep:04d}.json.gz'
                with gzip.open(d/name,'rt',encoding='utf-8') as stream:row=json.load(stream)
                assert row['model_seed']==model_seed and row['seed']==train_start+ep
                check_episode(row,base.Chapter3Config());states,_=verify_arrays(d,name,row);episode_count+=1
                training_workloads.setdefault((fit_id,ep),[]).append(row['workload_sha256'])
            validation_expected=5*(1+episodes//50 if episodes>=50 else 0)
            if phase=='development':
                validation_files=list(d.glob('validation_*.json.gz'))
                assert len(validation_files)==validation_expected
                for path in validation_files:
                    with gzip.open(path,'rt',encoding='utf-8') as stream:row=json.load(stream)
                    check_episode(row,base.Chapter3Config());episode_count+=1
            rows=[]
            for seed in plan['evaluation_seeds']:
                name=f'evaluation_{seed}.json.gz'
                with gzip.open(d/name,'rt',encoding='utf-8') as stream:row=json.load(stream)
                assert row['model_seed']==model_seed and row['seed']==seed
                check_episode(row,base.Chapter3Config());states,_=verify_arrays(d,name,row);episode_count+=1
                x=torch.tensor(states);actions=torch.tensor([s['action'] for s in row['slots']])
                model=runner.make_model(arm,model_seed);model.load_state_dict(ck['online'])
                target=runner.make_model(arm,model_seed);target.load_state_dict(ck['target'])
                with torch.inference_mode():
                    q=runner.chosen(model,x,actions,arm).numpy();tq=runner.chosen(target,x,actions,arm).numpy()
                np.testing.assert_allclose(q,row['chosen_q'],rtol=1e-5,atol=1e-5)
                np.testing.assert_allclose(tq,row['target_q'],rtol=1e-5,atol=1e-5)
                if runner.bounded(arm):assert np.max(q)<=0 and np.max(tq)<=0
                evaluation_workloads.setdefault((fit_id,seed),[]).append(row['workload_sha256'])
                rows.append(runner.compact_eval(row))
            evaluations[arm].extend(rows)
    assert all(len(set(v))==1 for v in training_workloads.values())
    assert all(len(set(v))==1 for v in evaluation_workloads.values())
    for name in p['comparisons']['heuristics']:
        heuristic_rows=[]
        for seed in plan['evaluation_seeds']:
            with gzip.open(folder/f'heuristic_{name}_{seed}.json.gz','rt',encoding='utf-8') as stream:row=json.load(stream)
            check_episode(row,base.Chapter3Config());episode_count+=1;heuristic_rows.append(runner.compact_eval(row))
        heuristics[name]=heuristic_rows
    analysis=runner.read(folder/'comparisons_and_value_stability.json')
    assert analysis['comparisons']==runner.compare_all(evaluations,heuristics)
    values={arm:[runner.value_summary_by_time([x for x in evaluations[arm] if x['model_seed']==seed]) for seed in plan['model_seeds']] for arm in runner.ARMS}
    assert analysis['value_stability']==values
    screens=runner.quality_screens(analysis['comparisons'],values,5,analysis['final_primary_bootstrap'],phase)
    assert analysis['quality_screens']==screens
    if phase=='final':
        bootstrap=runner.crossed_bootstrap(evaluations['amortized_workload_price24'],evaluations['additive128_unbounded'])
        assert analysis['final_primary_bootstrap']==bootstrap
    inf=audit_frozen_inference(folder,fits)
    result=dict(state='passed_v3_research_saved_artifact_audit',phase=phase,run_kind=plan['kind'],
        checkpoints_checked=len(fit_hashes),episodes_checked=episode_count,optimizer_updates_checked=update_count,
        training_workload_blocks_paired=True,evaluation_workloads_paired=True,
        all_failed_fits_retained=True,quality_screens_recomputed=True,final_inference=inf,
        manifest_entries=len(manifest),research_claim_established=all(screens.values()),
        thesis_changed=False,scope=exclusion_check())
    base.write(folder/'saved_artifact_audit.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--research',action='store_true');a=p.parse_args()
    print(json.dumps(audit_research(a.folder) if a.research else audit_fixture(a.folder),indent=2))
