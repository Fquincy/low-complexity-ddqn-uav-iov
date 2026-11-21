"""Independent seventeen-arm audit including transitive factory provenance."""
import argparse,gzip,json
from pathlib import Path
import numpy as np,torch
import onnxruntime as ort
import run_anchor_research_route_v3 as runner
import run_controller_redesign_engineering as base
import controller_v3_fused_runtime as fused
import compiled_controller_inference as compiled
from audit_controller_redesign_engineering import check_episode,verify_arrays,exclusion_check,verify_manifest

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
    audit_timing_freeze(folder)
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
                    if mode=='native':b,v=runner.runtime_values((session,extra),arm,state)
                    else:b,v=compiled.compiled_values(session,state)
                    np.testing.assert_array_equal(a[0].numpy(),b[0])
                    np.testing.assert_allclose(q.numpy(),v,rtol=1e-5,atol=1e-6)
                    if runner.bounded(arm):assert float(v[0])<=0
                    vectors+=1
        for round_id in range(spec['timing_rounds']):
            order=list(runner.ARMS);rng.shuffle(order);orders.append(dict(fit=fit_id,round=round_id,order=order))
            for arm in order:
                for index in range(len(states)):
                    expected.append((fit_id,round_id,arm,index))
    assert report['orders']==orders and len(raw)==len(expected)==len(fits)*spec['timing_rounds']*len(runner.ARMS)*128
    for actual,want in zip(raw,expected):
        fit_id,round_id,arm,index=want
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
        primary=computed['price_both'][fit_id];compact=computed['additive64_nonpositive'][fit_id];wide=computed['additive128_unbounded'][fit_id]
        screens[str(fit_id)]=dict(storage=primary['parameter_bytes']<=.5*compact['parameter_bytes'],
            median=primary['median_ms']<=wide['median_ms'],p95=primary['p95_ms']<=1.)
    assert report['screens_per_fit']==screens
    assert report['all_fit_cost_screens_pass']==all(all(x.values()) for x in screens.values())
    return dict(complete_actions_replayed=vectors,timing_rows_recomputed=len(raw),
        frozen_fit_count=len(fits),all_weight_hashes_verified=True,all_screens_recomputed=True)


def audit_timing_freeze(folder):
    pre=runner.read(folder/'PRE_TIMING_FREEZE.json')
    stage=runner.read(folder/'TIMING_STAGE_ORDER.json')
    assert stage['pre_timing_freeze_sha256']==base.old.sha(folder/'PRE_TIMING_FREEZE.json')
    assert pre['frozen_at_ns']<stage['first_timed_call_ns']<=stage['completed_at_ns']
    payloads={str(p.relative_to(folder)).replace('\\','/') for p in (folder/'exports').rglob('*') if p.is_file()}
    assert payloads==set(pre['payload_hashes'])
    for name,digest in pre['payload_hashes'].items():assert base.old.sha(folder/name)==digest
    assert pre['orders']==runner.read(folder/'verification.json')['orders']
    assert pre['inference_freeze_sha256']==base.old.sha(folder/'PRE_INFERENCE_FREEZE.json')
    spec=runner.protocol()['frozen_checkpoint_inference']
    assert (pre['rounds'],pre['warmups'],pre['calls'],pre['timing_order_seed'])==(
        spec['timing_rounds'],spec['warmups'],spec['calls_per_round'],spec['order_seed'])
    assert pre['backend']==runner.read(folder/'PRE_INFERENCE_FREEZE.json')['backend']
    return True

def audit_research(folder):
    """Audit a completed full development/final run. Never run from CLI in this gate."""
    base.configure();runner.patch_training_core();folder=Path(folder);manifest=verify_manifest(folder);plan=runner.read(folder/'runtime_plan.json')
    p=runner.protocol();phase='final' if plan.get('phase')=='final' else 'development'
    synthetic=plan['kind']=='SYNTHETIC_NOT_RESEARCH'
    runner.validate_plan(plan)
    if synthetic:
        assert plan['fixture'] and not plan['training_launched'] and not plan['quality_screen_authority']
        if len(plan['model_seeds'])==2:
            assert {k:plan[k] for k in runner.synthetic_multifit_plan()}==runner.synthetic_multifit_plan()
        else:
            assert plan['model_seeds']==[99131] and plan['train_blocks']==[dict(start=991310000,count=1)]
            assert plan['evaluation_seeds']==[991310100] and plan['cfg']==runner.FIXTURE_CFG
            assert not plan['diagnostic_validation_seeds'] and plan['episodes']==1 and plan['validation_interval']==50
    else:
        assert plan['kind']=='RESEARCH' and not plan['fixture'] and plan['training_launched']
        assert plan['cfg']==runner.research_cfg() and plan['episodes']==200 and plan['validation_interval']==50
        assert plan['diagnostic_validation_seeds']==p[phase].get('diagnostic_validation_seeds',[])
    assert plan['protocol_sha256']==base.old.sha(runner.PROTOCOL)
    assert plan['runtime_route_amendment_sha256']==base.old.sha(runner.ROOT/'ANCHOR_RUNTIME_ROUTE_AMENDMENT_2026-11-06.json')
    expected=plan if synthetic else p[phase];fits=[(seed,block['start'],i) for i,(seed,block) in enumerate(zip(expected['model_seeds'],expected['train_blocks']))]
    assert plan['model_seeds']==expected['model_seeds'] and plan['train_blocks']==expected['train_blocks']
    assert plan['evaluation_seeds']==(list(range(expected['evaluation_seeds']['start'],expected['evaluation_seeds']['start']+expected['evaluation_seeds']['count'])) if isinstance(expected['evaluation_seeds'],dict) else expected['evaluation_seeds'])
    runner.verify_factory_provenance()
    assert set(plan['source_sha256'])==set(runner.SOURCES)
    for name,digest in plan['source_sha256'].items():assert base.old.sha(runner.ROOT/name)==base.old.sha(folder/'source_snapshot'/name)==digest
    assert plan['inherited_v2_protocol_sha256']==base.old.sha(runner.ENGINEERING_PROTOCOL)
    assert plan['compatibility_protocol_sha256']==base.old.sha(runner.COMPAT_PROTOCOL)
    runner.verify_prior_engineering()
    assert plan['prior_engineering_completion_manifest_sha256']==base.old.sha(runner.ROOT/'validation/controller_v3_runtime_compatibility_2026-11-05_01_COMPLETION_MANIFEST.json')
    assert plan['prior_engineering_audit_sha256']==base.old.sha(runner.ROOT/'validation/controller_v3_runtime_compatibility_2026-11-05_01/saved_artifact_audit.json')
    frozen=runner.read(folder/'ALL_FINAL_CHECKPOINTS_FREEZE.json')
    expected_keys={f'{arm}_fit_{i:02d}' for i in range(len(fits)) for arm in runner.ARMS}
    assert set(frozen['checkpoints'])==expected_keys and frozen['count']==len(expected_keys) and frozen['before_evaluation']
    assert set(str(x.relative_to(folder)).replace('\\','/') for x in folder.rglob('*.pt'))=={
        f'{arm}/fit_{i:02d}/final.pt' for i in range(len(fits)) for arm in runner.ARMS}
    events=runner.read(folder/'events.json')
    freeze_event=next(x['sequence'] for x in events if x['stage']=='all_final_checkpoints_frozen')
    inference_event=next(x['sequence'] for x in events if x['stage']=='frozen_checkpoint_inference_gate_complete')
    heldout_event=next(x['sequence'] for x in events if x['stage']=='heldout_evaluation_started')
    assert freeze_event<inference_event<heldout_event
    fit_hashes={};training_workloads={};evaluation_workloads={};evaluations={a:[] for a in runner.ARMS};heuristics={}
    episode_count=0;update_count=0
    episodes=plan['episodes'];cfg=plan['cfg'];expected_slots=episodes*40;expected_updates=max(0,expected_slots-cfg['warmup']+1)
    target_interval=cfg['target_interval'];actor_streams={}
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
            assert ck['cfg']==cfg
            assert len(ck['actor_history'])==expected_slots
            actor_streams.setdefault(fit_id,[]).append(base.old.digest((ck['actor_history'],ck['action_rng_state'],ck['replay_rng_state'])))
            fit_hashes[key]=digest
            log_path=d/'optimizer.jsonl';logs=[json.loads(x) for x in log_path.read_text().splitlines()]
            assert len(logs)==expected_updates
            for i,row in enumerate(logs):
                assert row['step']==i+1 and row['physical_slots']==i+cfg['warmup']
                assert row['target_copy']==((i+1)%target_interval==0)
                assert len(row['replay_indices'])==cfg['batch'] and min(row['replay_indices'])>=0 and max(row['replay_indices'])<min(cfg['capacity'],i+cfg['warmup'])
                assert np.isfinite([row['loss'],row['grad_norm_before_clip']]).all()
                for metric in ['chosen_q','bellman_target','td_residual']:
                    assert np.isfinite(list(row[metric].values())).all()
                if arm in runner.CORRECTED:
                    slots=row['physical_slots'];capacity=cfg['capacity']
                    expected_pool={j%capacity for j in range(max(0,slots-capacity),slots) if j%40==39}
                    assert set(row['terminal_pool'])==expected_pool
                    anchored=runner.anchor.ARMS[arm][2]
                    assert all(j in expected_pool for j in row['anchor_indices'])
                    assert len(row['anchor_indices'])==(32 if anchored and expected_pool else 0)
                    assert abs(row['loss']-row['uniform_loss']-row['anchor_loss'])<1e-3
                    assert row['empty_anchor']==bool(anchored and not expected_pool)
                update_count+=1
            for ep in range(episodes):
                name=f'train_{ep:04d}.json.gz'
                with gzip.open(d/name,'rt',encoding='utf-8') as stream:row=json.load(stream)
                assert row['model_seed']==model_seed and row['seed']==train_start+ep and row['episode']==ep and row['run_kind']==plan['kind']
                check_episode(row,base.Chapter3Config());states,_=verify_arrays(d,name,row);episode_count+=1
                training_workloads.setdefault((fit_id,ep),[]).append(row['workload_sha256'])
            expected_validation={f'validation_initial_{seed}.json.gz':(seed,'validation_initial') for seed in plan['diagnostic_validation_seeds']}
            for ep in range(plan['validation_interval'],episodes+1,plan['validation_interval']):
                expected_validation.update({f'validation_{ep}_{seed}.json.gz':(seed,f'validation_{ep}') for seed in plan['diagnostic_validation_seeds']})
            assert {x.name for x in d.glob('validation_*.json.gz')}==set(expected_validation)
            for name,(seed,episode_id) in expected_validation.items():
                with gzip.open(d/name,'rt',encoding='utf-8') as stream:row=json.load(stream)
                assert row['seed']==seed and row['episode']==episode_id and row['model_seed']==model_seed and row['run_kind']==plan['kind']
                check_episode(row,base.Chapter3Config());verify_arrays(d,name,row);episode_count+=1
            assert {x.name for x in d.glob('train_*.json.gz')}=={f'train_{i:04d}.json.gz' for i in range(episodes)}
            assert {x.name for x in d.glob('evaluation_*.json.gz')}=={f'evaluation_{seed}.json.gz' for seed in plan['evaluation_seeds']}
            rows=[]
            for seed in plan['evaluation_seeds']:
                name=f'evaluation_{seed}.json.gz'
                with gzip.open(d/name,'rt',encoding='utf-8') as stream:row=json.load(stream)
                assert row['model_seed']==model_seed and row['seed']==seed and row['run_kind']==plan['kind']
                check_episode(row,base.Chapter3Config());states,_=verify_arrays(d,name,row);episode_count+=1
                x=torch.tensor(states);actions=torch.tensor([s['action'] for s in row['slots']])
                model=runner.make_model(arm,model_seed);model.load_state_dict(ck['online'])
                target=runner.make_model(arm,model_seed);target.load_state_dict(ck['target'])
                with torch.inference_mode():
                    q=runner.chosen(model,x,actions,arm).numpy();tq=runner.chosen(target,x,actions,arm).numpy()
                    # Reproduce original batch-one decisions to preserve FP32 tie behavior.
                    for state,action in zip(x,actions):
                        ga=runner.greedy(model,state[None],arm)[0][0]
                        np.testing.assert_array_equal(ga.numpy(),action.numpy())
                np.testing.assert_allclose(q,row['chosen_q'],rtol=1e-5,atol=1e-5)
                np.testing.assert_allclose(tq,row['target_q'],rtol=1e-5,atol=1e-5)
                if runner.bounded(arm):assert np.max(q)<=0 and np.max(tq)<=0
                evaluation_workloads.setdefault((fit_id,seed),[]).append(row['workload_sha256'])
                rows.append(runner.compact_eval(row))
            evaluations[arm].extend(rows)
    assert all(len(v)==len(runner.ARMS) and len(set(v))==1 for v in training_workloads.values())
    assert all(len(v)==len(runner.ARMS) and len(set(v))==1 for v in actor_streams.values())
    assert all(len(set(v))==1 for v in evaluation_workloads.values())
    for name in p['comparisons']['heuristics']:
        heuristic_rows=[]
        for seed in plan['evaluation_seeds']:
            with gzip.open(folder/f'heuristic_{name}_{seed}.json.gz','rt',encoding='utf-8') as stream:row=json.load(stream)
            assert row['seed']==seed and row['model_seed'] is None and row['run_kind']==plan['kind']
            check_episode(row,base.Chapter3Config());verify_arrays(folder,f'heuristic_{name}_{seed}.json.gz',row);episode_count+=1;heuristic_rows.append(runner.compact_eval(row))
        heuristics[name]=heuristic_rows
    analysis=runner.read(folder/'comparisons_and_value_stability.json')
    assert analysis['comparisons']==runner.compare_all(evaluations,heuristics)
    values={arm:[runner.value_summary_by_time([x for x in evaluations[arm] if x['model_seed']==seed]) for seed in plan['model_seeds']] for arm in runner.ARMS}
    assert analysis['value_stability']==values
    screens=runner.quality_screens(analysis['comparisons'],values,len(fits),analysis['final_primary_bootstrap'],phase) if not synthetic else dict(evaluated=False,reason='synthetic_fixture_excluded')
    assert analysis['quality_screens']==screens
    if not synthetic and phase=='final':
        bootstrap=runner.crossed_bootstrap(evaluations['price_both'],evaluations['additive128_unbounded'])
        assert analysis['final_primary_bootstrap']==bootstrap
    inf=audit_frozen_inference(folder,fits)
    invariance=runner.read(folder/'evaluation_invariance.json')
    assert {(r['arm'],r['fit']) for r in invariance}=={(a,i) for a in runner.ARMS for i in range(len(fits))}
    assert len(invariance)==len(expected_keys) and all(r['before']==r['after'] for r in invariance)
    verification=runner.read(folder/'verification.json')
    passed=False if synthetic else runner.aggregate_screens(screens,runner.read(folder/'FROZEN_CHECKPOINT_INFERENCE.json'))
    assert verification['empirical_protocol_screens_passed']==passed
    expected_state='completed_synthetic_full_fixture_not_research' if synthetic else phase+('_screen_passed' if passed else '_screen_failed')
    assert verification['state']==expected_state and verification['checkpoint_count']==len(expected_keys)
    assert verification['research_training_launched']==(not synthetic)
    assert not verification['policy_quality_claim'] and not verification['thesis_changed']
    assert analysis['quality_claim'] is False and analysis['selection_performed'] is False and analysis['all_fits_retained'] is True
    if synthetic:assert analysis['final_primary_bootstrap'] is None
    for key in expected_keys:
        count=verification['counts'][key]
        assert (count['slots'],count['updates'],count['target_copies'],count['checkpoints'],count['optimizer_rows'])==(
            expected_slots,expected_updates,expected_updates//target_interval,1,expected_updates)
    assert verification['training_slots']==expected_slots*len(expected_keys) and verification['optimizer_updates']==update_count
    assert verification['target_copies']==(expected_updates//target_interval)*len(expected_keys)
    result=dict(state='passed_v3_synthetic_shared_research_audit' if synthetic else 'passed_v3_research_saved_artifact_audit',phase=phase,run_kind=plan['kind'],
        checkpoints_checked=len(fit_hashes),episodes_checked=episode_count,optimizer_updates_checked=update_count,
        training_workload_blocks_paired=True,evaluation_workloads_paired=True,
        all_failed_fits_retained=True,quality_screens_recomputed=True,final_inference=inf,
        manifest_entries=len(manifest),empirical_protocol_screens_passed=passed,research_quality_evidence=not synthetic,
        thesis_changed=False,scope=exclusion_check())
    base.write(folder/'saved_artifact_audit.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--research',action='store_true');a=p.parse_args()
    print(json.dumps(audit_research(a.folder),indent=2))
