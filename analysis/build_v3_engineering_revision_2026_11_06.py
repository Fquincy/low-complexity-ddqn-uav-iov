"""Create separately versioned orchestration fixes; retain November 5 sources."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
s = (root/'run_v3_research_preparation.py').read_text()
s = s.replace('run_v3_research_preparation', 'run_v3_research_gate_v2').replace('audit_v3_research_preparation', 'audit_v3_research_gate_v2').replace('test_v3_research_preparation', 'test_v3_research_gate_v2')
s = s.replace("'controller_v3_runtime_compatibility.py',", "'controller_v3_runtime_compatibility.py','compiled_controller_inference.py',")
s = s.replace("def compare_all(evaluations,heuristics):", "def compare_all(evaluations,heuristics):\n    validate_comparison_rows(evaluations,heuristics)")
s = s.replace("def crossed_bootstrap(primary,reference,draws=5000,seed=520600002):", "def crossed_bootstrap(primary,reference,draws=5000,seed=520600002):\n    validate_unique_rows(primary);validate_unique_rows(reference)")
s = s.replace("p=protocol();screens={};primary='amortized_workload_price24'", "p=protocol();screens={};primary='amortized_workload_price24'\n    required=set(p['comparisons']['primary_against_each_of']+p['comparisons']['heuristics']+['amortized_no_price24'])\n    if not required.issubset(comparisons) or any(len(comparisons[a]['per_fit'])!=5 for a in required):\n        raise ValueError('incomplete comparison fit set')")
# Export all frozen models, then freeze graph/decoder bytes and order, then time.
s = s.replace('orders=[]\n    for fit_id', 'orders=[];all_sessions={}\n    for fit_id', 1)
start = s.index("        for round_id in range(spec['timing_rounds']):", s.index('def frozen_inference_gate'))
end = s.index('    metrics={}', start)
s = s[:start] + '''        all_sessions[fit_id]=sessions
        del models
    for fit_id in range(len(fits)):
        for round_id in range(spec['timing_rounds']):
            order=list(ARMS);order_rng.shuffle(order)
            orders.append(dict(fit=fit_id,round=round_id,order=order))
    graph_hashes={fit:{arm:data['graph_sha256'] for arm,data in items.items()} for fit,items in exports.items()}
    payload_hashes={str(path.relative_to(folder)).replace('\\\\','/'):base.old.sha(path)
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
                return fused.action(runtime[1],arm,state) if runtime[0]=='native' else compiled.compiled_action(runtime[1],state)
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
''' + s[end:]
s = s.replace("configure();patch_training_core();p=protocol();out=Path(output);out.mkdir(parents=True,exist_ok=False)", "if not fixture and not RESEARCH_ENABLED:raise RuntimeError('research execution disabled in implementation gate')\n    configure();patch_training_core();p=protocol();out=Path(output)")
s = s.replace('def run_plan(output,fixture=False,phase=None):','def run_plan(output,fixture=False,phase=None,multi_fixture=False):')
start=s.index('    cost_completion=read(');end=s.index('    cost_audit=read(',start)
s=s[:start]+'''    if multi_fixture:
        if not fixture:raise ValueError('synthetic override requires fixture mode')
        plan=synthetic_multifit_plan()
    plan.setdefault('validation_interval',50)
    validate_plan(plan)
    verify_prior_engineering()
    out.mkdir(parents=True,exist_ok=False)
'''+s[end:]
s=s.replace("kind='RESEARCH_NOT_LAUNCHED'", "kind='RESEARCH'")
s=s.replace("if (episode_index+1)%50==0 or", "if (episode_index+1)%plan['validation_interval']==0 or")
s=s.replace("and (episode_index+1)%50==0:", "and (episode_index+1)%plan['validation_interval']==0:")
s=s.replace("screens=quality_screens(comparisons,values,5,bootstrap,phase) if not fixture", "screens=quality_screens(comparisons,values,len(fits),bootstrap,phase) if not fixture")
s=s.replace("('development_screen_passed' if all(screens.values()) else 'development_screen_failed')", "(phase+'_screen_passed' if aggregate_screens(screens,inference_report) else phase+'_screen_failed')")
s=s.replace("thesis_changed=False,all_failed_fits_retained=True)", "thesis_changed=False,all_failed_fits_retained=True,\n            empirical_protocol_screens_passed=False if fixture else aggregate_screens(screens,inference_report))")
s=s.replace("choices=['fixture','development','final']", "choices=['fixture','integration-fixture','development','final']")
s=s.replace("if a.mode!='fixture':", "if a.mode not in ('fixture','integration-fixture'):")
s=s.replace("result=training_fixture(a.output)\n    from audit_v3_research_gate_v2 import audit_fixture\n    print(json.dumps(dict(runner=result,audit=audit_fixture(a.output)),indent=2))", "result=run_plan(a.output,fixture=True,multi_fixture=a.mode=='integration-fixture')\n    from audit_v3_research_gate_v2 import audit_research\n    print(json.dumps(dict(runner=result,audit=audit_research(a.output)),indent=2))")
helpers='''
PRIOR_STATUS_ARCHIVE=ROOT/'validation/controller_v3_research_fixture_2026-11-05_04/PROJECT_STATUS_BEFORE.json'

def verify_prior_engineering():
    manifest=read(ROOT/'validation/controller_v3_runtime_compatibility_2026-11-05_01_COMPLETION_MANIFEST.json')
    for name,digest in manifest['files'].items():
        path=PRIOR_STATUS_ARCHIVE if name=='LOW_COMPLEXITY_PROJECT_STATUS.json' else ROOT/name
        if base.old.sha(path)!=digest:raise ValueError('engineering evidence changed:'+name)
    return manifest

def synthetic_multifit_plan():
    return dict(kind='SYNTHETIC_NOT_RESEARCH',model_seeds=[99261,99267],
        train_blocks=[dict(start=992610000,count=2),dict(start=992670000,count=2)],
        evaluation_seeds=[992690100,992690101],cfg=dict(FIXTURE_CFG,episodes=2,epsilon_decay=80),
        episodes=2,diagnostic_validation_seeds=[992690200],validation_interval=1,
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

'''
s=s.replace('def main():',helpers+'def main():')
(root/'run_v3_research_gate_v2.py').write_text(s,encoding='utf-8')

a=(root/'audit_v3_research_preparation.py').read_text().replace('import run_v3_research_preparation as runner','import run_v3_research_gate_v2 as runner')
# Keep legacy fixture auditor out of the new entrypoint; all fixture dimensions
# go through the same plan-driven research auditor below.
start=a.index('def audit_fixture(');end=a.index('def audit_frozen_inference(',start)
a=a[:start]+a[end:]
start=a.index("                mode,session,extra=sessions[arm]",a.index("        for round_id in range(spec['timing_rounds']):"))
end=a.index("    assert report['orders']",start)
a=a[:start]+'''                for index in range(len(states)):
                    expected.append((fit_id,round_id,arm,index))
'''+a[end:]
a=a.replace('fit_id,round_id,arm,index,elapsed=want','fit_id,round_id,arm,index=want')
a=a.replace("    exports=report['exports'];vectors=0", "    audit_timing_freeze(folder)\n    exports=report['exports'];vectors=0")
a=a.replace("    assert plan['kind']=='RESEARCH_NOT_LAUNCHED' and not plan['fixture']", "    synthetic=plan['kind']=='SYNTHETIC_NOT_RESEARCH'\n    runner.validate_plan(plan)\n    if synthetic:\n        assert plan['fixture'] and not plan['training_launched'] and not plan['quality_screen_authority']\n        assert {k:plan[k] for k in runner.synthetic_multifit_plan()}==runner.synthetic_multifit_plan() if len(plan['model_seeds'])==2 else plan['model_seeds']==[99131]\n    else:\n        assert plan['kind']=='RESEARCH' and not plan['fixture'] and plan['training_launched']")
a=a.replace("expected=p[phase];fits=", "expected=plan if synthetic else p[phase];fits=")
start=a.index("    for name,digest in runner.read(runner.ROOT/'validation/controller_v3_runtime_compatibility")
end=a.index("    events=runner.read",start)
a=a[:start]+'''    runner.verify_prior_engineering()
    frozen=runner.read(folder/'ALL_FINAL_CHECKPOINTS_FREEZE.json')
    expected_keys={f'{arm}_fit_{i:02d}' for i in range(len(fits)) for arm in runner.ARMS}
    assert set(frozen['checkpoints'])==expected_keys and frozen['count']==len(expected_keys) and frozen['before_evaluation']
    assert set(str(x.relative_to(folder)).replace('\\\\','/') for x in folder.rglob('*.pt'))=={
        f'{arm}/fit_{i:02d}/final.pt' for i in range(len(fits)) for arm in runner.ARMS}
'''+a[end:]
a=a.replace("episodes=plan['episodes'];expected_slots=episodes*40;expected_updates=expected_slots-1024+1\n    target_interval=200", "episodes=plan['episodes'];cfg=plan['cfg'];expected_slots=episodes*40;expected_updates=max(0,expected_slots-cfg['warmup']+1)\n    target_interval=cfg['target_interval'];actor_streams={}")
a=a.replace("            fit_hashes[key]=digest", "            assert ck['cfg']==cfg\n            assert len(ck['actor_history'])==expected_slots\n            actor_streams.setdefault(fit_id,[]).append(base.old.digest((ck['actor_history'],ck['action_rng_state'],ck['replay_rng_state'])))\n            fit_hashes[key]=digest")
a=a.replace("i+1024", "i+cfg['warmup']").replace("len(row['replay_indices'])==128", "len(row['replay_indices'])==cfg['batch']")
a=a.replace("max(row['replay_indices'])<i+cfg['warmup']", "min(row['replay_indices'])>=0 and max(row['replay_indices'])<min(cfg['capacity'],i+cfg['warmup'])")
start=a.index('            validation_expected=');end=a.index('            rows=[]',start)
a=a[:start]+'''            expected_validation={f'validation_initial_{seed}.json.gz':(seed,'validation_initial') for seed in plan['diagnostic_validation_seeds']}
            for ep in range(plan['validation_interval'],episodes+1,plan['validation_interval']):
                expected_validation.update({f'validation_{ep}_{seed}.json.gz':(seed,f'validation_{ep}') for seed in plan['diagnostic_validation_seeds']})
            assert {x.name for x in d.glob('validation_*.json.gz')}==set(expected_validation)
            for name,(seed,episode_id) in expected_validation.items():
                with gzip.open(d/name,'rt',encoding='utf-8') as stream:row=json.load(stream)
                assert row['seed']==seed and row['episode']==episode_id and row['model_seed']==model_seed and row['run_kind']==plan['kind']
                check_episode(row,base.Chapter3Config());verify_arrays(d,name,row);episode_count+=1
            assert {x.name for x in d.glob('train_*.json.gz')}=={f'train_{i:04d}.json.gz' for i in range(episodes)}
            assert {x.name for x in d.glob('evaluation_*.json.gz')}=={f'evaluation_{seed}.json.gz' for seed in plan['evaluation_seeds']}
'''+a[end:]
a=a.replace("                assert row['model_seed']==model_seed and row['seed']==train_start+ep", "                assert row['model_seed']==model_seed and row['seed']==train_start+ep and row['episode']==ep and row['run_kind']==plan['kind']")
a=a.replace("                assert row['model_seed']==model_seed and row['seed']==seed", "                assert row['model_seed']==model_seed and row['seed']==seed and row['run_kind']==plan['kind']")
a=a.replace("                    q=runner.chosen(model,x,actions,arm).numpy();tq=runner.chosen(target,x,actions,arm).numpy()", "                    q=runner.chosen(model,x,actions,arm).numpy();tq=runner.chosen(target,x,actions,arm).numpy()\n                    # Reproduce original batch-one decisions to preserve FP32 tie behavior.\n                    for state,action in zip(x,actions):\n                        ga=runner.greedy(model,state[None],arm)[0][0]\n                        np.testing.assert_array_equal(ga.numpy(),action.numpy())")
a=a.replace("    assert all(len(set(v))==1 for v in training_workloads.values())", "    assert all(len(v)==len(runner.ARMS) and len(set(v))==1 for v in training_workloads.values())\n    assert all(len(v)==len(runner.ARMS) and len(set(v))==1 for v in actor_streams.values())")
a=a.replace("            check_episode(row,base.Chapter3Config());episode_count+=1;heuristic_rows.append(runner.compact_eval(row))", "            assert row['seed']==seed and row['model_seed'] is None and row['run_kind']==plan['kind']\n            check_episode(row,base.Chapter3Config());verify_arrays(folder,f'heuristic_{name}_{seed}.json.gz',row);episode_count+=1;heuristic_rows.append(runner.compact_eval(row))")
a=a.replace("screens=runner.quality_screens(analysis['comparisons'],values,5,analysis['final_primary_bootstrap'],phase)", "screens=runner.quality_screens(analysis['comparisons'],values,len(fits),analysis['final_primary_bootstrap'],phase) if not synthetic else dict(evaluated=False,reason='synthetic_fixture_excluded')")
a=a.replace("    if phase=='final':", "    if not synthetic and phase=='final':")
a=a.replace("    result=dict(state='passed_v3_research_saved_artifact_audit'", "    invariance=runner.read(folder/'evaluation_invariance.json')\n    assert {(r['arm'],r['fit']) for r in invariance}=={(a,i) for a in runner.ARMS for i in range(len(fits))}\n    assert len(invariance)==len(expected_keys) and all(r['before']==r['after'] for r in invariance)\n    verification=runner.read(folder/'verification.json')\n    passed=False if synthetic else runner.aggregate_screens(screens,runner.read(folder/'FROZEN_CHECKPOINT_INFERENCE.json'))\n    assert verification['empirical_protocol_screens_passed']==passed\n    assert verification['training_slots']==expected_slots*len(expected_keys) and verification['optimizer_updates']==update_count\n    assert verification['target_copies']==(expected_updates//target_interval)*len(expected_keys)\n    result=dict(state='passed_v3_synthetic_shared_research_audit' if synthetic else 'passed_v3_research_saved_artifact_audit'")
a=a.replace('research_claim_established=all(screens.values())','empirical_protocol_screens_passed=passed,research_quality_evidence=not synthetic')
a=a.replace('audit_research(a.folder) if a.research else audit_fixture(a.folder)','audit_research(a.folder)')
helpers='''
def audit_timing_freeze(folder):
    pre=runner.read(folder/'PRE_TIMING_FREEZE.json')
    stage=runner.read(folder/'TIMING_STAGE_ORDER.json')
    assert stage['pre_timing_freeze_sha256']==base.old.sha(folder/'PRE_TIMING_FREEZE.json')
    assert pre['frozen_at_ns']<stage['first_timed_call_ns']<=stage['completed_at_ns']
    payloads={str(p.relative_to(folder)).replace('\\\\','/') for p in (folder/'exports').rglob('*') if p.is_file()}
    assert payloads==set(pre['payload_hashes'])
    for name,digest in pre['payload_hashes'].items():assert base.old.sha(folder/name)==digest
    assert pre['orders']==runner.read(folder/'verification.json')['orders']
    assert pre['inference_freeze_sha256']==base.old.sha(folder/'PRE_INFERENCE_FREEZE.json')
    return True

'''
a=a.replace('def audit_research(folder):',helpers+'def audit_research(folder):')
(root/'audit_v3_research_gate_v2.py').write_text(a,encoding='utf-8')
print('Created separately versioned runner and auditor; original files unchanged.')
