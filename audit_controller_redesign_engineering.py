"""Independent saved-artifact checks. No simulator steps or optimizer updates."""
import argparse,ast,gzip,json
from pathlib import Path
import numpy as np
import torch
import run_controller_redesign_engineering as r
from audit_value_bound_development import check_episode

def read(p):
    with (gzip.open(p,'rt',encoding='utf-8') if str(p).endswith('.gz') else open(p,encoding='utf-8')) as f:return json.load(f)

def verify_manifest(folder):
    manifest=read(folder/'run_manifest.json')['files']
    actual={str(p.relative_to(folder)).replace('\\','/') for p in folder.rglob('*') if p.is_file()}
    assert actual-set(manifest)<={'run_manifest.json','saved_artifact_audit.json'},'unmanifested artifact'
    for name,h in manifest.items():assert r.old.sha(folder/name)==h,name
    return manifest

def exclusion_check():
    cfg=r.Chapter3Config()
    paths=['Source/chapter3_iov_env.py','Source/chapter3_iov_env_v2.py','controller_redesign_v2.py']
    checked=[]
    for name in paths:
        tree=ast.parse((r.ROOT/name).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            identifier=node.id if isinstance(node,ast.Name) else node.attr if isinstance(node,ast.Attribute) else ''
            assert 'cloud' not in identifier.lower() and 'charg' not in identifier.lower(),(name,identifier)
        checked.append(name)
    assert cfg.num_vehicles==30
    assert not any('cloud' in k.lower() or 'charg' in k.lower() for k in vars(cfg))
    return dict(excluded=True,active_source_paths=checked,execution_destinations=['local','UAV1','UAV2','RSU1','RSU2'],
                cloud_nodes=0,charging_stations=0,security_trust_enabled=False)

def verify_arrays(folder,name,row):
    p=(folder/name).with_suffix('').with_suffix('.npz')
    arrays=np.load(p)
    states=arrays['states'];following=arrays['next_states']
    assert states.shape==following.shape==(40,533)
    assert np.isfinite(states).all() and np.isfinite(following).all()
    np.testing.assert_array_equal(following[:-1],states[1:])
    return states,following

def audit_fixture(folder):
    r.configure();folder=Path(folder);manifest=verify_manifest(folder)
    plan=read(folder/'runtime_plan.json');assert plan['run_kind']=='SYNTHETIC_NOT_RESEARCH'
    assert plan['model_seed']==99001 and plan['train_seeds']==[990000000] and plan['evaluation_seeds']==[990000100]
    assert plan['cfg']==r.fixture_cfg() and plan['design_sha256']==r.SPEC_SHA and not plan['research_launch']
    for name,h in plan['source_sha256'].items():assert r.old.sha(folder/'source_snapshot'/name)==h
    frozen=read(folder/'FRESH_GATE_FREEZE.json');assert set(frozen)==set(r.ARMS)
    events=read(folder/'events.json')
    freeze_index=next(e['sequence'] for e in events if e['phase']=='all_checkpoints_frozen')
    assert all(e['sequence']>freeze_index for e in events if e['phase']=='synthetic_post_freeze_evaluation')
    summary=read(folder/'verification.json');assert summary['training_slots']==320 and summary['optimizer_updates']==264
    invariance=read(folder/'evaluation_invariance.json');assert len(invariance)==24
    assert all(x['before']==x['after'] for x in invariance)
    fingerprints={};actors=[];fresh={};heuristics={};episodes=0;optimizer_rows=0;cfg=r.Chapter3Config()
    for arm in r.ARMS:
        d=folder/arm;assert r.old.sha(d/'final.pt')==frozen[arm]==summary['checkpoints'][arm]
        ck=torch.load(d/'final.pt',map_location='cpu',weights_only=False)
        assert (ck['arm'],ck['model_seed'],ck['slots'],ck['updates'],ck['copies'])==(arm,99001,40,33,4)
        assert ck['cfg']==plan['cfg'] and ck['replay_position']==40
        actors.append(r.old.digest(ck['actor_history']))
        assert len(ck['actor_history'])==40
        agent=r.build(arm,99001,plan['cfg'])
        initial=agent['online']
        # Initial validation was generated before the sole synthetic fit.
        row=read(d/'validation_0.json.gz');states,_=verify_arrays(d,'validation_0.json.gz',row)
        with torch.inference_mode():
            at=torch.tensor([s['action'] for s in row['slots']],dtype=torch.long)
            np.testing.assert_allclose(r.chosen(initial,torch.tensor(states),at,arm).numpy(),row['chosen_q'],rtol=1e-5,atol=1e-5)
        agent['online'].load_state_dict(ck['online']);agent['target'].load_state_dict(ck['target'])
        agent['opt'].load_state_dict(ck['optimizer'])
        for replay_name,value in ck['replay'].items():assert np.isfinite(value).all()
        for name in ['train_ep1.json.gz','validation_0.json.gz','validation_1.json.gz','evaluation.json.gz']:
            row=read(d/name);assert row['run_kind']=='SYNTHETIC_NOT_RESEARCH';check_episode(row,cfg);episodes+=1
            states,ns=verify_arrays(d,name,row)
            fingerprints.setdefault(name,[]).append(row['workload_sha256'])
            if name=='train_ep1.json.gz':
                np.testing.assert_array_equal(ck['replay']['states'],states)
                np.testing.assert_array_equal(ck['replay']['next_states'],ns)
                np.testing.assert_array_equal(ck['replay']['actions'],[s['action'] for s in row['slots']])
                np.testing.assert_allclose(ck['replay']['rewards'],row['slot_rewards'],rtol=1e-6,atol=1e-6)
                np.testing.assert_array_equal(ck['replay']['dones'],[False]*39+[True])
            else:
                assert len(row['chosen_q'])==len(row['target_q'])==40
                np.testing.assert_allclose(row['realized_return_to_go'],np.cumsum(row['slot_rewards'][::-1])[::-1],rtol=0,atol=1e-9)
            if name in ['validation_1.json.gz','evaluation.json.gz']:
                at=torch.tensor([s['action'] for s in row['slots']],dtype=torch.long)
                with torch.inference_mode():
                    q=r.chosen(agent['online'],torch.tensor(states),at,arm).numpy()
                    target=r.chosen(agent['target'],torch.tensor(states),at,arm).numpy()
                np.testing.assert_allclose(q,row['chosen_q'],rtol=1e-5,atol=1e-5)
                np.testing.assert_allclose(target,row['target_q'],rtol=1e-5,atol=1e-5)
                if r.bounded(arm):assert max(row['chosen_q'])<=0 and max(row['target_q'])<=0
            if name=='evaluation.json.gz':fresh[arm]=[row]
        logs=[json.loads(x) for x in (d/'optimizer.jsonl').read_text().splitlines()]
        assert len(logs)==33
        for i,x in enumerate(logs):
            assert x['step']==i+1 and x['physical_slots']==i+8
            assert x['target_copy']==((i+1)%8==0)
            assert len(x['replay_indices'])==4 and min(x['replay_indices'])>=0 and max(x['replay_indices'])<i+8
            assert np.isfinite([x['loss'],x['grad_norm_before_clip']]).all()
            for field in ['chosen_q','bellman_target','td_residual']:
                assert np.isfinite(list(x[field].values())).all() and x[field]['count']==4
        optimizer_rows+=len(logs)
    assert len(set(actors))==1
    assert all(len(set(v))==1 for v in fingerprints.values())
    for policy in ['all_local','nearest_rsu','uav_selective']:
        row=read(folder/f'heuristic_{policy}.json.gz');check_episode(row,cfg);episodes+=1
        assert row['workload_sha256']==next(iter(fresh.values()))[0]['workload_sha256'];heuristics[policy]=[row]
    analysis=read(folder/'comparisons_and_values.json')
    assert analysis['comparisons']==r.comparisons(fresh,heuristics)
    assert analysis['value_stability']=={arm:r.values_summary(rows) for arm,rows in fresh.items()}
    for arm,rows in fresh.items():
        qs=np.array([q for row in rows for q in row['chosen_q']]);gs=np.array([g for row in rows for g in row['realized_return_to_go']])
        stats=analysis['value_stability'][arm];residual=qs-gs
        assert abs(stats['absolute_bias']-abs(float(residual.mean())))<1e-9
        assert abs(stats['normalized_rmse']-float(np.sqrt((residual**2).mean())/max(1.,np.sqrt((gs**2).mean()))))<1e-9
        assert stats['near_zero_large_return_fraction']==float(((abs(qs)<=1e-6)&(abs(gs)>=1.)).mean())
    result=dict(state='passed_synthetic_saved_artifact_audit',run_kind='SYNTHETIC_NOT_RESEARCH',research_quality_evidence=False,
        checkpoints_checked=8,episodes_checked=episodes,optimizer_rows_checked=optimizer_rows,
        manifest_entries_checked=len(manifest),training_slots_checked=320,simulator_steps_added=0,optimizer_updates_added=0,
        source_scope=exclusion_check())
    r.write(folder/'saved_artifact_audit.json',result)
    return result

def audit_cost(folder):
    r.configure();folder=Path(folder);manifest=verify_manifest(folder);result=read(folder/'verification.json')
    frozen=read(folder/'PRE_TIMING_FREEZE.json');s=r.spec()
    assert frozen['design_sha256']==r.SPEC_SHA and frozen['hardware']['backend']==result['hardware']['backend']
    assert frozen['rounds']==5 and frozen['warmups']==32 and frozen['calls']==128
    assert result['training_slots']==result['optimizer_updates']==0 and result['probe_slots']==320
    raw=read(folder/'raw_timings.json');assert len(raw)==5*128*8
    common=np.load(folder/'common_observations.npz')['observations'];assert common.shape==(128,533)
    probe_states=[]
    for seed in s['development']['probes']:
        name=f'probe_{seed}.json.gz';row=read(folder/name);check_episode(row,r.Chapter3Config())
        states,_=verify_arrays(folder,name,row);probe_states.extend(states)
    np.testing.assert_array_equal(common,np.stack(probe_states)[np.linspace(0,len(probe_states)-1,128,dtype=int)])
    metrics=result['metrics']
    for arm in r.ARMS:
        readings=[x for x in raw if x['arm']==arm];assert len(readings)==640
        assert {(x['round'],x['observation']) for x in readings}=={(j,i) for j in range(5) for i in range(128)}
        ns=np.array([x['nanoseconds'] for x in readings],dtype=float);assert (ns>0).all()
        assert metrics[arm]['median_ms']==float(np.median(ns/1e6))
        assert metrics[arm]['p95_ms']==float(np.quantile(ns/1e6,.95))
        model=r.build(arm,99001,r.fixture_cfg())['online']
        assert metrics[arm]['parameter_bytes']==r.parameter_bytes(model)
    assert result['screens']==r.cost_screens(metrics,s)
    assert result['state']==('cost_gate_passed' if all(result['screens'].values()) else 'cost_gate_failed')
    audit=dict(state='passed_cost_artifact_audit',cost_gate_state=result['state'],raw_timing_rows=len(raw),manifest_entries_checked=len(manifest),
        simulator_steps_added=0,optimizer_updates_added=0,source_scope=exclusion_check())
    r.write(folder/'saved_artifact_audit.json',audit)
    return audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--cost',action='store_true');a=p.parse_args()
    print(json.dumps(audit_cost(a.folder) if a.cost else audit_fixture(a.folder),indent=2))
