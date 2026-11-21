"""Saved-artifact audit of screened paired synthetic simulator fixture."""
import gzip,json
from pathlib import Path
import numpy as np,torch
import run_deadline_screen_iov_fixture as r
from audit_controller_redesign_engineering import check_episode,verify_arrays

def audit(out):
    out=Path(out);r.configure();p=r.read(r.SPEC);manifest=r.read(out/'run_manifest.json')['files']
    actual={str(x.relative_to(out)).replace('\\','/') for x in out.rglob('*') if x.is_file()}
    assert actual-set(manifest)<={'run_manifest.json','saved_artifact_audit.json'}
    for name,digest in manifest.items():assert r.sha(out/name)==digest,name
    pre=r.read(out/'PRE_RUN_FREEZE.json');timing=r.read(out/'PRE_TIMING_FREEZE.json')
    assert pre['protocol_sha256']==r.sha(r.SPEC)
    assert pre['source_sha256']==timing['source_sha256']
    for name,digest in pre['source_sha256'].items():assert r.sha(r.R/name)==r.sha(out/'source_snapshot'/name)==digest,name
    frozen=r.read(out/'ALL_FINAL_CHECKPOINTS_FREEZE.json');assert frozen['before_evaluation']
    keys={f'{arm}_fit_{fit:02d}' for fit in range(2) for arm in p['arms']}
    assert set(frozen['checkpoints'])==keys and len(list(out.rglob('*.pt')))==14
    assert frozen['checkpoints']==timing['checkpoints']
    states=np.load(out/'engineering_observations.npz')['states'];assert r.sha(out/'engineering_observations.npz')==timing['observations_sha256']
    predictions=r.read(out/'predictions.json');workloads={};actors={};replays={};tasks=0;episodes=0;updates=0;actions_checked=0;excluded_checks=0
    for fit,seed in enumerate(p['model_seeds']):
        for arm in p['arms']:
            key=f'{arm}_fit_{fit:02d}';folder=out/arm/f'fit_{fit:02d}';path=folder/'final.pt'
            assert r.sha(path)==frozen['checkpoints'][key]
            ck=torch.load(path,weights_only=False,map_location='cpu');assert (ck['arm'],ck['seed'],ck['slots'],ck['updates'],ck['copies'])==(arm,seed,80,73,9)
            assert len(ck['actor_history'])==80;actors.setdefault(fit,[]).append(ck['actor_history']);replays.setdefault(fit,[]).append(ck['replay_rng_state'])
            model=r.screen.Critic(arm,seed).eval();model.load_state_dict(ck['online']);target=r.screen.Critic(arm,seed).eval();target.load_state_dict(ck['target']);runtime=r.fast.Runtime(model)
            for i,state in enumerate(states):
                a,q=runtime.values(state)
                with torch.inference_mode():b,v=model.greedy(torch.from_numpy(state)[None])
                np.testing.assert_array_equal(a,b.numpy());np.testing.assert_array_equal(a[0],predictions[key]['actions'][i]);np.testing.assert_allclose(q,v.numpy(),rtol=1e-5,atol=1e-6)
                assert float(q[0])==predictions[key]['q'][i];actions_checked+=1
            saved=[]
            for ep_id in range(2):saved.append((f'train_{ep_id:02d}.json.gz',p['training_starts'][fit]+ep_id,True,ep_id))
            for task_seed in p['evaluation_seeds']:saved.append((f'evaluation_{task_seed}.json.gz',task_seed,False,'evaluation'))
            for name,task_seed,training,ep_id in saved:
                with gzip.open(folder/name,'rt',encoding='utf-8') as stream:row=json.load(stream)
                assert row['run_kind']=='SYNTHETIC_NOT_RESEARCH' and row['seed']==task_seed and row['model_seed']==seed and row['episode']==ep_id
                check_episode(row,r.ep.Chapter3Config());x,ns=verify_arrays(folder,name,row);episodes+=1
                workloads.setdefault((fit,task_seed),[]).append(row['workload_sha256'])
                masks=r.screen.allowed(x)
                for i,slot in enumerate(row['slots']):
                    action=np.array(slot['action']);assert masks[i,np.arange(32),action].all()
                    np.testing.assert_array_equal(slot['excluded_local'],~masks[i,:30,0])
                    for task in slot['tasks_arrived']:
                        vehicle=task['vehicle_id'];tasks+=1
                        if not masks[i,vehicle,0]:assert task['required_cycles']/1.5e9>task['deadline_s']+1e-10;excluded_checks+=1
                    if training:
                        entry=ck['actor_history'][ep_id*40+i]
                        if entry['explore']:
                            expected=r.screen.explore(x[i:i+1],np.array(entry['branch_uniforms'])[None])[0];np.testing.assert_array_equal(action,expected)
                    else:
                        a,q=runtime.values(x[i]);np.testing.assert_array_equal(action,a[0])
                        with torch.inference_mode():
                            act=torch.from_numpy(action)[None];tq=target(torch.from_numpy(x[i:i+1]),act)
                        np.testing.assert_allclose(q[0],row['chosen_q'][i],rtol=1e-5,atol=1e-6);np.testing.assert_allclose(tq.numpy()[0],row['target_q'][i],rtol=1e-5,atol=1e-6)
            rng=np.random.default_rng(seed+100000)
            for slot,entry in enumerate(ck['actor_history']):
                u=float(rng.random());uniforms=rng.random(32);eps=1. if slot<8 else 1-.95*min(slot/40,1.)
                assert entry==dict(warmup=slot<8,epsilon=eps,explore=bool(slot<8 or u<eps),epsilon_uniform=u,branch_uniforms=uniforms.tolist())
            assert rng.bit_generator.state==ck['action_rng_state']
            logs=[json.loads(x) for x in (folder/'optimizer.jsonl').read_text().splitlines()];assert len(logs)==73
            rr=np.random.default_rng(seed+200000);ar=np.random.default_rng(seed+300000)
            for step,entry in enumerate(logs,1):
                slot=step+7;assert entry['step']==step and entry['physical_slots']==slot and entry['target_copy']==(step%8==0)
                np.testing.assert_array_equal(rr.integers(0,min(64,slot),4),entry['replay_indices'])
                pool={i%64 for i in range(max(0,slot-64),slot) if i%40==39};assert set(entry['terminal_pool'])==pool
                expected=ar.choice(sorted(pool),32,replace=True).tolist() if model.anchored and pool else []
                assert entry['anchor_indices']==expected and entry['empty_anchor']==bool(model.anchored and not pool)
                assert np.isfinite([entry['loss'],entry['grad_norm_before_clip'],*entry['bellman_targets']]).all()
                assert abs(entry['loss']-entry['uniform_loss']-entry['anchor_loss'])<1e-4
                updates+=1
            assert rr.bit_generator.state==ck['replay_rng_state']
    assert all(len(v)==7 and len(set(v))==1 for v in workloads.values())
    assert all(all(row==v[0] for row in v) for v in actors.values())
    assert all(all(row==v[0] for row in v) for v in replays.values())
    inv=r.read(out/'evaluation_invariance.json');assert {x['key'] for x in inv}==keys and len(inv)==14 and all(x['before']==x['after'] for x in inv)
    raw=r.read(out/'raw_timings.json');cost=r.read(out/'cost.json');rng=np.random.default_rng(p['order_seed']);orders=[]
    for fit in range(2):
        for round_id in range(5):
            arms=list(p['arms']);rng.shuffle(arms);orders.append(dict(fit=fit,round=round_id,arms=arms))
    assert orders==timing['orders'];expected=[(x['fit'],x['round'],arm,i) for x in orders for arm in x['arms'] for i in range(128)]
    assert [(x['fit'],x['round'],x['arm'],x['index']) for x in raw]==expected and len(raw)==8960
    for key,m in cost['metrics'].items():
        arm,fit=key.rsplit('_fit_',1);ns=np.array([x['nanoseconds'] for x in raw if x['arm']==arm and x['fit']==int(fit)],np.float64)
        assert len(ns)==640 and np.all(ns>0)
        assert m['median_ms']==float(np.median(ns)/1e6) and m['p95_ms']==float(np.quantile(ns,.95)/1e6)
        ck=torch.load(out/arm/f'fit_{fit}'/'final.pt',weights_only=False,map_location='cpu')
        model=r.screen.Critic(arm,p['model_seeds'][int(fit)]);model.load_state_dict(ck['online'])
        assert m['parameter_bytes']==r.screen.existing.base.parameter_bytes(model)
    screens=[]
    for fit in range(2):
        pr=cost['metrics'][f'price_both_fit_{fit:02d}'];c=cost['metrics'][f'compact_both_fit_{fit:02d}'];w=cost['metrics'][f'additive128_unbounded_fit_{fit:02d}']
        screens.append(dict(fit=fit,storage=pr['parameter_bytes']<=.5*c['parameter_bytes'],median=pr['median_ms']<=w['median_ms'],p95=pr['p95_ms']<=1.))
    assert screens==cost['screens']==r.read(out/'verification.json')['cost_screens']
    assert (episodes,tasks,updates,actions_checked)==(56,67200,1022,1792)
    result=dict(state='passed_paired_synthetic_simulator_artifact_audit',checkpoints=14,episodes=episodes,task_records_checked=tasks,
        certified_infeasible_local_records_checked=excluded_checks,optimizer_rows_checked=updates,engineering_action_checks=actions_checked,
        workload_pairing=True,action_uniform_rng_pairing=True,replay_rng_pairing=True,evaluation_invariance=True,
        timing_rows_recomputed=8960,cost_screens=screens,research_launched=False)
    r.write(out/'saved_artifact_audit.json',result);return result
if __name__=='__main__':
    import sys
    print(json.dumps(audit(Path(sys.argv[1]))))
