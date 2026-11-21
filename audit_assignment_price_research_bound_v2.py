"""Saved full multi-fit protocol audit: no simulation, training or retiming."""
import json
from pathlib import Path
import numpy as np
import torch
import onnx
import run_assignment_price_research_bound as w
import audit_assignment_price_fixture_v2 as old
import audit_horizon_loss_research as analysis_audit

def audit(output):
    out=Path(output);p=w.base.protocol();pl=w.r.read(out/'runtime_plan.json');assert pl==w.plan(pl['mode'])
    manifest=w.r.read(out/'manifest.json')
    assert {x.relative_to(out).as_posix() for x in out.rglob('*') if x.is_file()}==set(manifest)|{'manifest.json'}
    for name,digest in manifest.items():assert w.r.sha(out/name)==digest,name
    freeze=w.r.read(out/'PRE_RUN_FREEZE.json');assert freeze['protocol_sha256']==w.base.SPEC_SHA and freeze['plan_sha256']==w.r.sha(out/'runtime_plan.json')
    for name,digest in freeze['sources'].items():assert w.r.sha(w.r.ROOT/name)==w.r.sha(out/'source_snapshot'/name)==digest
    jobs=[(arm,fit) for arm in w.base.ARMS for fit in range(len(pl['model_seeds']))]
    np.random.default_rng(pl['order_seed']).shuffle(jobs);assert w.r.read(out/'job_order.json')==[list(j) for j in jobs]
    events=w.r.read(out/'events.json');assert [e['sequence'] for e in events]==list(range(len(events)))
    stages=[e['stage'] for e in events]
    assert max(i for i,s in enumerate(stages) if s=='training')<stages.index('all_checkpoints_frozen')<min(i for i,s in enumerate(stages) if s=='export')
    assert max(i for i,s in enumerate(stages) if s=='export')<stages.index('all_exports_frozen')<stages.index('timing')<stages.index('timing_complete')<min(i for i,s in enumerate(stages) if s=='evaluation')
    checkpoints=w.r.read(out/'ALL_CHECKPOINT_FREEZE.json');exports=w.r.read(out/'ALL_EXPORT_FREEZE.json');timing_freeze=w.r.read(out/'PRE_TIMING_FREEZE.json')
    assert checkpoints==exports['checkpoints']==timing_freeze['checkpoints']
    assert exports['graphs']==timing_freeze['graphs'] and exports['equivalence_checks']==len(jobs)*pl['timing']['states']
    startup=w.r.read(out/'startup.json');assert len(startup['fresh_process'])==len(jobs)
    # Windows can reuse a PID after an earlier fresh child exits. Each saved
    # record must instead correspond to one distinct arm/fit subprocess job.
    assert {(x['arm'],x['fit']) for x in startup['fresh_process']}==set(jobs)
    assert len({(x['arm'],x['fit']) for x in startup['fresh_process']})==len(jobs)
    for row in startup['fresh_process']:
        key=f"{row['arm']}/{row['fit']:02d}"
        saved=w.r.read(out/row['arm']/f"fit_{row['fit']:02d}"/'fresh_startup.json')
        assert saved=={k:v for k,v in row.items() if k not in ('arm','fit')}
        assert row['process_id']>0 and row['equivalent'] and row['includes_native_first_compilation']
        assert row['checkpoint_sha256']==checkpoints[key] and row['graph_sha256']==exports['graphs'][key]
        assert all(row[k]>0 for k in ['parent_process_wall_ns','model_load_ns','session_load_ns','first_action_ns','module_start_to_first_action_ns'])
    rng=np.random.default_rng(pl['timing_seed']);orders=[]
    for rnd in range(pl['timing']['rounds']):order=list(jobs);rng.shuffle(order);orders.append(order)
    assert timing_freeze['orders']==[[list(j) for j in order] for order in orders]
    timing=w.r.read(out/'timings.json')
    assert [(x['arm'],x['fit'],x['round'],x['index']) for x in timing]==[(a,f,r,i) for r,order in enumerate(orders) for a,f in order for i in range(pl['timing']['states'])]
    assert all(x['ns']>0 for x in timing)
    counts=w.r.read(out/'counters.json');storage=w.r.read(out/'storage.json');costs={a:[] for a in w.base.ARMS}
    evaluations={a:[] for a in w.base.ARMS};heuristics={};workloads={};eval_workloads={};episodes=0;tasks=0;updates=0;diagnostics=w.r.read(out/'optimizer_distributions.json')
    for arm in w.base.ARMS:
        for fit in range(len(pl['model_seeds'])):
            folder=out/arm/f'fit_{fit:02d}';key=f'{arm}/{fit:02d}';a,ck=w.load_agent(out,arm,fit,pl)
            assert ck['arm']==arm and ck['fit']==fit and ck['model_seed']==pl['model_seeds'][fit] and ck['kind']==pl['kind']
            assert ck['plan_sha256']==w.r.sha(out/'runtime_plan.json') and w.r.sha(folder/'final.pt')==checkpoints[key]
            assert len(list(folder.glob('*.pt')))==1
            slots=pl['train_blocks'][fit]['count']*40;steps=slots-pl['cfg']['warmup']+1;copies=steps//pl['cfg']['target_interval_updates']
            assert (ck['slots'],ck['updates'],ck['copies'])==(slots,steps,copies)
            assert counts[key]['slots']==slots and counts[key]['updates']==steps and counts[key]['copies']==copies and counts[key]['training_wall_ns']>0
            assert all(int(v['step'])==steps for v in ck['optimizer']['state'].values())
            history=[];fingerprints=[]
            for ep in range(pl['train_blocks'][fit]['count']):
                row=old.load(folder/f'train_{ep:03d}.json.gz');assert row['seed']==pl['train_blocks'][fit]['start']+ep and row['run_kind']==pl['kind']
                old.check_episode(row,w.r.Chapter3Config());arrays=np.load(folder/f'train_{ep:03d}.npz');s=arrays['states'];ns=arrays['next_states']
                assert s.shape==ns.shape==(40,533);np.testing.assert_array_equal(ns[:-1],s[1:])
                old.close(w.base.adp.remaining(torch.from_numpy(s)),np.arange(40,0,-1));old.close(w.base.adp.remaining(torch.from_numpy(ns)),np.arange(39,-1,-1))
                history.extend(dict(**slot,state=s[i],next=ns[i]) for i,slot in enumerate(row['slots']))
                fingerprints.append(row['workload_sha256']);episodes+=1;tasks+=len(row['tasks'])
            workloads.setdefault(fit,[]).append(fingerprints)
            replay=np.load(folder/'replay.npz');capacity=pl['cfg']['replay_capacity'];size=min(slots,capacity)
            current={i%capacity:history[i] for i in range(max(0,slots-capacity),slots)}
            assert int(replay['size'])==size and int(replay['position'])==slots%capacity
            assert replay['terminal'].tolist()==sorted(i for i,v in current.items() if v['done'])
            for i,v in current.items():
                np.testing.assert_array_equal(replay['states'][i],v['state']);np.testing.assert_array_equal(replay['next_states'][i],v['next'])
                np.testing.assert_array_equal(replay['actions'][i],v['action']);old.close(replay['rewards'][i],np.float32(v['reward']))
                assert replay['dones'][i]==v['done']
            seed=pl['model_seeds'][fit];actor_rng=np.random.default_rng(seed+100000);assert len(ck['actor_history'])==slots
            for i,record in enumerate(ck['actor_history']):
                u=float(actor_rng.random());uniforms=actor_rng.random(32);warm=i<pl['cfg']['warmup'];eps=1. if warm else 1-.95*min(i/pl['cfg']['epsilon_decay'],1.)
                assert record==dict(warmup=warm,epsilon=eps,explore=bool(warm or u<eps),epsilon_uniform=u,branch_uniforms=uniforms.tolist())
                if record['explore']:np.testing.assert_array_equal(w.base.adp.explore(history[i]['state'][None],uniforms[None],a['definition']['screened'])[0],history[i]['action'])
            replay_rng=np.random.default_rng(seed+200000);anchor_rng=np.random.default_rng(seed+300000)
            anchored=True if arm in w.base.adp.ARMS else a['online'].anchored
            fields=['chosen_Q','bellman_targets','TD_residuals','grad_norm_before_clip','loss','uniform_loss','anchor_loss'];values={k:[] for k in fields};anchor_values=[];n=0;clip=0;optimizer_ns=0
            with (folder/'optimizer.jsonl').open() as stream:
                for line in stream:
                    row=json.loads(line)
                    # Original BDQ uses branchwise Huber, all other arms scalar Huber.
                    scalar=old.huber
                    if row.get('branch_values'):
                        dtype=np.float32;y=np.asarray(row['sample_rewards'],dtype=dtype)+np.asarray(row['next_target_values'],dtype=dtype)
                        old.close(old.huber(np.asarray(row['branch_values'],dtype=dtype)-y[:,None]),row['uniform_loss'])
                        # check_update's scalar-loss check is bypassed with its exact expected scalar,
                        # then restore actual branchwise total for separate audit below.
                        checked=dict(row,uniform_loss=scalar(np.asarray(row['chosen_Q'],dtype=dtype)-y))
                        checked['loss']=checked['uniform_loss']+pl['cfg']['anchor']['weight']*row['anchor_loss']
                        old.check_update(checked,history,pl,a['definition']['screened'],anchored,replay_rng,anchor_rng)
                        old.close(row['loss'],row['uniform_loss']+pl['cfg']['anchor']['weight']*row['anchor_loss'])
                    else:old.check_update(row,history,pl,a['definition']['screened'],anchored,replay_rng,anchor_rng)
                    n+=1;clip+=row['clipped'];optimizer_ns+=row['optimizer_ns'];anchor_values.extend(row['anchor_Q'])
                    for field in fields:values[field].extend(row[field] if isinstance(row[field],list) else [row[field]])
            assert n==steps;updates+=n
            expected=dict(distributions={k:analysis_audit.dist(v) for k,v in values.items()},anchor_Q=None if not anchor_values else analysis_audit.dist(anchor_values),updates=n,clipping_fraction=clip/n,optimizer_ns=optimizer_ns)
            assert diagnostics['arms'][arm][fit]==expected
            graph=onnx.load(str(folder/'final.onnx'));onnx.checker.check_model(graph);assert w.r.sha(folder/'final.onnx')==exports['graphs'][key]
            for tensor,parameter in zip(graph.graph.initializer,a['online'].parameters()):np.testing.assert_array_equal(onnx.numpy_helper.to_array(tensor),parameter.detach().numpy())
            st=storage[key];assert st['parameter_bytes']==sum(t.numel()*t.element_size() for t in a['online'].parameters())
            assert st['buffer_bytes']==sum(t.numel()*t.element_size() for t in a['online'].buffers())
            assert st['target_tensor_bytes']==sum(t.numel()*t.element_size() for t in a['target'].state_dict().values())
            assert st['optimizer_tensor_bytes']==sum(v.numel()*v.element_size() for item in ck['optimizer']['state'].values() for v in item.values() if isinstance(v,torch.Tensor))
            per_record=2*533*4+32+8
            assert st['replay_allocated_tensor_bytes']==capacity*per_record and st['replay_populated_tensor_bytes']==size*per_record
            assert st['checkpoint_file_bytes']==(folder/'final.pt').stat().st_size and st['replay_file_bytes']==(folder/'replay.npz').stat().st_size and st['serialized_graph_bytes']==(folder/'final.onnx').stat().st_size
            rows=[]
            for ep,seed in enumerate(pl['evaluation_seeds']):
                row=old.load(folder/f'eval_{ep:03d}.json.gz');assert row['seed']==seed and row['run_kind']==pl['kind'];old.check_episode(row,w.r.Chapter3Config())
                arrays=np.load(folder/f'eval_{ep:03d}.npz')
                for i,state in enumerate(arrays['states']):
                    with torch.inference_mode():action,q=a['online'].greedy(torch.from_numpy(state)[None]);target=a['target'](torch.from_numpy(state)[None],action)
                    np.testing.assert_array_equal(action.numpy()[0],row['slots'][i]['action']);old.close(q.numpy()[0],row['chosen_q'][i]);old.close(target.numpy()[0],row['target_q'][i])
                old.close(np.cumsum(np.asarray(row['slot_rewards'])[::-1])[::-1],row['realized_return_to_go'])
                rows.append(analysis_audit.compact_verified(row));eval_workloads.setdefault(seed,[]).append(row['workload_sha256']);episodes+=1;tasks+=len(row['tasks'])
            evaluations[arm].append(rows)
            ns=[x['ns'] for x in timing if x['arm']==arm and x['fit']==fit]
            costs[arm].append(dict(parameter_bytes=st['parameter_bytes'],median_ms=float(np.median(ns)/1e6),p95_ms=float(np.quantile(ns,.95)/1e6)))
    for policy in w.base.HEURISTICS:
        rows=[]
        for ep,seed in enumerate(pl['evaluation_seeds']):
            row=old.load(out/'heuristics'/policy/f'eval_{ep:03d}.json.gz');assert row['seed']==seed and row['policy']==policy
            old.check_episode(row,w.r.Chapter3Config());rows.append(analysis_audit.compact_verified(row));eval_workloads[seed].append(row['workload_sha256']);episodes+=1;tasks+=len(row['tasks'])
        heuristics[policy]=rows
    assert all(all(x==group[0] for x in group) for group in workloads.values()) and all(len(set(group))==1 for group in eval_workloads.values())
    assert costs==w.r.read(out/'costs.json')
    import copy
    ap=copy.deepcopy(p);ap['fixture']=dict(engineering_seeds=dict(bootstrap=pl['bootstrap_seed']))
    analysis_pl=dict(mode='integration-fixture' if pl['mode']=='fixture' else 'development',model_seeds=pl['model_seeds'],evaluation_seeds=pl['evaluation_seeds'])
    analysis_audit.analysis_audit(ap,analysis_pl,evaluations,heuristics,costs,w.r.read(out/'analysis.json'))
    verified=w.r.read(out/'verification.json');assert verified['training_slots']==sum(x['slots'] for x in counts.values()) and verified['optimizer_updates']==updates
    assert verified['raw_timing_rows']==len(timing) and verified['final_checkpoints']==len(jobs)
    return dict(passed=True,mode=pl['mode'],fits=len(pl['model_seeds']),episodes=episodes,task_records=tasks,optimizer_rows=updates,
        checkpoint_graphs=len(jobs),fresh_process_startups=len(jobs),timing_rows=len(timing),
        saved_eager_evaluation_states=len(jobs)*len(pl['evaluation_seeds'])*40,global_freeze_order_verified=True,
        replay_and_training_storage_verified=True,all_comparisons_distributions_audited=True,
        new_simulator_steps=0,new_optimizer_updates=0,retiming=False,research_launched=pl['mode']=='development')
