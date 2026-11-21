"""Independent saved-file arithmetic/trace audit; no simulator or optimizer."""
import gzip
import json
from pathlib import Path
import numpy as np
import torch
import onnx
import run_assignment_price_fixture as workflow
from audit_value_bound_development import check_episode

r=workflow.r

def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
def huber(x):
    x=np.abs(np.asarray(x,np.float64))
    return float(np.where(x<1,.5*x*x,x-.5).mean())
def load(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:return json.load(stream)

def check_update(row,history,pl,screened,anchored,replay_rng,anchor_rng):
    cfg=pl['cfg'];slot=row['physical_slots'];capacity=cfg['replay_capacity'];size=min(slot,capacity)
    indices=replay_rng.integers(0,size,cfg['batch_size']);assert indices.tolist()==row['replay_indices']
    current={i%capacity:history[i] for i in range(max(0,slot-capacity),slot)}
    chosen=[current[i] for i in indices]
    close(row['sample_rewards'],[np.float32(v['reward']) for v in chosen]);close(row['sample_dones'],[v['done'] for v in chosen])
    terminal=sorted(i for i,v in current.items() if v['done']);assert terminal==row['terminal_pool']
    dones=np.asarray(row['sample_dones'],bool);next_values=np.asarray(row['next_target_values'])
    actions=np.asarray(row['next_actions']);assert actions.shape==(cfg['batch_size'],32)
    assert (actions[dones]==-1).all() and (next_values[dones]==0).all()
    assert ((actions[~dones]>=0)&(actions[~dones]<5)).all()
    for j,transition in enumerate(chosen):
        if not dones[j] and screened:
            mask=workflow.adp.allowed(transition['next'][None])[0]
            assert mask[np.arange(32),actions[j]].all()
    y=np.asarray(row['sample_rewards'])+next_values;close(y,row['bellman_targets'])
    q=np.asarray(row['chosen_Q']);close(q-y,row['TD_residuals'])
    close(huber(q-y),row['uniform_loss'])
    if terminal and anchored:
        anchor_indices=anchor_rng.choice(terminal,size=cfg['anchor']['batch'],replace=True)
        assert anchor_indices.tolist()==row['anchor_indices']
        close(row['anchor_rewards'],[np.float32(current[i]['reward']) for i in anchor_indices])
        close(huber(np.asarray(row['anchor_Q'])-row['anchor_rewards']),row['anchor_loss'])
    else:
        assert row['anchor_indices']==[] and row['anchor_Q']==[] and row['anchor_rewards']==[]
        close(row['anchor_loss'],0.)
    close(row['loss'],row['uniform_loss']+cfg['anchor']['weight']*row['anchor_loss'])
    assert row['step']==slot-cfg['warmup']+1
    assert row['target_copy']==(row['step']%cfg['target_interval_updates']==0)
    norm=row['grad_norm_before_clip'];assert np.isfinite(norm) and norm>=0
    assert row['clipped']==(norm>cfg['gradient_clip']) and row['optimizer_ns']>0
    assert all(np.isfinite(np.asarray(row[k])).all() for k in ['chosen_Q','bellman_targets','TD_residuals','anchor_Q'])

def audit(output):
    out=Path(output);pl=r.read(out/'plan.json');assert pl==workflow.plan()
    assert pl['kind']==workflow.KIND and not pl['quality_evidence']
    manifest=r.read(out/'manifest.json')
    assert {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}==set(manifest)|{'manifest.json'}
    for name,digest in manifest.items():assert r.sha(out/name)==digest,name
    freeze=r.read(out/'PRE_RUN_FREEZE.json');assert freeze['plan_sha256']==r.sha(out/'plan.json')
    for name,digest in freeze['sources'].items():
        assert r.sha(out/'source_snapshot'/name)==digest and r.sha(r.ROOT/name)==digest
    expected_order=list(workflow.ARMS);np.random.default_rng(pl['order_seed']).shuffle(expected_order)
    assert r.read(out/'arm_order.json')==expected_order
    frozen=r.read(out/'FINAL_CHECKPOINT_FREEZE.json');assert set(frozen)==set(workflow.ARMS)
    all_workloads={};eval_workloads=[];counts={};values={};updates=0;task_count=0;episodes=0;evictions=0
    for arm in workflow.ARMS:
        folder=out/arm;checkpoint=torch.load(folder/'final.pt',map_location='cpu',weights_only=True)
        assert checkpoint['kind']==workflow.KIND and checkpoint['arm']==arm and checkpoint['model_seed']==pl['model_seed']
        assert checkpoint['plan_sha256']==r.sha(out/'plan.json') and r.sha(folder/'final.pt')==frozen[arm]
        assert len(list(folder.glob('*.pt')))==1
        assert (checkpoint['slots'],checkpoint['updates'],checkpoint['copies'])==(80,73,9)
        counts[arm]=dict(slots=80,updates=73,copies=9)
        # Independent terminal-pool reconstruction includes wraparound/eviction.
        history=[];fingerprints=[]
        for index,seed in enumerate(pl['train_seeds']):
            row=load(folder/f'train_{index}.json.gz');assert row['seed']==seed and row['training']
            check_episode(row,r.Chapter3Config());episodes+=1;task_count+=len(row['tasks']);fingerprints.append(row['workload_sha256'])
            arrays=np.load(folder/f'train_{index}.npz');s=arrays['states'];ns=arrays['next_states']
            assert s.shape==ns.shape==(40,533) and np.isfinite(s).all() and np.isfinite(ns).all()
            np.testing.assert_array_equal(ns[:-1],s[1:]);close(workflow.adp.remaining(torch.from_numpy(s)),np.arange(40,0,-1))
            close(workflow.adp.remaining(torch.from_numpy(ns)),np.arange(39,-1,-1))
            for i,slot in enumerate(row['slots']):history.append(dict(**slot,state=s[i],next=ns[i]))
        all_workloads[arm]=fingerprints
        rng=np.random.default_rng(pl['model_seed']+100000);actor=checkpoint['actor_history'];assert len(actor)==80
        a=workflow.build(arm,pl);screened=a['definition']['screened'];anchored=True if arm in workflow.adp.ARMS else a['online'].anchored
        for i,record in enumerate(actor):
            u=float(rng.random());uniforms=rng.random(32);warm=i<pl['cfg']['warmup'];epsilon=1. if warm else 1-.95*min(i/pl['cfg']['epsilon_decay'],1.)
            assert record==dict(warmup=warm,epsilon=epsilon,explore=bool(warm or u<epsilon),epsilon_uniform=u,branch_uniforms=uniforms.tolist())
            if record['explore']:
                expected=workflow.adp.explore(history[i]['state'][None],uniforms[None],screened)[0]
                np.testing.assert_array_equal(expected,history[i]['action'])
            if screened:
                mask=workflow.adp.allowed(history[i]['state'][None])[0]
                assert mask[np.arange(32),history[i]['action']].all()
        replay_rng=np.random.default_rng(pl['model_seed']+200000);anchor_rng=np.random.default_rng(pl['model_seed']+300000)
        logs=[json.loads(line) for line in (folder/'optimizer.jsonl').read_text().splitlines()];assert len(logs)==73
        for row in logs:
            check_update(row,history,pl,screened,anchored,replay_rng,anchor_rng)
            if row['physical_slots']==72:assert row['terminal_pool']==[];evictions+=1
        updates+=len(logs)
        assert len(checkpoint['optimizer']['state'])>0
        assert all(int(v['step'])==73 for v in checkpoint['optimizer']['state'].values())
        a['online'].load_state_dict(checkpoint['online']);a['target'].load_state_dict(checkpoint['target'])
        onnx_model=onnx.load(str(folder/'final.onnx'));onnx.checker.check_model(onnx_model)
        tensors=[onnx.numpy_helper.to_array(t) for t in onnx_model.graph.initializer]
        parameters=[p.detach().numpy() for p in a['online'].parameters()]
        assert len(tensors)==len(parameters)
        for tensor,parameter in zip(tensors,parameters):np.testing.assert_array_equal(tensor,parameter)
        row=load(folder/'evaluation.json.gz');assert row['seed']==pl['evaluation_seed'] and not row['training']
        check_episode(row,r.Chapter3Config());episodes+=1;task_count+=len(row['tasks']);eval_workloads.append(row['workload_sha256'])
        arrays=np.load(folder/'evaluation.npz');s=torch.from_numpy(arrays['states'])
        actions=torch.tensor([x['action'] for x in row['slots']])
        with torch.inference_mode():
            selected,q=a['online'].greedy(s);target=a['target'](s,actions)
        np.testing.assert_array_equal(selected.numpy(),actions.numpy());close(q.numpy(),row['chosen_q']);close(target.numpy(),row['target_q'])
        close(np.cumsum(np.asarray(row['slot_rewards'])[::-1])[::-1],row['realized_return_to_go'])
        values[arm]=dict(return_sum=row['return_sum'],deadline_success=row['deadline_success'],terminal_error=float(row['chosen_q'][-1]-row['slot_rewards'][-1]),
            q_min=min(row['chosen_q']),q_max=max(row['chosen_q']),quality_evidence=False)
    assert len(set(tuple(x) for x in all_workloads.values()))==1
    for policy in workflow.HEURISTICS:
        row=load(out/f'heuristic_{policy}.json.gz');assert row['seed']==pl['evaluation_seed'] and row['policy']==policy
        check_episode(row,r.Chapter3Config());episodes+=1;task_count+=len(row['tasks']);eval_workloads.append(row['workload_sha256'])
    assert len(set(eval_workloads))==1
    assert r.read(out/'counters.json')==counts
    verification=r.read(out/'verification.json')
    assert verification==dict(kind=workflow.KIND,quality_evidence=False,research_launched=False,training_slots=640,evaluation_slots=440,
        updates=584,copies=72,deployment_equivalence_states=320,final_checkpoint_only=True,evaluation_no_updates=True)
    assert (episodes,task_count,updates,evictions)==(27,32400,584,8)
    return dict(passed=True,kind=workflow.KIND,episodes=episodes,task_records=task_count,optimizer_rows=updates,
        replay_terminal_evictions_checked=evictions,trained_checkpoint_graphs=8,recorded_deployment_checks=320,
        independent_saved_eager_value_checks=320,quality_evidence=False,research_launched=False,
        new_simulator_steps=0,new_optimizer_updates=0,fixture_descriptive_values=values)
