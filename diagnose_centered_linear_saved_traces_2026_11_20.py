"""Offline diagnostics only: immutable traces/checkpoints, no simulator or optimizer."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import centered_linear_comparison_models as variants
import heuristic_initialized_ddqn_baseline as baseline

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'development_runs/centered_linear_correction_comparison_2026-11-20_01'
RUN = SOURCE/'run'
OUT = ROOT/'validation/centered_linear_saved_trace_diagnosis_2026-11-20_01'
PRIMARY = 'centered_linear_bc_ddqn'
BC = 'centered_linear_bc_only'
CONTROLS = [PRIMARY,BC,'old_softplus_bc_ddqn','uncentered_linear_bc_ddqn']
consumed = {}
decoded = 0


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def track(p):
    name=p.relative_to(ROOT).as_posix()
    if name not in consumed: consumed[name]=sha(p)
    return p


def read(p): return json.loads(track(p).read_text(encoding='utf-8-sig'))


def arrays(p):
    with np.load(track(p)) as data: return {k:data[k] for k in data.files}


def stats(x):
    x=np.asarray(x,dtype=float).ravel()
    return dict(n=len(x),mean=float(x.mean()),quantiles=np.quantile(x,[0,.05,.5,.95,1]).tolist()) if len(x) else dict(n=0)


def predict(model,states):
    global decoded
    with torch.no_grad():
        x=torch.from_numpy(states); a,q,_=model.greedy(x)
        p,m,v,_=model.tables(x)
        score=torch.cat([p,m],1)
        mask=torch.from_numpy(baseline.allowed(x))
        ordered=score.masked_fill(~mask,torch.inf).sort(2).values
        margins=(ordered[:,:,1]-ordered[:,:,0]).numpy()
    decoded+=len(x)
    return dict(actions=a.numpy(),q=q.numpy(),value=v.numpy(),margin=margins)


def summarize_samples(rows,selector=None):
    pred=np.array([r['prediction'] for r in rows]); target=np.array([r['target'] for r in rows])
    reward=np.array([r['rewards'] for r in rows]); tq=np.array([r['target_q'] for r in rows]); done=np.array([r['dones'] for r in rows],bool)
    mask=np.ones_like(done) if selector is None else (done if selector=='terminal' else ~done)
    # Saved FP32 addition can differ slightly from float64 addition here.
    assert np.allclose(target,reward+tq,rtol=2e-6,atol=2e-5)
    assert (tq[done]==0).all()
    delta=(target-pred)[mask]
    return dict(prediction=stats(pred[mask]),reward=stats(reward[mask]),target=stats(target[mask]),
                target_q=stats(tq[mask]),target_minus_prediction=stats(delta),
                upward_huber_fraction=float((delta>0).mean()) if len(delta) else None,
                mean_clipped_target_minus_prediction=float(np.clip(delta,-1,1).mean()) if len(delta) else None)


def task_summary(records):
    rows=list(records)
    return dict(tasks=len(rows),misses=sum(not r['on_time'] for r in rows),
                destination_counts=np.bincount([r['destination'] for r in rows],minlength=5).tolist(),
                by_destination={str(d):dict(tasks=sum(r['destination']==d for r in rows),
                    misses=sum(r['destination']==d and not r['on_time'] for r in rows),
                    queue_delay=stats([r['queue_delay_s'] for r in rows if r['destination']==d]),
                    delay=stats([r['delay_s'] for r in rows if r['destination']==d and r['delay_s'] is not None])) for d in range(5)})


def main():
    if OUT.exists(): raise RuntimeError('Preserve previous diagnoses')
    torch.set_num_threads(1)
    manifest=read(RUN/'manifest.json'); analysis=read(RUN/'analysis.json'); freeze=read(RUN/'CHECKPOINT_FREEZE.json')
    result=dict(kind='OFFLINE_POSTHOC_DIAGNOSIS_NOT_NEW_RESEARCH',probes={},training={},same_state={},tasks={})
    for fit in range(3):
        models={}
        for arm in CONTROLS:
            ck=torch.load(track(RUN/arm/f'fit_{fit}/final.pt'),weights_only=True)
            model=variants.Critic(arm).eval();model.load_state_dict(ck['online']);models[arm]=model
        assert freeze['bc_end_hashes'][f'{PRIMARY}/{fit}']==freeze['bc_end_hashes'][f'{BC}/{fit}']
        demo=[arrays(RUN/f'demo_{fit}'/f'episode_{ep:03d}.npz') for ep in range(10)]
        states=np.concatenate([d['states'] for d in demo]);labels=np.concatenate([d['actions'] for d in demo])
        for arm in variants.ARMS:
            key=f'{arm}/{fit}'; probes=read(RUN/arm/f'fit_{fit}/probes.json')
            stages={}
            for stage,p in probes.items():
                a=np.asarray(p['actions']);q=np.asarray(p['q']);v=np.asarray(p['scalar']['value_output'])
                stages[stage]=dict(imitation_loss=p['imitation_loss'],branch_agreement=p['branch_agreement'],
                    joint_agreement=p['joint_agreement'],q=stats(q),value_head=stats(v),q_minus_value=stats(q-v),
                    scalar_slope=stats(p['scalar']['scalar_slope']),
                    placement_destination_counts=np.bincount(a[:,:30].ravel(),minlength=5).tolist())
            a0=np.array(probes['after_bc']['actions']);a1=np.array(probes['final']['actions'])
            result['probes'][key]=dict(stages=stages,placement_changed_fraction=float((a0[:,:30]!=a1[:,:30]).mean()),
                motion_changed_fraction=float((a0[:,30:]!=a1[:,30:]).mean()))
            if arm in models:
                fixed=predict(models[arm],states)
                assert np.array_equal(fixed['actions'],a1)
                assert np.allclose(fixed['q'],probes['final']['q'],rtol=1e-5,atol=1e-5)
                result['probes'][key]['final_margin']=stats(fixed['margin'])
            if arm==BC: continue
            logs=read(RUN/arm/f'fit_{fit}/optimizer.json');assert len(logs)==1872
            gradients=np.array([r['gradient_norm'] for r in logs])
            windows=[]
            for lo in range(0,len(logs),200):
                block=logs[lo:lo+200]
                windows.append(dict(first_update=lo+1,last_update=lo+len(block),
                    all=summarize_samples(block),terminal=summarize_samples(block,'terminal'),
                    nonterminal=summarize_samples(block,'nonterminal')))
            first=logs[0]
            # Within a sample, successor greedy Q versus current sampled-action Q
            # includes both action choice and state changes, not a pure max bias.
            result['training'][key]=dict(first_update=summarize_samples([first]),
                first_terminal=summarize_samples([first],'terminal'),
                first_nonterminal=summarize_samples([first],'nonterminal'),
                gradient_norm=stats(gradients),clipped_fraction=float((gradients>10).mean()),
                windows=windows,terminal=summarize_samples(logs,'terminal'),nonterminal=summarize_samples(logs,'nonterminal'),
                terminal_sample_fraction=float(np.array([r['dones'] for r in logs]).mean()))
            del logs
        for population in [PRIMARY,BC]:
            episodes=[arrays(RUN/'evaluation'/population/str(fit)/f'episode_{ep:03d}.npz') for ep in range(16)]
            x=np.concatenate([d['states'] for d in episodes]);saved=np.concatenate([d['actions'] for d in episodes])
            predictions={arm:predict(model,x) for arm,model in models.items()}
            assert np.array_equal(predictions[population]['actions'],saved)
            assert np.allclose(predictions[population]['q'],np.concatenate([d['q'] for d in episodes]),rtol=1e-5,atol=1e-5)
            base=predictions[BC];row={}
            for arm,p in predictions.items():
                change=p['actions']!=base['actions']
                row[arm]=dict(q=stats(p['q']),value_head=stats(p['value']),q_minus_value=stats(p['q']-p['value']),
                    placement_change_vs_bc=float(change[:,:30].mean()),motion_change_vs_bc=float(change[:,30:].mean()),
                    destination_counts=np.bincount(p['actions'][:,:30].ravel(),minlength=5).tolist(),
                    transitions_from_bc=np.bincount((5*base['actions'][:,:30]+p['actions'][:,:30]).ravel(),minlength=25).reshape(5,5).tolist())
            result['same_state'][f'{population}/{fit}']=row
        task_records={};episode_records={}
        for arm in CONTROLS+['teacher']:
            basepath=RUN/'evaluation'/('teacher' if arm=='teacher' else f'{arm}/{fit}')
            es=[read(basepath/f'episode_{ep:03d}.json') for ep in range(16)]
            episode_records[arm]=es
            task_records[arm]=[r for e in es for r in e['records']]
            # Exactly 30 new tasks per slot; no unfinished tasks in these controls.
            assert len(task_records[arm])==19200
        taskrow={arm:task_summary(rs) for arm,rs in task_records.items()}
        paired=[];rprimary=task_records[PRIMARY];rbc=task_records[BC]
        for p,b in zip(rprimary,rbc):
            assert all(p[k]==b[k] for k in ['task_id','vehicle_id','arrival_s','input_bits','required_cycles','deadline_s'])
            paired.append((p,b))
        taskrow['primary_vs_bc_paired']=dict(
            primary_only_misses=sum(not p['on_time'] and b['on_time'] for p,b in paired),
            bc_only_misses=sum(p['on_time'] and not b['on_time'] for p,b in paired),
            both_miss=sum(not p['on_time'] and not b['on_time'] for p,b in paired),
            destination_changed=sum(p['destination']!=b['destination'] for p,b in paired),
            changed_primary=task_summary(p for p,b in paired if p['destination']!=b['destination']),
            unchanged_primary=task_summary(p for p,b in paired if p['destination']==b['destination']),
            newly_missed=task_summary(p for p,b in paired if not p['on_time'] and b['on_time']))
        # New deadline misses by relative episode quarter, from realized task records.
        taskrow['misses_by_quarter']={arm:[sum(not r['on_time'] and q*10<=r['arrival_s']<(q+1)*10 for r in rs) for q in range(4)] for arm,rs in task_records.items()}
        result['tasks'][str(fit)]=taskrow
        print(f'Completed saved-data fit {fit}',flush=True)
    # Bind every consumed artifact back to the already audited immutable manifest.
    for name,digest in consumed.items():
        p=ROOT/name
        assert sha(p)==digest,name
        if p.is_relative_to(RUN) and p!=RUN/'manifest.json':
            assert manifest[p.relative_to(RUN).as_posix()]==digest,name
    result.update(consumed_files_verified_unchanged=len(consumed),consumed_sha256=consumed,
        source_sha256={n:sha(ROOT/n) for n in ['diagnose_centered_linear_saved_traces_2026_11_20.py','centered_linear_comparison_models.py','heuristic_initialized_ddqn_baseline.py']},
        decoded_saved_states=decoded,new_simulator_steps=0,new_optimizer_updates=0,word_thesis_changed=False,
        limitations=['Only before/after-BC/final probes and final checkpoints retained; no intermediate greedy policy checkpoints.',
          'Replay prediction trajectories involve changing sampled transitions; not fixed-state trajectories.',
          'Paired workloads evolve under different queues/movement; task destination/outcome associations are not causal counterfactuals.',
          'No gradient replay, no teacher calls, no new evaluation episodes, no prospective scientific screen.'])
    OUT.mkdir(parents=True)
    (OUT/'diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='OFFLINE_DIAGNOSIS_COMPLETE',files=len(consumed),decoded_saved_states=decoded)))


if __name__=='__main__': main()
