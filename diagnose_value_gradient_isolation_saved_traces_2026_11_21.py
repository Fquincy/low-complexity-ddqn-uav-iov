"""Saved-data diagnosis only: no simulator, teacher, optimizer or autograd."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import return_calibrated_handoff as models
import heuristic_initialized_ddqn_baseline as baseline

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'development_runs/value_gradient_isolation_comparison_2026-11-20_01'
RUN=SOURCE/'run'
OUT=ROOT/'validation/value_gradient_isolation_saved_trace_diagnosis_2026-11-21_01'
SPEC=ROOT/'VALUE_GRADIENT_ISOLATION_COMPARISON_PROTOCOL_2026-11-20.json'
COMPLETION=ROOT/'validation/value_gradient_isolation_comparison_completion_2026-11-21_01/verification.json'
P='value_gradient_isolated_ddqn';R='dueling_shared_td_rescale_ddqn'
A='calibrated_retained_ddqn';E='calibrated_bc_only';F='old_softplus_bc_ddqn'
CORE=[P,R,A,E,F]
consumed={}
decoded=0


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def track(path):
    key=path.relative_to(ROOT).as_posix()
    if key not in consumed:consumed[key]=sha(path)
    return path


def read(path):return json.loads(track(path).read_text(encoding='utf-8-sig'))
def arrays(path):
    with np.load(track(path),allow_pickle=False) as data:return {k:data[k] for k in data.files}
def write_new(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def stats(value):
    x=np.asarray(value,dtype=np.float64).ravel()
    if not len(x):return dict(n=0)
    assert np.isfinite(x).all()
    return dict(n=len(x),mean=float(x.mean()),quantiles=np.quantile(x,[0,.05,.5,.95,1]).tolist())


def task_stats(tasks):
    return dict(tasks=len(tasks),misses=sum(not t['on_time'] for t in tasks),
        unfinished=sum(t['unfinished_at_horizon'] for t in tasks),
        by_destination={str(d):dict(tasks=sum(t['destination']==d for t in tasks),
            misses=sum(t['destination']==d and not t['on_time'] for t in tasks),
            **{k:stats([t[k] for t in tasks if t['destination']==d and t[k] is not None]) for k in
                ['delay_s','queue_delay_s','uplink_wait_s','transmission_delay_s','computation_delay_s']}) for d in range(5)})


def paired_tasks(primary,control):
    fields=['task_id','vehicle_id','arrival_s','input_bits','required_cycles','deadline_s']
    pairs=[];new=[];avoided=[];both=[];transitions=np.zeros((5,5),np.int64)
    for episode,(pp,cc) in enumerate(zip(primary,control)):
        assert pp['seed']==cc['seed']
        assert len(pp['records'])==len(cc['records'])==1200
        for p,c in zip(pp['records'],cc['records']):
            assert all(p[k]==c[k] for k in fields)
            row=(episode,p,c);pairs.append(row);transitions[c['destination'],p['destination']]+=1
            if not p['on_time'] and c['on_time']:new.append(row)
            elif p['on_time'] and not c['on_time']:avoided.append(row)
            elif not p['on_time'] and not c['on_time']:both.append(row)
    assert len(pairs)==19200
    same_rsu={}
    for destination in [3,4]:
        selected=[(ep,p,c) for ep,p,c in new if p['destination']==c['destination']==destination]
        deltas=[];residual=[];component_differences={k:[] for k in ['uplink_wait_s','transmission_delay_s','computation_delay_s']}
        for ep,p,c in selected:
            delta=p['queue_delay_s']-c['queue_delay_s'];delay=p['delay_s']-c['delay_s']
            assert np.isclose(delay,delta,rtol=1e-7,atol=1e-8)
            residual.append(delay-delta);deltas.append(delta)
            for k in component_differences:
                difference=p[k]-c[k];assert abs(difference)<1e-8;component_differences[k].append(difference)
        same_rsu[str(destination)]=dict(new_misses=len(selected),extra_CPU_wait_s=stats(deltas),
            delay_minus_queue_delta_s=stats(residual),other_component_differences={k:stats(v) for k,v in component_differences.items()},
            primary_delay_s=stats([p['delay_s'] for ep,p,c in selected]),
            control_delay_s=stats([c['delay_s'] for ep,p,c in selected]),
            task_ids=[[ep,p['task_id']] for ep,p,c in selected])
    new_transitions=np.zeros((5,5),np.int64)
    for ep,p,c in new:new_transitions[c['destination'],p['destination']]+=1
    return dict(primary_only_misses=len(new),control_only_misses=len(avoided),both_miss=len(both),
        primary_only_same_destination=sum(p['destination']==c['destination'] for ep,p,c in new),
        changed_destinations=sum(p['destination']!=c['destination'] for ep,p,c in pairs),
        all_control_to_primary_destinations=transitions.tolist(),new_miss_control_to_primary_destinations=new_transitions.tolist(),
        same_RSU_queue_accounting=same_rsu,new_miss_task_ids=[[ep,p['task_id']] for ep,p,c in new],
        avoided_miss_task_ids=[[ep,p['task_id']] for ep,p,c in avoided],
        primary_misses_by_arrival_quarter=[sum(not p['on_time'] and q*10<=p['arrival_s']<(q+1)*10 for ep,p,c in pairs) for q in range(4)])


def server_load(episodes,cfg):
    result={}
    for destination in [3,4]:
        counts=[];cycles=[];misses=[]
        for record in episodes:
            for slot in range(40):
                part=[t for t in record['records'] if t['arrival_s']==slot and t['destination']==destination]
                counts.append(len(part));cycles.append(sum(t['required_cycles'] for t in part));misses.append(sum(not t['on_time'] for t in part))
        service=np.array(cycles)/cfg['rsu_cpu_hz'][destination-3]
        result[str(destination)]=dict(tasks_per_slot=stats(counts),CPU_service_s_per_slot=stats(service),
            slots_with_misses=sum(v>0 for v in misses),misses=sum(misses),
            service_s_in_slots_with_misses=stats(service[np.array(misses)>0]),
            service_s_in_other_slots=stats(service[np.array(misses)==0]))
    return result


@torch.inference_mode()
def predict(model,states):
    global decoded
    output={k:[] for k in ['actions','q','value','scores','margins','hidden']}
    for state in states:
        x=torch.from_numpy(state)[None];actions,q,_=model.greedy(x);p,m,v,_=model.tables(x)
        scores=np.concatenate([p.numpy(),m.numpy()],axis=1);mask=baseline.allowed(x)
        ranked=np.sort(np.where(mask,scores,np.inf),axis=2)
        assert np.array_equal(actions.numpy(),np.where(mask,scores,np.inf).argmin(2))
        hidden=model.student.net[1](model.student.net[0](x)).numpy()
        for k,value in dict(actions=actions.numpy(),q=q.numpy(),value=v.numpy(),scores=scores,
            margins=ranked[:,:,1]-ranked[:,:,0],hidden=hidden).items():output[k].append(value)
        decoded+=1
    return {k:np.concatenate(v) for k,v in output.items()}


@torch.inference_mode()
def fixed_actions_q(model,prediction,actions):
    scores=torch.from_numpy(prediction['scores'])
    return model.value(scores[:,:30],scores[:,30:],torch.from_numpy(prediction['value']),torch.from_numpy(actions)).numpy()


def training_summary(logs):
    pred=np.asarray([v['prediction'] for v in logs]);target=np.asarray([v['target'] for v in logs])
    rewards=np.asarray([v['rewards'] for v in logs]);tq=np.asarray([v['target_q'] for v in logs])
    done=np.asarray([v['dones'] for v in logs],bool)
    demo=np.asarray([[i.startswith('demo:') for i in v['transition_ids']] for v in logs],bool)
    assert np.allclose(target,rewards+tq,rtol=3e-5,atol=1e-5) and np.all(tq[done]==0)
    def group(mask):
        delta=(target-pred)[mask]
        return dict(prediction=stats(pred[mask]),target=stats(target[mask]),reward=stats(rewards[mask]),
            target_q=stats(tq[mask]),target_minus_prediction=stats(delta),
            positive_TD_target_direction=float(np.mean(delta>0)) if len(delta) else None,
            huber_target_direction=stats(np.clip(delta,-1,1)))
    norms={k:np.asarray([v['gradient_norms'][k] for v in logs]) for k in ['total','shared_hidden','branch_output','value_output']}
    assert np.allclose(norms['total']**2,sum(norms[k]**2 for k in norms if k!='total'),rtol=1e-8,atol=1e-7)
    return dict(first_update=logs[0]['update'],last_update=logs[-1]['update'],
        td_loss=stats([v['td_loss'] for v in logs]),retention_loss=stats([v['retention_loss'] for v in logs]),
        norms={k:stats(v) for k,v in norms.items()},clipped_fraction=float(np.mean([v['clipped'] for v in logs])),
        value_output_squared_gradient_fraction=stats(norms['value_output']**2/np.maximum(norms['total']**2,1e-30)),
        shared_hidden_squared_gradient_fraction=stats(norms['shared_hidden']**2/np.maximum(norms['total']**2,1e-30)),
        all=group(np.ones_like(done)),terminal=group(done),nonterminal=group(~done),
        demonstration=group(demo),online=group(~demo))


def main():
    if OUT.exists():raise RuntimeError('Preserve previous diagnosis')
    torch.set_num_threads(1)
    completion=read(COMPLETION);spec=read(SPEC);manifest=read(RUN/'manifest.json')
    assert completion['policy_screen'] is False and completion['value_screen'] is True and completion['promising'] is False
    assert sha(SPEC)==completion['protocol_sha256']
    for name,digest in completion['evidence_sha256'].items():assert sha(track(SOURCE/name))==digest
    for name,digest in spec['definition_sha256'].items():assert sha(track(ROOT/name))==digest
    analysis=read(RUN/'analysis.json');audit=read(SOURCE/'saved_artifact_audit.json');freeze=read(RUN/'CHECKPOINT_FREEZE.json')
    result=dict(kind='POSTHOC_SAVED_TRACE_DIAGNOSIS_NOT_NEW_EXPERIMENT',paired_tasks={},server_loads={},
        reward_components={},fixed_state={},value_errors={},training={},probes={},observed_states={},
        completed_first_update_audit_contexts=audit['first_update_reconstructions'])
    raw={};cfg=spec['simulator_defaults']
    reward_weights=dict(normalized_delay_accrual=-cfg['delay_reward_weight'],normalized_uav_energy=-cfg['uav_energy_reward_weight'],
        new_deadline_misses_per_vehicle=-cfg['deadline_miss_penalty'])
    for fit in range(3):
        fitted={};eps={};saved={};tasks={};deployed={}
        for arm in CORE:
            path=track(RUN/arm/f'fit_{fit}/deployment_weights.pt')
            assert sha(path)==freeze['deployments'][f'{arm}/{fit}']
            with torch.random.fork_rng(devices=[]):model=models.Critic(A if arm in [P,R] else arm)
            state=torch.load(path,weights_only=True);model.load_state_dict(state);model.eval().requires_grad_(False)
            fitted[arm]=model;deployed[arm]=state
            eps[arm]=[read(RUN/'evaluation'/arm/str(fit)/f'episode_{i:03d}.json') for i in range(16)]
            saved[arm]=[arrays(RUN/'evaluation'/arm/str(fit)/f'episode_{i:03d}.npz') for i in range(16)]
            tasks[arm]=[t for ep in eps[arm] for t in ep['records']]
            assert len(tasks[arm])==19200
            parts=[]
            for ep in eps[arm]:
                part={k:sum(v['reward_components'][k] for v in ep['step_info'])*weight for k,weight in reward_weights.items()}
                assert np.isclose(sum(part.values()),ep['return_value'],rtol=1e-10,atol=1e-8);parts.append(part)
            result['reward_components'][f'{arm}/{fit}']=dict(episodes=parts,mean={k:float(np.mean([v[k] for v in parts])) for k in reward_weights},
                task_outcomes=task_stats(tasks[arm]),maximum_delay_s=max(t['delay_s'] for t in tasks[arm] if t['delay_s'] is not None))
            result['server_loads'][f'{arm}/{fit}']=server_load(eps[arm],cfg)
            s=np.concatenate([v['states'] for v in saved[arm]])
            qindices=list(range(240,514))+[514,515,517,518,520,521,523,524]
            result['observed_states'][f'{arm}/{fit}']=dict(max_abs_observed_queue_feature=float(np.abs(s[:,qindices]).max()),
                all_tasks_completed_before_next_slot=all(t['delay_s'] is not None and t['delay_s']<1 for t in tasks[arm]))
            errors=np.concatenate([v['q']-np.cumsum(v['rewards'][::-1],dtype=np.float64)[::-1] for v in saved[arm]])
            result['value_errors'][f'{arm}/{fit}']=dict(all=stats(errors),terminal=stats(errors[39::40]),
                nonterminal=stats(errors.reshape(16,40)[:,:39]),quarters=[stats(errors.reshape(16,40)[:,j*10:(j+1)*10]) for j in range(4)])
        assert models.tensor_hash(fitted[E])==freeze['handoff_hashes'][f'{P}/{fit}']==freeze['handoff_hashes'][f'{R}/{fit}']==freeze['handoff_hashes'][f'{A}/{fit}']
        result['paired_tasks'][str(fit)]={arm:paired_tasks(eps[P],eps[arm]) for arm in [E,A,R,F]}
        result['reward_components'][f'contrasts/{fit}']={arm:{k:result['reward_components'][f'{P}/{fit}']['mean'][k]-result['reward_components'][f'{arm}/{fit}']['mean'][k] for k in reward_weights} for arm in [E,A,R,F]}
        for arm in CORE:
            probes=read(RUN/arm/f'fit_{fit}/probes.json')['stages'];start=probes['AFTER_VALUE_FIT' if arm!=F else 'AFTER_BC'];end=probes['FINAL']
            change=np.asarray(start['actions'])!=np.asarray(end['actions'])
            result['probes'][f'{arm}/{fit}']=dict(handoff_imitation_loss=start['imitation_loss'],final_imitation_loss=end['imitation_loss'],
                placement_change_fraction=float(change[:,:30].mean()),movement_change_fraction=float(change[:,30:].mean()),
                stages={k:dict(q=stats(v['greedy_q']),value=stats(v['value']),imitation_loss=v['imitation_loss']) for k,v in probes.items() if not v.get('skipped')})
        demos=[arrays(RUN/f'demo_{fit}'/f'episode_{i:03d}.npz') for i in range(10)]
        populations={arm:np.concatenate([v['states'] for v in saved[arm]]) for arm in CORE}
        populations['demonstrations']=np.concatenate([v['states'] for v in demos])
        eweights=deployed[E]['student.net.2.weight'][160].numpy().astype(np.float64);ebias=float(deployed[E]['student.net.2.bias'][160])
        for population,states in populations.items():
            pred={arm:predict(model,states) for arm,model in fitted.items()};ref=pred[E];entries={}
            for arm,pr in pred.items():
                prefix=f'fit{fit}_{population}_{arm}'
                if population==arm:
                    assert np.array_equal(pr['actions'],np.concatenate([v['actions'] for v in saved[arm]]))
                    assert np.array_equal(pr['q'],np.concatenate([v['q'] for v in saved[arm]]).astype(np.float32))
                if population=='demonstrations':
                    probe=read(RUN/arm/f'fit_{fit}/probes.json')['stages']['FINAL']
                    with torch.inference_mode():ba,bq,_=fitted[arm].greedy(torch.from_numpy(states))
                    assert np.array_equal(ba.numpy(),probe['actions']) and np.array_equal(bq.numpy(),np.asarray(probe['greedy_q'],np.float32))
                qe=fixed_actions_q(fitted[arm],pr,ref['actions']);dv=pr['value']-ref['value'];dq=qe-ref['q']
                branch=(qe-pr['value'])-(ref['q']-ref['value'])
                if arm!=F:assert np.allclose(dq,dv+branch,rtol=3e-5,atol=3e-5)
                changed=pr['actions']!=ref['actions'];transition=np.bincount((5*ref['actions'][:,:30]+pr['actions'][:,:30]).ravel(),minlength=25).reshape(5,5)
                w=deployed[arm]['student.net.2.weight'][160].numpy().astype(np.float64);bias=float(deployed[arm]['student.net.2.bias'][160])
                hp=pr['hidden'].astype(np.float64);he=ref['hidden'].astype(np.float64)
                # Symmetric algebraic attribution; not a causal intervention.
                readout=.5*(hp+he)@(w-eweights)+(bias-ebias)
                feature=.5*(hp-he)@(w+eweights)
                v64delta=(hp@w+bias)-(he@eweights+ebias)
                assert np.allclose(readout+feature,v64delta,rtol=1e-9,atol=1e-9)
                entries[arm]=dict(q=stats(pr['q']),value=stats(pr['value']),
                    joint_change_vs_E=float(changed.any(1).mean()),placement_change_vs_E=float(changed[:,:30].mean()),movement_change_vs_E=float(changed[:,30:].mean()),
                    placement_margin=stats(pr['margins'][:,:30]),E_to_model_destinations=transition.tolist(),
                    greedy_gain_over_E_action=stats(pr['q']-qe),fixed_E_action_Q_difference=stats(dq),value_difference=stats(dv),
                    centered_branch_difference=stats(branch) if arm!=F else None,
                    symmetric_value_readout_contribution=stats(readout),symmetric_value_feature_contribution=stats(feature),
                    explicit64_minus_FP32_value_difference=stats(v64delta-dv),
                    value_weight_norm=float(np.linalg.norm(w)),value_weight_change_norm=float(np.linalg.norm(w-eweights)))
                for name,value in dict(actions=pr['actions'],q=pr['q'],value=pr['value'],q_on_E=qe,margins=pr['margins'],
                    readout=readout,feature=feature,explicit_value_delta=v64delta).items():raw[prefix+'_'+name]=value
            result['fixed_state'][f'{population}/{fit}']=entries
        for control in [E,A,R,F]:
            ps=populations[P];cs=populations[control];assert np.array_equal(ps[:,:240],cs[:,:240]) and np.array_equal(ps[:,-1],cs[:,-1])
            result['observed_states'][f'P-minus-{control}/{fit}']=dict(task_features_equal=True,time_equal=True,
                remote_and_queue_max_abs_difference=float(np.abs(ps[:,240:526]-cs[:,240:526]).max()),
                UAV_max_abs_difference=float(np.abs(ps[:,526:532]-cs[:,526:532]).max()))
        for arm in [P,R,A,F]:
            logs=read(RUN/arm/f'fit_{fit}/optimizer.json');assert len(logs)==1872
            assert all(v['gradient_rule']==({P:'isolate_value',R:'rescale_shared_td'}.get(arm,'ordinary')) for v in logs)
            result['training'][f'{arm}/{fit}']=dict(full=training_summary(logs),windows=[training_summary(logs[i:i+200]) for i in range(0,len(logs),200)],
                first_update_decomposition=logs[0]['first_update_decomposition'])
            del logs
        print(f'Completed fit {fit}: saved tasks/queues/rewards, fixed-state deployment inference and TD histories.',flush=True)
    for name,digest in consumed.items():
        path=ROOT/name;assert sha(path)==digest,name
        if path.is_relative_to(RUN) and path!=RUN/'manifest.json':assert manifest[path.relative_to(RUN).as_posix()]==digest,name
    assert analysis['policy_screen'] is False and analysis['value_screen'] is True and analysis['promising'] is False
    result.update(protocol_sha256=sha(SPEC),consumed_sha256=consumed,consumed_files_verified_unchanged=len(consumed),
        decoded_saved_states=decoded,script_sha256=sha(Path(__file__)),new_simulator_steps=0,new_optimizer_updates=0,new_teacher_calls=0,
        new_autograd_evaluations=0,policy_screen=False,value_screen=True,promising=False,new_research_launch_authorized=False,
        final_phase_launched=False,word_thesis_changed=False,
        limitations=['Posthoc description; no new acceptance rules or scientific retuning.',
            'No alternative allocation simulation or causal inference from same-state scores/task matching.',
            'Window samples/replay change over time; intermediate checkpoint trajectories are unavailable.',
            'Logged gradients combine TD/retention; only the completed first-update auditor separates the declared mechanism.',
            'Symmetric readout/feature attribution is an algebraic identity, not independent retraining interventions.',
            'FP64 attribution and FP32 inference differ by reported rounding; original inference boundaries are retained.',
            'Common V cancels directly from branch choices; shared-feature/branch changes can alter them.',
            'Teacher/training/final evaluation are not run; G receives no new demonstration probes.'])
    OUT.mkdir(parents=True);np.savez_compressed(OUT/'fixed_state_predictions.npz',**raw)
    write_new(OUT/'diagnosis.json',result)
    receipt=dict(status='SAVED_TRACE_DIAGNOSIS_COMPLETE_NO_NEW_RESEARCH',consumed_files_verified_unchanged=len(consumed),decoded_saved_states=decoded,
        artifacts_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},script_sha256=sha(Path(__file__)),
        new_simulator_steps=0,new_optimizer_updates=0,new_teacher_calls=0,new_autograd_evaluations=0,
        policy_screen=False,value_screen=True,promising=False,word_thesis_changed=False)
    write_new(OUT/'verification.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
