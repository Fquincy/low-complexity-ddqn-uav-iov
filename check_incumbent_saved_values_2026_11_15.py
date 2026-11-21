"""Fixed checkpoints/consumed trajectories only. No simulator stepping or fitting."""
import copy
import gzip
import json
from pathlib import Path
import numpy as np
import torch
import run_horizon_loss_research as r

OUT=r.ROOT/'validation/incumbent_evidence_stage_2026-11-15_01/value_check'
RUN=r.ROOT/'development_runs/horizon_loss_development_2026-11-14_01/comparison/run'
ARMS=['screened_price24','screened_no_price24','screened_plain24_anchored']

def dist(x):
    v=np.asarray(x,np.float64)
    return dict(count=int(v.size),mean=float(v.mean()),rmse=float(np.sqrt(np.mean(v*v))),
        median=float(np.median(v)),p05=float(np.quantile(v,.05)),p95=float(np.quantile(v,.95)),
        min=float(v.min()),max=float(v.max()))

def check():
    OUT.mkdir(parents=True,exist_ok=False);r.configure();p=r.protocol()
    manifest=r.read(RUN/'run_manifest.json')['files'];frozen=r.read(RUN/'ALL_FINAL_CHECKPOINTS_FREEZE.json')
    plan=r.read(RUN/'runtime_plan.json');defs={x['id']:x for x in p['learned_arms']}
    results={};inputs={};states_checked=0
    for arm in ARMS:
        for fit,seed in enumerate(plan['model_seeds']):
            folder=RUN/arm/f'fit_{fit:02d}';key=f'{arm}/fit_{fit:02d}/final.pt'
            assert r.sha(folder/'final.pt')==frozen[key]==manifest[key]
            ck=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
            online=r.models.Critic(defs[arm],seed).eval();online.load_state_dict(ck['online'])
            target=copy.deepcopy(online);target.load_state_dict(ck['target'])
            before=r.models.original.existing.base.old.digest((online.state_dict(),target.state_dict()))
            collected={name:[] for name in ['Q','target_current','G','reward','DDQN_residual','online_self_residual',
                'target_greedy_disagreement_fraction','target_regret_online_action','branch_cost_margin',
                'price_removed_action_disagreement_fraction']}
            terminal_decodes=0;price_states=0;price_changes=0
            for task_seed in plan['evaluation_seeds']:
                path=folder/f'evaluation_{task_seed}.json.gz';npz=path.with_suffix('').with_suffix('.npz')
                for file in (path,npz):
                    rel=file.relative_to(RUN).as_posix();h=r.sha(file);assert h==manifest[rel];inputs[rel]=h
                with gzip.open(path,'rt') as stream:row=json.load(stream)
                with np.load(npz) as z:states=z['states'].copy();following=z['next_states'].copy()
                actions=np.array([s['action'] for s in row['slots']],np.int64)
                episode={name:[] for name in collected}
                with torch.inference_mode():
                    for t in range(40):
                        s=torch.from_numpy(states[t:t+1]);act=torch.from_numpy(actions[t:t+1])
                        chosen,q=online.greedy(s);np.testing.assert_array_equal(chosen.numpy(),act.numpy())
                        tq=target(s,act)
                        np.testing.assert_allclose(q.numpy(),[row['chosen_q'][t]],rtol=1e-5,atol=1e-6)
                        np.testing.assert_allclose(tq.numpy(),[row['target_q'][t]],rtol=1e-5,atol=1e-6)
                        reward=row['slot_rewards'][t]
                        nextq=nexttarget=0.
                        if t<39:
                            np.testing.assert_array_equal(following[t],states[t+1])
                            ns=torch.from_numpy(following[t:t+1]);na,nq=online.greedy(ns)
                            np.testing.assert_array_equal(na.numpy(),actions[t+1:t+2])
                            nextq=float(nq[0]);nexttarget=float(target(ns,na)[0])
                        else:
                            assert r.models.original.remaining(torch.from_numpy(following[t:t+1])).item()==0
                        ta,tgreedy=target.greedy(s)
                        regret=float(tgreedy[0]-tq[0]);assert regret>=-1e-5
                        scores,context=online.tables(s);mask=torch.from_numpy(r.models.original.allowed(s))
                        sorted_scores=scores.masked_fill(~mask,float('inf')).sort(2).values
                        margin=(sorted_scores[:,:,1]-sorted_scores[:,:,0]).numpy()[0]
                        price_disagreement=0.
                        if arm=='screened_price24':
                            core=online.model.model
                            placement,motion,bias,extra=core.tables(s)
                            without=torch.cat((placement-extra['price'],motion),1).masked_fill(~mask,float('inf')).argmin(2)
                            price_disagreement=float((without!=act).float().mean())
                            price_states+=1;price_changes+=int(bool((without!=act).any()))
                        values=dict(Q=float(q[0]),target_current=float(tq[0]),G=row['realized_return_to_go'][t],
                            reward=reward,DDQN_residual=float(q[0])-reward-nexttarget,
                            online_self_residual=float(q[0])-reward-nextq,
                            target_greedy_disagreement_fraction=float((ta!=act).float().mean()),
                            target_regret_online_action=max(0.,regret),branch_cost_margin=margin.tolist(),
                            price_removed_action_disagreement_fraction=price_disagreement)
                        for name,value in values.items():episode[name].append(value)
                        states_checked+=1
                q=np.array(episode['Q']);g=np.array(episode['G']);e=np.array(episode['online_self_residual'])
                np.testing.assert_allclose(np.cumsum(e[::-1])[::-1],q-g,rtol=1e-5,atol=1e-6)
                for name,values in episode.items():collected[name].append(values)
            assert before==r.models.original.existing.base.old.digest((online.state_dict(),target.state_dict()))
            arrays={k:np.array(v) for k,v in collected.items()}
            def summarize(start,end):
                sub={k:v[:,start:end] for k,v in arrays.items()}
                return dict(start=start,end=end-1,metrics={k:dist(v) for k,v in sub.items()},
                    Q_minus_G=dist(sub['Q']-sub['G']),target_minus_Q=dist(sub['target_current']-sub['Q']))
            results[f'{arm}/{fit}']=dict(all_slots=summarize(0,40),terminal=summarize(39,40),
                time_bins=[summarize(t,t+10) for t in range(0,40,10)],
                price_removal=dict(states=price_states,states_with_any_action_change=price_changes,
                    scope='Same frozen primary core; remove explicit price table only. Diagnostic inference, not retrained ablation or policy rollout.'),
                terminal_next_state_decodes=terminal_decodes,weights_unchanged=True)
            print(f'Checked {arm} fit{fit}: 1920 consumed states',flush=True)
    r.write(OUT/'results.json',dict(fits=results,state_checks=states_checked,
        simulator_steps=0,optimizer_updates=0,terminal_next_state_decodes=0,final_seeds_used=False,
        interpretation='Own-policy realized returns and frozen one-step checks, not conditional expected Q-star labels. Target greedy regret is in model units and not true policy regret.'))
    r.write(OUT/'verification.json',dict(passed=True,fits=15,episodes=720,state_checks=states_checked,
        saved_predictions_match=True,trajectory_next_actions_match=True,self_residual_telescoping=True,
        weights_unchanged=True,new_simulator_steps=0,new_optimizer_updates=0,final_seeds_used=False))
    r.write(OUT/'input_manifest.json',inputs)
    r.write(OUT/'manifest.json',{x.relative_to(OUT).as_posix():r.sha(x) for x in OUT.rglob('*') if x.is_file()})

if __name__=='__main__':check()
