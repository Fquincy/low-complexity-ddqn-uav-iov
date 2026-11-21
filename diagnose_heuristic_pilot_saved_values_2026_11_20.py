"""Offline frozen-checkpoint/log diagnosis. No rollout, backward or optimizer."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
import heuristic_initialized_ddqn_baseline as baseline

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'development_runs/heuristic_initialization_paired_pilot_2026-11-19_01/run'
OUT=ROOT/'validation/heuristic_initialization_saved_value_diagnosis_2026-11-20_01'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def distribution(v):
    a=np.asarray(v,np.float64)
    return dict(count=int(a.size),mean=float(a.mean()),quantiles=np.quantile(a,[0,.05,.5,.95,1]).tolist()) if a.size else None


def main():
    if OUT.exists():raise RuntimeError('Preserve diagnosis output')
    torch.set_num_threads(1)
    manifest=read(RUN/'manifest.json');inputs={};results={};reproduced=0
    def use(p):
        name=p.relative_to(RUN).as_posix()
        if name not in inputs:
            assert sha(p)==manifest[name],name
            inputs[name]=manifest[name]
        return p
    for fit in range(3):
        demo_states=np.concatenate([np.load(use(p))['states'] for p in sorted((RUN/f'demo_{fit}').glob('*.npz'))])
        for arm in ['teacher_bc_then_ddqn','common_demo_no_bc']:
            folder=RUN/arm/f'fit_{fit}';key=f'{arm}/{fit}'
            model=baseline.TeacherInitializedCompactDDQN().eval()
            model.load_state_dict(torch.load(use(folder/'final.pt'),weights_only=True)['online'])
            probes=read(use(folder/'probes.json'));log=read(use(folder/'optimizer.json'))
            train_states=np.concatenate([np.load(use(p))['states'] for p in sorted(folder.glob('train_*.npz'))])
            score_stats={}
            for domain in ['demo','evaluation']:
                if domain=='demo':states=demo_states;saved_actions=np.asarray(probes['final']['actions'])
                else:
                    files=sorted((RUN/'evaluation'/arm/str(fit)).glob('*.npz'))
                    arrays=[dict(np.load(use(p))) for p in files]
                    states=np.concatenate([a['states'] for a in arrays]);saved_actions=np.concatenate([a['actions'] for a in arrays])
                cost=[];qq=[];slope=[];gaps=[];parts=[]
                with torch.no_grad():
                    # Preserve original batch-one inference semantics.
                    for s,a in zip(states,saved_actions):
                        x=torch.from_numpy(s)[None];aa,q,_=model.greedy(x)
                        assert np.array_equal(aa.numpy()[0],a),'greedy reproduction'
                        p,m,b,_=model.tables(x)
                        selected=p.gather(2,aa[:,:30,None]).sum()+m.gather(2,aa[:,30:,None]).sum()
                        c=b[0]+selected
                        assert np.allclose(float(-F.softplus(c)),float(q[0]),rtol=1e-5,atol=1e-6)
                        cost.append(float(c));qq.append(float(q[0]));slope.append(float(torch.sigmoid(c)))
                        mask=torch.from_numpy(baseline.allowed(x)[:,:30]);ordered=p.masked_fill(~mask,float('inf')).sort(2).values
                        gaps.extend((ordered[:,:,1]-ordered[:,:,0]).flatten().tolist())
                        parts.append([float(b[0]),float(p.gather(2,aa[:,:30,None]).sum()),float(m.gather(2,aa[:,30:,None]).sum())])
                        reproduced+=1
                score_stats[domain]=dict(preactivation=distribution(cost),Q=distribution(qq),
                    softplus_slope=distribution(slope),chosen_vehicle_gap=distribution(gaps),
                    abs_Q_below_1e_6_fraction=float(np.mean(np.abs(qq)<1e-6)),
                    slope_below_1e_6_fraction=float(np.mean(np.array(slope)<1e-6)),
                    mean_cost_parts=dict(zip(['bias','placement_sum','motion_sum'],np.mean(parts,axis=0).tolist())))
            bins={};by_horizon={};first_demo=[];first_online=[]
            for row in log:
                bin_id=min((row['update']-1)//468,3)
                target=np.array(row['target']);pred=np.array(row['prediction']);td=pred-target
                bucket=bins.setdefault(str(bin_id),dict(pred=[],td=[],loss=[],norm=[],nearzero=[]))
                bucket['pred'].extend(pred);bucket['td'].extend(td);bucket['loss'].append(row['loss']);bucket['norm'].append(row['gradient_norm']);bucket['nearzero'].extend(np.abs(pred)<1e-6)
                for j,identifier in enumerate(row['transition_ids']):
                    origin,index=identifier.split(':');s=demo_states[int(index)] if origin=='demo' else train_states[int(index)]
                    horizon_bin=str(min(int(round(float(s[-1])*40))//10,3))
                    by_horizon.setdefault(horizon_bin,[]).append(float(td[j]))
                    if row['update']==1:(first_demo if origin=='demo' else first_online).append(pred[j])
            timeline={k:dict(prediction=distribution(v['pred']),td_residual=distribution(v['td']),
                loss=distribution(v['loss']),preclip_norm=distribution(v['norm']),
                nearzero_prediction_fraction=float(np.mean(v['nearzero']))) for k,v in bins.items()}
            changed=np.mean(np.any(np.array(probes['after_bc']['actions'])!=np.array(probes['final']['actions']),axis=1))
            results[key]=dict(final_scores=score_stats,update_quarters=timeline,
                training_TD_by_current_slot_bin={k:distribution(v) for k,v in by_horizon.items()},
                first_update=dict(demo_predictions=distribution(first_demo),online_predictions=distribution(first_online)),
                demo_joint_actions_changed_after_initialization_fraction=float(changed),
                demo_agreement={k:dict(branch=v['branch_agreement'],joint=v['joint_agreement']) for k,v in probes.items()})
    # Exact invariance: adding the same constant to all logits in one branch
    # changes neither softmax preferences nor its argmin, but shifts joint cost.
    # This is an algebraic observation, not a fitted correction or new controller.
    OUT.mkdir(parents=True)
    result=dict(results=results,reproduced_batch_one_actions=reproduced,input_sha256=inputs,
                interpretation_limits=['No intermediate checkpoints: BC-end scalar values cannot be reconstructed',
                    'Logged first DDQN update occurs after128 exploratory online transitions',
                    'TD log statistics combine policy/replay distributions, not a controlled causal intervention',
                    'No BC-only closed-loop outcome exists in this pilot'],
                new_simulator_steps=0,new_optimizer_updates=0,new_backward_calls=0,retiming=False)
    (OUT/'diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for name,h in inputs.items():assert sha(RUN/name)==h
    print(json.dumps(dict(reproduced_actions=reproduced,consumed_files_verified_unchanged=len(inputs),new_optimizer_updates=0)))
    for key,row in results.items():
        print(json.dumps({'fit':key,'evaluation':row['final_scores']['evaluation'],
            'first_update':row['first_update'],'demo_actions_changed':row['demo_joint_actions_changed_after_initialization_fraction']}))


if __name__=='__main__':main()
