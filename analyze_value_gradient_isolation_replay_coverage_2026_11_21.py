"""Saved behavior/evaluation distribution comparison, no simulation or inference."""
import json
from pathlib import Path

import numpy as np

import verify_value_gradient_isolation_diagnosis_2026_11_21 as verify

ROOT=verify.ROOT;RUN=verify.RUN;OUT=verify.OUT
CORE=[verify.P,verify.R,verify.A,verify.F]
consumed={}


def track(path):
    name=path.relative_to(ROOT).as_posix()
    if name not in consumed:consumed[name]=verify.sha(path)
    return path
def read(path):return json.loads(track(path).read_text(encoding='utf-8-sig'))
def arrays(path):
    with np.load(track(path),allow_pickle=False) as data:return {k:data[k] for k in data.files}
def stats(x):
    x=np.asarray(x,dtype=np.float64).ravel()
    if not len(x):return dict(n=0)
    assert np.isfinite(x).all()
    return dict(n=len(x),mean=float(x.mean()),quantiles=np.quantile(x,[0,.05,.5,.95,1]).tolist())
def empty_queue(states):
    columns=list(range(240,514))+[514,515,517,518,520,521,523,524]
    return (np.abs(states[:,columns]).max(1)==0)
def describe(episodes):
    states=np.concatenate([e['states'] for e in episodes]);rewards=np.concatenate([e['rewards'] for e in episodes])
    return dict(states=len(states),empty_queue_state_fraction=float(empty_queue(states).mean()),
        reward=stats(rewards),terminal_reward=stats([e['rewards'][-1] for e in episodes]),
        mean_episode_return=float(np.mean([e['rewards'].sum(dtype=np.float64) for e in episodes])),
        UAV_battery_min=float(states[:,[528,531]].min()),UAV_battery_mean=float(states[:,[528,531]].mean()))


def main():
    path=OUT/'behavior_replay_coverage.json'
    if path.exists():raise RuntimeError('Preserve previous replay coverage analysis')
    diagnosis=read(OUT/'diagnosis.json');manifest=read(RUN/'manifest.json');record=dict(kind='POSTHOC_SAVED_BEHAVIOR_REPLAY_COVERAGE_NOT_CAUSAL_TEST',fits={})
    for fit in range(3):
        demos=[arrays(RUN/f'demo_{fit}'/f'episode_{i:03d}.npz') for i in range(10)]
        dstate=np.concatenate([e['states'] for e in demos]);record['fits'][str(fit)]={}
        for arm in CORE:
            training=[arrays(RUN/arm/f'fit_{fit}'/f'train_{i:03d}.npz') for i in range(50)]
            evaluation=[arrays(RUN/'evaluation'/arm/str(fit)/f'episode_{i:03d}.npz') for i in range(16)]
            online_state=np.concatenate([e['states'] for e in training]);online_r=np.concatenate([e['rewards'] for e in training])
            rows=read(RUN/arm/f'fit_{fit}/optimizer.json');summary={}
            for label,logs in [('all',rows),('last_200',rows[-200:])]:
                sampled=[];is_demo=[]
                for update in logs:
                    for transition in update['transition_ids']:
                        kind,number=transition.split(':');index=int(number)
                        sampled.append(dstate[index] if kind=='demo' else online_state[index]);is_demo.append(kind=='demo')
                qs=empty_queue(np.stack(sampled));done=np.asarray([r['dones'] for r in logs],bool).ravel()
                pred=np.asarray([r['prediction'] for r in logs]).ravel();target=np.asarray([r['target'] for r in logs]).ravel()
                reward=np.asarray([r['rewards'] for r in logs]).ravel();tq=np.asarray([r['target_q'] for r in logs]).ravel()
                dm=np.asarray(is_demo,bool);assert len(qs)==len(dm)==len(pred)
                assert np.all(tq[done]==0) and np.allclose(target,reward+tq,rtol=3e-5,atol=1e-5)
                summary[label]=dict(sampled_transitions=len(qs),demo_fraction=float(dm.mean()),empty_queue_fraction=float(qs.mean()),
                    online_empty_queue_fraction=float(qs[~dm].mean()),
                    terminal_online_reward=stats(reward[done&~dm]),terminal_demo_reward=stats(reward[done&dm]),
                    terminal_online_prediction=stats(pred[done&~dm]),terminal_demo_prediction=stats(pred[done&dm]),
                    empty_queue_prediction=stats(pred[qs]),nonempty_queue_prediction=stats(pred[~qs]),
                    positive_prediction_fraction=float((pred>0).mean()),online_reward=stats(reward[~dm]))
                assert summary[label]['sampled_transitions']==len(logs)*64
            result=dict(demonstrations=describe(demos),online_behavior=describe(training),greedy_evaluation=describe(evaluation),
                replay_windows=summary)
            record['fits'][str(fit)][arm]=result
            if arm==verify.P:
                assert result['greedy_evaluation']['empty_queue_state_fraction']==1.
                print(f'Fit {fit} P: online empty queues {result["online_behavior"]["empty_queue_state_fraction"]:.4f}; evaluation 1.0; online terminal reward {result["online_behavior"]["terminal_reward"]["mean"]:.6f}.',flush=True)
            del rows
    for name,digest in consumed.items():
        p=ROOT/name;assert verify.sha(p)==digest,name
        if p.is_relative_to(RUN) and p!=RUN/'manifest.json':assert manifest[p.relative_to(RUN).as_posix()]==digest,name
    record.update(consumed_sha256=consumed,consumed_files_verified_unchanged=len(consumed),script_sha256=verify.sha(Path(__file__)),
        diagnosis_sha256=verify.sha(OUT/'diagnosis.json'),new_simulator_steps=0,new_models_created=0,new_optimizer_updates=0,
        new_teacher_calls=0,new_autograd_evaluations=0,policy_screen=False,value_screen=True,promising=False,
        limitations=['Observed distribution mismatch is not proof of a unique learning-failure cause.',
            'Training chosen actions include exploration; terminal rewards differ for state AND action distribution.',
            'DDQN is off-policy, so mismatch alone does not imply an invalid update or prove a proposed cure.'])
    with path.open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='SAVED_REPLAY_COVERAGE_ANALYSIS_COMPLETE',files=len(consumed),new_training=False),indent=2))


if __name__=='__main__':main()
