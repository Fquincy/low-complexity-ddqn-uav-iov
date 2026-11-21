"""Read saved development data only. No environment steps or optimizer calls."""
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'development_runs/controller_v3_development_2026-11-06_01'
ARMS=['amortized_workload_price24','amortized_no_price24','additive64_nonpositive','additive128_unbounded','plain24_nonpositive']
HEURISTICS=['all_local','nearest_rsu','uav_selective']
NAMES=['local','UAV1','UAV2','RSU1','RSU2']

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dist(values):
    x=np.asarray(values,dtype=float)
    if not len(x):return dict(count=0)
    assert np.isfinite(x).all()
    return dict(count=len(x),mean=float(x.mean()),p05=float(np.quantile(x,.05)),median=float(np.median(x)),p95=float(np.quantile(x,.95)),minimum=float(x.min()),maximum=float(x.max()))

def episode_rows(paths):
    for p in paths:
        with gzip.open(p,'rt',encoding='utf-8') as f:yield json.load(f)

def policy_summary(rows):
    destinations={name:dict(tasks=0,on_time=0,missed=0,unfinished=0,upload_wait=[],tx=[],cpu_wait=[],compute=[],delay=[],missed_wait=[],missed_tx=[]) for name in NAMES}
    q=[];g=[];tq=[];reward=[];terminal=[];nonterminal=[];motion=defaultdict(int);returns=[];deadlines=[];by_time=defaultdict(list)
    for row in rows:
        returns.append(row['return_sum']);deadlines.append(row['deadline_success'])
        if row.get('chosen_q'):
            q+=row['chosen_q'];g+=row['realized_return_to_go'];tq+=row['target_q'];reward+=row['slot_rewards']
            terminal.append(row['chosen_q'][-1]-row['slot_rewards'][-1])
            nonterminal.extend(row['chosen_q'][i]-(row['slot_rewards'][i]+row['target_q'][i+1]) for i in range(39))
            # target_q at i+1 evaluates the online greedy action on the same trajectory.
            for i in range(40):by_time[i].append((row['chosen_q'][i],row['realized_return_to_go'][i]))
        for slot in row['slots']:
            motion[str(slot['action'][-2:])]+=1
        for task in row['tasks']:
            d=destinations[NAMES[task['destination']]];d['tasks']+=1;d['on_time']+=int(task['on_time'])
            d['missed']+=int(not task['on_time']);d['unfinished']+=int(task['unfinished_at_horizon'])
            for key,field in [('upload_wait','uplink_wait_s'),('tx','transmission_delay_s'),('cpu_wait','queue_delay_s'),('compute','computation_delay_s'),('delay','observed_elapsed_s')]:
                if task.get(field) is not None:d[key].append(task[field])
            if not task['on_time']:
                if task.get('uplink_wait_s') is not None:d['missed_wait'].append(task['uplink_wait_s'])
                if task.get('transmission_delay_s') is not None:d['missed_tx'].append(task['transmission_delay_s'])
    total=sum(d['tasks'] for d in destinations.values());missed=sum(d['missed'] for d in destinations.values())
    for d in destinations.values():
        d['task_fraction']=d['tasks']/total;d['deadline_success']=d['on_time']/max(1,d['tasks'])
        d['share_of_all_missed']=d['missed']/max(1,missed)
        for k in ['upload_wait','tx','cpu_wait','compute','delay','missed_wait','missed_tx']:d[k]=dist(d[k])
    result=dict(episodes=len(returns),mean_return=float(np.mean(returns)),deadline_success=float(np.mean(deadlines)),
        destinations=destinations,movement_counts=dict(motion))
    if q:
        result.update(chosen_q=dist(q),realized_return_to_go=dist(g),target_q=dist(tq),reward=dist(reward),
            terminal_q_minus_reward=dist(terminal),nonterminal_bellman_residual=dist(nonterminal),
            residual_label='Qonline(s,pi(s))-[r+Qtarget(s_next,pi_online(s_next))] on saved own-policy greedy trajectory',
            time_bins={str(k):dict(q_mean=float(np.mean([x[0] for x in v])),g_mean=float(np.mean([x[1] for x in v]))) for k,v in by_time.items()})
    return result

def optimizer_summary(path):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    total_samples=0;terminal_samples=0;count_disagreement=0;phases={}
    for index,row in enumerate(rows):
        inferred=sum(i%40==39 for i in row['replay_indices'])
        count_disagreement+=int(inferred!=row['terminal_samples'])
        terminal_samples+=inferred;total_samples+=len(row['replay_indices'])
    assert count_disagreement==0
    for name,selected in [('first500',rows[:500]),('last500',rows[-500:])]:
        phases[name]=dict(loss=dist([r['loss'] for r in selected]),gradient_norm=dist([r['grad_norm_before_clip'] for r in selected]),
            clipped_fraction=float(np.mean([r['gradient_clipped'] for r in selected])),
            chosen_q_mean=dist([r['chosen_q']['mean'] for r in selected]),
            bellman_target_mean=dist([r['bellman_target']['mean'] for r in selected]),
            td_residual_mean=dist([r['td_residual']['mean'] for r in selected]))
    return dict(updates=len(rows),terminal_fraction=terminal_samples/total_samples,
        terminal_counts_reconstructed_from_replay_indices=True,phases=phases,
        limitation='Logs have mixed-minibatch summaries, not per-sample gradients or terminal/nonterminal losses.')

def main():
    plan=read(OUT/'runtime_plan.json');manifest_path=OUT/'run_manifest.json';manifest_sha=sha(manifest_path)
    audit_sha=sha(OUT/'saved_artifact_audit.json');protocol_sha=sha(ROOT/'LOW_COMPLEXITY_V3_RESEARCH_PROTOCOL_2026-11-05.json')
    results={};training={}
    for arm in ARMS:
        results[arm]=[];training[arm]=[]
        for fit,seed in enumerate(plan['model_seeds']):
            folder=OUT/arm/f'fit_{fit:02d}'
            paths=[folder/f'evaluation_{s}.json.gz' for s in plan['evaluation_seeds']]
            summary=policy_summary(episode_rows(paths));summary['model_seed']=seed;results[arm].append(summary)
            training[arm].append(dict(model_seed=seed,**optimizer_summary(folder/'optimizer.jsonl')))
            print(f'Checked {arm} fit {seed}',flush=True)
    heuristic={name:policy_summary(episode_rows([OUT/f'heuristic_{name}_{s}.json.gz' for s in plan['evaluation_seeds']])) for name in HEURISTICS}
    assert sha(manifest_path)==manifest_sha and sha(OUT/'saved_artifact_audit.json')==audit_sha
    assert sha(ROOT/'LOW_COMPLEXITY_V3_RESEARCH_PROTOCOL_2026-11-05.json')==protocol_sha
    data=dict(state='saved_data_diagnosis_no_new_training',evaluation=results,optimizer=training,heuristics=heuristic,
        preserved_run_manifest_sha256=manifest_sha,preserved_artifact_audit_sha256=audit_sha,
        protocol_sha256=protocol_sha,training_updates_added=0,simulator_steps_added=0,final_seeds_consumed=0,
        limitations=['Destination statistics are descriptive, not counterfactual causal effects.',
            'Own-policy Bellman residual uses target Q for next online-greedy action, not a fresh maximization.',
            'Task delay component summaries exclude unavailable values and may reflect completion censoring.',
            'No per-sample training gradients or intermediate weights were saved; cause cannot be proven from aggregates.'])
    dest=ROOT/'validation/v3_saved_data_diagnosis_2026-11-06.json';dest.write_text(json.dumps(data,indent=2),encoding='utf-8')
    report=['VERSION 3 SAVED-DATA DIAGNOSIS','Date: November 6, 2026','','SCOPE',
        'Read saved development traces only: five fits of primary, no-price,',
        'compact64, wide128 and plain24 plus all three heuristics. No simulation,',
        'training, model mutation or final-evaluation seed use. Completed-run',
        'manifest, artifact-audit and frozen-protocol hashes remained unchanged.','',
        'TERMINAL AND NONTERMINAL VALUE EVIDENCE',
        'At termination gamma bootstrap is masked correctly in the inspected',
        'DDQN target code. The observation already includes normalized time as',
        'its last component. Missing time or a missing done mask is not supported',
        'as an explanation by this source inspection. The current architecture',
        'does not enforce a remaining-horizon value scale.',
        'Rows show terminal signed Q-minus-reward, nonterminal signed Bellman',
        'residual, and Q versus realized return-to-go at slot39.','',
        'arm | seed | terminal error mean | nonterminal TD mean | terminal Q mean | terminal G mean']
    for arm in ARMS:
        for row in results[arm]:
            report.append(f"{arm} | {row['model_seed']} | {row['terminal_q_minus_reward']['mean']:.4f} | {row['nonterminal_bellman_residual']['mean']:.4f} | {row['time_bins']['39']['q_mean']:.4f} | {row['time_bins']['39']['g_mean']:.4f}")
    report+=['','TRAINING TRACE EVIDENCE',
        'Replay indices reconstruct terminal samples exactly because each episode',
        'is40 slots and the8,000-slot run does not wrap its50,000-slot buffer.',
        'Huber loss clips the derivative magnitude for large TD errors; extreme',
        'terminal errors therefore do not receive squared-error gradient growth.',
        'Uniform replay gives terminal states only roughly one fortieth of samples.',
        'This is a plausible imbalance mechanism, not a proven sole cause.',
        'arm | seed | terminal sample % | last500 loss mean | grad norm mean | clipping % | chosen Q mean | target mean']
    for arm in ARMS:
        for row in training[arm]:
            p=row['phases']['last500'];report.append(f"{arm} | {row['model_seed']} | {100*row['terminal_fraction']:.3f} | {p['loss']['mean']:.4f} | {p['gradient_norm']['mean']:.4f} | {100*p['clipped_fraction']:.2f} | {p['chosen_q_mean']['mean']:.4f} | {p['bellman_target_mean']['mean']:.4f}")
    report+=['','DESTINATION AND DEADLINE EVIDENCE',
        'Task fractions and success rates are descriptive. Different policies',
        'create different queues; comparing destinations cannot isolate causality.',
        'arm | seed | destination | task % | on-time % | share of all misses % | upload wait mean s | CPU queue wait mean s']
    for arm,rows in {**results,**{k:[dict(v,model_seed='heuristic')] for k,v in heuristic.items()}}.items():
        for row in rows:
            for name,d in row['destinations'].items():
                report.append(f"{arm} | {row['model_seed']} | {name} | {100*d['task_fraction']:.3f} | {100*d['deadline_success']:.3f} | {100*d['share_of_all_missed']:.3f} | {d['upload_wait'].get('mean',0):.5f} | {d['cpu_wait'].get('mean',0):.5f}")
    report+=['','ARCHITECTURAL IMPLICATION',
        'The price term uses CPU workload and CPU queue features. It does not',
        'explicitly compute upload wait, wireless rate or deadline feasibility.',
        'The learned placement head can use radio/deadline observations, so this',
        'is an omitted explicit inductive bias, not absence of those state inputs.',
        'A separable critic also approximates shared capacity and decisions from',
        '30 vehicles; efficient decoding alone cannot ensure deadline performance.','',
        'DECISION',
        'Keep final evaluation untouched. The saved evidence supports investigating',
        'finite-horizon terminal anchoring first, with matched baseline/ablation',
        'controls. It does not justify claiming that this alone repairs deadlines.',
        'Do not tune or retrain the frozen Version3 run. A prospective amendment',
        'must separate the anchoring mechanism from any later feasibility decoder.',
        'Details and distributions: validation/v3_saved_data_diagnosis_2026-11-06.json']
    (ROOT/'LOW_COMPLEXITY_V3_SAVED_DATA_DIAGNOSIS_2026-11-06.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(dict(state=data['state'],output=str(dest)),indent=2))

if __name__=='__main__':main()
