"""Offline final-checkpoint gradients and saved task timelines; no optimizer steps."""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import assignment_price_prototype as adp

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'development_runs/assignment_price_development_2026-11-16_01/run'
OUT = ROOT / 'validation/assignment_price_offline_diagnosis_2026-11-19_01'
ARMS = ('coupled_sweep', 'coupled_no_sweep', 'separable64', 'fixed_price_plain_sweep')
EPISODES = (0, 12, 24, 36)
USED = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def track(path):
    USED[str(path)] = digest(path)
    return path


def read(path):
    return json.loads(track(path).read_text(encoding='utf-8'))


def episode(path):
    with gzip.open(track(path), 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    assert a.size and np.isfinite(a).all()
    return dict(count=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                p05=float(np.quantile(a, .05)), p95=float(np.quantile(a, .95)),
                min=float(a.min()), max=float(a.max()))


def gradient(model, loss):
    grads = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    return torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
                      for p, g in zip(model.parameters(), grads)]).double()


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=False)
    plan = dict(kind='POST_HOC_OFFLINE_DIAGNOSTICS_NOT_NEW_POLICY_EVIDENCE',
                arms=ARMS, fits=list(range(5)), episodes=EPISODES,
                state_source='primary saved evaluation traces; identical states and actions for each arm',
                bins=[[0, 9], [10, 19], [20, 29], [30, 39]],
                gradient_probe='mean original-unit scalar Huber with frozen target; final checkpoints only',
                sensitivity_probe='gradient of mean Q and mean Q/h; analytical diagnostic only, no alternative training loss',
                timeline_scope='all 48 evaluation episodes, five primary fits, matched saved heuristic tasks',
                no_simulation=True, no_optimizer_steps=True, no_retiming=True,
                final_seeds_used=False, historical_gradient_reconstruction=False)
    (OUT/'DIAGNOSTIC_PLAN.json').write_text(json.dumps(plan, indent=2)+'\n')
    protocol = read(ROOT/'ASSIGNMENT_PRICE_RESEARCH_PROTOCOL_2026-11-15.json')
    seeds = protocol['development']['model_seeds']
    gradients = []
    for fit in range(5):
        all_states, all_next, all_actions, all_rewards, all_dones = [], [], [], [], []
        for ep in EPISODES:
            folder = RUN/'coupled_sweep'/f'fit_{fit:02d}'
            row = episode(folder/f'eval_{ep:03d}.json.gz')
            with np.load(track(folder/f'eval_{ep:03d}.npz')) as data:
                all_states.append(data['states']); all_next.append(data['next_states'])
            all_actions.extend(x['action'] for x in row['slots'])
            all_rewards.extend(row['slot_rewards']); all_dones.extend(x['done'] for x in row['slots'])
        states = torch.from_numpy(np.concatenate(all_states))
        next_states = torch.from_numpy(np.concatenate(all_next))
        actions = torch.tensor(all_actions, dtype=torch.int64)
        rewards = torch.tensor(all_rewards, dtype=torch.float64)
        dones = torch.tensor(all_dones, dtype=torch.bool)
        time = np.tile(np.arange(40), len(EPISODES))
        for arm in ARMS:
            path = track(RUN/arm/f'fit_{fit:02d}'/'final.pt')
            ck = torch.load(path, map_location='cpu', weights_only=True)
            online = adp.Critic(arm, seeds[fit]).eval()
            target = adp.Critic(arm, seeds[fit]).eval().requires_grad_(False)
            online.load_state_dict(ck['online']); target.load_state_dict(ck['target'])
            online_before = {k: v.detach().clone() for k,v in online.state_dict().items()}
            terminal = torch.from_numpy(time == 39)
            anchor_loss = F.smooth_l1_loss(online(states[terminal], actions[terminal]), rewards[terminal])
            anchor_grad = gradient(online, anchor_loss)
            for start in range(0, 40, 10):
                ix = torch.from_numpy((time >= start) & (time < start+10))
                s, act = states[ix], actions[ix]
                y = adp.targets(online, target, rewards[ix], next_states[ix], dones[ix])
                q = online(s, act)
                loss = F.smooth_l1_loss(q, y)
                g = gradient(online, loss)
                sensitivity = gradient(online, online(s,act).mean())
                unit = gradient(online, (online(s,act)/adp.remaining(s).double()).mean())
                norm = float(g.norm()); anorm = float(anchor_grad.norm())
                cosine = float(torch.dot(g, anchor_grad)/(g.norm()*anchor_grad.norm())) if norm and anorm else None
                record = dict(arm=arm, fit=fit, slots=[start,start+9], states=int(ix.sum()),
                              loss=float(loss.detach()), Q=stats(q.detach().numpy()),
                              Bellman_target=stats(y.numpy()), residual=stats((q.detach()-y).numpy()),
                              gradient_norm=norm, anchor_gradient_norm=anorm,
                              uniform_anchor_cosine=cosine,
                              combined_gradient_norm=float((g+anchor_grad).norm()),
                              hypothetical_clip_multiplier=min(1., 10./max(norm,1e-30)),
                              mean_Q_gradient_norm=float(sensitivity.norm()),
                              mean_Q_per_h_gradient_norm=float(unit.norm()))
                assert all(torch.equal(v, online_before[k]) for k,v in online.state_dict().items())
                gradients.append(record)
            print(f'gradients complete: {arm} fit {fit}', flush=True)
    heuristic = [episode(RUN/'heuristics/observed_load_radio'/f'eval_{ep:03d}.json.gz') for ep in range(48)]
    timelines, examples = [], []
    for fit in range(5):
        counts = Counter(); delays = {d: {k: [] for k in ('transmission_delay_s','uplink_wait_s','queue_delay_s','computation_delay_s','delay_s')} for d in range(5)}
        reward_primary, reward_heuristic = Counter(), Counter()
        for ep in range(48):
            row = episode(RUN/'coupled_sweep'/f'fit_{fit:02d}'/f'eval_{ep:03d}.json.gz')
            ref = heuristic[ep]
            assert (row['seed'], row['workload_sha256']) == (ref['seed'],ref['workload_sha256'])
            lookup = {t['task_id']: t for t in ref['tasks']}
            assert len(lookup) == len(row['tasks']) == 1200
            for ours, theirs in zip(row['slots'],ref['slots']):
                for k,v in ours['reward_components'].items(): reward_primary[k] += v
                for k,v in theirs['reward_components'].items(): reward_heuristic[k] += v
                counts['primary_nonzero_movement_branches'] += sum(a != 0 for a in ours['action'][30:])
            for task in row['tasks']:
                other = lookup[task['task_id']]
                for key in ('arrival_s','input_bits','required_cycles','deadline_s','vehicle_id'):
                    assert task[key] == other[key]
                miss, refmiss = not task['on_time'], not other['on_time']
                counts['tasks'] += 1; counts['primary_not_on_time'] += miss; counts['heuristic_not_on_time'] += refmiss
                counts['primary_miss_heuristic_success'] += miss and not refmiss
                counts['primary_success_heuristic_miss'] += not miss and refmiss
                counts['both_miss'] += miss and refmiss
                if miss:
                    d = task['destination']; counts[f'not_on_time_destination_{d}'] += 1
                    counts['unfinished'] += task['unfinished_at_horizon']
                    for k in delays[d]:
                        if task[k] is not None: delays[d][k].append(task[k])
                    components = {k:task[k] for k in delays[d]}
                    for k,val in components.items():
                        if val is not None and k != 'delay_s': counts[f'{k}_alone_exceeds_deadline'] += val > task['deadline_s']
                    margin = (task['delay_s'] if task['delay_s'] is not None else task['observed_elapsed_s'])-task['deadline_s']
                    examples.append(dict(fit=fit, episode=ep, task_id=task['task_id'],
                        arrival_s=task['arrival_s'], destination=d, deadline_s=task['deadline_s'],
                        primary_components=components, primary_status=task['status'], margin_s=margin,
                        heuristic_destination=other['destination'], heuristic_on_time=other['on_time'],
                        heuristic_delay_s=other['delay_s']))
        timelines.append(dict(fit=fit, counts=dict(counts),
            missed_task_component_distributions={str(d):{k:stats(vals) for k,vals in fields.items() if vals} for d,fields in delays.items()},
            reward_component_sums_primary=dict(reward_primary),reward_component_sums_heuristic=dict(reward_heuristic)))
        print(f'timelines complete: fit {fit}',flush=True)
    examples.sort(key=lambda x:x['margin_s'], reverse=True)
    # Hash every read input again: checkpoint/trace/report integrity is unchanged.
    assert all(digest(Path(path)) == sha for path,sha in USED.items())
    result = dict(plan=plan, gradients=gradients, task_timelines=timelines,
                  largest_observed_lateness_examples=examples[:20],
                  examples_are_post_hoc_not_representative=True,
                  input_sha256=USED, inputs_unchanged=True,
                  new_training_slots=0,new_optimizer_updates=0,new_evaluation_slots=0,
                  final_phase_launched=False,word_thesis_changed=False)
    (OUT/'diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT),gradient_batches=len(gradients),input_files=len(USED),inputs_unchanged=True)),flush=True)


if __name__ == '__main__':
    main()
