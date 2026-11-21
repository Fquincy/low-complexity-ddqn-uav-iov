"""Offline diagnostics from saved evidence; no policy inputs or model changes."""
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return dict(count=0)
    assert np.isfinite(a).all()
    return dict(count=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                p05=float(np.quantile(a, .05)), p95=float(np.quantile(a, .95)),
                min=float(a.min()), max=float(a.max()))

def generate(out, plan, protocol):
    out = Path(out)
    residuals = {}
    decisions = {}
    for arm in protocol['learned_arms']:
        for fit in range(len(plan['model_seeds'])):
            folder = out / arm['id'] / f'fit_{fit:02d}'
            buckets = defaultdict(lambda: dict(original=[], normalized=[]))
            with (folder / 'optimizer.jsonl').open() as stream:
                for line in stream:
                    row = json.loads(line)
                    ix = np.asarray(row['replay_indices'], dtype=np.int64)
                    cap = plan['cfg']['replay_capacity']
                    inserted = ix + cap * ((row['physical_slots'] - 1 - ix) // cap)
                    horizons = 40 - inserted % 40
                    if 'current_h' in row:
                        np.testing.assert_array_equal(row['current_h'], horizons)
                    for h, value in zip(horizons.tolist(), row['TD_residuals']):
                        buckets[h]['original'].append(value)
                        buckets[h]['normalized'].append(value / h)
            residuals[f"{arm['id']}/{fit}"] = {
                str(h): {unit: distribution(values) for unit, values in b.items()}
                for h, b in sorted(buckets.items())}
    names = [x['id'] for x in protocol['learned_arms'] + protocol['heuristics']]
    learned = {x['id'] for x in protocol['learned_arms']}
    for arm in names:
        for fit in range(len(plan['model_seeds'])) if arm in learned else [None]:
            counts = np.zeros((40, 5), np.int64)
            misses = counts.copy()
            moves = np.zeros((2, 5), np.int64)
            stages = Counter()
            assigned = [[] for _ in range(5)]
            cpu = [[] for _ in range(4)]
            queues = [[] for _ in range(4)]
            distance = [[], []]
            delays = {key: [] for key in ('uplink_wait_s', 'queue_delay_s', 'computation_delay_s')}
            for seed in plan['evaluation_seeds']:
                path = (out / arm / f'fit_{fit:02d}' / f'evaluation_{seed}.json.gz'
                        if fit is not None else out / f'heuristic_{arm}_{seed}.json.gz')
                with gzip.open(path, 'rt') as stream:
                    row = json.load(stream)
                per_episode = np.zeros((40, 5), np.int64)
                for task in row['tasks']:
                    t, d = int(task['task_id'] // 30), int(task['destination'])
                    counts[t, d] += 1
                    per_episode[t, d] += 1
                    if task['on_time']:
                        continue
                    misses[t, d] += 1
                    if d == 0:
                        stage = 'local_deadline_miss'
                    else:
                        deadline = task['arrival_s'] + task['deadline_s']
                        up, start = task['upload_finish_s'], task['compute_start_s']
                        stage = ('unfinished_upload' if up is None else
                                 'upload_finished_after_deadline' if up > deadline + 1e-10 else
                                 'cpu_not_started' if start is None else
                                 'cpu_started_after_deadline' if start > deadline + 1e-10 else
                                 'cpu_completed_late')
                        for key in delays:
                            if task[key] is not None:
                                delays[key].append(task[key])
                    stages[f'{t}/{d}/{stage}'] += 1
                for d in range(5):
                    assigned[d].extend(per_episode[:, d].tolist())
                npz = path.with_suffix('').with_suffix('.npz')
                with np.load(npz) as arrays:
                    states = arrays['states']
                    for d in range(4):
                        cpu[d].extend(states[:, 270+d].tolist())
                        queues[d].extend(states[:, 514+3*d].tolist())
                for slot in row['slots']:
                    for u in range(2):
                        moves[u, slot['action'][30+u]] += 1
                        distance[u].append(slot['actual_distance_m'][u])
            decisions[f'{arm}/{fit}'] = dict(
                task_counts_by_time_destination=counts.tolist(),
                misses_by_time_destination=misses.tolist(),
                failure_stages_by_time_destination=dict(sorted(stages.items())),
                assigned_tasks_per_slot=[distribution(x) for x in assigned],
                observed_log_cpu_workload_remote=[distribution(x) for x in cpu],
                observed_log_remote_queue_length=[distribution(x) for x in queues],
                remote_miss_delays={k: distribution(v) for k, v in delays.items()},
                movement_counts=moves.tolist(), movement_distance_m=[distribution(x) for x in distance])
    common = dict(kind=plan['kind'], quality_evidence=plan['mode'] != 'integration-fixture')
    reports = {
        'training_residuals_by_horizon.json': dict(**common, fits=residuals,
            units='Original TD and TD/current_h; normalization is diagnostic for unchanged control losses'),
        'remote_decision_diagnostics.json': dict(**common, policies=decisions,
            interpretation='Observed task lifecycle categories, not causal proofs. Counts by arrival slot/destination; queue/workload fields retain observed log units.')}
    for name, value in reports.items():
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
