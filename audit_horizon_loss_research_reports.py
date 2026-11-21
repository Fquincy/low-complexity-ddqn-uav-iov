"""Independent diagnostic reconstruction. No calls to the report generator."""
import gzip
import json
from pathlib import Path
import numpy as np

def dist(x):
    v = np.asarray(x, np.float64)
    if len(v) == 0:
        return {'count': 0}
    assert np.isfinite(v).all()
    return {'count': len(v), 'mean': float(np.mean(v)), 'median': float(np.median(v)),
            'p05': float(np.percentile(v, 5)), 'p95': float(np.percentile(v, 95)),
            'min': float(np.min(v)), 'max': float(np.max(v))}

def audit(out, pl, p):
    out = Path(out)
    residual = json.loads((out / 'training_residuals_by_horizon.json').read_text())
    remote = json.loads((out / 'remote_decision_diagnostics.json').read_text())
    for report in (residual, remote):
        assert report['kind'] == pl['kind']
        assert report['quality_evidence'] == (pl['mode'] != 'integration-fixture')
    residual_keys, decision_keys = set(), set()
    for definition in p['learned_arms']:
        for fit in range(len(pl['model_seeds'])):
            key = f"{definition['id']}/{fit}"
            residual_keys.add(key)
            grouped = {}
            for line in (out / definition['id'] / f'fit_{fit:02d}' / 'optimizer.jsonl').open():
                row = json.loads(line)
                # Track the most recent chronological insertion for each circular index.
                last = row['physical_slots'] - 1
                cap = pl['cfg']['replay_capacity']
                assert len(row['replay_indices']) == len(row['chosen_Q']) == pl['cfg']['batch_size']
                for j, index in enumerate(row['replay_indices']):
                    chronological = last - (last - index) % cap
                    assert 0 <= chronological <= last
                    h = 40 - chronological % 40
                    # Recompute original FP32 residual, rather than trusting saved TD.
                    error = float(np.float32(row['chosen_Q'][j]) - np.float32(row['bellman_targets'][j]))
                    assert error == row['TD_residuals'][j]
                    b = grouped.setdefault(str(h), {'original': [], 'normalized': []})
                    b['original'].append(error)
                    b['normalized'].append(error / h)
            expected = {h: {unit: dist(values) for unit, values in b.items()} for h, b in grouped.items()}
            assert expected == residual['fits'][key], 'TD horizon distribution'
    learned = {d['id'] for d in p['learned_arms']}
    for name in list(learned) + [d['id'] for d in p['heuristics']]:
        for fit in range(len(pl['model_seeds'])) if name in learned else [None]:
            key = f'{name}/{fit}'
            decision_keys.add(key)
            counts = [[0]*5 for _ in range(40)]
            misses = [[0]*5 for _ in range(40)]
            stages, movements = {}, [[0]*5 for _ in range(2)]
            assignments, cpu, queue, distances = ([[] for _ in range(n)] for n in (5, 4, 4, 2))
            delays = {k: [] for k in ('uplink_wait_s', 'queue_delay_s', 'computation_delay_s')}
            for seed in pl['evaluation_seeds']:
                path = out / name / f'fit_{fit:02d}' / f'evaluation_{seed}.json.gz' if fit is not None else out / f'heuristic_{name}_{seed}.json.gz'
                with gzip.open(path, 'rt') as stream:
                    row = json.load(stream)
                episode_counts = [[0]*5 for _ in range(40)]
                for task in row['tasks']:
                    t, d = divmod(task['task_id'], 30)[0], task['destination']
                    counts[t][d] += 1
                    episode_counts[t][d] += 1
                    if not task['on_time']:
                        misses[t][d] += 1
                        if d == 0:
                            cause = 'local_deadline_miss'
                        else:
                            cutoff = task['arrival_s'] + task['deadline_s'] + 1e-10
                            if task['upload_finish_s'] is None:
                                cause = 'unfinished_upload'
                            elif task['upload_finish_s'] > cutoff:
                                cause = 'upload_finished_after_deadline'
                            elif task['compute_start_s'] is None:
                                cause = 'cpu_not_started'
                            elif task['compute_start_s'] > cutoff:
                                cause = 'cpu_started_after_deadline'
                            else:
                                cause = 'cpu_completed_late'
                            for field in delays:
                                if task[field] is not None:
                                    delays[field].append(task[field])
                        label = '/'.join(map(str, (t, d, cause)))
                        stages[label] = stages.get(label, 0) + 1
                assert sum(map(sum, episode_counts)) == 1200
                assert sum(not task['on_time'] for task in row['tasks']) == 1200 - row['tasks_on_time']
                for d in range(5):
                    assignments[d] += [slot[d] for slot in episode_counts]
                with np.load(path.with_suffix('').with_suffix('.npz')) as arrays:
                    for state in arrays['states']:
                        for d in range(4):
                            cpu[d].append(float(state[240+30+d]))
                            queue[d].append(float(state[240+34+240+3*d]))
                for slot in row['slots']:
                    for u, move in enumerate(slot['action'][-2:]):
                        movements[u][move] += 1
                        before, after = np.array(slot['uav_before'][u]), np.array(slot['uav_after'][u])
                        assert np.isclose(np.linalg.norm(after-before), slot['actual_distance_m'][u])
                        distances[u].append(slot['actual_distance_m'][u])
            expected = dict(task_counts_by_time_destination=counts, misses_by_time_destination=misses,
                failure_stages_by_time_destination=stages, assigned_tasks_per_slot=list(map(dist, assignments)),
                observed_log_cpu_workload_remote=list(map(dist, cpu)), observed_log_remote_queue_length=list(map(dist, queue)),
                remote_miss_delays={k: dist(v) for k, v in delays.items()}, movement_counts=movements,
                movement_distance_m=list(map(dist, distances)))
            assert expected == remote['policies'][key], 'remote diagnostic reconstruction'
    assert set(residual['fits']) == residual_keys
    assert set(remote['policies']) == decision_keys
    return dict(training_fit_reports=len(residual_keys), policy_reports=len(decision_keys),
                auditor_simulator_steps=0, auditor_optimizer_steps=0)
