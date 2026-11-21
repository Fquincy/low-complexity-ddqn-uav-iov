"""Posthoc diagnosis from frozen saved data; no simulator/teacher/optimizer calls."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import return_calibrated_handoff as models
import heuristic_initialized_ddqn_baseline as baseline

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'development_runs/return_calibrated_handoff_comparison_2026-11-20_01'
RUN = SOURCE/'run'
OUT = ROOT/'validation/return_calibrated_saved_trace_diagnosis_2026-11-20_01'
SPEC = ROOT/'RETURN_CALIBRATED_HANDOFF_COMPARISON_PROTOCOL_2026-11-20.json'
COMPLETION = ROOT/'validation/return_calibrated_comparison_completion_2026-11-20_01/verification.json'
A, B, C, D, E, F, G = list(models.ARMS)
CONTROLS = [A, B, C, D, E, F]
consumed = {}
decoded = 0


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def track(p):
    name = p.relative_to(ROOT).as_posix()
    if name not in consumed: consumed[name] = sha(p)
    return p


def read(p): return json.loads(track(p).read_text(encoding='utf-8-sig'))


def arrays(p):
    with np.load(track(p), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def stats(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    if not len(x): return dict(n=0)
    assert np.isfinite(x).all()
    return dict(n=len(x), mean=float(x.mean()), quantiles=np.quantile(x, [0, .05, .5, .95, 1]).tolist())


def task_stats(rows):
    rows = list(rows)
    by_dest = {}
    for d in range(5):
        part = [r for r in rows if r['destination'] == d]
        by_dest[str(d)] = dict(tasks=len(part), misses=sum(not r['on_time'] for r in part),
            **{k: stats([r[k] for r in part if r[k] is not None]) for k in
               ['delay_s', 'queue_delay_s', 'uplink_wait_s', 'transmission_delay_s', 'computation_delay_s', 'deadline_s']},
            tardiness=stats([max(0., r['delay_s']-r['deadline_s']) for r in part if r['delay_s'] is not None]))
    return dict(tasks=len(rows), misses=sum(not r['on_time'] for r in rows),
        unfinished=sum(r['unfinished_at_horizon'] for r in rows), by_destination=by_dest)


def paired_tasks(primary, other):
    assert len(primary) == len(other) == 19200
    pairs = list(zip(primary, other))
    fields = ['task_id', 'vehicle_id', 'arrival_s', 'input_bits', 'required_cycles', 'deadline_s']
    assert all(all(p[k] == b[k] for k in fields) for p, b in pairs)
    new = [(p, b) for p, b in pairs if not p['on_time'] and b['on_time']]
    reverse = [(p, b) for p, b in pairs if p['on_time'] and not b['on_time']]
    new_transitions = np.zeros((5, 5), dtype=int)
    for p, b in new: new_transitions[b['destination'], p['destination']] += 1
    return dict(primary_only_misses=len(new), control_only_misses=len(reverse),
        both_miss=sum(not p['on_time'] and not b['on_time'] for p, b in pairs),
        changed_destination=sum(p['destination'] != b['destination'] for p, b in pairs),
        primary_only_misses_same_destination=sum(p['destination'] == b['destination'] for p, b in new),
        primary_only_miss_control_to_primary_destinations=new_transitions.tolist(),
        new_miss_primary=task_stats(p for p, b in new), new_miss_control=task_stats(b for p, b in new),
        primary_misses_by_arrival_quarter=[sum(not p['on_time'] and 10*q <= p['arrival_s'] < 10*(q+1) for p in primary) for q in range(4)])


def predict(model, states):
    global decoded
    batches = []
    # Match the original evaluation boundary. GEMM batch size can change FP32
    # cancellation in the calibrated value head even with identical actions.
    for lo in range(len(states)):
        s = torch.from_numpy(states[lo:lo+1])
        with torch.no_grad():
            action, q, _ = model.greedy(s)
            p, m, v, _ = model.tables(s)
        scores = np.concatenate([p.numpy(), m.numpy()], axis=1)
        legal = baseline.allowed(s)
        sorted_scores = np.sort(np.where(legal, scores, np.inf), axis=2)
        assert np.array_equal(action.numpy(), np.where(legal, scores, np.inf).argmin(2))
        batches.append(dict(actions=action.numpy(), q=q.numpy(), value=v.numpy(),
            scores=scores, margins=sorted_scores[:, :, 1]-sorted_scores[:, :, 0]))
        decoded += len(s)
    return {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}


def selected_q(model, prediction, actions):
    scores = torch.from_numpy(prediction['scores'])
    with torch.no_grad():
        return model.value(scores[:, :30], scores[:, 30:], torch.from_numpy(prediction['value']),
                           torch.from_numpy(actions)).numpy()


def summarize_training(logs, demo_only=None, terminal=None):
    pred = np.array([v['prediction'] for v in logs])
    target = np.array([v['target'] for v in logs])
    rewards = np.array([v['rewards'] for v in logs])
    tq = np.array([v['target_q'] for v in logs])
    done = np.array([v['dones'] for v in logs], dtype=bool)
    demo = np.array([[x.startswith('demo:') for x in v['transition_ids']] for v in logs])
    assert np.allclose(target, rewards+tq, rtol=2e-6, atol=2e-5)
    assert (tq[done] == 0).all()
    mask = np.ones_like(done)
    if demo_only is not None: mask &= demo if demo_only else ~demo
    if terminal is not None: mask &= done if terminal else ~done
    delta = (target-pred)[mask]
    return dict(prediction=stats(pred[mask]), target=stats(target[mask]), reward=stats(rewards[mask]),
        target_q=stats(tq[mask]), target_minus_prediction=stats(delta),
        greedy_minus_sampled_Q=stats((np.array([v['current_greedy_q'] for v in logs])-pred)[mask]),
        upward_target_fraction=float(np.mean(delta > 0)) if len(delta) else None,
        mean_huber_target_direction=float(np.mean(np.clip(delta, -1, 1))) if len(delta) else None)


def training_window(logs):
    grads = {k: np.array([v['gradient_norms'][k] for v in logs]) for k in
             ['total', 'branch_output', 'value_output', 'shared_hidden']}
    assert np.allclose(grads['total']**2, sum(grads[k]**2 for k in grads if k != 'total'), rtol=1e-8, atol=1e-7)
    assert np.allclose([v['total_loss'] for v in logs],
        [v['td_loss']+v['retention_loss'] for v in logs], rtol=1e-6, atol=1e-6)
    return dict(first_update=logs[0]['update'], last_update=logs[-1]['update'],
        td_loss=stats([v['td_loss'] for v in logs]), retention_loss=stats([v['retention_loss'] for v in logs]),
        gradient_norms={k: stats(v) for k, v in grads.items()},
        shared_squared_norm_fraction=stats((grads['shared_hidden']/np.maximum(grads['total'], 1e-30))**2),
        clipped_fraction=float(np.mean([v['clipped'] for v in logs])),
        all=summarize_training(logs), terminal=summarize_training(logs, terminal=True),
        nonterminal=summarize_training(logs, terminal=False))


def main():
    if OUT.exists(): raise RuntimeError('Preserve previous diagnosis; output already exists')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
    completion = read(COMPLETION); spec = read(SPEC)
    assert completion['status'] == 'AUDIT_AND_ARCHIVE_REVERIFIED_POLICY_FAILED_VALUE_PASSED'
    assert sha(SPEC) == completion['protocol_sha256']
    manifest = read(RUN/'manifest.json'); freeze = read(RUN/'CHECKPOINT_FREEZE.json')
    for name, digest in spec['definition_sha256'].items():
        assert sha(track(ROOT/name)) == digest, name
    for name, digest in completion['evidence_sha256'].items():
        assert sha(track(SOURCE/name)) == digest, name
    analysis = read(RUN/'analysis.json')
    cfg = spec['simulator_defaults']
    weights = dict(normalized_delay_accrual=-cfg['delay_reward_weight'],
        normalized_uav_energy=-cfg['uav_energy_reward_weight'],
        new_deadline_misses_per_vehicle=-cfg['deadline_miss_penalty'])
    result = dict(kind='POSTHOC_SAVED_DATA_DIAGNOSIS_NOT_NEW_RESEARCH', fixed_state={}, probes={},
        training={}, paired_tasks={}, episode_reward_components={}, value_errors={})
    raw_predictions = {}
    for fit in range(3):
        fitted = {}
        for arm in CONTROLS:
            folder = RUN/arm/f'fit_{fit}'
            ckpath = track(folder/'final.pt'); ck = torch.load(ckpath, weights_only=True)
            assert sha(ckpath) == freeze['checkpoints'][f'{arm}/{fit}']
            m = models.Critic(arm).eval(); m.load_state_dict(ck['online']); fitted[arm] = m
        assert models.tensor_hash(fitted[E]) == freeze['handoff_hashes'][f'{A}/{fit}']
        assert freeze['handoff_hashes'][f'{A}/{fit}'] == freeze['handoff_hashes'][f'{B}/{fit}'] == freeze['handoff_hashes'][f'{E}/{fit}']
        demo = [arrays(RUN/f'demo_{fit}'/f'episode_{i:03d}.npz') for i in range(10)]
        demo_states = np.concatenate([v['states'] for v in demo]); demo_labels = np.concatenate([v['actions'] for v in demo])
        for arm in CONTROLS:
            key = f'{arm}/{fit}'; stages = read(RUN/arm/f'fit_{fit}/probes.json')['stages']
            start = stages['AFTER_VALUE_FIT' if models.ARMS[arm].calibration else 'AFTER_BC']
            end = stages['FINAL']; changed = np.array(start['actions']) != np.array(end['actions'])
            result['probes'][key] = dict(initial_imitation_loss=start['imitation_loss'],
                final_imitation_loss=end['imitation_loss'], placement_changed_fraction=float(changed[:, :30].mean()),
                motion_changed_fraction=float(changed[:, 30:].mean()),
                stages={stage: dict(imitation_loss=v['imitation_loss'], q=stats(v['greedy_q']), value=stats(v['value']),
                    margin=stats(v['margins'])) for stage, v in stages.items() if not v.get('skipped')})
        episodes = {}; tasks = {}; saved_arrays = {}
        for arm in list(models.ARMS)+['teacher']:
            base = RUN/'evaluation'/('teacher' if arm == 'teacher' else f'{arm}/{fit}')
            eps = [read(base/f'episode_{i:03d}.json') for i in range(16)]
            episodes[arm] = eps; tasks[arm] = [t for ep in eps for t in ep['records']]
            if arm in [A, E, F]:
                saved_arrays[arm] = [arrays(base/f'episode_{i:03d}.npz') for i in range(16)]
            parts = []
            for ep in eps:
                components = {k: sum(v['reward_components'][k] for v in ep['step_info'])*w for k, w in weights.items()}
                assert np.isclose(sum(components.values()), ep['return_value'], rtol=1e-10, atol=1e-8)
                assert np.isclose(-components['new_deadline_misses_per_vehicle']*30,
                    ep['summary']['deadline_misses_observed'], rtol=1e-10, atol=1e-8)
                parts.append(components)
            means = {k: float(np.mean([p[k] for p in parts])) for k in weights}
            result['episode_reward_components'][f'{arm}/{fit}'] = dict(episodes=parts, mean=means,
                mean_return=float(np.mean([ep['return_value'] for ep in eps])), tasks=task_stats(tasks[arm]))
            if arm != A:
                # Independently verify exogenous workload pairing for all controls.
                fields = ['task_id', 'vehicle_id', 'arrival_s', 'input_bits', 'required_cycles', 'deadline_s']
                assert all(all(p[k] == b[k] for k in fields) for p, b in zip(tasks[A], tasks[arm]))
                assert len(tasks[A]) == len(tasks[arm]) == 19200
        result['paired_tasks'][str(fit)] = {control: paired_tasks(tasks[A], tasks[control]) for control in [E, F]}
        result['episode_reward_components'][f'contrasts/{fit}'] = {
            control: {k: result['episode_reward_components'][f'{A}/{fit}']['mean'][k]-
                result['episode_reward_components'][f'{control}/{fit}']['mean'][k] for k in weights}
            for control in [E, F]}
        populations = dict(demonstrations=demo_states)
        populations.update({arm: np.concatenate([v['states'] for v in saved_arrays[arm]]) for arm in [A, E, F]})
        for population, states in populations.items():
            predictions = {arm: predict(m, states) for arm, m in fitted.items()}
            reference = predictions[E]; row = {}
            for arm, p in predictions.items():
                if population == arm:
                    source_episodes = saved_arrays[arm]
                    assert np.array_equal(p['actions'], np.concatenate([v['actions'] for v in source_episodes]))
                    assert np.allclose(p['q'], np.concatenate([v['q'] for v in source_episodes]), rtol=1e-5, atol=1e-5)
                if population == 'demonstrations':
                    original = read(RUN/arm/f'fit_{fit}/probes.json')['stages']['FINAL']
                    # Demonstration probes were recorded as one batch of 400.
                    with torch.no_grad(): probe_a, probe_q, _ = fitted[arm].greedy(torch.from_numpy(states))
                    assert np.array_equal(probe_a.numpy(), original['actions'])
                    assert np.allclose(probe_q.numpy(), original['greedy_q'], rtol=1e-5, atol=1e-5)
                    result['probes'][f'{arm}/{fit}']['single_vs_batch_probe_action_disagreements'] = int(np.sum(p['actions'] != probe_a.numpy()))
                    result['probes'][f'{arm}/{fit}']['single_vs_batch_probe_max_Q_difference'] = float(np.max(np.abs(p['q']-probe_q.numpy())))
                q_on_E = selected_q(fitted[arm], p, reference['actions'])
                changed = p['actions'] != reference['actions']
                q_delta = q_on_E-reference['q']; value_delta = p['value']-reference['value']
                adv_delta = (q_on_E-p['value'])-(reference['q']-reference['value'])
                if arm != F: assert np.allclose(q_delta, value_delta+adv_delta, rtol=2e-5, atol=3e-5)
                transition = np.bincount((5*reference['actions'][:, :30]+p['actions'][:, :30]).ravel(), minlength=25).reshape(5, 5)
                row[arm] = dict(q=stats(p['q']), value=stats(p['value']), branch_contribution=stats(p['q']-p['value']),
                    placement_change_vs_E=float(changed[:, :30].mean()), motion_change_vs_E=float(changed[:, 30:].mean()),
                    joint_change_vs_E=float(changed.any(1).mean()), placement_margin=stats(p['margins'][:, :30]),
                    motion_margin=stats(p['margins'][:, 30:]), E_to_model_destinations=transition.tolist(),
                    greedy_gain_over_E_action_Q=stats(p['q']-q_on_E),
                    fixed_E_action_Q_difference=stats(q_delta), value_difference=stats(value_delta),
                    centered_branch_difference=stats(adv_delta) if arm != F else None,
                    demo_label_placement_agreement=float((p['actions'][:, :30] == demo_labels[:, :30]).mean()) if population == 'demonstrations' else None)
                prefix = f'fit{fit}_{population}_{arm}'
                for name, val in dict(actions=p['actions'], q=p['q'], value=p['value'], q_on_E=q_on_E, margins=p['margins']).items():
                    raw_predictions[prefix+'_'+name] = val
            result['fixed_state'][f'{population}/{fit}'] = row
        for arm in [A, E, F]:
            eps = saved_arrays[arm]
            errors = np.concatenate([v['q']-np.cumsum(v['rewards'][::-1], dtype=np.float64)[::-1] for v in eps])
            for key in ['q', 'rewards']: assert len(np.concatenate([v[key] for v in eps])) == 640
            result['value_errors'][f'{arm}/{fit}'] = dict(all=stats(errors),
                terminal=stats(errors[39::40]), nonterminal=stats(errors[np.tile(np.arange(40) != 39, 16)]),
                by_slot_quarter=[stats(errors.reshape(16, 40)[:, q*10:(q+1)*10]) for q in range(4)])
        for arm in models.ARMS:
            if arm == E: continue
            logs = read(RUN/arm/f'fit_{fit}/optimizer.json'); assert len(logs) == 1872
            first = logs[0]['first_update_decomposition']
            assert np.allclose(first['residual'], np.array(first['reward'])+first['same_state_gap']+np.array(first['successor_minus_current']), rtol=2e-5, atol=2e-5)
            result['training'][f'{arm}/{fit}'] = dict(full=training_window(logs),
                windows=[training_window(logs[i:i+200]) for i in range(0, len(logs), 200)],
                demo=summarize_training(logs, demo_only=True), online=summarize_training(logs, demo_only=False),
                first_update={k: stats(first[k]) for k in ['reward', 'same_state_gap', 'successor_minus_current', 'residual']})
            del logs
        print(f'Completed fit {fit}: workload pairing, reward decomposition, fixed-state inference and saved update histories.', flush=True)
    for name, digest in consumed.items():
        p = ROOT/name
        assert sha(p) == digest, name
        if p.is_relative_to(RUN) and p != RUN/'manifest.json':
            assert manifest[p.relative_to(RUN).as_posix()] == digest, name
    assert analysis['policy_screen'] is False and analysis['value_screen'] is True and analysis['promising'] is False
    result.update(consumed_sha256=consumed, consumed_files_verified_unchanged=len(consumed), decoded_saved_states=decoded,
        protocol_sha256=sha(SPEC), script_sha256=sha(Path(__file__)),
        policy_screen=False, value_screen=True, promising=False,
        new_simulator_steps=0, new_optimizer_updates=0, new_teacher_calls=0,
        intermediate_checkpoint_reconstruction=False, new_research_launch_authorized=False,
        word_thesis_changed=False,
        limitations=['Posthoc descriptive diagnosis; no new prospective pass criteria.',
            'Final E equals A/B calibrated handoff; no intermediate training checkpoints retained.',
            'Saved replay sample populations change over time; windows are not fixed-state policy trajectories.',
            'Logged gradient norms combine TD and retention; separate historical gradient directions are unavailable.',
            'Changing queue/movement trajectories mean paired task outcomes are not causal counterfactuals.',
            'Same-state alternative choices were not simulated; their realized return/deadlines are unknown.',
            'Fixed-action Q decomposition is algebraic; direct common value output cannot change branch argmin, but shared-hidden training can.',
            'Single-state evaluation inference is retained; batch-400 demonstration probes are reproduced at their own boundary. Earlier batch-128 diagnosis assertion failure preserved separately.'])
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT/'fixed_state_predictions.npz', **raw_predictions)
    (OUT/'diagnosis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    receipt = dict(status='OFFLINE_DIAGNOSIS_COMPLETE', consumed_files_verified_unchanged=len(consumed),
        decoded_saved_states=decoded, artifacts_sha256={p.name: sha(p) for p in OUT.iterdir() if p.is_file()},
        script_sha256=result['script_sha256'], new_simulator_steps=0, new_optimizer_updates=0, new_teacher_calls=0,
        policy_screen=False, value_screen=True, promising=False, word_thesis_changed=False)
    (OUT/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
