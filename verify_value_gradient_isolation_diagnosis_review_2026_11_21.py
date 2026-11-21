"""Bind the offline diagnosis pack and independently count primary replay coverage.

Reads saved arrays/logs only. Does not import models, torch, simulator or teacher.
The scientific decisions are inherited unchanged from the completed experiment.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'validation/value_gradient_isolation_saved_trace_diagnosis_2026-11-21_01'
SOURCE = ROOT / 'development_runs/value_gradient_isolation_comparison_2026-11-20_01'
RUN = SOURCE / 'run'
P = 'value_gradient_isolated_ddqn'
SPEC = ROOT / 'VALUE_GRADIENT_ISOLATION_COMPARISON_PROTOCOL_2026-11-20.json'
SPEC_SHA = '2a3a4cd58f3f04eabf1d54958e058af7294fbb73a2f41000c3242999bcc4f683'
COMPLETION = ROOT / 'validation/value_gradient_isolation_comparison_completion_2026-11-21_01/verification.json'
COMPLETION_SHA = 'd0420a2504e10278fcad7cfbc29f2fb373afb7efd01bbe24dfb1cd1cd294c9fd'
PREVIOUS_REVIEW = ROOT / 'VALUE_GRADIENT_ISOLATION_DEVELOPMENT_RESULTS_REVIEW_2026-11-21.txt'
PREVIOUS_REVIEW_SHA = 'a7d321765bd03d395a83c40829f1174d0c5e2831600b3de686a28c26d8bf8a69'
REVIEW = ROOT / 'VALUE_GRADIENT_ISOLATION_SAVED_TRACE_DIAGNOSIS_2026-11-21.txt'
SCRIPTS = [
    'diagnose_value_gradient_isolation_saved_traces_2026_11_21.py',
    'verify_value_gradient_isolation_diagnosis_2026_11_21.py',
    'analyze_value_gradient_isolation_replay_coverage_2026_11_21.py',
    Path(__file__).name,
]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected):
    assert np.isclose(actual, expected, rtol=1e-10, atol=1e-10), (actual, expected)


def check_stats(values, expected):
    values = np.asarray(values, dtype=np.float64)
    assert len(values) == expected['n'] and np.isfinite(values).all()
    if len(values):
        close(values.mean(), expected['mean'])
        assert np.allclose(np.quantile(values, [0, .05, .5, .95, 1]),
                           expected['quantiles'], rtol=1e-10, atol=1e-10)


def episode_lookup(folder, prefix, count, inputs):
    """Separate boolean/counter implementation; no concatenated state matrix."""
    empty, terminal, rewards, terminal_rewards = [], [], [], []
    for index in range(count):
        path = folder / f'{prefix}_{index:03d}.npz'
        assert path.relative_to(ROOT).as_posix() in inputs
        with np.load(path, allow_pickle=False) as data:
            state = data['states']
            assert state.shape == (40, 533) and np.isfinite(state).all()
            flags = np.ones(len(state), dtype=bool)
            # Queue layout: 240-513 vehicles/UAVs, then paired RSU queues.
            for column in (*range(240, 514), 514, 515, 517, 518, 520, 521, 523, 524):
                flags &= state[:, column] == 0
            done = data['dones']
            assert done.shape == (40,) and done[-1] and not done[:-1].any()
            empty.extend(flags.tolist())
            terminal.extend(done.tolist())
            rewards.extend(data['rewards'].tolist())
            terminal_rewards.append(float(data['rewards'][-1]))
    return dict(empty=empty, terminal=terminal, rewards=rewards,
                terminal_rewards=terminal_rewards)


def verify_primary_coverage(coverage, inputs):
    result = {}
    for fit in range(3):
        expected = coverage['fits'][str(fit)][P]
        populations = {
            'demonstrations': episode_lookup(RUN / f'demo_{fit}', 'episode', 10, inputs),
            'online_behavior': episode_lookup(RUN / P / f'fit_{fit}', 'train', 50, inputs),
            'greedy_evaluation': episode_lookup(RUN / 'evaluation' / P / str(fit), 'episode', 16, inputs),
        }
        population_summary = {}
        for label, data in populations.items():
            record = expected[label]
            assert len(data['empty']) == record['states']
            close(sum(data['empty']) / len(data['empty']), record['empty_queue_state_fraction'])
            check_stats(data['terminal_rewards'], record['terminal_reward'])
            population_summary[label] = dict(states=len(data['empty']), empty_states=sum(data['empty']),
                empty_fraction=sum(data['empty']) / len(data['empty']),
                terminal_reward_mean=float(np.mean(data['terminal_rewards'])))
        log_path = RUN / P / f'fit_{fit}/optimizer.json'
        assert log_path.relative_to(ROOT).as_posix() in inputs
        logs = read(log_path)
        assert len(logs) == 1872
        windows = {}
        for label, rows in [('all', logs), ('last_200', logs[-200:])]:
            total = demo = empty = online_empty = online = 0
            values = {key: [] for key in ['terminal_online_reward', 'terminal_demo_reward',
                                          'terminal_online_prediction', 'terminal_demo_prediction']}
            terminal_targets = {'demo': [], 'online': []}
            for update in rows:
                assert all(len(update[key]) == 64 for key in
                           ['transition_ids', 'dones', 'prediction', 'target', 'target_q', 'rewards'])
                for pos, identifier in enumerate(update['transition_ids']):
                    kind, number = identifier.split(':')
                    assert kind in ['demo', 'online']
                    sample = populations['demonstrations' if kind == 'demo' else 'online_behavior']
                    index = int(number)
                    assert 0 <= index < len(sample['empty'])
                    is_empty = sample['empty'][index]
                    is_terminal = sample['terminal'][index]
                    assert bool(update['dones'][pos]) == is_terminal
                    assert update['rewards'][pos] == sample['rewards'][index]
                    total += 1
                    empty += int(is_empty)
                    if kind == 'demo':
                        demo += 1
                    else:
                        online += 1
                        online_empty += int(is_empty)
                    if is_terminal:
                        assert update['target_q'][pos] == 0
                        assert update['target'][pos] == update['rewards'][pos]
                        values[f'terminal_{kind}_reward'].append(update['rewards'][pos])
                        values[f'terminal_{kind}_prediction'].append(update['prediction'][pos])
                        terminal_targets[kind].append(update['target'][pos])
            record = expected['replay_windows'][label]
            assert total == record['sampled_transitions'] == len(rows) * 64
            close(demo / total, record['demo_fraction'])
            close(empty / total, record['empty_queue_fraction'])
            close(online_empty / online, record['online_empty_queue_fraction'])
            for key, data in values.items():
                check_stats(data, record[key])
            windows[label] = dict(sampled_transitions=total, demo_samples=demo, empty_samples=empty,
                online_samples=online, online_empty_samples=online_empty,
                terminal={kind: dict(count=len(terminal_targets[kind]),
                    target_mean=float(np.mean(terminal_targets[kind])),
                    prediction_mean=float(np.mean(values[f'terminal_{kind}_prediction'])))
                    for kind in ['demo', 'online']})
        result[str(fit)] = dict(populations=population_summary, replay_windows=windows)
        print(f'Independent primary coverage counters verified for fit {fit}.', flush=True)
    return result


def main():
    receipt_path = OUT / 'review_verification.json'
    manifest_path = OUT / 'manifest.json'
    assert not receipt_path.exists() and not manifest_path.exists(), 'Preserve completed review evidence'
    assert sha(SPEC) == SPEC_SHA and sha(COMPLETION) == COMPLETION_SHA
    assert sha(PREVIOUS_REVIEW) == PREVIOUS_REVIEW_SHA
    diagnosis = read(OUT / 'diagnosis.json')
    coverage = read(OUT / 'behavior_replay_coverage.json')
    receipt = read(OUT / 'verification.json')
    independent = read(OUT / 'independent_verification.json')
    completion = read(COMPLETION)
    manifest = read(RUN / 'manifest.json')
    assert receipt['status'] == 'SAVED_TRACE_DIAGNOSIS_COMPLETE_NO_NEW_RESEARCH'
    for name, digest in receipt['artifacts_sha256'].items():
        assert sha(OUT / name) == digest
    for name, record in zip(SCRIPTS[:3], [receipt, independent, coverage]):
        assert sha(ROOT / name) == record['script_sha256']
    assert independent['diagnosis_receipt_sha256'] == sha(OUT / 'verification.json')
    assert coverage['diagnosis_sha256'] == sha(OUT / 'diagnosis.json')
    assert diagnosis['script_sha256'] == receipt['script_sha256']
    assert diagnosis['protocol_sha256'] == SPEC_SHA
    assert receipt['decoded_saved_states'] == diagnosis['decoded_saved_states'] == 54000
    assert independent['fixed_state_population_model_entries_verified'] == 90
    assert independent['paired_task_comparisons_verified'] == 12
    assert independent['same_RSU_new_miss_queue_identities_verified'] == 723
    assert len(diagnosis['consumed_sha256']) == 602
    assert len(coverage['consumed_sha256']) == 836
    inputs = dict(diagnosis['consumed_sha256'])
    for name, digest in coverage['consumed_sha256'].items():
        assert name not in inputs or inputs[name] == digest
        inputs[name] = digest
    assert len(inputs) == 1203
    for name, digest in inputs.items():
        path = ROOT / name
        assert sha(path) == digest, name
        if path.is_relative_to(RUN) and path != RUN / 'manifest.json':
            assert manifest[path.relative_to(RUN).as_posix()] == digest, name
    for name, digest in completion['evidence_sha256'].items():
        assert sha(SOURCE / name) == digest
    for record in [diagnosis, coverage, receipt, independent, completion, read(RUN / 'analysis.json')]:
        assert record['policy_screen'] is False and record['value_screen'] is True and record['promising'] is False
    freeze = read(RUN / 'CHECKPOINT_FREEZE.json')
    assert len(freeze['checkpoints']) == len(freeze['deployments']) == 27
    for label, filename in [('checkpoints', 'final.pt'), ('deployments', 'deployment_weights.pt')]:
        for key, digest in freeze[label].items():
            arm, fit = key.split('/')
            assert sha(RUN / arm / f'fit_{fit}' / filename) == digest
    prefreeze = read(RUN / 'PRE_RUN_FREEZE.json')
    assert prefreeze['protocol_sha256'] == SPEC_SHA
    for name, digest in prefreeze['source_sha256'].items():
        assert sha(RUN / 'source_snapshot' / name) == digest
        assert sha(ROOT / name) == digest
    counters = verify_primary_coverage(coverage, inputs)
    # Bind the exact review and scripts after successful checks, before issuing the receipt.
    bound_files = {name: sha(OUT / name) for name in
                   ['diagnosis.json', 'fixed_state_predictions.npz', 'verification.json',
                    'independent_verification.json', 'behavior_replay_coverage.json']}
    source_hashes = {name: sha(ROOT / name) for name in [*SCRIPTS, REVIEW.name]}
    for name, digest in inputs.items():
        assert sha(ROOT / name) == digest, name
    snapshot = OUT / 'review_source_snapshot'
    snapshot.mkdir(exist_ok=False)
    for name, digest in source_hashes.items():
        with (snapshot / name).open('xb') as stream:
            stream.write((ROOT / name).read_bytes())
        assert sha(snapshot / name) == digest
    result = dict(status='SAVED_TRACE_DIAGNOSIS_REVIEW_VERIFIED_NO_NEW_RESEARCH',
        protocol_sha256=SPEC_SHA, prior_completion_receipt_sha256=COMPLETION_SHA,
        prior_results_review_sha256=PREVIOUS_REVIEW_SHA,
        main_input_files=602, coverage_input_files=836, union_input_files_rehashed=1203,
        union_input_map_sha256=hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        decoded_saved_states_in_main_analysis=54000, fixed_state_population_model_entries_verified=90,
        paired_task_comparisons_verified=12, overlapping_comparator_task_queue_identities_verified=723,
        final_checkpoint_hashes_verified=27, deployment_hashes_verified=27,
        frozen_source_files_verified=len(prefreeze['source_sha256']),
        primary_coverage_independent_counters=counters, evidence_sha256=bound_files,
        review_and_source_sha256=source_hashes, script_sha256=sha(Path(__file__)),
        new_models_created_by_this_verifier=0, new_inference_evaluations_by_this_verifier=0,
        new_simulator_steps=0, new_optimizer_updates=0, new_teacher_calls=0, new_autograd_evaluations=0,
        policy_screen=False, value_screen=True, promising=False, single_failure_cause_proven=False,
        final_seeds_consumed=False, new_research_launch_authorized=False, final_phase_launched=False,
        word_thesis_changed=False,
        archive_verification_scope='Prior independent archive receipt remains bound; no full archive rehash repeated in this diagnosis gate.',
        next_gate='prospective_controller_design_decision_brief_only')
    with receipt_path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    pack = {path.relative_to(OUT).as_posix(): sha(path) for path in sorted(OUT.rglob('*'))
            if path.is_file() and path != manifest_path}
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(pack, stream, indent=2, allow_nan=False)
        stream.write('\n')
    assert set(pack) == {path.relative_to(OUT).as_posix() for path in OUT.rglob('*')
                         if path.is_file() and path != manifest_path}
    assert all(sha(OUT / name) == digest for name, digest in pack.items())
    print(json.dumps(dict(status=result['status'], union_inputs=1203, manifest_files=len(pack),
                          receipt_sha256=sha(receipt_path), manifest_sha256=sha(manifest_path)), indent=2))


if __name__ == '__main__':
    main()
