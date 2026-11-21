"""Tiny packaging fixture only; no research seeds, learner or optimizer."""
from pathlib import Path
import ast
from dataclasses import replace
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / 'Source'
sys.path.insert(0, str(SIM))
from chapter3_iov_env_v2 import Chapter3Config, Chapter3IoVEnv


def run_fixture():
    config = replace(Chapter3Config(), num_vehicles=3, episode_steps=2)
    env = Chapter3IoVEnv(config, seed=0)  # engineering fixture only, no research reservation
    observations = [env.reset(seed=0)]
    rewards = []
    for index in range(2):
        state, reward, done, info = env.step(dict(
            destinations=np.zeros(3, dtype=np.int64),
            uav_movement=np.zeros(2, dtype=np.int64)))
        assert state.shape == (17 * 3 + 23,) and np.isfinite(state).all()
        assert np.isfinite(reward) and done == (index == 1)
        observations.append(state)
        rewards.append(reward)
    for name in ['cloud_server', 'charging_stations', 'is_charging']:
        assert not hasattr(env, name), name
    assert all(record['destination'] == 0 for record in env.export_task_records())
    return np.stack(observations), np.array(rewards), env.episode_summary()


def main():
    for filename in ['chapter3_iov_env.py', 'chapter3_iov_env_v2.py']:
        tree = ast.parse((SIM / filename).read_text(encoding='utf-8-sig'))
        identifiers = {node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)}
        identifiers |= {node.attr.lower() for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert not any('charg' in name or 'cloud' in name for name in identifiers)
    first, second = run_fixture(), run_fixture()
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    assert first[2] == second[2]
    print(json.dumps(dict(status='ENGINEERING_ONLY_ENVIRONMENT_CHECK_PASSED',
        observation_size=74, destinations=['vehicle_local','UAV1','UAV2','RSU1','RSU2'],
        two_repeated_fixture_episodes=2, engineering_slots=4, model_training=False,
        reserved_research_seeds_used=False, numpy_version=np.__version__,
        source_sha256={name:hashlib.sha256((SIM/name).read_bytes()).hexdigest()
                       for name in ['chapter3_iov_env.py','chapter3_iov_env_v2.py']}), indent=2))


if __name__ == '__main__':
    main()
