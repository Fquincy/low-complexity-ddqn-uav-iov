"""Read-only recomputation of saved engineering cost evidence; no model timing."""
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys

import numpy as np
import numba
import torch

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'validation/assignment_price_prototype_2026-11-15_01'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def percentile(values, fraction):
    ordered = sorted(values)
    location = (len(ordered) - 1) * fraction
    lo, hi = math.floor(location), math.ceil(location)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (location - lo)


def main():
    out = BASE / 'completion'
    out.mkdir(exist_ok=False)
    verified = {}
    for directory, filename in [(BASE, 'tests_manifest.json'), (BASE/'cost_gate', 'manifest.json')]:
        manifest = read(directory/filename)
        for relative, digest in manifest.items():
            assert sha(directory/relative) == digest, relative
        verified[str(directory.relative_to(BASE)/filename)] = dict(files=len(manifest), sha256=sha(directory/filename))
    cost = BASE/'cost_gate'
    tests = read(BASE/'tests.json')
    assert tests['passed'] and tests['tests'] == 6 and tests['synthetic_optimizer_steps'] == 4
    plan = read(cost/'PRE_BENCHMARK_PLAN.json')
    freeze = read(cost/'PRE_TIMING_FREEZE.json')
    summary = read(cost/'cost_summary.json')
    assert freeze['plan_sha256'] == sha(cost/'PRE_BENCHMARK_PLAN.json')
    for relative, digest in plan['source_hashes'].items():
        assert sha(ROOT/relative) == digest
    observations = ROOT/'validation/controller_redesign_engineering_2026-11-05_01/cost_gate/common_observations.npz'
    assert sha(observations) == plan['observation_sha256']
    assert np.load(observations)['observations'].shape == (128, 533)
    checkpoint_freeze = read(cost/'CHECKPOINT_FREEZE.json')
    assert freeze['checkpoints'] == checkpoint_freeze and len(checkpoint_freeze) == 16
    payloads = {}
    for filename, digest in checkpoint_freeze.items():
        assert sha(cost/filename) == digest
        checkpoint = torch.load(cost/filename, map_location='cpu', weights_only=True)
        assert checkpoint['engineering_only'] is True
        arm, fit_text = Path(filename).stem.rsplit('_', 1)
        fit = int(fit_text)
        assert checkpoint['seed'] == plan['model_seeds'][fit]
        assert checkpoint['arm'] == arm
        payload = sum(t.numel()*t.element_size() for t in checkpoint['online'].values())
        reported = summary['costs'][arm][fit]
        assert payload == reported['parameter_bytes'] + reported['buffer_bytes']
        payloads[filename] = payload
    expected_order = [(arm, fit, rnd, index) for rnd, order in enumerate(freeze['orders'])
                      for arm, fit in order for index in range(128)]
    jobs = [(arm, fit) for arm in plan['arms'] for fit in range(2)]
    rng = np.random.default_rng(plan['order_seed'])
    for order in freeze['orders']:
        expected = list(jobs)
        rng.shuffle(expected)
        assert expected == [tuple(x) for x in order]
    rows = read(cost/'raw_timings.json')
    assert len(rows) == len(expected_order) == plan['readings'] == 10240
    assert [(x['arm'], x['fit'], x['round'], x['index']) for x in rows] == expected_order
    assert all(isinstance(x['ns'], int) and x['ns'] > 0 for x in rows)
    recalculated = {}
    for arm, fit in jobs:
        samples = [x['ns'] for x in rows if x['arm'] == arm and x['fit'] == fit]
        assert len(samples) == 640
        result = dict(median_ms=statistics.median(samples)/1e6, p95_ms=percentile(samples, .95)/1e6)
        for key, value in result.items():
            assert math.isclose(value, summary['costs'][arm][fit][key], rel_tol=1e-12, abs_tol=1e-12)
        recalculated[arm, fit] = result
    thresholds = plan['thresholds']
    gates = []
    for fit in range(2):
        primary = recalculated['coupled_sweep', fit]
        gates.append(dict(
            storage=payloads[f'coupled_sweep_{fit}.pt']-20 <= thresholds['parameter_bytes_max_fraction_of_unmasked_compact64']*payloads[f'unmasked_compact64_{fit}.pt'],
            median=all(primary['median_ms'] <= thresholds['median_ratio_max']*recalculated[arm, fit]['median_ms'] for arm in thresholds['median_ms_max_fraction_of_each']),
            p95=primary['p95_ms'] <= thresholds['p95_ms_max']))
    assert gates == summary['primary_screens']
    assert summary['all_primary_cost_screens_pass'] == all(all(g.values()) for g in gates)
    assert summary['engineering_equivalence_checks'] == len(jobs)*plan['observations'] == 2048
    environment = dict(scope='Audit process environment; historical benchmark settings verified from saved source, not independently captured during timing',
                       executable=sys.executable, python=sys.version, platform=platform.platform(),
                       packages={name: dict(version=module.__version__, path=module.__file__) for name, module in [('torch',torch),('numpy',np),('numba',numba)]},
                       benchmark_declared_threads=dict(torch=1, torch_interop=1, numba=1),
                       benchmark_thread_source='run_horizon_loss_research.py configure() called by benchmark; no retrospective timing')
    result = dict(passed=True, scope='Saved-file integrity, timing recomputation and checkpoint tensor payload; runtime equivalence count is recorded test evidence, not rerun here',
                  manifests=verified, checkpoints_verified=16, raw_timing_rows_verified=10240,
                  primary_screens=gates, all_primary_cost_screens_pass=False,
                  recorded_equivalence_checks=2048, tensor_payload_bytes=payloads,
                  no_retiming=True, new_simulator_steps=0, new_optimizer_updates=0)
    (out/'environment.json').write_text(json.dumps(environment,indent=2)+'\n',encoding='utf-8')
    (out/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
