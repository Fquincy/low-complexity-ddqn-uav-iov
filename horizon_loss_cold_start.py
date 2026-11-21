"""Separate fresh-process startup measurement; never repeats steady latency timing."""
import time
_ENTRY = time.perf_counter_ns()
import hashlib
import json
from pathlib import Path
import subprocess
import sys

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def worker(saved, arm, fit, output):
    # Imports are intentionally inside the measured fresh Python process.
    import run_horizon_loss_research as r
    import numpy as np
    import torch
    r.configure()
    p = r.protocol()
    checkpoint = Path(saved)/arm/f'fit_{fit:02d}'/'final.pt'
    frozen = read(Path(saved)/'ALL_FINAL_CHECKPOINTS_FREEZE.json')
    assert sha(checkpoint) == frozen[f'{arm}/fit_{fit:02d}/final.pt']
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    definition = next(d for d in p['learned_arms'] if d['id'] == arm)
    model = r.models.Critic(definition, ck['model_seed']).eval()
    model.load_state_dict(ck['online'])
    runtime = r.models.Runtime(model)
    with np.load(Path(saved)/'engineering_observations.npz') as arrays:
        state = arrays['observations'][0].copy()
    action, value = runtime.values(state)
    ready = time.perf_counter_ns() - _ENTRY
    result = dict(arm=arm, fit=fit, checkpoint_sha256=sha(checkpoint),
        process_id=__import__('os').getpid(), ready_ns=ready,
        action=action[0].tolist(), Q=float(value[0]),
        worker_source_sha256=sha(__file__), protocol_sha256=r.sha(r.SPEC))
    Path(output).write_text(json.dumps(result, indent=2)+'\n')

def generate(saved, output):
    saved, output = Path(saved), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan = read(saved/'runtime_plan.json')
    frozen = read(saved/'ALL_FINAL_CHECKPOINTS_FREEZE.json')
    rows = []
    for relative in sorted(frozen):
        arm, fit_name, _ = relative.split('/')
        fit = int(fit_name.split('_')[1])
        destination = output/f'{arm}_{fit:02d}.json'
        started = time.perf_counter_ns()
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                        str(saved.resolve()), arm, str(fit), str(destination.resolve())],
                       check=True, capture_output=True, text=True, timeout=120)
        row = read(destination)
        row['process_to_exit_ns'] = time.perf_counter_ns()-started
        destination.write_text(json.dumps(row, indent=2)+'\n')
        rows.append(row)
    report = dict(kind=plan['kind'], quality_evidence=plan['mode']!='integration-fixture',
        measurements=rows, extra_optimizer_steps=0, extra_simulator_steps=0,
        steady_latency_repeated=False,
        definition='ready_ns: fresh Python entry, imports, source/checkpoint validation, reconstruction, first action. process_to_exit_ns: parent spawn to child exit including overhead.',
        limitation='Fresh process, not a cold machine or empty disk/Numba cache. Python executable/OS launch before script entry is excluded from ready_ns and included in process_to_exit_ns. No gate or model selection uses these readings.')
    (output/'startup_report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report

def audit(saved, output):
    saved, output = Path(saved), Path(output)
    report = read(output/'startup_report.json')
    frozen, predictions = read(saved/'ALL_FINAL_CHECKPOINTS_FREEZE.json'), read(saved/'predictions.json')
    seen = set()
    for row in report['measurements']:
        key = f"{row['arm']}/fit_{row['fit']:02d}/final.pt"
        assert key not in seen and key in frozen
        seen.add(key)
        assert row['checkpoint_sha256'] == frozen[key] == sha(saved/key)
        assert row['worker_source_sha256'] == sha(__file__)
        assert 0 < row['ready_ns'] < row['process_to_exit_ns']
        expected = predictions[f"{row['arm']}/{row['fit']}"][0]
        assert row['action'] == expected['action']
        assert abs(row['Q']-expected['Q']) <= 1e-6 + 1e-5*abs(expected['Q'])
        assert row == read(output/f"{row['arm']}_{row['fit']:02d}.json")
    assert seen == set(frozen)
    assert report['extra_optimizer_steps'] == report['extra_simulator_steps'] == 0
    assert report['steady_latency_repeated'] is False
    assert report['kind'] == read(saved/'runtime_plan.json')['kind']
    assert report['quality_evidence'] == (report['kind']!='SYNTHETIC_NOT_RESEARCH')
    return dict(passed=True, models=len(seen), no_training=True, no_simulator_steps=True,
                independent_actions_and_checkpoint_hashes_verified=True)

if __name__ == '__main__':
    assert sys.argv[1] == '--worker'
    worker(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
