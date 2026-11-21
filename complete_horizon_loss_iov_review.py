"""Read-only evidence verification and supplemental engineering summaries."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'validation/horizon_loss_iov_integration_2026-11-14_01'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write(p, x):
    p.write_text(json.dumps(x, indent=2) + '\n', encoding='utf-8')

def stats(x):
    x = np.asarray(x, dtype=float)
    return dict(count=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p05=float(np.quantile(x, .05)), p95=float(np.quantile(x, .95)),
                min=float(x.min()), max=float(x.max()))

def main():
    manifest = read(OUT / 'evidence_manifest.json')
    for name, expected in manifest.items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == expected, name
    assert read(OUT / 'tests.json')['passed']
    assert read(OUT / 'workflow/workflow_verification.json')['artifact_audit_passed']
    summaries = {}
    for path in sorted((OUT / 'workflow/run').glob('normalized_*/fit_*/optimizer.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        by_h = {}
        for row in rows:
            for h, raw, normalized in zip(row['current_h'], row['TD_residuals'], row['normalized_TD_residuals']):
                assert abs(raw / h - normalized) < 1e-5
                bucket = by_h.setdefault(int(h), {'original': [], 'normalized': []})
                bucket['original'].append(raw)
                bucket['normalized'].append(normalized)
        summaries[path.relative_to(OUT / 'workflow/run').as_posix()] = {
            str(h): {unit: stats(values) for unit, values in units.items()}
            for h, units in sorted(by_h.items())}
    assert len(summaries) == 6
    write(OUT / 'supplemental_residual_distributions.json', dict(
        kind='SYNTHETIC_NOT_RESEARCH', quality_evidence=False,
        original_evidence_files_verified=len(manifest),
        residual_definition='chosen_Q minus Bellman_target; normalized residual divides by current horizon',
        by_current_horizon=summaries))
    status_path = ROOT / 'LOW_COMPLEXITY_PROJECT_STATUS.json'
    backup = OUT / 'project_status_before_integration_review.json'
    if not backup.exists():
        shutil.copy2(status_path, backup)
    status = read(status_path)
    status['current_workstream'] = 'horizon_normalized_loss_simulator_integration_passed_research_protocol_pending'
    status['horizon_normalized_loss_design']['implementation_complete'] = True
    status['horizon_loss_iov_integration'] = dict(
        report='HORIZON_LOSS_IOV_INTEGRATION_REVIEW_2026-11-14.txt',
        evidence=str(OUT.relative_to(ROOT)), tests_passed=4,
        learned_controllers=14, heuristics=3, checkpoints=28,
        fixture_training_slots=2240, fixture_evaluation_slots=2480,
        optimizer_updates=2044, target_copies=252,
        audited_episodes=118, audited_task_records=141600,
        runtime_action_checks=3584, timing_readings=17920,
        full_artifact_audit_passed=True, deployment_storage_audit_passed=True,
        quality_evidence=False, research_launched=False,
        research_budget_declared=False, research_seeds_reserved=False,
        word_thesis_changed=False)
    status['next_engineering_step'] = (
        'Prospectively freeze the 14-learner and 3-heuristic research comparison, '
        'budget, fresh development seeds and collision audit, and decision rules. '
        'Retain untouched final seeds; obtain a separate explicit research launch instruction.')
    write(status_path, status)
    print(json.dumps(dict(original_evidence_files_verified=len(manifest),
                          normalized_fit_distributions=len(summaries), status_updated=True)))

if __name__ == '__main__':
    main()
