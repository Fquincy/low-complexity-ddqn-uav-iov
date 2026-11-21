"""Read-only launch checks; no simulator, optimizer or research seeds used."""
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def manifest_check(path):
    files = read(path)['files']
    failures = [name for name, digest in files.items()
                if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    return dict(entries=len(files), mismatches=failures)

def line_of(lines, text):
    return next(i for i, line in enumerate(lines, 1) if text in line)

def main():
    runner = (ROOT / 'run_v3_research_preparation.py').read_text()
    auditor = (ROOT / 'audit_v3_research_preparation.py').read_text()
    tests = (ROOT / 'test_v3_research_preparation.py').read_text()
    lines = runner.splitlines()
    timing_call = line_of(lines, 'start=time.perf_counter_ns();action=infer(state)')
    timing_freeze = line_of(lines, "write(folder/'PRE_TIMING_FREEZE.json'")
    assert timing_freeze > timing_call
    assert "'development_screen_passed' if all(screens.values())" in runner
    assert 'research_claim_established=all(screens.values())' in auditor
    assert 'audit_research(' not in tests
    fixture = ROOT / 'validation/controller_v3_research_fixture_2026-11-05_04'
    plan = read(fixture / 'runtime_plan.json')
    previous = read(ROOT / 'validation/v3_research_preparation_tests_2026-11-05.json')
    source_mismatches = [name for name, digest in previous['source_sha256'].items()
                         if sha(ROOT / name) != digest]
    guards = []
    for phase in ['development', 'final']:
        for confirmed in [False, True]:
            with tempfile.TemporaryDirectory(prefix='V3_READINESS_NO_TRAINING_') as td:
                out = Path(td) / 'blocked'
                command = [sys.executable, str(ROOT / 'run_v3_research_preparation.py'),
                           '--mode', phase, '--output', str(out)]
                if confirmed:
                    command.append('--confirm-' + phase + '-run')
                proc = subprocess.run(command, capture_output=True, text=True, timeout=60)
                assert proc.returncode != 0 and not out.exists()
                assert 'disabled' in proc.stderr
                guards.append(dict(phase=phase, confirmed=confirmed,
                                   exit_code=proc.returncode, output_created=out.exists()))
    result = dict(
        date='2026-11-06', decision='defer_launch_engineering_corrections_required',
        preparation_manifest=manifest_check(ROOT / 'validation/v3_research_preparation_2026-11-05_COMPLETION_MANIFEST.json'),
        compatibility_manifest_against_current_root=manifest_check(ROOT / 'validation/controller_v3_runtime_compatibility_2026-11-05_01_COMPLETION_MANIFEST.json'),
        archived_status_matches_compatibility_manifest=(sha(fixture / 'PROJECT_STATUS_BEFORE.json') ==
            read(ROOT / 'validation/controller_v3_runtime_compatibility_2026-11-05_01_COMPLETION_MANIFEST.json')['files']['LOW_COMPLEXITY_PROJECT_STATUS.json']),
        prior_test_evidence=dict(passed=previous['passed'], tests=previous['tests'],
                                 current_source_mismatches=source_mismatches,
                                 tests_rerun_this_review=False),
        timing_order_source_evidence=dict(measurement_line=timing_call,
                                         freeze_write_line=timing_freeze),
        cost_omitted_from_overall_runner_and_auditor_decisions=True,
        full_research_auditor_called_by_existing_tests=False,
        retained_fixture_dimensions=dict(fits=len(plan['model_seeds']),
            episodes=plan['episodes'], evaluation_seeds=len(plan['evaluation_seeds']),
            diagnostic_validation_seeds=len(plan['diagnostic_validation_seeds'])),
        cli_guards=guards, refreshed_seed_consumption_audit_completed=False,
        protocol_sha256=sha(ROOT / 'LOW_COMPLEXITY_V3_RESEARCH_PROTOCOL_2026-11-05.json'),
        research_training_launched=False, new_simulator_slots=0, optimizer_updates=0,
        thesis_changed=False)
    assert not result['preparation_manifest']['mismatches']
    assert result['compatibility_manifest_against_current_root']['mismatches'] == ['LOW_COMPLEXITY_PROJECT_STATUS.json']
    assert result['archived_status_matches_compatibility_manifest'] and not source_mismatches
    output = ROOT / 'validation/v3_launch_readiness_review_2026-11-06.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
