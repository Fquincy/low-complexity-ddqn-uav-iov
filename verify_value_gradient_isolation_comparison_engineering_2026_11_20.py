"""Rehash completed fixture/tests and bind engineering evidence; no fitting."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import run_value_gradient_isolation_comparison as runner
from freeze_heuristic_initialization_pilot_plan_2026_11_19 import ranges

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write_new(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main(output):
    out=Path(output).resolve();root=out/'run'
    if (out/'engineering_verification.json').exists():raise ValueError('Preserve prior engineering receipt')
    spec=runner.protocol();tests=read(out/'tests.json');audit=read(out/'saved_artifact_audit.json')
    completion=read(out/'completion.json');pl=read(root/'runtime_plan.json')
    assert pl==runner.plan(spec,'fixture')
    assert tests['passed'] and tests['tests']==20 and tests['failures']==tests['errors']==0
    assert audit['passed'] and completion['status']=='ENGINEERING_AUDITED_COMPLETE'
    assert not completion['research_launch_authorized'] and not audit['research_launch_authorized']
    expected=dict(simulation_slots_recorded=2200,historical_bc_updates=32,historical_ddqn_updates=512,
        retention_batches=256,ridge_solves=10,final_checkpoints=18,deployment_weight_files=18,
        measured_action_calls=76,warmup_action_calls=19,audit_gradient_evaluations=12)
    for key,value in expected.items():assert audit[key]==value,(key,audit[key])
    assert len(audit['first_update_reconstructions'])==6
    assert len(audit['value_distributions'])==len(audit['storage'])==len(audit['allocation_diagnostics'])==18
    assert all(v['parameter_bytes']==67364 and v['buffer_bytes']==20 for v in audit['storage'].values())
    manifest=read(root/'manifest.json');freeze=read(root/'PRE_RUN_FREEZE.json')
    assert sha(root/'manifest.json')==completion['manifest_sha256']
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p!=root/'manifest.json'}
    assert actual==set(manifest)
    for name,digest in manifest.items():assert sha(root/name)==digest,name
    for name,digest in freeze['source_sha256'].items():
        assert sha(ROOT/name)==sha(root/'source_snapshot'/name)==digest,name
    assert freeze['environment']==spec['numerical_environment']
    assert not runner.DEVELOPMENT_RELEASED
    assert not runner.vgi.DEVELOPMENT_RELEASED and not runner.vgi.FINAL_RELEASED
    assert not (ROOT/spec['artifact_retention']['reserved_output']).exists()
    assert not Path(spec['artifact_retention']['reserved_archive']).exists()
    documents=[];aliases=[];collisions=[]
    for path in sorted(ROOT.glob('*.json'))+sorted((ROOT/'validation').glob('*/run/runtime_plan.json'))+sorted((ROOT/'development_runs').glob('*/run/runtime_plan.json')):
        value=read(path);entry=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path))
        if path==runner.SPEC:
            assert value['fixture']['seeds']==pl['seeds'];aliases.append(entry);continue
        if path==root/'runtime_plan.json':
            assert value['seeds']==pl['seeds'];aliases.append(entry);continue
        documents.append(entry)
        for lo,hi,label in ranges(pl['seeds']):
            for a,b,old in ranges(value):
                if max(lo,a)<=min(hi,b):collisions.append(dict(path=entry['path'],fixture=label,other=old))
    assert not collisions,collisions
    previous=read(ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json')['return_calibrated_handoff_results_2026_11_20']
    assert previous['policy_screen'] is False and previous['value_screen'] is True and previous['promising'] is False
    names=['run_value_gradient_isolation_comparison.py','audit_value_gradient_isolation_comparison.py',
        'test_value_gradient_isolation_comparison.py','verify_value_gradient_isolation_comparison_engineering_2026_11_20.py',
        'VALUE_GRADIENT_ISOLATION_COMPARISON_ENGINEERING_REVIEW_2026-11-20.txt']
    snapshot=out/'engineering_source_snapshot';snapshot.mkdir(exist_ok=False)
    for name in names:(snapshot/name).write_bytes((ROOT/name).read_bytes())
    receipt=dict(status='NINE_ARM_WORKFLOW_AND_INDEPENDENT_AUDITOR_PASSED_RESEARCH_DISABLED',
        protocol_sha256=sha(runner.SPEC),source_sha256={name:sha(ROOT/name) for name in names},
        run_manifest_files_verified=len(manifest),tests_passed=20,test_failures=0,
        fixture_exposure=expected,first_update_contexts=6,
        audit_gradient_evaluations_in_initial_full_audit=audit['audit_gradient_evaluations'],
        audit_gradient_evaluations_during_corruption_tests=tests['audit_gradient_evaluations_during_tests'],
        initial_full_audit_teacher_recomputations=audit['teacher_reference_recomputations'],
        environment=freeze['environment'],python_executable=sys.executable,
        seed_audit=dict(documents=documents,same_study_fixture_aliases=aliases,collisions=collisions),
        prior_mechanics_and_frozen_protocol_unchanged=True,all_fits_and_closed_evidence_preserved=True,
        research_runner_implemented=True,independent_auditor_implemented=True,fixture_completed=True,
        new_research_simulator_steps=0,new_research_optimizer_updates=0,
        reserved_research_seeds_used=False,scientific_quality_established=False,novelty_established=False,
        research_launch_authorized=False,development_release_enabled=False,final_release_enabled=False,
        final_seeds_consumed=False,word_thesis_changed=False,
        next_gate='Separate source/environment/seed/storage/monitoring launch-readiness review; no launch in engineering gate',
        limitations=['Independent gradients cover first P/R/A update per fit only; no full training trajectory replay',
            'BC branches are approximately inverse-scaled in auditor; source/hashes bind BC parity',
            'Engineering fixture is not policy quality, novelty, compression, or thesis claim evidence'])
    write_new(out/'engineering_verification.json',receipt)
    write_new(out/'engineering_manifest.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps(dict(status=receipt['status'],tests=20,manifest_files=len(manifest),
        seed_documents_checked=len(documents),fixture_slots=2200,research_launched=False),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');main(parser.parse_args().output)
