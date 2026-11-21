"""Prelaunch evidence/seed/storage review; no models, episodes or optimizer steps."""
import copy
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch
import run_value_gradient_isolation_comparison as runner
from freeze_heuristic_initialization_pilot_plan_2026_11_19 import ranges
import value_gradient_isolation_progress as monitor

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation/value_gradient_isolation_comparison_readiness_2026-11-21_01'


def main():
    if OUT.exists():raise RuntimeError('Preserve previous readiness evidence')
    spec=runner.protocol();fixture=ROOT/'validation/value_gradient_isolation_comparison_fixture_2026-11-20_01'
    receipt=runner.read(fixture/'engineering_verification.json');tests=runner.read(fixture/'tests.json')
    audit=runner.read(fixture/'saved_artifact_audit.json');completion=runner.read(fixture/'completion.json')
    assert tests['passed'] and tests['tests']==20 and audit['passed'] and completion['audit_passed']
    assert receipt['protocol_sha256']==runner.sha(runner.SPEC)
    for name,digest in receipt['source_sha256'].items():assert runner.sha(ROOT/name)==digest,name
    engineering=runner.read(fixture/'engineering_manifest.json')
    actual={p.relative_to(fixture).as_posix() for p in fixture.rglob('*') if p.is_file() and p!=fixture/'engineering_manifest.json'}
    assert actual==set(engineering)
    for name,digest in engineering.items():assert runner.sha(fixture/name)==digest,name
    freeze=runner.read(fixture/'run/PRE_RUN_FREEZE.json');manifest=runner.read(fixture/'run/manifest.json')
    assert runner.sha(fixture/'run/manifest.json')==completion['manifest_sha256']
    for name,digest in freeze['source_sha256'].items():assert runner.sha(ROOT/name)==digest,name
    torch.set_num_threads(1)
    env=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,platform=platform.platform(),threads=torch.get_num_threads())
    assert env==freeze['environment']==receipt['environment']==spec['numerical_environment']
    assert Path(sys.executable).resolve()==Path(spec['python_executable']).resolve()
    assert runner.DEVELOPMENT_RELEASED is False and runner.vgi.DEVELOPMENT_RELEASED is False and runner.vgi.FINAL_RELEASED is False
    guards=[]
    with tempfile.TemporaryDirectory() as td:
        output=Path(td)/'blocked'
        with patch.object(runner.vgi,'Critic',side_effect=AssertionError('Model construction forbidden in readiness')):
            for mode,confirm in [('development',None),('development',runner.CONFIRM),('final',runner.CONFIRM)]:
                try:runner.run(output,mode,confirm)
                except RuntimeError as error:
                    expected='Explicit development confirmation' if confirm is None else 'disabled'
                    assert expected.lower() in str(error).lower(),str(error)
                    guards.append(dict(mode=mode,confirmation_provided=confirm is not None,error=str(error)))
                else:raise AssertionError('Research guard did not reject')
                assert not output.exists()
    reserved_output=ROOT/spec['artifact_retention']['reserved_output']
    reserved_archive=Path(spec['artifact_retention']['reserved_archive'])
    assert not reserved_output.exists() and not reserved_archive.exists(),'Research/archive output already exists'
    monitor_tests=ROOT/'validation/value_gradient_isolation_monitor_tests_2026-11-21_01'
    mt=runner.read(monitor_tests/'tests.json')
    assert mt['passed'] and mt['tests']==10 and mt['failures']==mt['errors']==0
    warning=runner.read(monitor_tests/'warning_child/child_result.json')
    failure=runner.read(monitor_tests/'nonzero_child/child_result.json')
    assert warning['child_exit_code']==0 and warning['stderr_bytes']>0
    assert failure['child_exit_code']==7 and failure['stderr_bytes']==0
    old=runner.read(ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json')['return_calibrated_handoff_results_2026_11_20']
    assert old['policy_screen'] is False and old['value_screen'] is True and old['promising'] is False
    proposed=ranges(spec['seeds']);documents=[];aliases=[];collisions=[]
    design_name='VALUE_GRADIENT_ISOLATION_DESIGN_STATUS_2026-11-20.json'
    paths=sorted(ROOT.glob('*.json'))+sorted((ROOT/'validation').glob('*/run/runtime_plan.json'))+sorted((ROOT/'development_runs').glob('*/run/runtime_plan.json'))
    for p in paths:
        value=json.loads(p.read_text(encoding='utf-8-sig'));entry=dict(path=p.relative_to(ROOT).as_posix(),sha256=runner.sha(p))
        if p==runner.SPEC or p==ROOT/design_name:
            assert value['seeds']==spec['seeds'];aliases.append(entry);continue
        documents.append(entry)
        for a,b,label in proposed:
            for c,d,old in ranges(value):
                if max(a,c)<=min(b,d):collisions.append(dict(proposed=label,existing=entry['path']+old))
    for i,(a,b,label) in enumerate(proposed):
        for c,d,old in proposed[:i]:
            if max(a,c)<=min(b,d):collisions.append(dict(proposed=label,existing=old))
    assert not collisions,collisions
    category=dict(episodes=0,optimizer=0,other=0)
    for p in (fixture/'run').rglob('*'):
        if p.is_file():
            key='optimizer' if p.name=='optimizer.json' else 'episodes' if p.name.startswith(('episode_','train_')) else 'other'
            category[key]+=p.stat().st_size
    projection=int(category['episodes']*(67120/2200)+category['optimizer']*(44928*64/(512*4))+category['other']*10)
    required=4*projection+1024**3
    volumes={drive:dict(free_bytes=shutil.disk_usage(drive).free,required_bytes=required) for drive in ['E:/','D:/']}
    assert all(row['free_bytes']>=required for row in volumes.values()),volumes
    runtime=dict(demo_wall_ns=0,student_episode_wall_ns=0,evaluation_wall_ns=0,bc_fit_wall_ns=0,calibration_fit_wall_ns=0,online_fit_wall_ns=0,
        online_actor_ns=0)
    for p in (fixture/'run').glob('demo_*/episode_*.json'):runtime['demo_wall_ns']+=runner.read(p)['wall_ns']
    for p in (fixture/'run').glob('*/fit_*/train_*.json'):
        row=runner.read(p);runtime['student_episode_wall_ns']+=row['wall_ns'];runtime['online_actor_ns']+=row['actor_ns']
    for p in (fixture/'run/evaluation').rglob('episode_*.json'):runtime['evaluation_wall_ns']+=runner.read(p)['wall_ns']
    for p in (fixture/'run').glob('*/fit_*/counts.json'):
        row=runner.read(p)
        for key,field in [('bc_fit_wall_ns','bc_wall_ns'),('calibration_fit_wall_ns','calibration_wall_ns'),('online_fit_wall_ns','online_wall_ns')]:runtime[key]+=row[field]
    # A disclosed planning interval only: compare scaling by decisions with
    # scaling all nonactor online time by sampled transition exposure. No ETA.
    decisions_ratio=48000/640;updates_ratio=(44928*64)/(512*4)
    actor=runtime['online_actor_ns'];rest=max(0,runtime['student_episode_wall_ns']-actor)
    common=runtime['demo_wall_ns']*1.5+runtime['evaluation_wall_ns']*(17920/760)+runtime['calibration_fit_wall_ns']*1.5
    low=common+runtime['bc_fit_wall_ns']*(4800/32)+runtime['student_episode_wall_ns']*decisions_ratio
    high=common+runtime['bc_fit_wall_ns']*(4800*64/(32*4))+actor*decisions_ratio+rest*updates_ratio
    row=monitor.snapshot(fixture)
    assert row['saved']==row['expected'] and row['audit_complete'] and not row['archive_receipt_present']
    assert row['saved']['total_slots']==2200 and row['saved']['final_checkpoints']==18
    # Monitor's development denominators checked on a temporary plan, not data.
    with tempfile.TemporaryDirectory() as td:
        testout=Path(td);(testout/'run').mkdir()
        runner.write(testout/'run/runtime_plan.json',runner.plan(spec,'development'))
        initial=monitor.snapshot(testout)
        assert initial['expected']['total_slots']==67120 and initial['expected']['completed_fit_td_updates']==44928
        assert initial['expected']['deployment_exports']==27 and initial['expected']['completed_fit_retention_batches']==22464
        assert initial['saved']['total_slots']==0
        runner.write(testout/'progress.json',dict(stage='artifacts_complete'))
        assert monitor.snapshot(testout)['stage']=='FULL_ARTIFACT_AUDIT_RUNNING_OR_PENDING'
    record=dict(status='TECHNICALLY_READY_PENDING_SEPARATE_EXPLICIT_LAUNCH',protocol_sha256=runner.sha(runner.SPEC),
        tested_sources=receipt['source_sha256'],readiness_sources={name:runner.sha(ROOT/name) for name in [Path(__file__).name,'value_gradient_isolation_progress.py','value_gradient_isolation_process_monitor.py','test_value_gradient_isolation_monitoring.py']},
        engineering_receipt_sha256=runner.sha(fixture/'engineering_verification.json'),
        fixture_tests=20,monitoring_tests_passed=10,monitoring_tests_sha256=runner.sha(monitor_tests/'tests.json'),fixture_manifest_files_verified=len(manifest),engineering_manifest_files_verified=len(engineering),
        environment=env,python_executable=sys.executable,guards=guards,
        seed_audit=dict(documents=documents,same_study_aliases=aliases,collisions=collisions,
            limitation='Root JSON and saved validation/development */run/runtime_plan.json; not exhaustive historical log scan. Refresh at launch.'),
        storage=dict(fixture_category_bytes=category,projected_bytes=projection,required_free_bytes=required,volumes=volumes,
            method='Episode bytes by slot ratio,optimizer logs by transition exposure,other data10x; require4xprojection+1GiB per drive',
            retention='E: primary; independent D: archive only after full audit; verify inventory and SHA256'),
        runtime=dict(fixture_recorded_ns=runtime,unvalidated_training_and_evaluation_planning_hours=[low/3.6e12,high/3.6e12],
            limitation='Not a confidence interval or completion ETA. Linear extrapolation from short fixture; full audit,archive,source hashing,thermal/IO variation and other overhead excluded. Large batch retention gradients may scale differently.'),
        output_absence_verified=dict(research=str(reserved_output),archive=str(reserved_archive)),
        monitoring=dict(process_wrapper='value_gradient_isolation_process_monitor.py',actual_python_exit_code_verified=True,stderr_warning_is_not_failure_verified=True,helper='value_gradient_isolation_progress.py',fixture_counts_verified=True,development_denominators_verified=True,
            audit_stage_verified=True,counters='saved episodes/completed fits; active work may be ahead',
            launch_requirement='Versioned launcher polls numerical counters and records audit/archive phase wall times outside immutable run data'),
        teacher_backend='Bound Python reference; compiled-runtime equivalence not established',
        quality_or_novelty_established=False,development_release_enabled=False,final_release_enabled=False,research_launch_authorized=False,launched=False,
        new_models_created=0,new_simulator_steps=0,new_optimizer_updates=0,new_teacher_calls=0,word_thesis_changed=False,
        launch_requirements=['Separate explicit authorization for this frozen67120slot comparison',
            'Versioned launcher validates receipt/exact sources/environment,refreshes seed/storage checks and chooses nonexistent output/archive',
            'Enable only bound runner in memory;keep on-disk DEVELOPMENT_RELEASED=False and final mode closed',
            'Numerical LIVE_STATUS plus direct child exit codes and separate audit/archive phase timing;retain every fit and failure with no automatic retry',
            'Full artifact audit then independent D: inventory/hash verification before claiming archived completion'])
    OUT.mkdir(parents=True)
    sources=OUT/'source_snapshot';sources.mkdir()
    for name in record['readiness_sources']:(sources/name).write_bytes((ROOT/name).read_bytes())
    for path in sorted(monitor_tests.rglob('*')):
        if path.is_file():
            target=OUT/'monitoring_test_evidence'/path.relative_to(monitor_tests)
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    runner.write(OUT/'readiness.json',record)
    runner.write(OUT/'manifest.json',{p.relative_to(OUT).as_posix():runner.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()})
    print(json.dumps({key:record[key] for key in ['status','fixture_tests','fixture_manifest_files_verified','engineering_manifest_files_verified','launched']},indent=2))
    print(json.dumps(record['storage'],indent=2));print(json.dumps(record['runtime'],indent=2))


if __name__=='__main__':main()

