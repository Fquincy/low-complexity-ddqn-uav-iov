"""Read-only launch-readiness checks; never creates models or runs episodes."""
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import run_heuristic_initialization_pilot as runner
from freeze_heuristic_initialization_pilot_plan_2026_11_19 import ranges

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation/heuristic_initialization_pilot_readiness_2026-11-19_01'


def main():
    if OUT.exists():raise RuntimeError('Preserve previous readiness records')
    spec=runner.protocol()
    fixture=ROOT/'validation/heuristic_initialization_pilot_fixture_2026-11-19_03'
    receipt=runner.read(fixture/'engineering_verification.json')
    tests=runner.read(fixture/'tests.json');audit=runner.read(fixture/'saved_artifact_audit.json')
    assert tests['passed'] and tests['tests']==8 and audit['passed']
    assert receipt['protocol_sha256']==runner.sha(runner.SPEC)
    for name,h in receipt['source_sha256'].items():assert runner.sha(ROOT/name)==h,name
    freeze=runner.read(fixture/'run/PRE_RUN_FREEZE.json')
    for name,h in freeze['source_sha256'].items():assert runner.sha(ROOT/name)==h,name
    manifest=runner.read(fixture/'run/manifest.json')
    actual={p.relative_to(fixture/'run').as_posix() for p in (fixture/'run').rglob('*') if p.is_file() and p.name!='manifest.json'}
    assert actual==set(manifest)
    for name,h in manifest.items():assert runner.sha(fixture/'run'/name)==h,name
    current=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),threads=1)
    assert current==freeze['environment'],(current,freeze['environment'])
    # The runner explicitly sets one thread and deterministic algorithms before work.
    assert runner.PILOT_RELEASED is False
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'blocked'
        for confirmation in [None,runner.CONFIRM]:
            try:runner.run(p,'pilot',confirmation)
            except RuntimeError as e:
                expected='Explicit pilot confirmation required' if confirmation is None else 'disabled pending'
                assert expected in str(e)
            else:raise AssertionError('pilot guard did not reject')
            assert not p.exists()
    proposed=ranges(spec['seeds']);collisions=[];documents=[]
    for p in sorted(ROOT.glob('*.json')):
        if p==runner.SPEC:continue
        value=json.loads(p.read_text(encoding='utf-8-sig'))
        existing=ranges(value);documents.append(dict(path=p.name,sha256=runner.sha(p)))
        for lo,hi,label in proposed:
            for a,b,old in existing:
                if max(lo,a)<=min(hi,b):collisions.append(dict(proposed=label,existing=p.name+old))
    fixture_plans=[]
    for p in sorted((ROOT/'validation').glob('heuristic_initialization_pilot_fixture_*/run/runtime_plan.json')):
        value=runner.read(p);fixture_plans.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=runner.sha(p)))
        for lo,hi,label in proposed:
            for a,b,old in ranges(value['seeds']):
                if max(lo,a)<=min(hi,b):collisions.append(dict(proposed=label,existing=str(p)+old))
    assert not collisions,collisions
    # A bounded output estimate by saved category, not a claim of exact disk use.
    run=fixture/'run'
    sizes={'episodes':0,'optimizer':0,'other':0}
    for p in run.rglob('*'):
        if not p.is_file():continue
        category='optimizer' if p.name=='optimizer.json' else 'episodes' if p.name.startswith(('episode_','train_')) else 'other'
        sizes[category]+=p.stat().st_size
    # 442 pilot episodes vs 11 fixture episodes; optimizer 11232*64 vs128*4
    # sample entries. Other data scales generously 10x. Require 4x plus 1 GiB.
    estimate=int(sizes['episodes']*(17680/440)+sizes['optimizer']*(11232*64/(128*4))+sizes['other']*10)
    required=4*estimate+1024**3
    volumes={v:dict(free_bytes=shutil.disk_usage(v).free) for v in ['E:/','D:/']}
    assert all(v['free_bytes']>=required for v in volumes.values()),volumes
    historical=runner.read(ROOT/'LOW_COMPLEXITY_CONSOLIDATED_EVIDENCE_LEDGER_2026-11-19.json')
    for row in historical['sources']:assert runner.sha(ROOT/row['path'])==row['sha256']
    record=dict(status='TECHNICALLY_READY_PENDING_SEPARATE_EXPLICIT_LAUNCH',
        protocol_sha256=runner.sha(runner.SPEC),tested_sources=receipt['source_sha256'],
        environment=current,python_executable=sys.executable,
        teacher_backend='Python reference of frozen teacher bodies; identity native decorators',
        compiled_teacher_equivalence_claim=False,quality_or_novelty_established=False,
        fixture_tests=8,fixture_manifest_files_verified=len(manifest),
        seed_audit=dict(root_documents=documents,fixture_plans=fixture_plans,collisions=collisions,
            limitation='Explicit seed values and start/count blocks in root JSON and this pilot fixture family; not exhaustive historical log scan. Refresh at launch.'),
        storage=dict(fixture_category_bytes=sizes,projected_bytes=estimate,required_free_bytes=required,volumes=volumes,
            retention='Keep primary E: output and independent D: copy after completed audit; verify hashes before claiming archive completion'),
        pilot_simulated_slots=17680,pilot_training_slots=12000,
        pilot_release_enabled=False,launch_authorized=False,launched=False,
        new_simulator_steps=0,new_optimizer_updates=0,word_thesis_changed=False,
        launch_requirements=['Explicit user instruction for this pilot',
            'Versioned launcher validates this receipt, exact source hashes and environment before authorizing runner in memory',
            'Refresh seed/storage checks and choose nonexistent output directory',
            'Leave default runner PILOT_RELEASED=False; final mode disabled',
            'After completion retain all fits, run full saved-artifact audit and verify independent archive'])
    OUT.mkdir(parents=True);runner.write(OUT/'readiness.json',record)
    print(json.dumps({k:record[k] for k in ['status','fixture_tests','fixture_manifest_files_verified','pilot_simulated_slots','launched']}))
    print(json.dumps(record['storage']))


if __name__=='__main__':main()
