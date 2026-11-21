"""Bind synthetic test evidence and verify frozen historical sources, no training."""
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import torch
import return_calibrated_handoff as handoff

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation/return_calibrated_handoff_mechanics_2026-11-20_02'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def main():
    if (OUT/'verification.json').exists():raise RuntimeError('Preserve completed evidence')
    tests=read(OUT/'tests.json')
    assert tests['passed'] and tests['tests']==15 and tests['failures']==tests['errors']==0
    first=read(ROOT/'validation/return_calibrated_handoff_mechanics_2026-11-20_01/tests.json')
    assert not first['passed'] and first['errors']==1
    assert all(row['new_simulator_steps']==row['new_teacher_calls']==0 and not row['reserved_research_seeds_used'] for row in [tests,first])
    design=ROOT/'RETURN_CALIBRATED_DDQN_HANDOFF_DESIGN_STATUS_2026-11-20.json'
    brief=ROOT/'RETURN_CALIBRATED_DDQN_HANDOFF_DESIGN_BRIEF_2026-11-20.txt'
    reservation=read(design)
    prior=read(ROOT/'validation/return_calibrated_handoff_brief_2026-11-20_01/verification.json')
    assert sha(design)==prior['design_status_sha256'] and sha(brief)==prior['brief_sha256']
    for arm in reservation['arms']:
        d=handoff.ARMS[arm['id']]
        assert all(getattr(d,name)==arm[name] for name in ['calibration','retention','bc','ddqn'])
    assert (handoff.RIDGE_RHO,handoff.RETENTION_LAMBDA,handoff.SCALE_FLOOR)==(.001,1.,1e-6)
    historical=read(ROOT/'development_runs/centered_linear_correction_comparison_2026-11-20_01/run/PRE_RUN_FREEZE.json')
    for name,h in historical['source_sha256'].items():assert sha(ROOT/name)==h,name
    definitions=set(historical['source_sha256'])|{
        'return_calibrated_handoff.py','test_return_calibrated_handoff.py',
        'verify_return_calibrated_handoff_mechanics_2026_11_20.py',
        'RETURN_CALIBRATED_HANDOFF_MECHANICS_REVIEW_2026-11-20.txt',design.name,brief.name}
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=False)
    for name in sorted(definitions):
        destination=snapshot/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination)
    torch.set_num_threads(1)
    env=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,platform=platform.platform(),threads=torch.get_num_threads())
    assert env==historical['environment']
    forbidden=[name for name in sys.modules if name.startswith(('chapter3_iov_env','heuristic_pilot_teacher_adapter'))]
    assert not forbidden,forbidden
    manifest={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    write(OUT/'manifest.json',manifest)
    for name,h in manifest.items():assert sha(OUT/name)==h,name
    updates=sum(row[k] for row in [first,tests] for k in ['synthetic_bc_optimizer_steps','synthetic_td_optimizer_steps','independent_reference_optimizer_steps'])
    assert updates==26
    record=dict(status='ISOLATED_MECHANICS_PASSED_RESEARCH_DISABLED',passed=True,tests=15,
        source_sha256={name:sha(ROOT/name) for name in sorted(definitions)},
        manifest_sha256=sha(OUT/'manifest.json'),manifest_files=len(manifest),
        manifest_scope='All saved mechanics artifacts except manifest.json and this verification.json',
        environment=env,python_executable=sys.executable,
        prior_failed_test_attempt_preserved=True,total_synthetic_optimizer_steps_both_attempts=updates,
        new_research_optimizer_updates=0,new_simulator_steps=0,new_teacher_calls=0,
        research_or_teacher_modules_imported=forbidden,reserved_research_seeds_used=False,
        actual_research_policy_evaluated=False,scientific_quality_established=False,
        research_protocol_frozen=False,research_launch_authorized=False,launched=False,
        next_gate='Freeze full seven-arm comparison; then separate runner/auditor integration fixture',
        word_thesis_changed=False)
    write(OUT/'verification.json',record)
    print(json.dumps({k:record[k] for k in ['status','tests','manifest_files','total_synthetic_optimizer_steps_both_attempts','new_simulator_steps','launched']},indent=2))


if __name__=='__main__':main()
