"""Bind completed integration tests and saved evidence; no simulation or training."""
import hashlib
import io
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch
import run_return_calibrated_comparison as runner
from freeze_heuristic_initialization_pilot_plan_2026_11_19 import ranges

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation/return_calibrated_comparison_fixture_2026-11-20_02'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write_new(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def main():
    if (OUT/'engineering_verification.json').exists():raise RuntimeError('Preserve prior engineering receipt')
    spec=runner.protocol();tests=read(OUT/'tests.json');audit=read(OUT/'saved_artifact_audit.json');completion=read(OUT/'completion.json')
    assert tests['passed'] and tests['tests']==12 and audit['passed'] and completion['status']=='ENGINEERING_AUDITED_COMPLETE'
    assert audit['simulation_slots_recorded']==1880 and audit['historical_ddqn_updates']==384 and audit['historical_bc_updates']==24
    assert audit['retention_batches']==128 and audit['ridge_solves']==6 and audit['final_checkpoints']==14
    freeze=read(OUT/'run/PRE_RUN_FREEZE.json');manifest=read(OUT/'run/manifest.json')
    assert sha(OUT/'run/manifest.json')==completion['manifest_sha256']
    actual={p.relative_to(OUT/'run').as_posix() for p in (OUT/'run').rglob('*') if p.is_file() and p!=OUT/'run/manifest.json'}
    assert actual==set(manifest)
    for name,digest in manifest.items():assert sha(OUT/'run'/name)==digest,name
    for name,digest in freeze['source_sha256'].items():
        assert sha(ROOT/name)==digest and sha(OUT/'run/source_snapshot'/name)==digest,name
    torch.set_num_threads(1)
    environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),threads=1)
    assert environment==freeze['environment']==spec['numerical_environment']
    assert not runner.DEVELOPMENT_RELEASED
    # Audit reservations without drawing from any stream. Prior fixture family
    # is an intentional same-fixture alias; check research roots separately.
    fixture=read(OUT/'run/runtime_plan.json')['seeds'];collisions=[];documents=[];aliases=[]
    for p in sorted(ROOT.glob('*.json'))+sorted((ROOT/'validation').glob('*/run/runtime_plan.json'))+sorted((ROOT/'development_runs').glob('*/run/runtime_plan.json')):
        v=read(p);entry=dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p))
        if p.parent.parent.name.startswith('return_calibrated_comparison_fixture_'):
            assert v['kind']=='ENGINEERING_FIXTURE_NOT_RESEARCH' and v['seeds']==fixture
            aliases.append(entry);continue
        documents.append(entry)
        for a,b,label in ranges(fixture):
            for c,d,old in ranges(v):
                if max(a,c)<=min(b,d):collisions.append(dict(path=entry['path'],fixture=label,existing=old))
    assert not collisions,collisions
    # Separate weights-only serialization size from optimizer/target checkpoints.
    # Bytes are round-tripped in memory; no deployment export/equivalence claim.
    serialized={}
    for p in sorted((OUT/'run').glob('*/fit_*/final.pt')):
        ck=torch.load(p,weights_only=True);stream=io.BytesIO();torch.save(ck['online'],stream)
        stream.seek(0);reopened=torch.load(stream,weights_only=True)
        assert all(torch.equal(v,reopened[k]) for k,v in ck['online'].items())
        serialized[p.relative_to(OUT/'run').as_posix()]=dict(weights_only_serialized_bytes=len(stream.getvalue()),
            tensor_parameter_bytes=67364,buffer_bytes=20,scope='in-memory torch state_dict serialization, not measured process memory')
    write_new(OUT/'deployment_storage_measurement.json',serialized)
    source_names=['run_return_calibrated_comparison.py','audit_return_calibrated_comparison.py','test_return_calibrated_comparison.py',
        'verify_return_calibrated_comparison_engineering_2026_11_20.py','RETURN_CALIBRATED_COMPARISON_ENGINEERING_REVIEW_2026-11-20.txt']
    snapshot=OUT/'engineering_source_snapshot';snapshot.mkdir(exist_ok=False)
    for name in source_names:(snapshot/name).write_bytes((ROOT/name).read_bytes())
    record=dict(status='PAIRED_WORKFLOW_AND_AUDITOR_PASSED_RESEARCH_DISABLED',tests_passed=12,
        protocol_sha256=sha(runner.SPEC),source_sha256={name:sha(ROOT/name) for name in source_names},
        run_manifest_files_verified=len(manifest),environment=environment,python_executable=sys.executable,
        fixture_exposure=dict(simulated_slots=1880,bc_updates=24,ddqn_updates=384,retention_batches=128,ridge_solves=6,final_checkpoints=14),
        prior_fixture_preserved=True,total_both_fixtures=dict(simulated_slots=3760,bc_updates=48,ddqn_updates=768,retention_batches=256,ridge_solves=12),
        seed_audit=dict(documents=documents,same_fixture_aliases=aliases,collisions=collisions),
        audit_teacher_reference_recomputations=audit['teacher_reference_recomputations'],
        new_research_simulator_steps=0,new_research_optimizer_updates=0,reserved_research_seeds_used=False,
        research_launch_authorized=False,development_release_enabled=False,launched=False,
        next_gate='Separate launch-readiness review; no research launch from engineering gate',
        limitations=['No replay of historical gradient/parameter trajectory',
            'BC branches reconstructed approximately by inverse alpha; hashes bind parity but do not independently replay BC',
            'Positive fixture/engineering outcomes do not establish scientific quality or novelty'],word_thesis_changed=False)
    write_new(OUT/'engineering_verification.json',record)
    inventory={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    write_new(OUT/'engineering_manifest.json',inventory)
    print(json.dumps(dict(status=record['status'],tests=12,run_manifest_files=len(manifest),engineering_artifacts=len(inventory),
        seed_documents_checked=len(documents),fixture_aliases=len(aliases),research_launched=False),indent=2))


if __name__=='__main__':main()
