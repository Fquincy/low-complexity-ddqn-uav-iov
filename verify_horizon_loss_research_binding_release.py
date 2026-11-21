"""Archive and verify the completed engineering gate; does not release training."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'validation/horizon_loss_research_binding_2026-11-14_01'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')

def check_startup_protocol():
    stage = OUT/'startup_fixture'
    saved = OUT/'workflow/run'
    report = read(stage/'startup/startup_report.json')
    expected_protocol = read(saved/'PRE_RUN_FREEZE.json')['protocol_sha256']
    for row in report['measurements']:
        assert row['protocol_sha256'] == expected_protocol
        assert row == read(stage/f"startup/{row['arm']}_{row['fit']:02d}.json")
    return len(report['measurements'])

def main():
    assert read(OUT/'tests.json')['passed']
    original = read(OUT/'evidence_manifest.json')
    for name,h in original.items():assert sha(OUT/name)==h,name
    startup = read(OUT/'startup_fixture/startup_verification.json')
    assert startup['passed'] and startup['models']==28
    assert check_startup_protocol()==28
    # A rehashed metadata label must not pass the independent release check.
    path=OUT/'startup_fixture/startup/startup_report.json'
    raw=path.read_bytes();data=read(path)
    try:
        data['measurements'][0]['protocol_sha256']='0'*64;write(path,data)
        try:check_startup_protocol()
        except AssertionError:pass
        else:raise AssertionError('Altered startup protocol accepted')
    finally:path.write_bytes(raw)
    complete = read(OUT/'startup_fixture/completion_manifest.json')
    for name,h in complete['files'].items():assert sha(OUT/'startup_fixture'/name)==h,name
    assert complete['comparison_workflow_manifest_sha256']==sha(OUT/'workflow/workflow_manifest.json')
    source = read(OUT/'workflow/run/PRE_RUN_FREEZE.json')['source_sha256']
    for name,h in source.items():assert sha(ROOT/name)==h,name
    for name,h in complete['sources'].items():assert sha(ROOT/name)==h,name
    for mode,token in [('development',None),('development','LAUNCH_FROZEN_HORIZON_LOSS_DEVELOPMENT'),('final',None)]:
        target=OUT/f'forbidden_outer_cli_{mode}_{token}'
        args=[sys.executable,str(ROOT/'horizon_loss_research_complete_workflow.py'),'--mode',mode,'--output',str(target)]
        if token:args+=['--confirm-development',token]
        proc=subprocess.run(args,capture_output=True,text=True,timeout=60)
        assert proc.returncode!=0 and not target.exists()
    additional=['horizon_loss_cold_start.py','horizon_loss_research_complete_workflow.py',
        Path(__file__).name,'HORIZON_LOSS_RESEARCH_BINDING_REVIEW_2026-11-14.txt']
    source.update({n:sha(ROOT/n) for n in additional})
    for name in source:
        dest=OUT/'release_source_snapshot'/name
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    write(OUT/'release_review_verification.json',dict(passed=True,core_tests=4,
        original_evidence_files_verified=len(original),source_files_frozen=len(source),
        separate_startup_models_verified=28,outer_CLI_guards_passed=True,
        startup_protocol_tamper_rejected=True,source_release_enabled=False,
        research_launched=False,reserved_development_seeds_consumed=False,
        word_thesis_changed=False))
    status_path=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json'
    shutil.copy2(status_path,OUT/'PROJECT_STATUS_BEFORE_BINDING_REVIEW.json')
    write(OUT/'ENGINEERING_RELEASE_MANIFEST.json',dict(
        status='engineering_verified_pending_separate_launch_review_and_explicit_instruction',
        protocol_sha256=sha(ROOT/'HORIZON_LOSS_RESEARCH_PROTOCOL_2026-11-14.json'),
        source_sha256=source,
        environment_sha256=sha(OUT/'workflow/run/environment.json'),
        source_release_enabled=False,
        files={x.relative_to(OUT).as_posix():sha(x) for x in OUT.rglob('*') if x.is_file()}))
    status=read(status_path)
    status['current_workstream']='horizon_loss_research_binding_and_full_engineering_workflow_verified_no_research_launch'
    status['next_engineering_step']='Review frozen source/environment release and refresh actual seed-use audit before separate explicit research launch decision; keep protocol and thresholds unchanged'
    status['next']=status['next_engineering_step']
    status['horizon_loss_research_freeze']['research_executable_bound']=True
    status['horizon_loss_research_binding']=dict(
        review='HORIZON_LOSS_RESEARCH_BINDING_REVIEW_2026-11-14.txt',evidence=str(OUT.relative_to(ROOT)),
        entrypoint='horizon_loss_research_complete_workflow.py',core_tests_passed=4,
        full_fixture_artifact_audit_passed=True,deployment_storage_audit_passed=True,
        training_fit_diagnostic_reports=28,policy_diagnostic_reports=31,
        startup_models_verified=28,release_manifest=str((OUT/'ENGINEERING_RELEASE_MANIFEST.json').relative_to(ROOT)),
        source_release_enabled=False,launch_authorized=False,research_launched=False,
        reserved_development_seeds_consumed=False,word_thesis_changed=False)
    write(status_path,status)
    print(json.dumps(read(OUT/'release_review_verification.json'),indent=2))

if __name__=='__main__':main()
