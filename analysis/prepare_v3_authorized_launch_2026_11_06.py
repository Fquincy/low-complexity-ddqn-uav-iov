"""Prepare and check the authorized launch; this script does not train."""
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'validation/v3_development_launch_2026-11-06_01'
OUTPUT=ROOT/'development_runs/controller_v3_development_2026-11-06_01'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')

def main():
    manifest=read(ROOT/'validation/v3_engineering_gate_v2_completion_2026-11-06/COMPLETION_MANIFEST.json')
    for name,digest in manifest['files'].items():assert sha(ROOT/name)==digest,name
    assert not OUTPUT.exists() and not CONTROL.exists()
    CONTROL.mkdir()
    shutil.copy2(ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json',CONTROL/'PROJECT_STATUS_BEFORE_LAUNCH.json')
    write(CONTROL/'LAUNCH_AUTHORIZATION.json',dict(
        user_instruction='Launch the frozen 440,000-slot Version 3 development comparison.',
        recorded_at=datetime.datetime.now().astimezone().isoformat(),phase='development',training_slots=440000,
        output=str(OUTPUT),protocol_sha256=sha(ROOT/'LOW_COMPLEXITY_V3_RESEARCH_PROTOCOL_2026-11-05.json'),
        engineering_manifest_sha256=sha(ROOT/'validation/v3_engineering_gate_v2_completion_2026-11-06/COMPLETION_MANIFEST.json'),
        entrypoint_hashes={name:sha(ROOT/name) for name in ['launch_v3_development.py','supervise_v3_development.py']},
        frozen_settings_changed=False,final_phase_authorized=False,thesis_changed=False))
    checks=[]
    for mode,confirm in [('development',False),('final',True),('development',True)]:
        command=[sys.executable,str(ROOT/'launch_v3_development.py'),'--mode',mode,'--check-only',
            '--control',str(CONTROL),'--output',str(OUTPUT)]
        if confirm:command.append('--confirm-development-run')
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=60)
        if mode=='development' and confirm:
            assert result.returncode==0,result.stderr
            assert json.loads(result.stdout)['state']=='preflight_passed'
        else:assert result.returncode!=0
        assert not OUTPUT.exists()
        checks.append(dict(mode=mode,confirmation=confirm,exit_code=result.returncode,output_created=False,
            stdout=result.stdout,stderr=result.stderr))
    write(CONTROL/'PREFLIGHT_AND_ENTRYPOINT_TESTS.json',dict(passed=True,checks=checks,
        source_sha256={name:sha(ROOT/name) for name in ['launch_v3_development.py','supervise_v3_development.py']},
        research_training_launched=False))
    print(json.dumps(dict(control=str(CONTROL),output=str(OUTPUT),checks_passed=True,
        disk_free_bytes=shutil.disk_usage(ROOT).free),indent=2))

if __name__=='__main__':main()
