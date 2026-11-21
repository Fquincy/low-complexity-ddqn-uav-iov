"""Record launch authority and test guards before starting a worker."""
import ast,datetime,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run_anchor_development_v4 as r
import launch_anchor_development as entry
CONTROL=ROOT/'validation/anchor_development_launch_2026-11-06_01'
OUTPUT=ROOT/'development_runs/terminal_anchor_development_2026-11-06_01'
def functions(path):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
def main():
    assert not CONTROL.exists() and not OUTPUT.exists()
    for before,after in [('run_anchor_research_route_v3.py','run_anchor_development_v4.py'),('audit_anchor_research_route_v3.py','audit_anchor_development_v4.py')]:
        old=functions(ROOT/before);new=functions(ROOT/after)
        for name,node in old.items():
            if name!='main':assert node==new[name],name
    CONTROL.mkdir()
    shutil.copy2(ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json',CONTROL/'PROJECT_STATUS_BEFORE_LAUNCH.json')
    entrypoints=['run_anchor_development_v4.py','audit_anchor_development_v4.py','launch_anchor_development.py','supervise_anchor_development.py']
    auth=dict(user_instruction='proceeding with the frozen **680,000-slot, seventeen-controller development comparison**',
        explicit_user_launch_authorization=True,recorded_at=datetime.datetime.now().astimezone().isoformat(),
        phase='development',training_slots=680000,output=str(OUTPUT),protocol_sha256=r.base.old.sha(r.PROTOCOL),
        entrypoint_hashes={n:r.base.old.sha(ROOT/n) for n in entrypoints},final_phase_authorized=False,
        scientific_settings_changed=False,word_thesis_changed=False)
    r.write(CONTROL/'LAUNCH_AUTHORIZATION.json',auth)
    preflight=entry.preflight(CONTROL)
    cmd=[sys.executable,str(ROOT/'launch_anchor_development.py'),'--mode','development','--control',str(CONTROL),'--output',str(OUTPUT)]
    denied=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    assert denied.returncode!=0 and '--confirm-development-run is required' in denied.stderr
    final_cmd=cmd.copy();final_cmd[final_cmd.index('development')]='final'
    final=subprocess.run(final_cmd+['--confirm-development-run'],capture_output=True,text=True,timeout=60)
    assert final.returncode!=0 and 'Final execution remains disabled' in final.stderr
    check=subprocess.run(cmd+['--confirm-development-run','--check-only'],capture_output=True,text=True,timeout=60)
    assert check.returncode==0,check.stderr
    assert not OUTPUT.exists() and r.RESEARCH_ENABLED is False
    r.write(CONTROL/'PREFLIGHT_AND_ENTRYPOINT_TESTS.json',dict(state='passed',preflight=preflight,
        confirmation_guard_passed=True,final_guard_passed=True,check_only_passed=True,
        shared_runner_and_auditor_functions_unchanged=True,source_snapshot_includes_entrypoints=True,
        training_launched=False))
    files={n:r.base.old.sha(ROOT/n) for n in r.SOURCES}
    files.update({str(p.relative_to(ROOT)).replace('\\','/'):r.base.old.sha(p) for p in CONTROL.iterdir() if p.is_file()})
    r.write(CONTROL/'LAUNCH_EVIDENCE_MANIFEST.json',dict(files=files))
    print(json.dumps(dict(control=str(CONTROL),output=str(OUTPUT),preflight=preflight,guards='passed')))
if __name__=='__main__':main()
