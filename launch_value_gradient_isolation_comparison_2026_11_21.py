"""Explicit source-bound research worker; preserves tested optimization sources."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

import audit_value_gradient_isolation_comparison as auditor
import check_value_gradient_isolation_comparison_readiness as readiness
import run_value_gradient_isolation_comparison as runner
import value_gradient_isolation_progress as monitor

ROOT=Path(__file__).resolve().parent
BINDING=ROOT/'VALUE_GRADIENT_ISOLATION_COMPARISON_LAUNCH_BINDING_2026-11-21.json'
AUTHORIZATION='USER_EXPLICIT_LAUNCH_VALUE_GRADIENT_ISOLATION_67120'
TZ=timezone(timedelta(hours=8))


def preflight():
    binding=runner.read(BINDING)
    if binding['authorization']!=AUTHORIZATION:raise RuntimeError('Missing explicit launch authorization')
    for name,digest in binding['sha256'].items():
        if runner.sha(ROOT/name)!=digest:raise RuntimeError('Launch-bound file changed: '+name)
    if Path(sys.executable).resolve()!=Path(binding['python_executable']).resolve():raise RuntimeError('Wrong interpreter')
    if runner.DEVELOPMENT_RELEASED or runner.vgi.DEVELOPMENT_RELEASED or runner.vgi.FINAL_RELEASED:
        raise RuntimeError('On-disk release defaults must remain disabled')
    spec=runner.protocol()
    if binding['output']!=spec['artifact_retention']['reserved_output'] or Path(binding['archive'])!=Path(spec['artifact_retention']['reserved_archive']):
        raise RuntimeError('Output/archive differs from frozen reservation')
    if (ROOT/binding['output']).exists() or Path(binding['archive']).exists():
        raise RuntimeError('Output/archive exists; preserve previous evidence')
    check=ROOT/'validation/value_gradient_isolation_comparison_launch_checks_2026-11-21_01'
    index=1
    while check.exists():
        index+=1;check=check.with_name(f'value_gradient_isolation_comparison_launch_checks_2026-11-21_{index:02d}')
    previous=readiness.OUT
    try:
        readiness.OUT=check;readiness.main()
    finally:readiness.OUT=previous
    return binding,check


def archive_completed(out,archive,audit_ns,begin):
    """Independent saved-file copy/inventory/hash verification; no model calls."""
    out=Path(out);archive=Path(archive);tick=time.perf_counter_ns()
    if not runner.read(out/'completion.json')['audit_passed'] or not runner.read(out/'saved_artifact_audit.json')['passed']:
        raise RuntimeError('Full artifact audit must pass before archive')
    archive.mkdir(parents=True,exist_ok=False)
    selected=['run','completion.json','saved_artifact_audit.json','launch_authorization.json','launch_binding.json','audit_phase_timing.json']
    expected=set()
    for name in selected:
        source=out/name;destination=archive/name
        if source.is_dir():
            expected.update(p.relative_to(out).as_posix() for p in source.rglob('*') if p.is_file())
            shutil.copytree(source,destination)
        else:expected.add(name);shutil.copy2(source,destination)
    actual={p.relative_to(archive).as_posix() for p in archive.rglob('*') if p.is_file()}
    if actual!=expected:raise RuntimeError('Archive inventory mismatch')
    inventory={}
    for name in sorted(expected):
        digest=runner.sha(archive/name)
        if runner.sha(out/name)!=digest:raise RuntimeError('Archive hash mismatch: '+name)
        inventory[name]=digest
    timings=dict(audit_wall_ns=audit_ns,archive_copy_and_base_verification_wall_ns=time.perf_counter_ns()-tick,
        elapsed_to_base_archive_verification_ns=time.perf_counter_ns()-begin,
        limitation='Archive phase excludes final small timing/verification receipt writes')
    runner.write(out/'phase_timings.json',timings);shutil.copy2(out/'phase_timings.json',archive/'phase_timings.json')
    digest=runner.sha(out/'phase_timings.json')
    if runner.sha(archive/'phase_timings.json')!=digest:raise RuntimeError('Archive phase timing mismatch')
    inventory['phase_timings.json']=digest
    runner.write(out/'archive_verification.json',dict(passed=True,archive=str(archive),files=len(inventory),sha256=inventory))
    shutil.copy2(out/'archive_verification.json',archive/'archive_verification.json')
    if runner.sha(out/'archive_verification.json')!=runner.sha(archive/'archive_verification.json'):
        raise RuntimeError('Archive receipt mismatch')
    return inventory


def execute(confirmation):
    if confirmation!=runner.CONFIRM:raise RuntimeError('Explicit development confirmation required')
    binding,checks=preflight();out=ROOT/binding['output'];archive=Path(binding['archive'])
    stop=threading.Event();begin=time.perf_counter_ns();phase=dict(name='starting',audit_ns=None)
    original_audit=auditor.audit
    def authorization():
        return dict(authorized=True,binding=BINDING.name,binding_sha256=runner.sha(BINDING),
            checks=str(checks.relative_to(ROOT)),scope='frozen67120slot value-gradient-isolation development comparison only',
            user_instruction=binding['user_instruction'],word_thesis_changed=False)
    def publish():
        if not (out/'run/runtime_plan.json').exists():return
        row=monitor.snapshot(out)
        if phase['name'] in ['full_artifact_audit','archiving','stopped']:row['stage']=phase['name'].upper()
        text=f'Updated: {datetime.now(TZ).isoformat()}\nElapsed hours: {(time.perf_counter_ns()-begin)/3.6e12:.3f}\nCurrent policy/fit: {row["progress"]}\n'+monitor.format_text(row)
        temp=out/'LIVE_STATUS.txt.tmp';temp.write_text(text,encoding='utf-8');temp.replace(out/'LIVE_STATUS.txt')
        runner.write(out/'LIVE_STATUS.json',row)
    def update_status():
        while not stop.wait(3):
            try:publish()
            except (OSError,ValueError,KeyError):pass
    def timed_audit(root):
        phase['name']='full_artifact_audit';start=time.perf_counter_ns()
        try:return original_audit(root)
        finally:
            phase['audit_ns']=time.perf_counter_ns()-start
            runner.write(out/'audit_phase_timing.json',dict(wall_ns=phase['audit_ns'],completed_at=datetime.now(TZ).isoformat(),
                scope='Full saved-artifact audit including independent first-update gradients and reference inferences'))
    worker=threading.Thread(target=update_status,daemon=True);worker.start()
    try:
        auditor.audit=timed_audit;runner.DEVELOPMENT_RELEASED=True;phase['name']='training_and_evaluation'
        receipt=runner.run(out,'development',runner.CONFIRM)
        if not receipt['passed']:raise RuntimeError('Full artifact audit did not pass')
        runner.write(out/'launch_authorization.json',authorization())
        (out/'launch_binding.json').write_bytes(BINDING.read_bytes())
        phase['name']='archiving';publish();archive_completed(out,archive,phase['audit_ns'],begin)
        stop.set();worker.join();phase['name']='complete';publish()
        return dict(status='AUDITED_AND_ARCHIVED_COMPLETE',output=str(out),archive=str(archive))
    except BaseException:
        stop.set();worker.join();phase['name']='stopped'
        if out.exists():
            (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            if not (out/'launch_authorization.json').exists():runner.write(out/'launch_authorization.json',authorization())
            try:publish()
            except (OSError,ValueError,KeyError):(out/'LIVE_STATUS.txt').write_text('STOPPED: inspect failure.txt and launch stderr. No automatic retry.\n',encoding='utf-8')
        raise
    finally:
        runner.DEVELOPMENT_RELEASED=False;auditor.audit=original_audit;stop.set();worker.join()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');parser.add_argument('--confirmation');args=parser.parse_args()
    if args.check_only:
        binding,checks=preflight();print(json.dumps(dict(status='PREFLIGHT_PASSED',output=binding['output'],checks=str(checks))))
    else:print(json.dumps(execute(args.confirmation)))
