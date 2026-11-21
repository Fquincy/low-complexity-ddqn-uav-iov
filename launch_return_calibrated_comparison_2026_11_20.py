"""Explicit source-bound development launch with numerical monitoring and archive."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

import audit_return_calibrated_comparison as auditor
import check_return_calibrated_comparison_readiness as readiness
import return_calibrated_progress as monitor
import run_return_calibrated_comparison as runner

ROOT=Path(__file__).resolve().parent
BINDING=ROOT/'RETURN_CALIBRATED_COMPARISON_LAUNCH_BINDING_2026-11-20.json'
TZ=timezone(timedelta(hours=8))


def preflight():
    binding=runner.read(BINDING)
    if binding['authorization']!='USER_EXPLICIT_LAUNCH_RETURN_CALIBRATED_51280':
        raise RuntimeError('Missing explicit launch authorization')
    for name,digest in binding['sha256'].items():
        if runner.sha(ROOT/name)!=digest:raise RuntimeError('Launch-bound file changed: '+name)
    if Path(sys.executable).resolve()!=Path(binding['python_executable']).resolve():raise RuntimeError('Wrong interpreter')
    if runner.DEVELOPMENT_RELEASED:raise RuntimeError('Default runner must remain disabled')
    if (ROOT/binding['output']).exists() or Path(binding['archive']).exists():raise RuntimeError('Output/archive exists; preserve prior evidence')
    check=ROOT/'validation/return_calibrated_comparison_launch_checks_2026-11-20_01'
    index=1
    while check.exists():
        index+=1;check=check.with_name(f'return_calibrated_comparison_launch_checks_2026-11-20_{index:02d}')
    readiness.OUT=check
    readiness.main()
    return binding,check


def execute(confirmation):
    if confirmation!=runner.CONFIRM:raise RuntimeError('Explicit development confirmation required')
    binding,checks=preflight();out=ROOT/binding['output'];archive=Path(binding['archive'])
    stop=threading.Event();begin=time.perf_counter_ns();phase=dict(name='starting',audit_ns=None)
    original_audit=auditor.audit

    def authorization():
        return dict(authorized=True,binding=BINDING.name,binding_sha256=runner.sha(BINDING),
            checks=str(checks.relative_to(ROOT)),scope='frozen51280slot return-calibrated development comparison only',
            word_thesis_changed=False,teacher_backend='Bound Python reference')

    def publish():
        if not (out/'run/runtime_plan.json').exists():return
        row=monitor.snapshot(out)
        if phase['name'] in ['full_artifact_audit','archiving','stopped']:
            row['stage']=phase['name'].upper()
        text=(f'Updated: {datetime.now(TZ).isoformat()}\n'
              f'Elapsed hours: {(time.perf_counter_ns()-begin)/3.6e12:.3f}\n'
              f'Current policy/fit: {row["progress"]}\n'+monitor.format_text(row))
        tmp=out/'LIVE_STATUS.txt.tmp';tmp.write_text(text,encoding='utf-8');tmp.replace(out/'LIVE_STATUS.txt')
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
            runner.write(out/'audit_phase_timing.json',dict(wall_ns=phase['audit_ns'],completed_at=datetime.now(TZ).isoformat(),scope='full saved-artifact audit including reference recomputation'))

    worker=threading.Thread(target=update_status,daemon=True);worker.start()
    try:
        auditor.audit=timed_audit
        runner.DEVELOPMENT_RELEASED=True
        phase['name']='training_and_evaluation'
        receipt=runner.run(out,'development',runner.CONFIRM)
        if not receipt['passed']:raise RuntimeError('Full artifact audit did not pass')
        runner.write(out/'launch_authorization.json',authorization())
        (out/'launch_binding.json').write_bytes(BINDING.read_bytes())
        phase['name']='archiving';publish();archive_begin=time.perf_counter_ns()
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
        timings=dict(audit_wall_ns=phase['audit_ns'],archive_copy_and_base_verification_wall_ns=time.perf_counter_ns()-archive_begin,
            elapsed_to_base_archive_verification_ns=time.perf_counter_ns()-begin,
            limitation='Archive phase excludes final small timing/verification receipt writes')
        runner.write(out/'phase_timings.json',timings);shutil.copy2(out/'phase_timings.json',archive/'phase_timings.json')
        digest=runner.sha(out/'phase_timings.json')
        if runner.sha(archive/'phase_timings.json')!=digest:raise RuntimeError('Phase timing archive mismatch')
        inventory['phase_timings.json']=digest
        runner.write(out/'archive_verification.json',dict(passed=True,archive=str(archive),files=len(inventory),sha256=inventory))
        shutil.copy2(out/'archive_verification.json',archive/'archive_verification.json')
        if runner.sha(out/'archive_verification.json')!=runner.sha(archive/'archive_verification.json'):raise RuntimeError('Archive receipt mismatch')
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
