"""Child-process supervision only; no research launcher or release switch.

A later explicitly authorized, source-bound launcher must choose the command.
This module never imports model code, evaluates stderr as a success criterion,
or interprets a completed training phase as an artifact audit/archive pass.
"""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import subprocess
import time

import value_gradient_isolation_progress as progress

TZ=timezone(timedelta(hours=8))


def write_atomic(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def supervise(command,records_dir,progress_output=None,cwd=None,poll_seconds=.5):
    """Capture actual child return code and independent stderr, without shell.

    records_dir is a NEW external monitoring directory, never an immutable run.
    Files/partial records are retained on failure; no retry, overwrite or kill.
    A zero child exit does not establish a full audit or archive result.
    """
    if not isinstance(command,list) or not command or any(not isinstance(x,str) for x in command):
        raise ValueError('Explicit argument list required')
    if poll_seconds<=0 or poll_seconds>30:raise ValueError('Poll interval must be in (0,30] seconds')
    records=Path(records_dir).resolve()
    if progress_output is not None:
        output=Path(progress_output).resolve()
        if records==output or records.is_relative_to(output/'run'):
            raise ValueError('Monitor records must be outside research output and immutable run')
    records.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter_ns();last_progress_error=None
    def publish(child_code):
        nonlocal last_progress_error
        row=None
        if progress_output is not None:
            try:row=progress.snapshot(progress_output);last_progress_error=None
            except (OSError,ValueError,KeyError) as error:last_progress_error=str(error)
        status=dict(updated_at=datetime.now(TZ).isoformat(),elapsed_ns=time.perf_counter_ns()-start,
            child_pid=child.pid,child_exit_code=child_code,process_running=child_code is None,
            phase='RUNNING' if child_code is None else ('CHILD_EXITED_ZERO_CHECK_AUDIT_AND_ARCHIVE' if child_code==0 else 'CHILD_FAILED_INSPECT_LOGS'),
            progress=row,progress_read_error=last_progress_error,
            stderr_is_not_exit_status=True,archive_independently_verified=False)
        write_atomic(records/'LIVE_STATUS.json',status)
        text=f"Updated: {status['updated_at']}\nElapsed hours: {status['elapsed_ns']/3.6e12:.3f}\nProcess: {status['phase']}\nPython child exit code: {child_code}\n"
        text+=progress.format_text(row) if row else 'Numeric progress unavailable: '+str(last_progress_error)+'\n'
        text+='A zero child exit is not an independent audit/archive verification.\n'
        temp=records/'LIVE_STATUS.txt.tmp';temp.write_text(text,encoding='utf-8');temp.replace(records/'LIVE_STATUS.txt')
        return status
    with (records/'stdout.log').open('xb') as stdout,(records/'stderr.log').open('xb') as stderr:
        flags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess,'CREATE_NO_WINDOW') else 0
        try:child=subprocess.Popen(command,cwd=cwd,stdout=stdout,stderr=stderr,creationflags=flags,shell=False)
        except OSError as error:
            write_atomic(records/'child_start_failure.json',dict(error=repr(error),child_created=False,automatic_retry=False))
            raise
        write_atomic(records/'child_started.json',dict(pid=child.pid,command=command,started_at=datetime.now(TZ).isoformat()))
        try:
            while True:
                code=child.poll();status=publish(code)
                if code is not None:break
                try:child.wait(timeout=poll_seconds)
                except subprocess.TimeoutExpired:pass
        except BaseException as error:
            write_atomic(records/'monitor_failure.json',dict(error=repr(error),child_pid=child.pid,
                child_exit_code=child.poll(),child_may_still_be_running=child.poll() is None,
                automatic_retry=False,automatic_child_termination=False))
            raise
    result=dict(child_exit_code=code,child_pid=child.pid,child_exited_successfully=code==0,
        stderr_bytes=(records/'stderr.log').stat().st_size,stdout_bytes=(records/'stdout.log').stat().st_size,
        audit_pass_claim=False,archive_verification_claim=False,elapsed_ns=status['elapsed_ns'])
    write_atomic(records/'child_result.json',result)
    return result
