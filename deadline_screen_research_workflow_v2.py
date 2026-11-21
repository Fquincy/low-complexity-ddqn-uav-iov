"""Complete staged workflow; authorization is enforced by the underlying runner."""
import argparse,json
from pathlib import Path
import run_deadline_screen_research_v2 as runner
import deadline_screen_research_storage_v2 as storage
import audit_deadline_screen_research_v2 as auditor

def finish(saved_run,storage_output,export=True):
    if export:storage.export(saved_run,storage_output)
    deployment=storage.audit(saved_run,storage_output)
    result=auditor.audit(saved_run)
    return dict(kind=result['kind'],artifact_audit_passed=True,deployment_storage_audit=deployment,
        quality_evidence=result['kind']!='SYNTHETIC_NOT_RESEARCH',research_launched=result['research_launched'])

def run(output,mode='integration-fixture',confirmation=None):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    runner.run(output/'run',mode,confirmation)  # Guard runs before output is created.
    result=finish(output/'run',output/'deployment_storage')
    runner.write(output/'workflow_verification.json',result)
    runner.write(output/'workflow_manifest.json',dict(files={x.relative_to(output).as_posix():runner.sha(x) for x in output.rglob('*') if x.is_file()}))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=['integration-fixture','development','final']);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--confirm-development',choices=[runner.CONFIRM]);args=parser.parse_args()
    print(json.dumps(run(args.output,args.mode,args.confirm_development)))
