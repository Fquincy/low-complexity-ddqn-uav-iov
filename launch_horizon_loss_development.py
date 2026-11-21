"""Explicit launch boundary. Check-only never creates models or steps a simulator."""
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
import run_horizon_loss_research as runner
import horizon_loss_research_complete_workflow as workflow

ROOT=Path(__file__).resolve().parent
RELEASE=ROOT/'HORIZON_LOSS_LAUNCH_RELEASE_2026-11-14.json'

def preflight(output=None):
    release=runner.read(RELEASE);p=runner.protocol()
    runner.require(release['status']=='engineering_release_requires_explicit_launch_confirmation','release not ready')
    runner.require(release['protocol_sha256']==runner.sha(runner.SPEC),'release protocol mismatch')
    for name,digest in release['source_sha256'].items():runner.require(runner.sha(ROOT/name)==digest,'release source changed: '+name)
    for name,digest in release['evidence_sha256'].items():runner.require(runner.sha(ROOT/name)==digest,'release evidence changed: '+name)
    import importlib.metadata
    recorded=runner.read(ROOT/release['environment_record'])
    packages=json.loads(json.dumps({d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))
    runner.require(packages==recorded['packages'] and sys.executable==recorded['executable'] and sys.version==recorded['python'],'environment changed')
    runner.require(str(runner.torch.__file__)==recorded['torch_path'] and str(runner.models.fast.decoder.numba.__file__)==recorded['numba_path'],'runtime module path changed')
    if output is not None and Path(output).exists():raise FileExistsError(output)
    return dict(passed=True,protocol=p['id'],training_slots=p['development']['training_slots'],research_launched=False)

def launch(output,confirmation=None):
    if confirmation!=runner.CONFIRM:raise RuntimeError('Explicit development confirmation required')
    preflight(output)
    # Actual consumption is checked afresh after confirmation and before any research seed use.
    folder=Path(tempfile.mkdtemp(prefix='horizon_loss_prelaunch_',dir=ROOT/'validation'))
    command=[sys.executable,str(ROOT/'analysis/audit_horizon_loss_launch_seeds_2026_11_14.py'),'--output',str(folder/'seed_refresh.json')]
    subprocess.run(command,check=True)
    runner.require(runner.read(folder/'seed_refresh.json')['passed'],'fresh seed audit failed')
    # Freeze launch adapter/release as well as the runner's own transitive source snapshot.
    import shutil
    shutil.copy2(RELEASE,folder/RELEASE.name)
    shutil.copy2(__file__,folder/Path(__file__).name)
    runner.write(folder/'launch_request.json',dict(output=str(Path(output).resolve()),confirmation=confirmation,
        release_sha256=runner.sha(RELEASE),protocol_sha256=runner.sha(runner.SPEC),seed_audit_sha256=runner.sha(folder/'seed_refresh.json')))
    for name,digest in runner.read(RELEASE)['source_sha256'].items():
        runner.require(runner.sha(ROOT/name)==digest,'source changed during prelaunch: '+name)
        target=folder/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    runner.write(folder/'COMPLETE_SOURCE_FREEZE.json',runner.read(RELEASE)['source_sha256'])
    runner.DEVELOPMENT_RELEASED=True
    try:return workflow.run(output,'development',confirmation)
    finally:runner.DEVELOPMENT_RELEASED=False

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--confirm-development',choices=[runner.CONFIRM]);args=parser.parse_args()
    if args.check_only:print(json.dumps(preflight(args.output)))
    else:
        if args.output is None:parser.error('--output is required for development launch')
        if args.confirm_development!=runner.CONFIRM:parser.error('Explicit development confirmation required')
        print(json.dumps(launch(args.output,args.confirm_development)))
