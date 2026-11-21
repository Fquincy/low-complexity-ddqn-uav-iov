"""Versioned outer workflow including separate startup reporting."""
import argparse
import json
from pathlib import Path
import horizon_loss_research_workflow as paired
import horizon_loss_cold_start as startup

def finish(comparison, output):
    comparison, output = Path(comparison), Path(output)
    assert paired.runner.read(comparison/'workflow_verification.json')['artifact_audit_passed']
    startup.generate(comparison/'run', output/'startup')
    result = startup.audit(comparison/'run', output/'startup')
    paired.runner.write(output/'startup_verification.json', result)
    paired.runner.write(output/'completion_manifest.json', dict(
        comparison_workflow_manifest_sha256=paired.runner.sha(comparison/'workflow_manifest.json'),
        sources={str(Path(__file__).name):paired.runner.sha(__file__),
                 'horizon_loss_cold_start.py':paired.runner.sha(startup.__file__)},
        files={x.relative_to(output).as_posix():paired.runner.sha(x) for x in output.rglob('*') if x.is_file()}))
    return result

def run(output, mode='integration-fixture', confirmation=None):
    output = Path(output)
    if output.exists():raise FileExistsError(output)
    paired.run(output/'comparison', mode, confirmation)
    return finish(output/'comparison', output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['integration-fixture','development','final'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--confirm-development', choices=[paired.runner.CONFIRM])
    args = parser.parse_args()
    print(json.dumps(run(args.output,args.mode,args.confirm_development)))
