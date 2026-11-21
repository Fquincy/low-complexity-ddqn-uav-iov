"""Explicitly authorized development entrypoint; final execution is unavailable."""
import argparse
import json
from pathlib import Path
import run_v3_research_gate_v2 as runner

ROOT=Path(__file__).resolve().parent
PROTOCOL_SHA='5feece24ac058b547140b0513f4991a3f8035536b59b4002893f0ae1fc8d96d9'

def preflight(control):
    control=Path(control)
    manifest=runner.read(ROOT/'validation/v3_engineering_gate_v2_completion_2026-11-06/COMPLETION_MANIFEST.json')
    for name,digest in manifest['files'].items():
        path=control/'PROJECT_STATUS_BEFORE_LAUNCH.json' if name=='LOW_COMPLEXITY_PROJECT_STATUS.json' else ROOT/name
        if runner.base.old.sha(path)!=digest:raise ValueError('engineering evidence changed:'+name)
    assert runner.base.old.sha(runner.PROTOCOL)==PROTOCOL_SHA
    spec=runner.protocol();assert len(spec['arms'])==11
    assert spec['development']['training_slots']==440000
    assert spec['development']['optimizer_updates']==383735
    assert spec['development']['target_copies']==1870
    assert spec['final']['status']=='reserved_untouched_no_launch'
    tests=runner.read(ROOT/'validation/v3_engineering_gate_v2_tests_2026-11-06.json')
    assert tests['passed'] and tests['tests']==7
    for name,digest in tests['source_sha256'].items():assert runner.base.old.sha(ROOT/name)==digest,name
    runner.verify_prior_engineering()
    seed=runner.read(ROOT/'validation/v3_seed_consumption_review_2026-11-06.json')
    assert not seed['errors'] and not seed['collisions'] and seed['protocol_sha256']==PROTOCOL_SHA
    authorization=runner.read(control/'LAUNCH_AUTHORIZATION.json')
    assert authorization['phase']=='development' and authorization['training_slots']==440000
    assert authorization['user_instruction']=='Launch the frozen 440,000-slot Version 3 development comparison.'
    for name,digest in authorization['entrypoint_hashes'].items():assert runner.base.old.sha(ROOT/name)==digest,name
    return dict(state='preflight_passed',engineering_manifest_files=len(manifest['files']),
        phase='development',training_slots=440000,model_fits=55,protocol_sha256=PROTOCOL_SHA)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['development','final'],required=True)
    parser.add_argument('--confirm-development-run',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.mode!='development':parser.error('Final execution remains disabled')
    if not args.confirm_development_run:parser.error('--confirm-development-run is required')
    authorization=runner.read(args.control/'LAUNCH_AUTHORIZATION.json')
    if args.output.resolve()!=Path(authorization['output']).resolve():parser.error('Output differs from authorized launch')
    result=preflight(args.control)
    if args.output.exists():raise FileExistsError('Output exists; automatic restart/overwrite is forbidden')
    if args.check_only:
        print(json.dumps(result,indent=2));return
    runner.SOURCES=list(dict.fromkeys([*runner.SOURCES,'launch_v3_development.py','supervise_v3_development.py']))
    runner.RESEARCH_ENABLED=True
    try:
        result=runner.run_plan(args.output,phase='development')
    finally:
        runner.RESEARCH_ENABLED=False
    from audit_v3_research_gate_v2 import audit_research
    audit=audit_research(args.output)
    runner.write(args.control/'COMPLETION.json',dict(runner=result,audit=audit))
    print(json.dumps(dict(runner=result,audit=audit),indent=2),flush=True)

if __name__=='__main__':main()
