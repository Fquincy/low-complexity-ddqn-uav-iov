"""Explicitly authorized seventeen-arm development launch; final disabled."""
import argparse,json
from pathlib import Path
import run_anchor_development_v4 as runner
ROOT=Path(__file__).resolve().parent
PROTOCOL_SHA='dae10a458a8de3234615e7f013e71730b52d12e9d87c477606de0f23a809f868'

def preflight(control):
    control=Path(control)
    manifest=runner.read(ROOT/'validation/anchor_launch_readiness_completion_2026-11-06.json')
    for name,digest in manifest['files'].items():
        if runner.base.old.sha(ROOT/name)!=digest:raise ValueError('readiness evidence changed:'+name)
    assert runner.base.old.sha(runner.PROTOCOL)==PROTOCOL_SHA
    p=runner.protocol();assert len(p['arms'])==17
    assert p['development']['training_slots']==680000
    assert p['development']['optimizer_updates']==593045
    assert p['development']['target_copies']==2890
    assert p['final']['status']=='reserved_untouched_no_launch'
    runner.verify_prior_engineering();runner.verify_factory_provenance()
    auth=runner.read(control/'LAUNCH_AUTHORIZATION.json')
    assert auth['phase']=='development' and auth['training_slots']==680000
    assert auth['explicit_user_launch_authorization'] is True
    assert auth['protocol_sha256']==PROTOCOL_SHA and not auth['final_phase_authorized']
    for name,digest in auth['entrypoint_hashes'].items():
        if runner.base.old.sha(ROOT/name)!=digest:raise ValueError('launch source changed:'+name)
    seed=runner.read(ROOT/'validation/anchor_research_seed_reservation_2026-11-06.json')
    assert seed['protocol_sha256']==PROTOCOL_SHA and not seed['collisions'] and not seed['errors']
    return dict(state='preflight_passed',phase='development',training_slots=680000,model_fits=85,protocol_sha256=PROTOCOL_SHA)

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
    auth=runner.read(args.control/'LAUNCH_AUTHORIZATION.json')
    if args.output.resolve()!=Path(auth['output']).resolve():parser.error('Output differs from authorized launch')
    result=preflight(args.control)
    if args.output.exists():raise FileExistsError('Output exists; automatic restart/overwrite forbidden')
    if args.check_only:print(json.dumps(result,indent=2));return
    runner.RESEARCH_ENABLED=True
    try:result=runner.run_plan(args.output,phase='development')
    finally:runner.RESEARCH_ENABLED=False
    from audit_anchor_development_v4 import audit_research
    audit=audit_research(args.output)
    runner.write(args.control/'COMPLETION.json',dict(runner=result,audit=audit))
    print(json.dumps(dict(runner=result,audit=audit),indent=2),flush=True)
if __name__=='__main__':main()
