"""Guarded development evaluation-only entry point. Never train; final disabled."""
import argparse,datetime,json
from pathlib import Path
import run_anchor_posttraining_continuation_v2 as c

ROOT=Path(__file__).resolve().parent
DECISION=ROOT/'validation/anchor_evaluation_readiness_decision_2026-11-11.json'

def preflight(source,output):
    if Path(output).exists():raise FileExistsError('Output exists; no overwrite/restart')
    source=Path(source).resolve();output=Path(output).resolve()
    if source==output or source in output.parents:raise ValueError('Output must be separate from original run')
    plan,freeze=c.preflight(source,False)
    return dict(state='evaluation_only_preflight_passed',source=str(source),output=str(output),
        checkpoints=85,evaluation_seeds=plan['evaluation_seeds'],new_training_slots=0,new_optimizer_updates=0,
        protocol_sha256=c.base.old.sha(c.PROTOCOL),checkpoint_freeze_sha256=c.base.old.sha(source/'ALL_FINAL_CHECKPOINTS_FREEZE.json'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['development','final'],required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--confirm-development-evaluation',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.mode!='development':parser.error('Final execution disabled')
    if not args.confirm_development_evaluation:parser.error('--confirm-development-evaluation required')
    checked=preflight(args.source,args.output)
    if args.check_only:print(json.dumps(checked,indent=2));return
    decision=c.read(DECISION)
    if decision['state']!='ready_for_development_evaluation_only':raise RuntimeError('readiness review is not passing')
    for name,digest in decision['source_sha256'].items():
        if c.base.old.sha(ROOT/name)!=digest:raise ValueError('reviewed source changed:'+name)
    args.control.mkdir(parents=True,exist_ok=False)
    c.write(args.control/'LAUNCH_AUTHORIZATION.json',dict(mode='development_evaluation_only',
        explicit_cli_confirmation=True,recorded_at=datetime.datetime.now().astimezone().isoformat(),
        preflight=checked,readiness_decision_sha256=c.base.old.sha(DECISION),
        original_launch_authorization_sha256=c.base.old.sha(ROOT/'validation/anchor_development_launch_2026-11-06_01/LAUNCH_AUTHORIZATION.json'),
        final_authorized=False,new_training_slots=0,new_optimizer_updates=0))
    c.CONTINUATION_RESEARCH_ENABLED=True
    try:result=c.run_continuation(args.source,args.output,fixture=False)
    finally:c.CONTINUATION_RESEARCH_ENABLED=False
    from audit_anchor_posttraining_continuation_v2 import audit_research
    audit=audit_research(args.output)
    c.write(args.control/'COMPLETION.json',dict(result=result,audit=audit))
    print(json.dumps(dict(result=result,audit=audit),indent=2),flush=True)

if __name__=='__main__':main()
