"""Authorized fixed-command supervisor; child exit code is captured without shell."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from value_gradient_isolation_process_monitor import supervise

ROOT=Path(__file__).resolve().parent
BINDING=ROOT/'VALUE_GRADIENT_ISOLATION_COMPARISON_LAUNCH_BINDING_2026-11-21.json'
CONFIRM='LAUNCH_FROZEN_VALUE_GRADIENT_ISOLATION_COMPARISON'


def execute(confirmation):
    if confirmation!=CONFIRM:raise RuntimeError('Explicit development confirmation required')
    binding=json.loads(BINDING.read_text(encoding='utf-8'))
    if binding['authorization']!='USER_EXPLICIT_LAUNCH_VALUE_GRADIENT_ISOLATION_67120':
        raise RuntimeError('Missing explicit launch authorization')
    for name,digest in binding['sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Launch-bound file changed: '+name)
    if Path(sys.executable).resolve()!=Path(binding['python_executable']).resolve():raise RuntimeError('Wrong interpreter')
    command=[sys.executable,str(ROOT/'launch_value_gradient_isolation_comparison_2026_11_21.py'),'--confirmation',CONFIRM]
    return supervise(command,ROOT/binding['monitoring_output'],progress_output=ROOT/binding['output'],cwd=ROOT,poll_seconds=3)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--confirmation');args=parser.parse_args()
    result=execute(args.confirmation);print(json.dumps(result));raise SystemExit(result['child_exit_code'])
