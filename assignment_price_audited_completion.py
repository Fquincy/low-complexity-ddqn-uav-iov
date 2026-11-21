"""Verified completion sidecar; never launches training or repeats timings."""
import math
from pathlib import Path
import statistics
import run_assignment_price_research_bound as runner
import audit_assignment_price_research_bound_v2 as audit

ROOT=runner.r.ROOT
HEURISTIC_EVIDENCE=ROOT/'validation/assignment_price_launch_review_2026-11-15_01'

def heuristic_evidence():
    plan=runner.r.read(HEURISTIC_EVIDENCE/'PRE_HEURISTIC_TIMING_PLAN.json')
    rows=runner.r.read(HEURISTIC_EVIDENCE/'heuristic_timings.json')
    summary=runner.r.read(HEURISTIC_EVIDENCE/'heuristic_cost_summary.json')
    assert runner.r.sha(ROOT/'assignment_price_load_radio_heuristic.py')==plan['source_sha256']
    assert runner.r.sha(ROOT/plan['input'])==plan['input_sha256']
    assert [(x['round'],x['index']) for x in rows]==[(r,i) for r in range(5) for i in range(128)]
    assert all(isinstance(x['ns'],int) and x['ns']>0 for x in rows)
    values=sorted(x['ns'] for x in rows);position=(len(values)-1)*.95;lo=math.floor(position);hi=math.ceil(position)
    assert math.isclose(summary['median_ms'],statistics.median(values)/1e6,abs_tol=1e-12,rel_tol=1e-12)
    assert math.isclose(summary['p95_ms'],(values[lo]+(values[hi]-values[lo])*(position-lo))/1e6,abs_tol=1e-12,rel_tol=1e-12)
    assert plan['timing_attempts']==1 and not plan['acceptance_screens']
    return dict(summary=summary,plan=plan,evidence_files={name:runner.r.sha(HEURISTIC_EVIDENCE/name) for name in [
        'PRE_HEURISTIC_TIMING_PLAN.json','heuristic_timings.json','heuristic_cost_summary.json']},
        scope='Preserved standalone engineering timing of same deterministic heuristic/source/input; no same-session learned-policy speedup comparison, no research retiming')

def complete(run_output,completion_output):
    # Run outputs stay frozen. A successful sidecar exists only after full audit.
    output=Path(completion_output)
    if output.exists():raise FileExistsError(output)
    supplemental=heuristic_evidence()
    report=audit.audit(run_output)
    assert report['passed']
    output.mkdir(parents=True,exist_ok=False)
    runner.r.write(output/'saved_artifact_audit.json',report)
    runner.r.write(output/'heuristic_cost_reference.json',supplemental)
    runner.r.write(output/'completion.json',dict(status='AUDITED_COMPLETE',mode=report['mode'],
        run_manifest_sha256=runner.r.sha(Path(run_output)/'manifest.json'),run_path=str(Path(run_output).resolve()),
        heuristic_cost_reported=True,all_saved_artifacts_audited=True,word_thesis_changed=False,final_phase_launched=False,
        no_new_simulator_steps=True,no_optimizer_steps=True,no_retiming=True))
    runner.r.write(output/'manifest.json',{p.relative_to(output).as_posix():runner.r.sha(p) for p in output.rglob('*') if p.is_file()})
    return output
