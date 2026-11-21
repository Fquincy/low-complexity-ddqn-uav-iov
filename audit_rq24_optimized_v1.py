"""Independent recomputation of the single optimized synthetic benchmark."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation/rq24_optimized_v1_2026-11-19_02'


def read(path):return json.loads(path.read_text())


def main():
    plan=read(OUT/'PLAN.json');v=read(OUT/'verification.json');eq=read(OUT/'EQUIVALENCE.json')
    rows=read(OUT/'timings.json');freeze=read(OUT/'PRE_TIMING_FREEZE.json')
    assert len(rows)==plan['raw_readings']==1728
    assert len({(r['round'],r['arm'],r['fit'],r['state']) for r in rows})==1728
    assert all(r['ns']>0 for r in rows)
    observed=[(r['round'],r['arm'],r['fit']) for i,r in enumerate(rows) if i%32==0]
    expected=[(rnd,arm,fit) for rnd,order in enumerate(freeze['orders']) for arm,fit in order]
    assert observed==expected
    assert hashlib.sha256((OUT/'PLAN.json').read_bytes()).hexdigest()==freeze['plan_sha256']
    for path,digest in freeze['input_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    assert eq['passed'] and eq['counts']['exact_actions']==456 and eq['counts']['finite_gradient_arms']==12
    screens=[]
    for fit in range(3):
        costs=v['costs'][str(fit)]
        for arm,c in costs.items():
            ns=[r['ns'] for r in rows if r['arm']==arm and r['fit']==fit]
            assert len(ns)==96
            assert c['median_ms']==float(np.median(ns)/1e6)
            assert c['p95_ms']==float(np.quantile(ns,.95)/1e6)
        n=costs['optimized_rq24']
        screens.append(dict(storage=n['parameter_bytes']<=.5*costs['unmasked_compact64']['parameter_bytes'],
            median=all(n['median_ms']<=costs[a]['median_ms'] for a in ('screened_wide128','unmasked_wide128')),p95=n['p95_ms']<=1.))
    assert screens==v['cost_screens']
    assert all(all(s.values()) for s in screens)==v['cost_gate_passed']
    failed=ROOT/'validation/rq24_optimized_v1_2026-11-19_01'
    assert read(failed/'equivalence_failure.json')['timing_started'] is False
    assert not (failed/'timings.json').exists()
    result=dict(saved_artifact_audit_passed=True,equivalence_checks=456,raw_readings_verified=1728,
        cost_screens=screens,cost_gate_passed=v['cost_gate_passed'],failed_precision_check_preserved=True,
        one_timed_attempt_verified=True,research_release=False,new_simulator_steps=0,new_optimizer_updates=0)
    (OUT/'saved_artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
