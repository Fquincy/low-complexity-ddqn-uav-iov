"""Independent saved-artifact cost/equivalence audit; no timing or models run."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation/rq24_engineering_2026-11-19_01/fixture'


def read(name):return json.loads((OUT/name).read_text())


def main():
    plan=read('PLAN.json');freeze=read('PRE_TIMING_FREEZE.json');v=read('verification.json');rows=read('timings.json')
    assert len(rows)==3*5*3*32==v['raw_timing_rows']
    assert len({(r['round'],r['arm'],r['fit'],r['state']) for r in rows})==len(rows)
    assert all(r['ns']>0 for r in rows)
    for path,h in freeze['sources'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
    for path,h in freeze['graphs'].items():assert hashlib.sha256((OUT/path).read_bytes()).hexdigest()==h
    assert len(freeze['graphs'])==24 and len(v['equivalence'])==24
    assert all(e['states']==32 and e['action_disagreements']==0 for e in v['equivalence'])
    assert hashlib.sha256((OUT/'synthetic_states.npy').read_bytes()).hexdigest()==freeze['synthetic_states_sha256']
    recomputed=[]
    for fit in range(3):
        costs=v['costs'][str(fit)]
        for arm,cost in costs.items():
            ns=[r['ns'] for r in rows if r['arm']==arm and r['fit']==fit]
            assert len(ns)==96
            assert cost['median_ms']==float(np.median(ns)/1e6)
            assert cost['p95_ms']==float(np.quantile(ns,.95)/1e6)
        recomputed.append(dict(storage=costs['rq24']['parameter_bytes']<=.5*costs['unmasked_compact64']['parameter_bytes'],
            median=all(costs['rq24']['median_ms']<=costs[a]['median_ms'] for a in ('screened_wide128','unmasked_wide128')),
            p95=costs['rq24']['p95_ms']<=1.))
    assert recomputed==v['cost_screens']
    report=dict(saved_artifact_audit_passed=True,cost_screens=recomputed,
                cost_gate_passed=all(all(x.values()) for x in recomputed),
                research_release=False,new_simulator_steps=0,new_optimizer_updates=0,new_timing_readings=0,
                quality_evidence=False)
    (OUT/'saved_artifact_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
