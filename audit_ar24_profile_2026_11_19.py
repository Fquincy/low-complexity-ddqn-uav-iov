"""Recompute diagnostic summaries and check unchanged inputs; no retiming."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation/ar24_runtime_profile_2026-11-19_01'


def read(name):return json.loads((OUT/name).read_text())


def main():
    p=read('PLAN.json');v=read('profile.json');r=read('raw_profile.json');f=read('INPUT_FREEZE.json')
    assert len(r)==v['readings']==3*7*5*32==3360
    assert len({(x['round'],x['fit'],x['stage'],x['state']) for x in r})==3360
    actual=[(x['round'],x['fit'],x['stage']) for i,x in enumerate(r) if i%32==0]
    expected=[(rnd,fit,stage) for rnd,order in enumerate(v['orders']) for fit,stage in order]
    assert actual==expected
    for fit in p['fits']:
        for stage in p['stages']:
            ns=np.array([x['ns'] for x in r if x['fit']==fit and x['stage']==stage],np.float64)/1e6
            assert len(ns)==160 and np.all(ns>0)
            s=v['summaries'][str(fit)][stage]
            assert s['median_ms']==float(np.median(ns))
            assert s['p95_ms']==float(np.quantile(ns,.95))
    for path,h in f.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    base=ROOT/'validation/ar24_engineering_2026-11-19_01'
    original=json.loads((base/'PRE_TIMING_FREEZE.json').read_text())
    for name,h in original['graphs'].items():assert hashlib.sha256((base/name).read_bytes()).hexdigest()==h
    assert not json.loads((base/'verification.json').read_text())['cost_gate_passed']
    result=dict(passed=True,raw_readings_verified=3360,input_hashes_unchanged=True,
        original_graphs_verified=30,original_failed_cost_preserved=True,
        diagnostic_not_acceptance_retiming=True,optimization_implemented=False,
        research_release=False,new_simulator_steps=0,new_optimizer_updates=0,new_timing_readings=0)
    (OUT/'saved_artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
