"""Recompute profile summaries and verify preserved inputs without retiming."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation/rq24_runtime_profile_2026-11-19_01'


def read(path):return json.loads(path.read_text())


def main():
    p=read(OUT/'PLAN.json');r=read(OUT/'raw_profile.json');v=read(OUT/'profile.json')
    hashes=read(OUT/'INPUT_FREEZE.json');source=Path(p['source'])
    assert len(r)==3*9*5*32==v['readings']==4320
    assert len({(x['round'],x['fit'],x['stage'],x['state']) for x in r})==len(r)
    for fit in p['fits']:
        for stage in p['stages']:
            readings=np.array([x['ns'] for x in r if x['fit']==fit and x['stage']==stage],np.float64)/1e6
            assert len(readings)==160 and np.all(readings>0)
            s=v['summaries'][str(fit)][stage]
            assert s['median_ms']==float(np.median(readings))
            assert s['p95_ms']==float(np.quantile(readings,.95))
            assert s['mean_ms']==float(np.mean(readings))
    for name,digest in hashes.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    old=read(source/'verification.json');assert not old['engineering_cost_passed']
    freeze=read(source/'PRE_TIMING_FREEZE.json')
    for name,digest in freeze['graphs'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    result=dict(passed=True,readings_verified=4320,source_and_graph_hashes_verified=True,
                original_failed_cost_result_preserved=True,original_graphs_verified=len(freeze['graphs']),
                profile_is_not_acceptance_retiming=True,research_released=False,
                new_simulator_steps=0,new_optimizer_updates=0,new_timing_readings=0)
    (OUT/'saved_artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
