"""Saved AR24 engineering audit and independently rebuilt reference checks."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import test_ar24_engineering_2026_11_19 as fixture
import run_assignment_price_research_bound as old

ROOT=Path(__file__).resolve().parent
OUT=fixture.OUT


def read(name):return json.loads((OUT/name).read_text())


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    p=read('PLAN.json');v=read('verification.json');c=read('correctness.json');f=read('PRE_TIMING_FREEZE.json');rows=read('timings.json')
    assert c['passed'] and len(c['checks'])==18
    assert c['counts']['eager_export_states']==666 and c['counts']['direct_decoder_states']==555
    assert c['counts']['autograd_cases']==18 and c['counts']['terminal_checks']==36
    assert len(rows)==v['raw_readings']==p['raw_readings']==1440
    assert len({(r['round'],r['arm'],r['fit'],r['state']) for r in rows})==1440
    assert all(r['ns']>0 for r in rows)
    actual=[(r['round'],r['arm'],r['fit']) for i,r in enumerate(rows) if i%32==0]
    expected=[(rnd,arm,fit) for rnd,order in enumerate(f['orders']) for arm,fit in order]
    assert actual==expected
    assert fixture.sha(OUT/'PLAN.json')==f['plan_sha256']
    assert fixture.sha(OUT/'synthetic_states.npy')==f['states_sha256']
    for name,h in f['sources'].items():assert fixture.sha(ROOT/name)==h
    assert len(f['graphs'])==30
    for name,h in f['graphs'].items():assert fixture.sha(OUT/name)==h
    screens=[]
    for fit in range(3):
        costs=v['costs'][str(fit)]
        for arm,cost in costs.items():
            ns=[r['ns'] for r in rows if r['arm']==arm and r['fit']==fit]
            assert len(ns)==96
            assert cost['median_ms']==float(np.median(ns)/1e6)
            assert cost['p95_ms']==float(np.quantile(ns,.95)/1e6)
        primary=costs['ar24']
        screens.append(dict(storage=primary['parameter_bytes']<=.5*costs['unmasked_compact64']['parameter_bytes'],
            median=all(primary['median_ms']<=costs[arm]['median_ms'] for arm in ('screened_wide128','unmasked_wide128')),p95=primary['p95_ms']<=1.))
    assert screens==v['cost_screens'] and all(all(s.values()) for s in screens)==v['cost_gate_passed']
    x=np.load(OUT/'synthetic_states.npy');equivalence=[]
    for fit,seed in enumerate(p['model_seeds']):
        for arm in fixture.REFS:
            model=old.base.build(arm,dict(model_seed=seed,cfg=copy.deepcopy(old.base.protocol()['training'])))['online'].eval()
            rt=old.base.references.Runtime.__new__(old.base.references.Runtime)
            rt.mode={'plain24_nonpositive':5,'additive128_unbounded':0,'additive64_nonpositive':1}[model.base_arm]
            rt.scaled=model.scaled;rt.screened=model.screened
            rt.session=fixture.ar.prior.deployed.session(OUT/f'{arm}_{fit}.onnx');maxdiff=0.
            for state in x:
                with torch.no_grad():a,q=model.greedy(torch.from_numpy(state)[None])[:2]
                da,dq=rt.values(state);np.testing.assert_array_equal(a.numpy(),da)
                np.testing.assert_allclose(q.numpy(),dq,rtol=1e-5,atol=2e-4)
                maxdiff=max(maxdiff,float(abs(q.item()-dq[0])))
            equivalence.append(dict(arm=arm,fit=fit,states=32,action_disagreements=0,max_Q_difference=maxdiff))
    protocol=json.loads((ROOT/'ASSIGNMENT_PRICE_RESEARCH_PROTOCOL_2026-11-15.json').read_text())
    for name,h in protocol['definition_sha256'].items():assert fixture.sha(ROOT/name)==h
    result=dict(saved_artifact_audit_passed=True,primary_and_variant_states=666,
        direct_decoder_states=555,reference_export_states=384,reference_equivalence=equivalence,
        raw_timing_readings_verified=1440,graphs_hashed=30,cost_screens=screens,
        engineering_cost_passed=v['cost_gate_passed'],research_release=False,quality_evidence=False,
        frozen_research_definitions_unchanged=True,new_simulator_steps=0,new_optimizer_updates=0,new_timing_readings=0)
    (OUT/'saved_artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='reference_equivalence'}))


if __name__=='__main__':main()
