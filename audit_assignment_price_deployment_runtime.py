"""Independent saved timing arithmetic and deployment artifact verification."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np
import torch
import onnx

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'validation/assignment_price_fused_2026-11-15_01'
OLD=ROOT/'validation/assignment_price_prototype_2026-11-15_01'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def quantile(values,p):
    x=sorted(values);position=(len(x)-1)*p;l=math.floor(position);h=math.ceil(position)
    return x[l]+(x[h]-x[l])*(position-l)

def main():
    out=BASE/'cost_gate';plan=read(out/'PRE_BENCHMARK_PLAN.json');summary=read(out/'cost_summary.json')
    verified={}
    for directory,name in [(out,'manifest.json'),(OLD,'tests_manifest.json'),(OLD/'cost_gate','manifest.json')]:
        for relative,digest in read(directory/name).items():assert sha(directory/relative)==digest,relative
        verified[str((directory/name).relative_to(ROOT))]=sha(directory/name)
    for name,digest in read(BASE/'extended_tests/tested_source_freeze_extended.json').items():assert sha(ROOT/name)==digest
    for name,digest in plan['sources'].items():assert sha(ROOT/name)==digest
    assert sha(BASE/'extended_tests/tests_extended.json')==plan['test_report_sha256'] and read(BASE/'extended_tests/tests_extended.json')['passed']
    freeze=read(out/'PRE_TIMING_FREEZE.json');assert freeze['plan_sha256']==sha(out/'PRE_BENCHMARK_PLAN.json')
    assert sha(OLD/'cost_gate/CHECKPOINT_FREEZE.json')==plan['checkpoint_freeze_sha256']
    assert freeze['checkpoints']==read(OLD/'cost_gate/CHECKPOINT_FREEZE.json')
    checkpoint_bytes={}
    for name,digest in freeze['checkpoints'].items():
        path=OLD/'cost_gate'/name;assert sha(path)==digest
        data=torch.load(path,weights_only=True,map_location='cpu');assert data['engineering_only']
        checkpoint_bytes[name]=sum(t.numel()*t.element_size() for t in data['online'].values())
    graph_payloads={}
    for name,digest in freeze['graphs'].items():
        path=out/name;assert sha(path)==digest
        model=onnx.load(str(path));onnx.checker.check_model(model)
        arm,fit=Path(name).stem.rsplit('_',1);fit=int(fit)
        checkpoint=torch.load(OLD/'cost_gate'/f'{arm.removeprefix("fused_")}_{fit}.pt',weights_only=True,map_location='cpu')
        arrays={t.name:onnx.numpy_helper.to_array(t) for t in model.graph.initializer}
        assert len(arrays)==4
        for key,array in arrays.items():
            assert array.dtype==np.float32
            np.testing.assert_array_equal(array,checkpoint['online']['core.'+key].numpy())
        payload=sum(t.nbytes for t in arrays.values())
        assert payload+20==checkpoint_bytes[f'{arm.removeprefix("fused_")}_{fit}.pt']
        assert path.stat().st_size==summary['costs'][arm][fit]['serialized_head_bytes']
        graph_payloads[name]=dict(parameter_bytes=payload,external_scale_buffer_bytes=20,serialized_bytes=path.stat().st_size)
    jobs=[(arm,fit) for arm in plan['arms'] for fit in range(2)]
    rng=np.random.default_rng(plan['order_seed'])
    for order in freeze['orders']:
        expected=list(jobs);rng.shuffle(expected);assert expected==[tuple(x) for x in order]
    expected=[(arm,fit,rnd,index) for rnd,order in enumerate(freeze['orders']) for arm,fit in order for index in range(plan['states'])]
    rows=read(out/'raw_timings.json');assert len(rows)==len(expected)==plan['readings']==15360
    assert [(x['arm'],x['fit'],x['round'],x['index']) for x in rows]==expected
    assert all(isinstance(x['ns'],int) and x['ns']>0 for x in rows)
    costs={}
    for arm,fit in jobs:
        samples=[x['ns'] for x in rows if x['arm']==arm and x['fit']==fit]
        assert len(samples)==640
        calculated=dict(median_ms=statistics.median(samples)/1e6,p95_ms=quantile(samples,.95)/1e6)
        reported=summary['costs'][arm][fit]
        for k,v in calculated.items():assert math.isclose(v,reported[k],rel_tol=1e-12,abs_tol=1e-12)
        assert checkpoint_bytes[f'{arm.removeprefix("fused_")}_{fit}.pt']==reported['parameter_bytes']+reported['buffer_bytes']
        costs[arm,fit]=dict(**calculated,parameter_bytes=reported['parameter_bytes'])
    t=plan['thresholds'];screens=[]
    for fit in range(2):
        x=costs['fused_coupled_sweep',fit]
        screens.append(dict(storage=x['parameter_bytes']<=t['parameter_bytes_max_fraction_of_unmasked_compact64']*costs['unmasked_compact64',fit]['parameter_bytes'],
            median=all(x['median_ms']<=t['median_ratio_max']*costs[arm,fit]['median_ms'] for arm in t['median_ms_max_fraction_of_each']),p95=x['p95_ms']<=t['p95_ms_max']))
    assert screens==summary['primary_screens']
    assert summary['all_primary_cost_screens_pass']==all(all(x.values()) for x in screens)
    assert freeze['equivalence_checks']==summary['equivalence_checks']==len(jobs)*plan['states']==3072
    result=dict(passed=True,manifests_verified=verified,checkpoints_verified=16,serialized_graphs_verified=8,
        graph_payloads=graph_payloads,raw_rows_recomputed=15360,primary_screens=screens,
        all_primary_cost_screens_pass=summary['all_primary_cost_screens_pass'],
        runtime_equivalence_scope='3072 recorded execution checks; audit independently recomputes saved cost arithmetic and integrity without retiming',
        new_simulator_steps=0,new_optimizer_updates=0,retiming=False)
    path=BASE/'independent_audit.json';assert not path.exists()
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
