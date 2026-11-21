"""Fixed family routes: preserve historical graphs; identical native principal arms."""
import shutil,time
from pathlib import Path
import numpy as np
import controller_v3_fused_runtime as native
import run_controller_v3_engineering as g

WHOLE_POLICY_ARMS=tuple(g.r.SHARED_ARMS)
NATIVE_ARMS=tuple(a for a in g.ARMS if a not in WHOLE_POLICY_ARMS)
ORIGINAL=g.ROOT/'validation/controller_v3_engineering_2026-11-05_01'

def route(arm):
    if arm not in g.ARMS:raise ValueError('unknown controller arm')
    return 'validated_complete_ORT_policy' if arm in WHOLE_POLICY_ARMS else 'native_core_and_decoder'

def export(model,arm,seed,path):
    path=Path(path);original_hash=g.r.old.digest(model.state_dict())
    if route(arm)=='native_core_and_decoder':
        runtime,meta=native.export(model,arm,path)
    else:
        source=ORIGINAL/'exports'/f'{arm}_{seed}.onnx'
        meta_source=g.read(ORIGINAL/'metadata.json')[f'{arm}_{seed}']
        assert meta_source['parameter_hash']==original_hash
        assert meta_source['graph_sha256']==g.r.old.sha(source)
        start=time.perf_counter();shutil.copy2(source,path);copy_seconds=time.perf_counter()-start
        start=time.perf_counter();session=g.c.session(path);startup=time.perf_counter()-start
        runtime=(session,np.zeros((2,5,2),np.float32))
        meta=dict(graph_sha256=g.r.old.sha(path),graph_bytes=path.stat().st_size,parameter_hash=original_hash,
            parameters=sum(v.numel() for v in model.parameters()),parameter_bytes=g.r.parameter_bytes(model),
            export_seconds=0.,retained_graph_copy_seconds=copy_seconds,startup_seconds=startup,
            reused_validated_full_graph=True,original_graph=str(source.relative_to(g.ROOT)).replace('\\','/'))
    assert g.r.old.digest(model.state_dict())==original_hash
    meta['route']=route(arm);meta['model_seed']=seed
    return runtime,meta

def values(runtime,arm,state):
    # Each route calls exactly one finite-input check and one ORT session.
    if route(arm)=='native_core_and_decoder':return native.values(runtime,arm,state)
    return g.compiled_values(runtime[0],state)

def action(runtime,arm,state):return values(runtime,arm,state)[0][0].copy()
