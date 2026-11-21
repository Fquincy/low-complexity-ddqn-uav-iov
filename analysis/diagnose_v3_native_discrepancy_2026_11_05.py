"""Read-only numerical diagnosis of the preserved pre-timing failure."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
import controller_v3_fused_runtime as f
import run_controller_v3_fused_gate as runner
import controller_redesign_v2 as old

g=runner.g;OUT=ROOT/'validation/controller_v3_runtime_optimization_2026-11-05_01/gate_01'
def trace_native(tables):
    u,w,L,beta,motion,base=[x[0] for x in tables];constant=base;moves=[]
    for j in range(2):
        moves.append(int(np.argmin(motion[j])));constant=np.float32(constant+motion[j,moves[-1]])
    a=np.argmin(u,axis=1);best=a.copy();current=f.score(a,u,w,L,beta,constant);candidates=[dict(action=a.tolist(),cost=float(current))]
    for _ in range(2):
        p=f.price_table(a,w,L,beta);a=np.argmin(u+p,axis=1);cost=f.score(a,u,w,L,beta,constant)
        candidates.append(dict(action=a.tolist(),cost=float(cost)))
        if cost<current:current=cost;best=a.copy()
    a=best;repairs=[]
    for _ in range(4):
        p=f.price_table(a,w,L,beta);delta=(u+p)-(u+p)[np.arange(len(a)),a,None]
        ix=int(delta.argmin());minimum=np.float32(delta.reshape(-1)[ix]);trial=a.copy()
        if minimum<np.float32(-1e-6):trial[ix//5]=ix%5
        cost=f.score(trial,u,w,L,beta,constant)
        accept=minimum<np.float32(-1e-6) and cost<np.float32(current-np.float32(1e-6))
        repairs.append(dict(before=a.tolist(),trial=trial.tolist(),predicted_delta=float(minimum),
            current_cost=float(current),trial_cost=float(cost),full_delta=float(np.float32(cost-current)),accepted=bool(accept)))
        if accept:a=trial;current=cost
    return dict(candidates=candidates,repairs=repairs,action=[*a.tolist(),*moves],cost=float(current))

g.r.configure();f.numba.set_num_threads(1)
failure=g.read(OUT/'FAILED.json');assert not failure['timing_started']
state=np.load(OUT/'observations.npz')['observations'][187]
model=g.build('shared_congestion_cost',99023)
session=g.c.session(OUT/'exports/shared_congestion_cost_99023.onnx')
native_tables=session.run(None,{'states':state[None]})
with torch.inference_mode():
    eager_tables=model.tables(torch.from_numpy(state)[None])
    a,q,details=model.greedy(torch.from_numpy(state)[None])
    at,qt,_=old.decode_cost_tables(*(torch.from_numpy(x) for x in native_tables))
eager_arrays=[x.numpy() for x in eager_tables]
an,qn=f.values((session,np.zeros((2,5,2),np.float32)),'shared_congestion_cost',state)
ane,qne=f.shared_decode(*(x[0] for x in eager_arrays),True,False)
legacy=g.c.session(runner.ORIGINAL/'exports/shared_congestion_cost_99023.onnx')
al,ql=g.compiled_values(legacy,state)
nt=trace_native(native_tables);et=trace_native(eager_arrays)
assert np.array_equal(nt['action'],an[0]) and np.array_equal(et['action'],ane)
def exact_table_cost(actions):
    u,w,L,beta,motion,base=[x[0].astype(np.float64) if x.ndim>1 else np.float64(x[0]) for x in eager_arrays]
    A=np.zeros(5,np.float64)
    for i,d in enumerate(actions[:30]):A[d]+=w[i,d]
    return float(base+sum(u[i,d] for i,d in enumerate(actions[:30]))+motion[0,actions[30]]+motion[1,actions[31]]+
        np.sum(beta*(2*L*A+A*A))/120)
result=dict(kind='read_only_diagnosis_not_fix_or_new_timing',arm='shared_congestion_cost',seed=99023,
    observation_index=187,observation_kind='previously_saved_synthetic_numerical_case',
    mismatched_vehicle_indices=np.flatnonzero(a[0].numpy()!=an[0]).tolist(),
    eager_action=a[0].tolist(),legacy_fullORT_action=al[0].tolist(),native_action=an[0].tolist(),
    torch_decoder_on_native_core_action=at[0].tolist(),native_decoder_on_eager_tables_action=ane.tolist(),
    eager_Q=float(q[0]),legacy_Q=float(ql[0]),native_Q=float(qn[0]),
    native_Q_on_eager_tables=float(qne),torch_Q_on_native_tables=-float(qt[0]),
    core_table_max_errors={k:float(np.max(abs(x-y))) for k,x,y in zip(['u','w','L','beta','motion','base'],eager_arrays,native_tables)},
    eager_decoder_details={k:v.tolist() for k,v in details.items()},native_trace=nt,native_trace_on_eager_tables=et,
    higher_precision_same_eager_table_costs=dict(eager_assignment=exact_table_cost(a[0].numpy()),native_assignment=exact_table_cost(an[0])),
    unchanged_source_hashes={n:g.r.old.sha(ROOT/n) for n in runner.SOURCES},
    optimizer_updates=0,new_simulator_steps=0,timing_started=False,no_retry_or_source_fix=True)
g.r.write(OUT/'discrepancy_diagnosis.json',result)
np.savez_compressed(OUT/'discrepancy_fixture.npz',state=state,**{f'eager_{k}':v for k,v in zip(['u','w','L','beta','motion','base'],eager_arrays)},
    **{f'core_{k}':v for k,v in zip(['u','w','L','beta','motion','base'],native_tables)})
print(json.dumps(result,indent=2))
