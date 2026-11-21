"""Independently reproduce the stopped gate, not an optimization retry."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
import controller_v3_fused_runtime as f
import run_controller_v3_fused_gate as runner
g=runner.g;out=ROOT/'validation/controller_v3_runtime_optimization_2026-11-05_01/gate_01'
g.r.configure();f.numba.set_num_threads(1)
before=g.read(out/'PRE_EXPORT_FREEZE.json');failed=g.read(out/'FAILED.json');diag=g.read(out/'discrepancy_diagnosis.json')
assert not failed['timing_started'] and not before['research_training_authorized']
for name in ['PRE_TIMING_FREEZE.json','raw_timings.json','verification.json']:
    assert not (out/name).exists(),name
for name,h in before['source_sha256'].items():assert g.r.old.sha(ROOT/name)==g.r.old.sha(out/'source_snapshot'/name)==h
for name,h in before['original_hashes'].items():
    if g.r.old.sha(ROOT/name)!=h:
        # The live tracker may advance after the audit; its frozen version is archived.
        assert name=='LOW_COMPLEXITY_PROJECT_STATUS.json'
        assert g.r.old.sha(out.parent/'PROJECT_STATUS_BEFORE_RUNTIME_OPTIMIZATION.json')==h
assert runner.dependency_info()==before['dependencies']
states=np.load(out/'observations.npz')['observations'];np.testing.assert_array_equal(states,np.load(runner.ORIGINAL/'observations.npz')['observations'])
test=g.read(out/'mechanics_tests.json');assert test['passed'] and test['tests']==8
results=[];vectors=[];native=[];eager_q=[];native_q=[]
items=[*failed['partial_equivalence'],dict(arm='shared_congestion_cost',seed=99023)]
assert len(items)==15
for item in items:
    arm=item['arm'];seed=item['seed'];key=f'{arm}_{seed}';model=g.build(arm,seed)
    path=out/'exports'/f'{key}.onnx';extra=np.load(out/'exports'/f'{key}_extra.npy')
    if key in failed['metadata']:
        meta=failed['metadata'][key]
        assert g.r.old.sha(path)==meta['graph_sha256'] and g.r.old.digest(model.state_dict())==meta['parameter_hash']
    expected=model.movement_embeddings.detach().numpy() if arm=='emc_q_nonpositive' else np.zeros((2,5,2),np.float32)
    np.testing.assert_array_equal(extra,expected);session=g.c.session(path);aa=[];qq=[];bb=[];vv=[]
    with torch.inference_mode():
        for state in states:
            a,q,_=g.greedy(model,arm,torch.from_numpy(state)[None]);b,v=f.values((session,extra),arm,state)
            aa.append(a[0].numpy());qq.append(float(q[0]));bb.append(b[0]);vv.append(float(v[0]))
    aa=np.array(aa);bb=np.array(bb);mismatch=np.argwhere(aa!=bb)
    assert np.isfinite(qq).all() and np.isfinite(vv).all()
    np.testing.assert_allclose(qq,vv,rtol=1e-5,atol=1e-6)
    if key=='shared_congestion_cost_99023':np.testing.assert_array_equal(mismatch,[[187,2]])
    else:assert len(mismatch)==0
    results.append(dict(arm=arm,seed=seed,observations=192,action_mismatches=mismatch.tolist(),Q_tolerance_passed=True))
    vectors.append(aa);native.append(bb);eager_q.append(qq);native_q.append(vv)
assert diag['torch_decoder_on_native_core_action']==diag['native_action']
assert diag['native_decoder_on_eager_tables_action']==diag['eager_action']==diag['legacy_fullORT_action']
assert diag['eager_Q']==diag['legacy_Q']==diag['native_Q']
saved_inputs=g.read(ROOT/'analysis/controller_redesign_2026-11-05/saved_trace_diagnosis.json')['input_sha256']
for name,h in saved_inputs.items():
    path=Path(name)
    if not path.is_absolute():path=ROOT/path
    assert g.r.old.sha(path)==h
np.savez_compressed(out/'failure_audit_vectors.npz',eager_actions=np.array(vectors),native_actions=np.array(native),eager_Q=np.array(eager_q),native_Q=np.array(native_q))
result=dict(state='passed_failure_record_audit_not_equivalence_or_cost_pass',research_gate='native_equivalence_failed_before_timing',
    complete_cases_replayed=2880,fully_equivalent_completed_models=14,fully_equivalent_cases=2688,
    failed_model_action_mismatch_indices=[[187,2]],native_core_changes_tie_sensitive_choice=True,
    original_completion_artifacts_verified=len(before['original_hashes']),original_research_inputs_verified=len(saved_inputs),
    original_code_and_results_unchanged=True,dependencies_verified=True,mechanics_tests=8,
    comparisons=results,new_timing_records=0,new_simulator_steps=0,optimizer_updates=0,
    research_training_started=False,word_thesis_changed=False,
    note='Audit verifies faithful failure recording; it does not approve the optimized runtime')
g.r.write(out/'failure_record_audit.json',result)
g.r.write(out/'FAILURE_ARTIFACT_MANIFEST.json',dict(files={str(p.relative_to(out)).replace('\\','/'):g.r.old.sha(p) for p in out.rglob('*') if p.is_file() and p.name!='FAILURE_ARTIFACT_MANIFEST.json'}))
print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2))
