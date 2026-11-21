"""Independent checks of exported diagnostics; no models, autograd or simulation."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'validation/return_calibrated_saved_trace_diagnosis_2026-11-20_01'
RUN = ROOT/'development_runs/return_calibrated_handoff_comparison_2026-11-20_01/run'
A='calibrated_retained_ddqn'; E='calibrated_bc_only'; F='old_softplus_bc_ddqn'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    dest=OUT/'independent_verification.json'
    if dest.exists(): raise RuntimeError('Preserve previous verification')
    main_receipt=read(OUT/'verification.json'); gradient_receipt=read(OUT/'first_update_verification.json')
    v=read(OUT/'diagnosis.json');g=read(OUT/'first_update_gradient_decomposition.json')
    for receipt in [main_receipt,gradient_receipt]:
        for name,digest in receipt['artifacts_sha256'].items():assert sha(OUT/name)==digest
    assert sha(ROOT/'diagnose_return_calibrated_saved_traces_2026_11_20.py')==main_receipt['script_sha256']
    assert sha(ROOT/'decompose_return_calibrated_first_update_2026_11_20.py')==gradient_receipt['script_sha256']
    for data in [v,g]:
        for name,digest in data['consumed_sha256'].items():assert sha(ROOT/name)==digest,name
    cross={}
    manifest=read(RUN/'manifest.json')
    with np.load(OUT/'fixed_state_predictions.npz',allow_pickle=False) as raw:
        for fit in range(3):
            # Check every exported same-state summary, independently of inference.
            for key,entries in v['fixed_state'].items():
                population,index=key.rsplit('/',1)
                if int(index)!=fit:continue
                e=raw[f'fit{fit}_{population}_{E}_actions']
                for arm,entry in entries.items():
                    prefix=f'fit{fit}_{population}_{arm}'
                    a=raw[prefix+'_actions'];change=a!=e
                    assert float(change[:,:30].mean())==entry['placement_change_vs_E']
                    assert float(change[:,30:].mean())==entry['motion_change_vs_E']
                    matrix=np.zeros((5,5),dtype=int)
                    for b,c in zip(e[:,:30].ravel(),a[:,:30].ravel()):matrix[b,c]+=1
                    assert matrix.tolist()==entry['E_to_model_destinations']
                    assert np.isclose(float(raw[prefix+'_q'].mean()),entry['q']['mean'],rtol=1e-6,atol=1e-6)
            states={}
            for population in [A,E]:
                es=[]
                for ep in range(16):
                    path=RUN/'evaluation'/population/str(fit)/f'episode_{ep:03d}.npz'
                    assert sha(path)==manifest[path.relative_to(RUN).as_posix()]
                    with np.load(path) as data:es.append(data['states'])
                states[population]=np.concatenate(es)
            assert np.array_equal(states[A][:,:240],states[E][:,:240])
            assert np.array_equal(states[A][:,-1],states[E][:,-1])
            diff=states[A]-states[E]
            blocks=dict(tasks=(0,240),local_work=(240,270),remote_work=(270,274),
                radio=(274,514),remote_summary=(514,526),uavs=(526,532),time=(532,533))
            cross[str(fit)]=dict(saved_A_E_states_different_fraction=float(np.any(diff!=0,axis=1).mean()),
                block_max_abs_difference={k:float(np.abs(diff[:,lo:hi]).max()) for k,(lo,hi) in blocks.items()},
                policy_changes_across_observed_A_E_states={})
            for arm in v['fixed_state'][f'{A}/{fit}']:
                aa=raw[f'fit{fit}_{A}_{arm}_actions'];ae=raw[f'fit{fit}_{E}_{arm}_actions']
                cross[str(fit)]['policy_changes_across_observed_A_E_states'][arm]=dict(
                    placement=float((aa[:,:30]!=ae[:,:30]).mean()),movement=float((aa[:,30:]!=ae[:,30:]).mean()))
            # Independent reward and workload/task-miss arithmetic for the core controls.
            for control in [E,F]:
                counts=np.zeros(4,dtype=int); parts=np.zeros(3)
                for ep in range(16):
                    records=[]
                    for arm in [A,control]:
                        path=RUN/'evaluation'/arm/str(fit)/f'episode_{ep:03d}.json'
                        assert sha(path)==manifest[path.relative_to(RUN).as_posix()]
                        records.append(read(path))
                    p,c=records
                    for pt,ct in zip(p['records'],c['records']):
                        assert (pt['task_id'],pt['required_cycles'],pt['input_bits'],pt['deadline_s'])==(ct['task_id'],ct['required_cycles'],ct['input_bits'],ct['deadline_s'])
                        counts[0]+=not pt['on_time'] and ct['on_time']
                        counts[1]+=pt['on_time'] and not ct['on_time']
                        counts[2]+=not pt['on_time'] and not ct['on_time']
                        counts[3]+=not pt['on_time'] and ct['on_time'] and pt['destination']==ct['destination']
                    keys=['normalized_delay_accrual','normalized_uav_energy','new_deadline_misses_per_vehicle']
                    for i,(k,w) in enumerate(zip(keys,[-.5,-.5,-1.])):
                        parts[i]+=w*(sum(x['reward_components'][k] for x in p['step_info'])-sum(x['reward_components'][k] for x in c['step_info']))/16
                saved=v['paired_tasks'][str(fit)][control]
                assert counts.tolist()==[saved[k] for k in ['primary_only_misses','control_only_misses','both_miss','primary_only_misses_same_destination']]
                assert np.allclose(parts,list(v['episode_reward_components'][f'contrasts/{fit}'][control].values()),rtol=1e-9,atol=1e-9)
    for fit,row in g['fits'].items():
        norms=row['norm_by_loss_and_group']
        for kind,group in norms.items():
            assert np.isclose(group['total']**2,sum(group[k]**2 for k in group if k!='total'),rtol=1e-8,atol=1e-8)
        assert norms['retention']['value_output']==0
        assert row['parameters_unchanged_after_analysis'] and row['logged_predictions_targets_losses_and_group_norms_reproduced']
    result=dict(status='DIAGNOSIS_EXPORT_HASH_ARITHMETIC_AND_WORKLOAD_CHECKS_PASSED',
        cross_population=cross, primary_evidence_files_rehashed=len(v['consumed_sha256']),
        first_update_evidence_files_rehashed=len(g['consumed_sha256']),
        input_receipts_sha256={p.name:sha(p) for p in [OUT/'verification.json',OUT/'first_update_verification.json']},
        script_sha256=sha(Path(__file__)),new_optimizer_steps=0,new_simulator_steps=0,new_teacher_calls=0,
        new_autograd_evaluations=0,policy_screen=False,value_screen=True,promising=False)
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
