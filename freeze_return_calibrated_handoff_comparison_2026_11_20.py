"""Freeze prospective comparison only: hashes, seed reservations and arithmetic."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

from freeze_heuristic_initialization_pilot_plan_2026_11_19 import ranges

ROOT=Path(__file__).resolve().parent
PLAN=ROOT/'RETURN_CALIBRATED_HANDOFF_COMPARISON_PLAN_2026-11-20.txt'
SPEC=ROOT/'RETURN_CALIBRATED_HANDOFF_COMPARISON_PROTOCOL_2026-11-20.json'
OUT=ROOT/'validation/return_calibrated_handoff_comparison_freeze_2026-11-20_01'
DESIGN=ROOT/'RETURN_CALIBRATED_DDQN_HANDOFF_DESIGN_STATUS_2026-11-20.json'
MECHANICS=ROOT/'validation/return_calibrated_handoff_mechanics_2026-11-20_02'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def main():
    if SPEC.exists() or OUT.exists():raise RuntimeError('Never overwrite a frozen comparison')
    m=read(MECHANICS/'verification.json');tests=read(MECHANICS/'tests.json')
    assert m['passed'] and tests['passed'] and tests['tests']==15
    assert sha(MECHANICS/'manifest.json')==m['manifest_sha256']
    manifest=read(MECHANICS/'manifest.json')
    actual={p.relative_to(MECHANICS).as_posix() for p in MECHANICS.rglob('*') if p.is_file() and p.name not in ['manifest.json','verification.json']}
    assert actual==set(manifest)
    for name,h in manifest.items():assert sha(MECHANICS/name)==h,name
    for name,h in m['source_sha256'].items():assert sha(ROOT/name)==h,name
    design=read(DESIGN);previous=read(ROOT/'CENTERED_LINEAR_CORRECTION_COMPARISON_PROTOCOL_2026-11-20.json')
    assert sha(ROOT/design['brief'])==design['brief_sha256']
    seeds=copy.deepcopy(design['seeds']);proposed=ranges(seeds)
    documents=[];collisions=[];aliases=[];existing=[]
    paths=sorted(ROOT.glob('*.json'))+sorted((ROOT/'validation').glob('*/run/runtime_plan.json'))+sorted((ROOT/'development_runs').glob('*/run/runtime_plan.json'))
    for p in paths:
        value=read(p);entry=dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p))
        if p==DESIGN:
            assert value['seeds']==seeds
            aliases.append(dict(**entry,reason='Same study design reservation copied exactly, not an independent use'))
            continue
        documents.append(entry)
        existing.extend((a,b,str(p.relative_to(ROOT))+label) for a,b,label in ranges(value))
    for i,(a,b,label) in enumerate(proposed):
        for x,y,old in existing+proposed[:i]:
            if max(a,x)<=min(b,y):collisions.append(dict(proposed=label,existing=old))
    assert not collisions,collisions
    arms=copy.deepcopy(design['arms'])
    for arm in arms:
        arm['value_formula']='-softplus(b+sum(z_selected))' if arm['id'] in ['old_softplus_bc_ddqn','plain_small_softplus_ddqn'] else 'V-sum(z_selected-mean_all5(z))'
        arm['demo_replay']=arm['bc'] and arm['ddqn']
        arm['retention_temperature']='fixed handoff alpha' if arm['calibration'] and arm['retention'] else (1. if arm['retention'] else None)
    online=copy.deepcopy(previous['online'])
    online['loss']='scalar smooth_l1 beta1 TD; plus fixed weight1 auxiliary CE only for A/C'
    online['append_before_update']=True
    online['epsilon_draws']='one scalar and32branch uniforms consumed each slot,including greedy slots'
    names=set(m['source_sha256'])|{PLAN.name,Path(__file__).name,
        'validation/return_calibrated_handoff_mechanics_2026-11-20_02/verification.json',
        'validation/return_calibrated_handoff_mechanics_2026-11-20_02/manifest.json'}
    spec=dict(protocol_id='return-calibrated-handoff-development-v1',date='2026-11-20',
        status='PLAN_FREEZE_NOT_EXECUTABLE_RELEASE',plan=PLAN.name,
        scope='Prospective handoff correction study; not full contribution/novelty/final validation',
        primary=arms[0]['id'],arms=arms,reference='observed_load_radio',fits=3,seeds=seeds,
        reservation_source=DESIGN.name,
        model=dict(architecture=[533,24,161],parameters=16841,parameter_bytes=67364,buffer_bytes=20,
            paired_raw_initial_tensors=True,centering='all five actions independent of mask',
            gamma=1.,horizon_scaling=False,terminal_anchor=False,new_mask=False,teacher_in_deployment=False),
        demonstrations=copy.deepcopy(previous['demonstrations']),bc=copy.deepcopy(previous['bc']),
        calibration=dict(fit_episode_indices=list(range(8)),withheld_return_episode_indices=[8,9],
            fit_states=320,withheld_return_states=80,all_labels_used_by_BC_and_retention=True,
            target='episode-local teacher realized G + scaled C(s,a_teacher)',
            reward_scale='max(1e-6,median(abs(r_fit)))',
            range='max over fit states of sum(all5 branch max-min)',
            alpha='S_r/(S_r+R_max);1 if R_max=0',branch_rows_scaled=list(range(160)),
            ridge_rho=.001,feature_floor=1e-6,return_rms_floor=1e-6,
            feature_standardization='fit-only means and RMS centered deviations',
            ridge='FP64 solve(H.T H/N+rho I,H.T(u-mean(u))/N);unpenalized intercept',
            modified_value_row=160,export='fold into existing FP32 weights, no new buffer',
            numerical_checks=dict(exact_action_preservation_states=400,
                coefficient_underflow_rejected=True,export_rtol=1e-5,export_atol=1e-4,
                range_tolerance='S_r+1e-5+S_r*1e-5'),
            withheld_fit_warning='after_value_fit RMSE > after_bc RMSE on80withheld return targets',
            withheld_fit_warning_is_acceptance_criterion=False,
            finite_poor_fit_handling='retain warning and complete declared schedule; no refitting/tuning',
            numerical_or_action_failure='STOP, preserve partial evidence, no replacement fit'),
        online=online,optimizer_common=copy.deepcopy(previous['optimizer_common']),
        retention_loss=dict(arms=['calibrated_retained_ddqn','uncalibrated_retained_ddqn'],
            weight=1.,batch_size=64,sampling='uniform with replacement from all400demo transitions',
            objective='masked mean vehicle CE plus mean movement CE',
            temperature='alpha for A;1 for C',independent_sampler=True,
            additional_optimizer_steps=0,one_joint_backward=True,no_teacher_inference=True),
        pairing=dict(all_six_BC_end_hashes_equal=True,A_B_E_handoff_hashes_and_alpha_equal=True,
            job_order='seeded permutation of all21arm/fit jobs before training',
            independent_optimizers=True,paired_action_replay_streams_per_fit=True),
        evaluation=copy.deepcopy(previous['evaluation']),
        diagnostics=dict(probe_stages=['BEFORE_BC','AFTER_BC','AFTER_SCALE','AFTER_VALUE_FIT','FINAL'],
            mark_inapplicable_stages_skipped=True,plain_G_probes='saved own online states; no demonstration inputs',
            signed_Q_report=True,nearzero_absQ_threshold=1e-6,
            nrmse='sqrt(mean((Q-G_own)^2))/max(sqrt(mean(G_own^2)),1e-12)',
            terminal_error='Q(last physical state,action)-last reward',
            distributions='terminal/nonterminal,slot groups,positive/nearzero Q,target/gradient tails',
            first_update_decomposition=True,per_destination_task_and_miss_counts=True),
        decision=dict(policy='A-E and A-F:both outcome differences>=0 every fit,equal-fit mean>0 both outcomes for each contrast',
            value='A own-policy NRMSE AND absolute mean Q-G bias strictly below D every fit',
            promising='policy AND value AND finite complete results AND full artifact audit',
            primary_comparisons=['A-E','A-F'],value_comparator='D',
            secondary_contrasts=['A-B','A-C','A-D','A-G','each arm minus teacher'],
            factorial_interaction='A-B-C+D for return,deadline,NRMSE and abs bias,per fit',
            component_claim='needs relevant ablation benefit; no benefit attributed if it merely matches',
            equality_tolerance='none for scientific screens; engineering float checks are separate',
            fit_selection=False,withheld_calibration_diagnostic_only=True,
            on_failure='HOLD expansion; preserve all fits; no retuning,retries,best-arm promotion or automatic final phase',
            statistical_confirmatory_claim=False,causal_calibration_claim=False),
        logging=dict(episode_arrays=['states','actions','rewards','next_states','dones','evaluation chosen Q'],
            episode_records=['seed','task records','reward components','unfinished accounting','destination/queue/misses'],
            immutable_demo_freeze_before_fitting=True,
            calibration=['input/split hashes','alpha','ridge fit scales/coefficients/rank/condition','FP32 export discrepancy',
                'stage probes/hashes','fit and withheld residuals','warning'],
            BC=['draw indices','loss','preclip gradient norm','BC-end hash'],
            TD=['replay IDs and indices','next actions','prediction','target_q','rewards','targets','dones',
                'TD/retention/total losses','temperature','total/shared/branch/V preclip norms','clip flag','target-copy flag'],
            final=['source snapshots','environment','job order','stage events','counts','final replay/RNG','checkpoints/global freeze']),
        auditor=dict(exact_inventory_and_hashes=True,independent_screen_and_comparison_recomputation=True,
            episode_and_training_data_permissions=True,paired_workloads=True,reconstruct_sampler_draws=True,
            independent_augmented_ridge_solve=True,calibration_action_and_hash_parity=True,
            recorded_gradient_target_arithmetic=True,final_checkpoint_actions_and_Q=True,
            global_checkpoint_freeze_before_evaluation=True,timing_inventory_and_actions=True,
            full_historical_gradient_replay=False),
        cost=copy.deepcopy(previous['cost']),
        totals=dict(student_online_slots=36000,teacher_demo_slots=1200,evaluation_slots=14080,
            all_simulated_slots=51280,BC_updates=3600,DDQN_updates=33696,all_gradient_updates=37296,
            retention_batches=11232,retention_examples=718848,ridge_solves=9,final_checkpoints=21,
            initial_target_copies=18,scheduled_target_copies=162),
        checkpoint_retention=dict(final_only=True,keep_all_fits=True,fit_retry=False,best_checkpoint_selection=False,
            intermediate_weight_checkpoints=False,stage_hashes_and_probes=True),
        artifact_retention=dict(primary_drive='E:',independent_archive_drive='D:',
            archive_after_full_audit=True,archive_inventory_and_sha256_required=True,
            partial_failures_preserved=True,automatic_overwrite=False),
        numerical_environment=copy.deepcopy(m['environment']),
        python_executable=m['python_executable'],
        simulator_defaults=copy.deepcopy(previous['simulator_defaults']),
        scope_exclusions=['security/trust','cloud','charging'],
        deferred_controls=['compact64','wide128','adapted_BDQ','full_DQfD'],
        deferred_claims=['full low-complexity contribution','algorithmic novelty','final generalization','real UAV energy or latency'],
        definition_sha256={name:sha(ROOT/name) for name in sorted(names)},
        research_launch_authorized=False,launched=False,runner_implemented=False,auditor_implemented=False,
        final_seeds_consumed=False,word_thesis_changed=False,
        next_gate='Implement paired runner and independent auditor; reduced isolated full-workflow fixture; research CLI disabled')
    spec['evaluation']['BC_only']='independently calibrated same-initial-policy arm E, no TD/online exposure'
    spec['cost'].update(warmup_calls=2112,all_calls_including_warmup=10560,
        states_selection='permutation of sorted saved evaluation episode states, first128, no replacement',
        order='then shuffle all66policy/round blocks with same timing RNG',
        storage_scope='parameters,buffers,deployment file separate from optimizer/replay/process memory',
        training_costs='teacher generation,BC,ridge,TD+retention,evaluation,audit separately')
    # Independent budget and control checks before creating any frozen output.
    assert len(arms)==7 and sum(a['bc'] for a in arms)==6 and sum(a['ddqn'] for a in arms)==6
    assert sum(a['retention'] for a in arms)==2 and sum(a['calibration'] for a in arms)==3
    assert [a['id'] for a in arms if a['demo_replay']]==[arms[i]['id'] for i in [0,1,2,3,5]]
    assert 3*10*40+6*3*50*40+(7*3+1)*16*40==51280
    assert 6*3*200==3600 and 6*3*(2000-128)==33696 and 3600+33696==37296
    assert 2*3*1872==11232 and 11232*64==718848 and 3*3==9
    assert 6*3==18 and (1872//200)*18==162
    assert 22*3*128==8448 and 22*3*32==2112
    for field in ['seeds','simulator_defaults']:
        assert spec[field]==(design['seeds'] if field=='seeds' else previous[field])
    assert spec['totals']['all_simulated_slots']==design['proposed_totals']['simulated_slots']
    assert not any(name.startswith(('torch','numpy','chapter3_iov_env','return_calibrated_handoff','heuristic_pilot_teacher_adapter')) for name in __import__('sys').modules)
    write(SPEC,spec)
    OUT.mkdir(parents=True)
    snapshot=OUT/'source_snapshot';snapshot.mkdir()
    for name in sorted(names|{SPEC.name}):
        dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    record=dict(status='COMPARISON_PROTOCOL_FROZEN_RESEARCH_DISABLED',
        protocol_sha256=sha(SPEC),plan_sha256=sha(PLAN),source_sha256=spec['definition_sha256'],
        mechanics_tests_verified=15,mechanics_manifest_files_verified=len(manifest),
        seed_audit=dict(documents=documents,same_study_aliases=aliases,collisions=collisions,
            internal_ranges_checked=True,coverage='Root JSON and validation/development */run/runtime_plan.json; not exhaustive historical logs'),
        arithmetic_verified=True,proposed_totals=spec['totals'],
        new_models_created=0,new_simulator_steps=0,new_optimizer_updates=0,new_teacher_calls=0,
        research_launch_authorized=False,launched=False,word_thesis_changed=False)
    write(OUT/'verification.json',record)
    inventory={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    write(OUT/'manifest.json',inventory)
    print(json.dumps(dict(status=record['status'],protocol_sha256=record['protocol_sha256'],
        seed_documents_checked=len(documents),same_study_reservations=len(aliases),seed_collisions=len(collisions),
        manifest_files=len(inventory),proposed_totals=spec['totals'],launched=False),indent=2))


if __name__=='__main__':main()
