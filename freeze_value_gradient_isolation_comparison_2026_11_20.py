"""Freeze source/seed/decision contracts only; never import models or simulate."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

from freeze_heuristic_initialization_pilot_plan_2026_11_19 import ranges

ROOT=Path(__file__).resolve().parent
PLAN=ROOT/'VALUE_GRADIENT_ISOLATION_COMPARISON_PLAN_2026-11-20.txt'
SPEC=ROOT/'VALUE_GRADIENT_ISOLATION_COMPARISON_PROTOCOL_2026-11-20.json'
OUT=ROOT/'validation/value_gradient_isolation_comparison_freeze_2026-11-20_01'
DESIGN=ROOT/'VALUE_GRADIENT_ISOLATION_DESIGN_STATUS_2026-11-20.json'
MECHANICS=ROOT/'validation/value_gradient_isolation_mechanics_2026-11-20_01'
P='value_gradient_isolated_ddqn';R='dueling_shared_td_rescale_ddqn'
A='calibrated_retained_ddqn';B='calibrated_no_retention_ddqn';C='uncalibrated_retained_ddqn'
D='uncalibrated_no_retention_ddqn';E='calibrated_bc_only';F='old_softplus_bc_ddqn';G='plain_small_softplus_ddqn'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def write(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def arithmetic(arms,fits,demo_episodes,online_episodes,eval_episodes,bc_steps,warmup,batch,target_interval,timing_states,timing_rounds,timing_warmups):
    bc=sum(a['bc'] for a in arms);td=sum(a['ddqn'] for a in arms)
    keep=sum(a['retention'] for a in arms);cal=sum(a['calibration'] for a in arms)
    updates=online_episodes*40-warmup;policies=len(arms)*fits+1
    result=dict(student_online_slots=td*fits*online_episodes*40,teacher_demo_slots=fits*demo_episodes*40,
        evaluation_slots=policies*eval_episodes*40,BC_updates=bc*fits*bc_steps,DDQN_updates=td*fits*updates,
        retention_batches=keep*fits*updates,retention_examples=keep*fits*updates*batch,
        ridge_solves=cal*fits,final_checkpoints=len(arms)*fits,deployment_weight_files=len(arms)*fits,
        initial_target_copies=td*fits,scheduled_target_copies=td*fits*(updates//target_interval),
        measured_action_calls=policies*timing_rounds*timing_states,warmup_action_calls=policies*timing_rounds*timing_warmups)
    result['all_simulated_slots']=sum(result[k] for k in ['student_online_slots','teacher_demo_slots','evaluation_slots'])
    result['all_gradient_updates']=result['BC_updates']+result['DDQN_updates']
    return result


def main():
    if SPEC.exists() or OUT.exists():raise RuntimeError('Preserve previous frozen study; refuse overwrite')
    mechanics=read(MECHANICS/'verification.json');tests=read(MECHANICS/'test_results.json')
    assert mechanics['passed'] and mechanics['tests_run']==tests['tests_run']==18
    assert tests['passed'] and tests['failures']==tests['errors']==tests['skipped']==0
    manifest=read(MECHANICS/'manifest.json')
    actual={p.relative_to(MECHANICS).as_posix() for p in MECHANICS.rglob('*') if p.is_file() and p!=MECHANICS/'manifest.json'}
    assert actual==set(manifest) and len(manifest)==17
    for name,digest in manifest.items():assert sha(MECHANICS/name)==digest,name
    for name,digest in mechanics['source_sha256'].items():assert sha(ROOT/name)==digest,name
    binding=read(ROOT/'RETURN_CALIBRATED_COMPARISON_LAUNCH_BINDING_2026-11-20.json')
    for name,digest in binding['sha256'].items():assert sha(ROOT/name)==digest,name
    design=read(DESIGN)
    assert sha(ROOT/design['brief'])==design['brief_sha256']
    previous_path=ROOT/'RETURN_CALIBRATED_HANDOFF_COMPARISON_PROTOCOL_2026-11-20.json'
    previous=read(previous_path)
    assert sha(previous_path)==design['inherited_protocol_sha256']
    # Read literals without importing the module or creating a model.
    literals={node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse((ROOT/'value_gradient_isolation.py').read_text()).body
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in
        {'PRIMARY','RESCALE','BASE','DEVELOPMENT_RELEASED','FINAL_RELEASED'}}
    assert literals==dict(PRIMARY=P,RESCALE=R,BASE=A,DEVELOPMENT_RELEASED=False,FINAL_RELEASED=False)
    seeds=copy.deepcopy(design['seeds'])
    fixture_seeds=dict(model_seeds=[988480001,988480011],
        demonstration_seed_blocks=[dict(start=988490000+1000*i,count=10) for i in range(2)],
        online_seed_blocks=[dict(start=988500000+1000*i,count=1) for i in range(2)],
        evaluation_seed_block=dict(start=988510000,count=1),
        bc_sampler_seeds=[988520001,988520011],epsilon_branch_seeds=[988530001,988530011],
        replay_sampler_seeds=[988540001,988540011],retention_sampler_seeds=[988550001,988550011],
        job_order_seed=988560001,timing_order_seed=988560011)
    proposed=ranges(dict(development_seeds=seeds,fixture_seeds=fixture_seeds))
    documents=[];aliases=[];history=[]
    paths=sorted(ROOT.glob('*.json'))+sorted((ROOT/'validation').glob('*/run/runtime_plan.json'))+sorted((ROOT/'development_runs').glob('*/run/runtime_plan.json'))
    for path in paths:
        value=read(path);entry=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path))
        if path==DESIGN:
            assert value['seeds']==seeds
            aliases.append(dict(**entry,reason='Exact same-study development reservation; no independent consumption'))
        else:
            documents.append(entry)
            history.extend((lo,hi,path.relative_to(ROOT).as_posix()+label) for lo,hi,label in ranges(value))
    collisions=[]
    for i,(lo,hi,label) in enumerate(proposed):
        for a,b,origin in history+proposed[:i]:
            if max(lo,a)<=min(hi,b):collisions.append(dict(proposed=label,existing=origin,intersection=[max(lo,a),min(hi,b)]))
    assert not collisions,collisions
    arms=copy.deepcopy(design['arms'])
    assert [a['id'] for a in arms]==[P,R,A,B,C,D,E,F,G]
    roles={P:'sole_primary',R:'adapted_known_gradient_rule_control',A:'unchanged_isolation_ablation',
        B:'no_retention_control',C:'no_calibration_control',D:'relative_value_control',E:'no_TD_imitation_control',
        F:'old_same_size_baseline',G:'ordinary_no_demonstration_DDQN'}
    for arm in arms:
        name=arm['id'];arm.update(role=roles[name],label={P:'P',R:'R',A:'A',B:'B',C:'C',D:'D',E:'E',F:'F',G:'G'}[name],
            gradient_rule='isolate_value' if name==P else 'rescale_shared_td' if name==R else 'ordinary',
            value_formula='-softplus(b+sum(z_selected))' if name in [F,G] else 'V-sum(z_selected-mean_all5(z))',
            demo_replay=arm['bc'] and arm['ddqn'],
            retention_temperature='fixed handoff alpha' if arm['retention'] and arm['calibration'] else 1. if arm['retention'] else None)
    assert [sum(a[k] for a in arms) for k in ['bc','ddqn','retention','calibration']]==[8,8,4,5]
    totals=arithmetic(arms,3,10,50,16,200,128,64,200,128,3,32)
    fixture_totals=arithmetic(arms,2,10,1,1,2,8,4,8,4,1,1)
    assert (totals['all_simulated_slots'],totals['BC_updates'],totals['DDQN_updates'],totals['retention_batches'],totals['ridge_solves'],totals['final_checkpoints'])==(67120,4800,44928,22464,15,27)
    assert (totals['initial_target_copies'],totals['scheduled_target_copies'],totals['measured_action_calls'],totals['warmup_action_calls'])==(24,216,10752,2688)
    assert (fixture_totals['all_simulated_slots'],fixture_totals['BC_updates'],fixture_totals['DDQN_updates'],fixture_totals['retention_batches'],fixture_totals['ridge_solves'],fixture_totals['final_checkpoints'])==(2200,32,512,256,10,18)
    assert (fixture_totals['measured_action_calls'],fixture_totals['warmup_action_calls'])==(76,19)
    assert totals['all_simulated_slots']==design['proposed_totals']['simulated_slots']
    names=set(previous['definition_sha256'])|set(mechanics['source_sha256'])|{PLAN.name,Path(__file__).name,
        previous_path.name,DESIGN.name,design['brief'],'VALUE_GRADIENT_ISOLATION_MECHANICS_REVIEW_2026-11-20.txt',
        'prepare_value_gradient_isolation_brief_2026_11_20.py','collect_value_gradient_isolation_literature_2026_11_20.py',
        'freeze_heuristic_initialization_pilot_plan_2026_11_19.py',
        'run_return_calibrated_comparison.py','audit_return_calibrated_comparison.py',
        'validation/value_gradient_isolation_mechanics_2026-11-20_01/verification.json',
        'validation/value_gradient_isolation_mechanics_2026-11-20_01/manifest.json',
        'validation/value_gradient_isolation_brief_2026-11-20_01/verification.json',
        'validation/value_gradient_isolation_brief_2026-11-20_01/manifest.json'}
    spec=copy.deepcopy(previous)
    spec.update(protocol_id='value-gradient-isolation-nine-arm-development-v1',date='2026-11-20',
        status='PLAN_FREEZE_NOT_EXECUTABLE_RELEASE',plan=PLAN.name,
        scope='Narrow prospective gradient-mechanism comparison; not full efficiency/novelty/final validation',
        primary=P,arms=arms,seeds=seeds,reservation_source=DESIGN.name,
        inherited_protocol=previous_path.name,inherited_protocol_sha256=sha(previous_path))
    spec['model'].update(gradient_routing_training_only=True,inference_class='return_calibrated_handoff.Critic',
        training_class='value_gradient_isolation.Critic',extra_deployment_parameters=0)
    spec['online'].update(loss='scalar smooth_l1 beta1 TD plus fixed weight1 auxiliary CE only for P/R/A/C',
        update_entrypoint='value_gradient_isolation.ddqn_retention_step',current_TD_entrypoint='Critic.td_prediction',
        next_selection_and_target_forward='ordinary inference under no_grad',
        readout_derivatives='ordinary for every arm',
        shared_TD_derivative={P:'branch only; omit V path',R:'(V+branch)/sqrt(2)',
            'others':'ordinary'},shared_imitation_derivative='ordinary,unscaled for all retention arms',
        higher_order_derivatives_supported=False)
    spec['retention_loss'].update(arms=[P,R,A,C],temperature='fixed handoff alpha for P/R/A;1 for C')
    spec['pairing']=dict(all_nine_raw_initial_tensors_equal=True,all_eight_BC_end_hashes_equal=True,
        calibrated_arms=[P,R,A,B,E],calibrated_handoff_hashes_and_alpha_equal=True,
        job_order='seeded permutation of all27arm/fit jobs before training',independent_optimizers=True,
        paired_action_replay_streams_per_fit=True,first_P_R_A_TD_batches_and_initial_targets_equal=True)
    spec['evaluation'].update(student_policies=27,teacher_policies=1,
        checkpoint_freeze='all27final training and27deployment weight hashes before any development evaluation',
        inference='original-class deployed models,epsilon0,no fitting',
        inference_batch_boundary='one state per greedy action/chosen-Q record')
    spec['deployment_export']=dict(entrypoint='value_gradient_isolation.deployment_model',
        class_name='return_calibrated_handoff.Critic',file='deployment_weights.pt',
        final_training_file='final.pt',one_training_and_one_deployment_file_per_endpoint=True,
        deployment_contains_optimizer_or_target=False,tensor_hash_matches_final_online=True,
        independent_storage=True,training_gradient_method_absent=True,
        parity_before_global_freeze=True,parity_states=128,
        parity_corpus='FP32((arange(128*533).reshape(128,533)%37)/80);cycle0.9/0.6 alternating;deadline0.9;time(row%40)/40',
        parity_checks='each of128single states and one128state batch,training versus original-class export',
        actions_exact=True,Q_exact_at_matched_batch_boundary=True,
        G_allowed_synthetic_parity_inputs=True,cross_batch_Q_exactness_claim=False)
    spec['decision']=dict(policy_controls=[E,F,A,R],
        policy='P-control return and deadline differences>=0 every fit;equal-fit means>0 in both outcomes for EACH required control',
        value_control=D,value='P own-policy NRMSE AND absolute mean Q-G bias strictly below D every fit',
        promising='policy AND value AND finite complete results AND all fits AND full artifact audit',
        primary_comparisons=['P-E','P-F','P-A','P-R'],
        secondary_contrasts=['P-minus-every-arm','R-A','A-B-C+D interaction','each student minus teacher'],
        metric_definitions=copy.deepcopy(previous['diagnostics']),
        primary_selection_after_results=False,equality_tolerance='none for scientific screens',
        poor_handoff_warning_diagnostic_only=True,
        on_failure='HOLD expansion;preserve all fits;no automatic retry,retuning,best-arm promotion or final phase',
        scientific_failure_does_not_abort_remaining_valid_outcomes=True,
        numerical_or_integrity_failure='STOP and preserve partial evidence,no replacement fit',
        statistical_confirmatory_claim=False,causal_calibration_claim=False)
    spec['diagnostics'].update(all_P_control_contrasts=True,
        paired_task_miss_and_wait_accounting=True,server_assigned_cycles_per_slot=True,
        reward_delay_energy_deadline_decomposition=True,
        G_probe_source='own saved online warmup,never demonstration states',
        original_probe_batch_boundary='400demo states for eligible arms;own128warmup states for G')
    spec['logging']['TD'].extend(['expected gradient_rule on every update','first-update action-gap decomposition'])
    spec['logging']['final'].extend(['deployment-only weights/hash','synthetic deployment parity states/results',
        'global training/deployment freeze','separate export/parity/audit computation counts'])
    spec['auditor'].update(exported_original_class_weights_and_inference=True,
        gradient_rule_tags_for_all_updates=True,expected_update_entrypoint_source_bound=True,
        first_update_reconstruction=dict(arms=[P,R,A],fits=3,contexts=9,
            initialization='retained E final checkpoint equals P/R/A calibrated handoff',
            data='saved replay/retention indices and pre-first-update transitions only',
            reference='independent full-affine plus explicit derivative equations OR independent detached-feature construction',
            production_custom_backward_for_reference_forbidden=True,
            verify=['predictions','targets','losses','parameter gradients','group norms','clipping','P V-feature block','R TD-only rescale','ordinary readout/imitation derivatives'],
            optimizer_steps=0,audit_gradient_evaluations_counted=True),
        full_historical_gradient_replay=False,
        limitation='Source/rule bindings,log arithmetic,first-update reconstruction and final policies;not full historical weight/optimizer/gradient replay')
    spec['engineering_tolerances']=dict(forward_actions_and_Q_same_batch='exact',
        independent_parameter_gradient=dict(rtol=3e-5,atol=2e-6),
        scalar_norm_and_target_arithmetic=dict(rtol=3e-5,atol=1e-5),
        ridge_export=dict(rtol=1e-5,atol=1e-4),
        scientific_comparison_tolerance=None,parity_failure='STOP and preserve;no relaxation from scientific results')
    spec['cost'].update(policies=28,timed_calls=10752,warmup_calls=2688,all_calls_including_warmup=13440,
        order='shuffle all84policy/round blocks after selecting states using same timing RNG',
        inference='original-class exported policies,eagerCPU FP32,one torch thread',
        training_costs='demo generation,BC,ridge,TD+retention,evaluation,export/parity,audit,archive separately',
        audit_gradient_work='record actual first-update reference/autograd work separately,never counted as new training')
    spec['totals']=totals
    spec['checkpoint_retention'].update(deployment_duplicates_allowed=True,
        expected_training_checkpoint_files=27,expected_deployment_weight_files=27,
        no_other_model_weight_files=True)
    spec['artifact_retention'].update(reserved_output='development_runs/value_gradient_isolation_comparison_2026-11-20_01',
        reserved_archive='D:\\Fitz Thesis Research Archive\\value_gradient_isolation_comparison_2026-11-20_01',
        directories_created_by_freeze=False,archive_failed_scientific_screens_if_artifact_audit_passed=True,
        readiness_rechecks=['sources','seeds','environment','output absence','E:/D: headroom'],
        separate_audited_complete_and_scientific_promising_status=True)
    spec['fixture']=dict(kind='ENGINEERING_SYNTHETIC_NOT_RESEARCH',fits=2,seeds=fixture_seeds,
        demo_episodes=10,online_episodes=1,eval_episodes=1,bc_steps=2,batch=4,warmup=8,
        replay_capacity=416,target_interval=8,timing_states=4,timing_rounds=1,timing_warmups=1,
        return_fit_split_unchanged=True,all_nine_arms_and_teacher=True,original_40_slot_horizon=True,
        totals=fixture_totals,first_update_reconstruction_contexts=6,
        excluded_from_scientific_quality=True,research_seeds_forbidden=True,
        full_orchestration_and_independent_auditor_required=True,executed_in_freeze_gate=False)
    spec['authorization']=dict(this_gate='protocol and source/seed/arithmetic freeze only',
        next_gate='NEW paired runner and independent auditor;full reduced isolated workflow fixture',
        development_release_enabled=False,final_release_enabled=False,
        development_confirmation='LAUNCH_FROZEN_VALUE_GRADIENT_ISOLATION_COMPARISON',
        confirmation_alone_is_sufficient=False,separate_readiness_and_explicit_launch_required=True,
        final_mode_disabled=True)
    spec.update(numerical_environment=copy.deepcopy(mechanics['environment']),
        controller_mechanics_implemented=True,runner_implemented=False,auditor_implemented=False,
        research_launch_authorized=False,launched=False,final_seeds_consumed=False,word_thesis_changed=False,
        definition_sha256={name:sha(ROOT/name) for name in sorted(names)},
        next_gate='Implement source-bound runner/auditor;full isolated nine-arm fixture;research CLI requires confirmation and remains disabled')
    # Verify unchanged scientific/physical definitions and inherited schedules.
    for key in ['demonstrations','bc','calibration','optimizer_common','simulator_defaults']:
        assert spec[key]==previous[key],key
    for key in previous['online']:
        if key!='loss':assert spec['online'][key]==previous['online'][key],key
    assert spec['seeds']==design['seeds']
    assert [a['id'] for a in arms if a['role']=='sole_primary']==[P]
    assert not any(n.startswith(('torch','numpy','chapter3_iov_env','value_gradient_isolation','heuristic_pilot_teacher_adapter')) for n in sys.modules)
    # Nothing above runs a model or creates a research directory.
    write(SPEC,spec);OUT.mkdir(parents=True)
    snapshot=OUT/'source_snapshot'
    for name in sorted(names|{SPEC.name}):
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    record=dict(status='NINE_ARM_COMPARISON_PROTOCOL_FROZEN_RESEARCH_DISABLED',protocol_sha256=sha(SPEC),
        plan_sha256=sha(PLAN),source_sha256=spec['definition_sha256'],
        mechanics_tests_verified=18,mechanics_manifest_files_rehashed=17,
        seed_audit=dict(documents=documents,same_study_aliases=aliases,collisions=collisions,
            internal_development_fixture_ranges_checked=True,
            coverage='Root JSON plus validation/development_runs */run/runtime_plan.json;not exhaustive historical logs'),
        totals=totals,fixture_totals=fixture_totals,arithmetic_verified=True,
        inherited_physical_reward_calibration_optimizer_settings_unchanged=True,
        frozen_primary=P,policy_controls=[E,F,A,R],value_control=D,
        new_models_created=0,new_simulator_steps=0,new_optimizer_updates=0,new_teacher_calls=0,
        fixture_executed=False,research_launch_authorized=False,launched=False,word_thesis_changed=False)
    write(OUT/'verification.json',record)
    write(OUT/'manifest.json',{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()})
    print(json.dumps(dict(status=record['status'],protocol_sha256=record['protocol_sha256'],
        definitions_bound=len(names),freeze_manifest_files=len(read(OUT/'manifest.json')),
        seed_documents_checked=len(documents),same_study_aliases=len(aliases),seed_collisions=len(collisions),
        totals=totals,fixture_totals=fixture_totals,launched=False),indent=2))


if __name__=='__main__':main()
