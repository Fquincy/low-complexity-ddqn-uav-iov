"""Freeze a correction-study plan and reserve seeds; no model execution."""
import hashlib
import json
from pathlib import Path
from freeze_heuristic_initialization_pilot_plan_2026_11_19 import ranges

ROOT=Path(__file__).resolve().parent
SPEC=ROOT/'CENTERED_LINEAR_CORRECTION_COMPARISON_PROTOCOL_2026-11-20.json'
OUT=ROOT/'validation/centered_linear_comparison_freeze_2026-11-20_01'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if SPEC.exists() or OUT.exists():raise RuntimeError('Do not overwrite frozen evidence')
    old=json.loads((ROOT/'HEURISTIC_INITIALIZED_DDQN_PAIRED_PILOT_PROTOCOL_2026-11-19.json').read_text())
    arms=[
        dict(id='centered_linear_bc_ddqn',formula='V-sum(z_selected-mean_all5(z))',bc=True,ddqn=True,demo_replay=True,role='primary'),
        dict(id='old_softplus_bc_ddqn',formula='-softplus(b+sum(z_selected))',bc=True,ddqn=True,demo_replay=True,role='old_baseline'),
        dict(id='centered_softplus_bc_ddqn',formula='-softplus(-V+sum(z_selected-mean_all5(z)))',bc=True,ddqn=True,demo_replay=True,role='transform_control'),
        dict(id='uncentered_linear_bc_ddqn',formula='V-sum(z_selected)',bc=True,ddqn=True,demo_replay=True,role='centering_control'),
        dict(id='centered_linear_bc_only',formula='V-sum(z_selected-mean_all5(z))',bc=True,ddqn=False,demo_replay=False,role='TD_benefit_control'),
        dict(id='centered_linear_demo_no_bc',formula='V-sum(z_selected-mean_all5(z))',bc=False,ddqn=True,demo_replay=True,role='BC_control'),
        dict(id='centered_linear_no_demo_no_bc',formula='V-sum(z_selected-mean_all5(z))',bc=False,ddqn=True,demo_replay=False,role='ordinary_no_demo_control')]
    seeds=dict(model_seeds=[948200001,948200011,948200021],
        demonstration_seed_blocks=[dict(start=948210000+i*1000,count=10) for i in range(3)],
        online_seed_blocks=[dict(start=948220000+i*1000,count=50) for i in range(3)],
        evaluation_seed_block=dict(start=948230000,count=16),
        bc_sampler_seeds=[948240001+10*i for i in range(3)],
        epsilon_branch_seeds=[948250001+10*i for i in range(3)],
        replay_sampler_seeds=[948260001+10*i for i in range(3)],
        job_order_seed=948270001,timing_order_seed=948270011)
    proposed=ranges(seeds);existing=[];documents=[]
    paths=list(ROOT.glob('*.json'))+list((ROOT/'validation').glob('heuristic_initialization_pilot_fixture_*/run/runtime_plan.json'))
    for p in sorted(paths):
        value=json.loads(p.read_text(encoding='utf-8-sig'));documents.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
        existing.extend((lo,hi,p.relative_to(ROOT).as_posix()+label) for lo,hi,label in ranges(value))
    collisions=[]
    for i,(lo,hi,name) in enumerate(proposed):
        for a,b,prior in existing+proposed[:i]:
            if max(lo,a)<=min(hi,b):collisions.append(dict(seed=name,existing=prior))
    if collisions:raise ValueError(collisions)
    files=['CENTERED_LINEAR_CORRECTION_COMPARISON_PLAN_2026-11-20.txt',
        'HEURISTIC_DDQN_VALUE_SEPARATION_CORRECTION_BRIEF_2026-11-20.txt',
        'centered_linear_ddqn_prototype.py','heuristic_initialized_ddqn_baseline.py',
        'controller_v3_amortized.py','heuristic_pilot_teacher_adapter.py',
        'assignment_price_load_radio_heuristic.py','assignment_price_fast_runtime.py',
        'Source/chapter3_iov_env.py','Source/chapter3_iov_env_v2.py']
    spec=dict(protocol_id='centered-linear-correction-development-v1',date='2026-11-20',
        status='PLAN_FREEZE_NOT_EXECUTABLE_RELEASE',plan=files[0],
        scope='Correction-study descriptive screen; not full low-complexity contribution validation',
        primary=arms[0]['id'],arms=arms,reference='observed_load_radio',fits=3,seeds=seeds,
        deferred_controls=['compact64','wide128','adapted_BDQ','full_DQfD'],
        deferred_claims=['superiority_or_speed_against_stronger_neural_controls','algorithmic_novelty','final_generalization'],
        model=dict(architecture=[533,24,161],parameters=16841,parameter_bytes=67364,buffer_bytes=20,
            paired_raw_initial_tensors=True,V_sign_flip=False,V_reset_after_BC=False,
            centering='all five declared actions, independent of mask',gamma=1.,horizon_scaling=False,terminal_anchor=False),
        demonstrations=dict(episodes_per_fit=10,slots_per_episode=40,shared_generation=True,
            teacher_backend='Python reference frozen teacher functions',teacher_stationary_motion=True,extra_empty_battery_restriction=True),
        bc=old['bc'],online=old['online'],optimizer_common=old['optimizer_common'],
        evaluation=dict(episodes_per_policy=16,student_policies=21,teacher_policies=1,
            epsilon=0,checkpoint_freeze='all21finalcheckpoints before evaluation',
            BC_only='separate final arm endpoint, not selected intermediate primary checkpoint'),
        diagnostics=dict(signed_Q_report=True,nearzero_absQ_threshold=1e-6,
            nrmse='sqrt(mean((Q-G)^2))/max(sqrt(mean(G^2)),1e-12)',
            terminal_error='Q(last physical state,action)-last reward',
            probe_stages=['before','after_bc','final'],BC_end_primary_equals_BC_only=True,
            clipnorm=10,horizon_distributions=True,softplus_derivative_where_applicable=True),
        decision=dict(policy='A-B and A-E: both outcome differences >=0 in every fit, mean >0 for both outcomes in each contrast',
            value='A NRMSE and absolute Q-G bias strictly below B in every fit',
            promising='policy AND value AND finite artifacts AND full audit',
            secondary_contrasts=['A-C','A-D','A-F','A-G','each arm minus teacher'],
            on_failure='HOLD expansion; no retuning, retries or best-arm promotion',
            calibration_guarantee=False,statistical_confirmatory_claim=False),
        cost=dict(states=128,warmups=32,rounds=3,policies=22,timed_calls=8448,
            boundary='validation,screen,forward,selection,chosenQ,copy',
            descriptive_only=True,acceptance_retiming=False),
        totals=dict(student_online_slots=36000,teacher_demo_slots=1200,evaluation_slots=14080,
            all_simulated_slots=51280,BC_updates=3000,DDQN_updates=33696,all_gradient_updates=36696,final_checkpoints=21),
        retention=dict(final_only=True,fit_retry=False,best_checkpoint_selection=False),
        definition_sha256={name:sha(ROOT/name) for name in files},simulator_defaults=old['simulator_defaults'],
        launch_authorized=False,launched=False,runner_implemented=False,auditor_implemented=False,
        final_seeds_consumed=False,word_thesis_changed=False,
        next_gate='Implement bound seven-arm adapters/runner/auditor; reduced isolated orchestration fixture and fail-closed CLI; no research launch')
    assert sum(a['ddqn'] for a in arms)==6 and sum(a['bc'] for a in arms)==5
    assert 6*3*2000+3*400+22*16*40==51280
    assert 6*3*1872+5*3*200==36696
    assert 22*128*3==8448
    SPEC.write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8');OUT.mkdir()
    audit=dict(passed=True,protocol_sha256=sha(SPEC),documents_checked=documents,
        seed_intervals=proposed,collisions=collisions,
        scope='Root JSON explicit seeds/start-count blocks plus prior pilot fixtures; refresh with future correction fixture before launch',
        arithmetic_verified=True,new_simulator_steps=0,new_optimizer_updates=0)
    (OUT/'verification.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    p=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json';s=json.loads(p.read_text(encoding='utf-8-sig'))
    s['next']=spec['next_gate'];s['centered_linear_comparison_freeze_2026_11_20']=dict(protocol=SPEC.name,
        protocol_sha256=sha(SPEC),plan=files[0],status=spec['status'],planned_slots=51280,
        seed_documents_checked=len(documents),seed_collisions=0,launch_authorized=False,word_thesis_changed=False)
    p.write_text(json.dumps(s,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(plan_frozen=True,arms=7,seed_documents_checked=len(documents),collisions=0,planned_slots=51280,launched=False)))


if __name__=='__main__':main()
