"""Create prospective protocol and audit declarations only; never train."""
import copy
import json
from pathlib import Path
import shutil
import run_horizon_loss_research as r

NAME='ASSIGNMENT_PRICE_RESEARCH_PROTOCOL_2026-11-15.json'
OUT=r.ROOT/'validation/assignment_price_research_freeze_2026-11-15_01'

def main():
    OUT.mkdir(exist_ok=False);parent=r.protocol()
    old=[copy.deepcopy(d) for d in parent['learned_arms'] if not d.get('normalized_horizon_loss')]
    new=[]
    for name,params in [('coupled_sweep',17066),('coupled_no_sweep',17066),('separable64',17066),('fixed_price_plain_sweep',16841)]:
        new.append(dict(id=name,model_module='assignment_price_prototype',model_key=name,
            screened=True,horizon_scaling=True,terminal_anchor=True,expected_online_parameter_count=params,
            expected_buffer_bytes=20,scalar_precision='FP64',neural_precision='FP32',
            deployment='assignment_price_native_heads_runtime.Runtime'))
    training=copy.deepcopy(parent['training'])
    training['loss_definitions']='All new arms: original-unit Huber(Q,y), gamma1, terminal anchor weight1. Original controls retain their frozen losses; no normalized-loss arms.'
    seeds=[87301,87311,87317,87323,87331]
    train=[dict(start=873000000+i*100000,count=200) for i in range(5)]
    evaluation=dict(start=874000000,count=48)
    rng=dict(arm_order_seed=875000001,timing_order_seed=875000003,bootstrap_seed=875000007)
    mechanisms=[dict(id=label,left='coupled_sweep',right=right,each_fit_return_difference_min=0.,
        each_fit_deadline_fraction_difference_min=0.,equal_fit_mean_return_difference_min=2.,
        equal_fit_mean_deadline_fraction_difference_min=.01) for label,right in [
            ('sweep','coupled_no_sweep'),('coupling_and_selection','separable64'),('learned_prices','fixed_price_plain_sweep')]]
    sources=['assignment_price_prototype.py','assignment_price_native_heads_runtime.py','assignment_price_fast_runtime.py',
        'assignment_price_deployment_runtime.py','assignment_price_reference_deployment.py','run_assignment_price_fixture.py',
        'audit_assignment_price_fixture_v2.py','verify_assignment_price_fixture.py',
        'ASSIGNMENT_DEPENDENT_PRICE_DESIGN_BRIEF_2026-11-15.txt','ASSIGNMENT_PRICE_NATIVE_HEADS_REVIEW_2026-11-15.txt',
        'ASSIGNMENT_PRICE_IOV_FIXTURE_REVIEW_2026-11-15.txt','ASSIGNMENT_PRICE_OBSERVED_LOAD_HEURISTIC_2026-11-15.txt',
        'ASSIGNMENT_PRICE_LITERATURE_POSITION_2026-11-15.txt']
    p=dict(protocol_id='ADP-DDQN-prospective-development-v1',created='2026-11-15',status='DESIGN_FREEZE_NOT_EXECUTABLE_RELEASE',
        primary='coupled_sweep',parent_protocol=r.SPEC.name,parent_sha256=r.sha(r.SPEC),
        learned_arms=new+old,heuristics=['nearest_rsu','uav_selective','observed_load_radio','all_local'],
        simulator=copy.deepcopy(parent['simulator']),training=training,
        development=dict(model_seeds=seeds,train_blocks=train,evaluation_seeds=evaluation,
            fits=5,training_slots=600000,optimizer_updates=523275,target_copies=2550,
            learned_evaluation_episodes=3600,heuristic_evaluation_episodes=192,evaluation_slots=151680,
            final_checkpoints=75,task_records=22550400,checkpoint_selection=False),rng=rng,
        comparisons=dict(primary_report_against_each=[d['id'] for d in new+old if d['id']!='coupled_sweep']+['nearest_rsu','uav_selective','observed_load_radio','all_local'],
            required_reference_arms=[d['id'] for d in old],reference_each_fit=copy.deepcopy(parent['comparisons']['reference_each_fit']),
            strong_heuristics=['nearest_rsu','uav_selective','observed_load_radio'],heuristic_each_fit=dict(return_difference_min=0.,deadline_fraction_difference_min=0.),
            diagnostic_only_heuristics=['all_local'],mechanisms=mechanisms,
            auxiliary_contrasts=[dict(left='separable64',right='screened_price24',question='FP64 scalar/runtime route characterization, not one-factor price benefit'),
                dict(left='coupled_no_sweep',right='separable64',question='Coupled evaluation/training without sweep'),
                dict(left='screened_price24',right='unmasked_price24',question='Original screening'),
                dict(left='screened_plain24',right='unmasked_plain24',question='Ordinary small DDQN screening'),
                dict(left='screened_wide128',right='unmasked_wide128',question='Wide reference screening')]),
        value_stability=copy.deepcopy(parent['value_stability']),action_cost=copy.deepcopy(parent['action_cost']),
        uncertainty=copy.deepcopy(parent['uncertainty']),final=copy.deepcopy(parent['final']),
        scope=dict(security_trust=False,cloud=False,charging=False,word_thesis_changed=False),
        decision=dict(all_scientific_screens_required=True,every_fit=True,best_initialization_selection=False,
            primary_value_failure_is_failure=True,fixture_outcomes_used_for_tuning=False,
            research_authorization=False,runner_bound=False,novelty_established=False,
            launch='Separate explicit user instruction only after runner/heuristic/auditor implementation, fixture, seed refresh and launch review',
            failure='Preserve all results; no outcome-driven retuning or unchanged-run retiming; new design requires a prospective amendment and fresh seeds'),
        definition_sha256={n:r.sha(r.ROOT/n) for n in sources},
        frozen_legacy_definition_sha256=copy.deepcopy(parent['definition_sha256']))
    cost=p['action_cost'];cost.update(backend='Matched ONNX CPU neural cores plus native Numba heads/decoders, one thread, fastmath false',
        route='All fifteen arms receive same graph/compiler optimization opportunity; extend adapters and verify all original controls before release',
        raw_timed_calls=48000,checkpoint_order='Freeze all 75 final online/target checkpoints; export/check action-value equivalence, freeze artifacts; time once before policy-quality evaluation',
        startup='Record graph export/session construction and true fresh-process first-action timing separately; shared-process startup labeled; never subtract startup from end-to-end storage claims')
    assert len(p['learned_arms'])==15 and len(set(d['id'] for d in p['learned_arms']))==15
    d=p['development'];assert d['training_slots']==15*5*200*40
    assert d['optimizer_updates']==15*5*(8000-training['warmup']+1)
    assert d['target_copies']==15*5*((8000-training['warmup']+1)//training['target_interval_updates'])
    assert d['evaluation_slots']==(15*5+4)*48*40
    assert d['task_records']==(d['training_slots']+d['evaluation_slots'])*30
    assert cost['raw_timed_calls']==15*5*5*128
    # Check all new reservations against root declared protocol/spec/amendment/release files.
    points=seeds+[s+offset for s in seeds for offset in [100000,200000,300000]]+list(rng.values())
    intervals=[(x['start'],x['start']+x['count']) for x in train+[evaluation]]
    def check(value):
        if isinstance(value,dict):
            if isinstance(value.get('start'),int) and isinstance(value.get('count'),int):
                lo=value['start'];hi=lo+value['count']
                assert not any(lo<=x<hi for x in points),(lo,hi)
                assert not any(max(lo,a)<min(hi,b) for a,b in intervals),(lo,hi)
            for v in value.values():check(v)
        elif isinstance(value,list):
            for v in value:check(v)
        elif isinstance(value,int) and not isinstance(value,bool):
            assert value not in points and not any(a<=value<b for a,b in intervals),value
    checked={}
    for path in r.ROOT.glob('*.json'):
        if any(s in path.name for s in ['PROTOCOL','SPEC','AMENDMENT','RELEASE']):
            check(json.loads(path.read_text(encoding='utf-8-sig')));checked[path.name]=r.sha(path)
    assert len(points)==len(set(points))
    assert all(max(a,c)>=min(b,d) for i,(a,b) in enumerate(intervals) for c,d in intervals[i+1:])
    assert not any(a<=s<b for s in points for a,b in intervals)
    r.write(OUT/'seed_declaration_audit.json',dict(passed=True,scope='Root declared reservation documents, all numeric values and start/count intervals; not all historical trace contents',
        checked_documents=checked,new_points=points,new_intervals=intervals,reserved_no_execution=True,refresh_before_launch=True))
    target=r.ROOT/NAME;assert not target.exists();r.write(target,p)
    for n in sources+[NAME]:
        dest=OUT/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(r.ROOT/n,dest)
    r.write(OUT/'verification.json',dict(passed=True,protocol_sha256=r.sha(target),learned_controllers=15,heuristics=4,
        training_slots=600000,evaluation_slots=151680,optimizer_updates=523275,checkpoints=75,
        thresholds_retained=True,seed_declaration_audit_passed=True,research_launched=False,runner_ready=False,novelty_established=False))
    r.write(OUT/'manifest.json',{path.relative_to(OUT).as_posix():r.sha(path) for path in OUT.rglob('*') if path.is_file()})
    print(json.dumps(r.read(OUT/'verification.json'),indent=2))

if __name__=='__main__':main()
