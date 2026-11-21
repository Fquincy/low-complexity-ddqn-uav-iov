"""Independent static protocol review: no model, simulator or optimizer imports."""
import ast
import hashlib
import json
from pathlib import Path
import sys

from freeze_heuristic_initialization_pilot_plan_2026_11_19 import ranges

ROOT=Path(__file__).resolve().parent
SPEC=ROOT/'VALUE_GRADIENT_ISOLATION_COMPARISON_PROTOCOL_2026-11-20.json'
FREEZE=ROOT/'validation/value_gradient_isolation_comparison_freeze_2026-11-20_01'
OUT=ROOT/'validation/value_gradient_isolation_comparison_freeze_review_2026-11-20_01'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    if OUT.exists():raise RuntimeError('Preserve previous independent review')
    p=read(SPEC);f=read(FREEZE/'verification.json');manifest=read(FREEZE/'manifest.json')
    actual={x.relative_to(FREEZE).as_posix() for x in FREEZE.rglob('*') if x.is_file() and x!=FREEZE/'manifest.json'}
    assert actual==set(manifest)
    for name,digest in manifest.items():assert sha(FREEZE/name)==digest,name
    assert sha(SPEC)==f['protocol_sha256']
    assert sha(ROOT/p['plan'])==f['plan_sha256']
    for name,digest in p['definition_sha256'].items():
        assert sha(ROOT/name)==sha(FREEZE/'source_snapshot'/name)==digest,name
    assert sha(FREEZE/'source_snapshot'/SPEC.name)==sha(SPEC)
    old=read(ROOT/p['inherited_protocol']);design=read(ROOT/p['reservation_source'])
    names=[a['id'] for a in p['arms']];P,R,A,B,C,D,E,F,G=names
    assert P=='value_gradient_isolated_ddqn' and R=='dueling_shared_td_rescale_ddqn'
    assert [a['id'] for a in p['arms'] if a['role']=='sole_primary']==[P]
    byname={a['id']:a for a in p['arms']}
    assert byname[A]['role']=='unchanged_isolation_ablation'
    for arm in old['arms']:
        for key in ['calibration','retention','bc','ddqn','value_formula','demo_replay','retention_temperature']:
            assert byname[arm['id']][key]==arm[key],(arm['id'],key)
    assert [a['gradient_rule'] for a in p['arms']]==['isolate_value','rescale_shared_td']+['ordinary']*7
    for key in ['demonstrations','bc','calibration','optimizer_common','simulator_defaults']:
        assert p[key]==old[key],key
    assert p['decision']['policy_controls']==[E,F,A,R] and p['decision']['value_control']==D
    assert p['decision']['equality_tolerance']=='none for scientific screens'
    assert p['seeds']==design['seeds']
    assert p['online']['update_entrypoint']=='value_gradient_isolation.ddqn_retention_step'
    assert p['online']['shared_imitation_derivative']=='ordinary,unscaled for all retention arms'
    assert p['auditor']['first_update_reconstruction']['contexts']==9
    assert p['auditor']['first_update_reconstruction']['production_custom_backward_for_reference_forbidden']
    assert not p['auditor']['full_historical_gradient_replay']
    assert p['deployment_export']['inference_class'] if 'inference_class' in p['deployment_export'] else p['deployment_export']['class_name']=='return_calibrated_handoff.Critic'
    # Recompute each arm's budget directly, not through the freeze's arithmetic helper.
    for mode,config,t in [('development',dict(fits=3,demo=10,online=50,evaluation=16,bc=200,warmup=128,batch=64,interval=200,states=128,rounds=3,warmups=32),p['totals']),
                          ('fixture',dict(fits=2,demo=10,online=1,evaluation=1,bc=2,warmup=8,batch=4,interval=8,states=4,rounds=1,warmups=1),p['fixture']['totals'])]:
        fits=config['fits'];policies=9*fits+1
        bcs=tds=keeps=ridges=online=scheduled=initial=0
        for arm in p['arms']:
            if arm['bc']:bcs+=fits*config['bc']
            if arm['calibration']:ridges+=fits
            if arm['ddqn']:
                updates=config['online']*40-config['warmup'];tds+=fits*updates
                online+=fits*config['online']*40;initial+=fits;scheduled+=fits*(updates//config['interval'])
                if arm['retention']:keeps+=fits*updates
        expected=dict(student_online_slots=online,teacher_demo_slots=fits*config['demo']*40,
            evaluation_slots=policies*config['evaluation']*40,BC_updates=bcs,DDQN_updates=tds,
            retention_batches=keeps,retention_examples=keeps*config['batch'],ridge_solves=ridges,
            final_checkpoints=9*fits,deployment_weight_files=9*fits,initial_target_copies=initial,
            scheduled_target_copies=scheduled,measured_action_calls=policies*config['rounds']*config['states'],
            warmup_action_calls=policies*config['rounds']*config['warmups'],
            all_simulated_slots=online+fits*config['demo']*40+policies*config['evaluation']*40,
            all_gradient_updates=bcs+tds)
        assert expected==t,mode
    # Refresh collisions after the new protocol itself exists. Only exact same-study aliases are exempt.
    proposed=ranges(dict(development_seeds=p['seeds'],fixture_seeds=p['fixture']['seeds']))
    history=[];documents=[];aliases=[]
    for path in sorted(ROOT.glob('*.json'))+sorted((ROOT/'validation').glob('*/run/runtime_plan.json'))+sorted((ROOT/'development_runs').glob('*/run/runtime_plan.json')):
        value=read(path)
        if path==SPEC:
            assert value['seeds']==p['seeds'] and value['fixture']['seeds']==p['fixture']['seeds'];aliases.append(path.name)
        elif path==ROOT/p['reservation_source']:
            assert value['seeds']==p['seeds'];aliases.append(path.name)
        else:
            documents.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path)))
            history.extend((lo,hi,path.name+label) for lo,hi,label in ranges(value))
    for i,(lo,hi,label) in enumerate(proposed):
        for a,b,origin in history+proposed[:i]:assert max(lo,a)>min(hi,b),(label,origin)
    literals={node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse((ROOT/'value_gradient_isolation.py').read_text()).body
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['DEVELOPMENT_RELEASED','FINAL_RELEASED']}
    assert not any(literals.values()) and not p['research_launch_authorized'] and not p['launched']
    assert not p['runner_implemented'] and not p['auditor_implemented']
    assert p['authorization']['confirmation_alone_is_sufficient'] is False
    assert not (ROOT/p['artifact_retention']['reserved_output']).exists()
    assert not Path(p['artifact_retention']['reserved_archive']).exists()
    assert not any(n.startswith(('torch','numpy','chapter3_iov_env','value_gradient_isolation')) for n in sys.modules)
    OUT.mkdir(parents=True)
    record=dict(status='FROZEN_PROTOCOL_INDEPENDENTLY_REVIEWED_RESEARCH_DISABLED',
        protocol_sha256=sha(SPEC),freeze_manifest_sha256=sha(FREEZE/'manifest.json'),
        freeze_manifest_files_verified=len(manifest),bound_definitions_verified=len(p['definition_sha256']),
        inherited_settings_and_legacy_controls_unchanged=True,sole_primary_and_required_comparisons_verified=True,
        development_and_fixture_all_budget_fields_recomputed=True,seed_documents_checked=len(documents),
        postfreeze_same_study_aliases=aliases,seed_collision_count=0,seed_documents=documents,
        research_and_archive_output_absence_verified=True,release_flags_false=True,
        verifier_sha256=sha(Path(__file__)),new_models_created=0,new_simulator_steps=0,new_optimizer_updates=0,
        new_teacher_calls=0,fixture_executed=False,research_launch_authorized=False,word_thesis_changed=False)
    with (OUT/'verification.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in record.items() if k!='seed_documents'},indent=2))


if __name__=='__main__':main()
