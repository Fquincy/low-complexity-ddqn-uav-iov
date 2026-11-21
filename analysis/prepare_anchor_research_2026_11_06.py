"""Create prospective protocol and separate runner/auditor revisions; never launch."""
import copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=json.loads((ROOT/'LOW_COMPLEXITY_V3_RESEARCH_PROTOCOL_2026-11-05.json').read_text())
extra=['price_both','no_price_both','price_horizon_only','price_anchor_only','plain_both','compact_both']
p.update(id='terminal-anchor-seventeen-arm-development-v1',date='2026-11-06',
    status='prospective_development_design_no_launch_authority',primary='price_both',arms=extra+p['arms'])
p['development'].update(model_seeds=[61001,61011,61017,61023,61039],
    train_blocks=[dict(start=610000000+i*1000000,count=200) for i in range(5)],
    diagnostic_validation_seeds=list(range(621000000,621000005)),evaluation_seeds=dict(start=620000000,count=48),
    training_slots=680000,optimizer_updates=593045,target_copies=2890,status='prospectively_reserved_pending_collision_audit')
p['final']['status']='reserved_untouched_no_launch'
p['final']['authorization']='not part of this prospective development protocol; existing reservations preserved only'
p['final_uncertainty']['status']='historical reference only; final amended protocol requires separate review'
p['paired_randomization']['arm_order_rng_seed']=620600001
p['paired_randomization'].pop('all_55_final_checkpoints_frozen_and_hashed_before_any_gate_evaluation')
p['paired_randomization']['all_85_final_checkpoints_frozen_and_hashed_before_any_gate_evaluation']=True
p['paired_randomization']['source_schedule_inherited_from_version2']=False
p['comparisons']['primary_against_each_of']+=['plain_both','compact_both']
p['comparisons']['no_price_arm']='no_price_both'
p['comparisons']['correction_ablations']=['price_horizon_only','price_anchor_only']
p['comparisons']['correction_ablations_rule']='each fit primary minus ablation return>=0 and deadline>=0; all contrasts reported'
p['value_screen_scope']=dict(gated_arms=['price_both'],report_all_arms=True,
    amendment='Candidate-specific gate prospectively replaces all-arm value gate ONLY for this new comparison; original run remains failed under original rules.',
    failures_of_comparators_do_not_excuse_candidate_failure=True)
p['terminal_anchor']=dict(source='terminal_anchor_prototype.py',horizon=40,loss='uniform DDQN Huber + 1.0 * mean terminal Huber',
    batch=32,terminal_replay='uniform with replacement; eviction checked',empty_pool='zero anchor loss and recorded event',
    separate_rng_offset=300000,one_optimizer_step=True,model_parameter_counts_unchanged=True)
p['frozen_checkpoint_inference']['order_seed']=623600003
p['frozen_checkpoint_inference'].pop('all_11_models_same_fixed_family_routes_as_compatibility_protocol')
p['frozen_checkpoint_inference']['routes']='original eleven retain fixed routes; six corrected use underlying native route plus complete validated horizon wrapper'
p['authorization']['final']='disabled: no final amended protocol or launch authority'
p['notes']=['No fixture policy outcome used for tuning or selection.','Old eleven rerun fresh; historical failed run retained.',
    'Comparators all reported, but candidate-specific value gating is explicitly prospective.',
    'No final execution in this protocol.','Peak memory not claimed; payload storage and measured training times reported.']
p['fixture']=dict(model_seeds=[99601,99607],training_starts=[996010000,996070000],evaluation_seeds=[996090100,996090101],
    validation_seeds=[996090200],episodes=2,capacity=64,warmup=8,batch=4,target_interval=8,validation_interval=1,
    excluded_from_quality_evidence=True)
protocol_name='TERMINAL_ANCHOR_RESEARCH_PROTOCOL_2026-11-06.json'
(ROOT/protocol_name).write_text(json.dumps(p,indent=2),encoding='utf-8')

s=(ROOT/'run_v3_research_gate_v2.py').read_text()
s=s.replace('run_v3_research_gate_v2','run_anchor_research_preparation').replace('audit_v3_research_gate_v2','audit_anchor_research_preparation').replace('test_v3_research_gate_v2','test_anchor_research_preparation')
s=s.replace('LOW_COMPLEXITY_V3_RESEARCH_PROTOCOL_2026-11-05.json',protocol_name)
s=s.replace('import compiled_controller_inference as compiled','import compiled_controller_inference as compiled\nimport terminal_anchor_prototype as anchor\nfrom torch.nn import functional as F')
s=s.replace('ARMS=(*NEW_ARMS,*BASE_ARMS)','CORRECTED=tuple(anchor.ARMS)\nARMS=(*CORRECTED,*NEW_ARMS,*BASE_ARMS)')
s=s.replace('_base_bounded=base.bounded','_base_bounded=base.bounded\n_base_update=base.update')
s=s.replace("'test_anchor_research_preparation.py',PROTOCOL.name", "'test_anchor_research_preparation.py','terminal_anchor_prototype.py',PROTOCOL.name")
s=s.replace('11*5*200*40==440000','17*5*200*40==680000').replace('11*5*(8000-1024+1)==383735','17*5*(8000-1024+1)==593045')
s=s.replace('or arm in NEW_ARMS','or arm in NEW_ARMS or arm in CORRECTED')
s=s.replace("replay=base.old.Replay(cfg['capacity']),", "replay=TerminalBuffer(cfg['capacity']) if arm in CORRECTED else base.old.Replay(cfg['capacity']),\n        anchor_rng=np.random.default_rng(seed+300000),")
s=s.replace('    if arm in NEW_ARMS:return AmortizedPriceQ(arm)','    if arm in CORRECTED:return anchor.Critic(arm,seed)\n    if arm in NEW_ARMS:return AmortizedPriceQ(arm)')
s=s.replace('    if arm in NEW_ARMS:return model.greedy(states)','    if arm in CORRECTED:return (*model.greedy(states),{})\n    if arm in NEW_ARMS:return model.greedy(states)')
s=s.replace('    if arm in NEW_ARMS:return model(states,actions)','    if arm in CORRECTED:return model(states,actions)\n    if arm in NEW_ARMS:return model(states,actions)')
s=s.replace('base.bounded=bounded','base.bounded=bounded;base.update=update')
s=s.replace("'amortized_workload_price24'","'price_both'").replace("'amortized_no_price24'","'no_price_both'")
# The replacements above touch only primary string literals in comparisons/costs;
# imported original NEW_ARMS and architecture definitions remain unchanged.
s=s.replace('for x in values[a]) for a in ARMS)',"for x in values[a]) for a in p['value_screen_scope']['gated_arms'])")
s=s.replace("    if phase=='final':\n        if bootstrap", "    screens['correction_ablations_each_fit']=all(all(r['return_difference']>=0 and r['deadline_difference']>=0 for r in comparisons[a]['per_fit']) for a in p['comparisons']['correction_ablations'])\n    if phase=='final':\n        if bootstrap")
s=s.replace("+['no_price_both'])", "+['no_price_both']+p['comparisons']['correction_ablations'])")
s=s.replace("runtime,meta=fused.export(model,arm,path)","runtime,meta=fused.export(model.model if arm in CORRECTED else model,anchor.ARMS[arm][0] if arm in CORRECTED else arm,path)")
s=s.replace('fused.values(runtime[1],arm,state)','runtime_values(runtime[1],arm,state)')
s=s.replace('fused.action(runtime[1],arm,state)','runtime_values(runtime[1],arm,state)[0][0].copy()')
s=s.replace("        if phase not in ('development','final'):raise ValueError('phase')","        if phase!='development':raise ValueError('final execution disabled')")
s=s.replace("model_seeds=[99261,99267]", "model_seeds=[99601,99607]").replace('992610000','996010000').replace('992670000','996070000').replace('992690100','996090100').replace('992690101','996090101').replace('992690200','996090200')
s=s.replace("p.add_argument('--mode',choices=['fixture','integration-fixture','development','final'],required=True)","p.add_argument('--mode',choices=['integration-fixture','development','final'],required=True)")
helpers='''
class TerminalBuffer(anchor.Replay):
    def add(self,*row):super().add(row)

def update(agent):
    if agent['arm'] not in CORRECTED:return _base_update(agent)
    cfg=agent['cfg']
    if agent['slots']<cfg['warmup']:return None
    batch=agent['replay'].sample(cfg['batch'],agent['replay_rng'])
    terminal=agent['replay'].sample(32,agent['anchor_rng'],True) if agent['online'].anchored else None
    start=time.perf_counter_ns()
    loss,detail=anchor.objective(agent['online'],agent['target'],batch,terminal)
    s,a,r,ns,d,indices=batch
    q=chosen(agent['online'],s,a,agent['arm']);y=torch.tensor(detail['target'],dtype=q.dtype)
    agent['opt'].zero_grad(set_to_none=True);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(agent['online'].parameters(),cfg['clip'],error_if_nonfinite=True)
    assert all(v.grad is None for v in agent['target'].parameters())
    agent['opt'].step();agent['updates']+=1
    assert all(torch.isfinite(v).all() for v in agent['online'].parameters())
    copied=agent['updates']%cfg['target_interval']==0
    if copied:agent['target'].load_state_dict(agent['online'].state_dict());agent['copies']+=1
    return dict(step=agent['updates'],physical_slots=agent['slots'],loss=float(loss.detach()),
        grad_norm_before_clip=float(norm),gradient_clipped=bool(norm>cfg['clip']),
        chosen_q=base.old.distribution(q),bellman_target=base.old.distribution(y),td_residual=base.old.distribution(q.detach()-y),
        replay_indices=indices.tolist(),target_copy=copied,terminal_samples=int(d.sum()),
        anchor_indices=[] if terminal is None else terminal[-1].tolist(),terminal_pool=sorted(agent['replay'].terminal),
        uniform_loss=detail['uniform'],anchor_loss=detail['anchor'],empty_anchor=detail['empty_anchor'],
        objective_update_ns=time.perf_counter_ns()-start)

def runtime_values(runtime,arm,state):
    if arm not in CORRECTED:return fused.values(runtime,arm,state)
    x=np.asarray(state,dtype=np.float32)
    if x.shape!=(533,) or not np.isfinite(x).all():raise ValueError('finite state required')
    t=float(x[-1])*40;rounded=round(t)
    if abs(t-rounded)>1e-4 or not 0<=rounded<40:raise ValueError('nonterminal grid time required')
    a,q=fused.values(runtime,anchor.ARMS[arm][0],x)
    return a,q*np.float32(40-rounded if anchor.ARMS[arm][1] else 1.)

'''
s=s.replace('def main():',helpers+'def main():')
(ROOT/'run_anchor_research_preparation.py').write_text(s,encoding='utf-8')

a=(ROOT/'audit_v3_research_gate_v2.py').read_text().replace('import run_v3_research_gate_v2 as runner','import run_anchor_research_preparation as runner')
a=a.replace('fused.values((session,extra),arm,state)','runner.runtime_values((session,extra),arm,state)')
a=a.replace("'amortized_workload_price24'","'price_both'")
a=a.replace("    if phase=='final'", "    if phase=='final'")
# Verify anchor replay membership from physical slots and ring-buffer indices.
a=a.replace('                update_count+=1', '''                if arm in runner.CORRECTED:
                    slots=row['physical_slots'];capacity=cfg['capacity']
                    expected_pool={j%capacity for j in range(max(0,slots-capacity),slots) if j%40==39}
                    assert set(row['terminal_pool'])==expected_pool
                    anchored=runner.anchor.ARMS[arm][2]
                    assert all(j in expected_pool for j in row['anchor_indices'])
                    assert len(row['anchor_indices'])==(32 if anchored and expected_pool else 0)
                    assert abs(row['loss']-row['uniform_loss']-row['anchor_loss'])<1e-3
                    assert row['empty_anchor']==bool(anchored and not expected_pool)
                update_count+=1''')
(ROOT/'audit_anchor_research_preparation.py').write_text(a,encoding='utf-8')
print('Created separate seventeen-arm protocol, runner and auditor. Research remains disabled.')
