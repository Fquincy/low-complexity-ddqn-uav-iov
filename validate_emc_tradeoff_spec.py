"""Validate a frozen development design; no simulation, backward or fitting."""
from pathlib import Path
import json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parent
SPEC=ROOT/'LOW_COMPLEXITY_EMC_TRADEOFF_SPEC_2026-11-04.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    s=json.loads(SPEC.read_text());t=s['training'];sm=s['smoke']
    groups={f'train{i}':set(range(b['start'],b['start']+b['count'])) for i,b in enumerate(s['seeds']['train_blocks'])}
    groups.update({k:set(v) for k,v in s['seeds'].items() if k!='train_blocks'})
    groups.update({f'smoke_{k}':set(sm[k]) for k in ('train_seeds','validation_seeds','probe_seeds','fresh_seeds')})
    for a in groups:
        for b in groups:
            if a!=b:assert not groups[a]&groups[b], f'Overlapping seed groups: {a},{b}'
    proposed=set.union(*groups.values());scanned=[]
    # Scan seed-named fields of known prior protocol files and expand seed blocks.
    # This is metadata scope, not a claim about every unregistered historical run.
    other=ROOT.parent/'October 31/Working/Project'
    paths=set()
    for folder in (ROOT,ROOT/'Source',other):
        for pattern in ('*SPEC*.json','*PLAN*.json'):
            paths.update(folder.glob(pattern))
    def seed_values(x,context=False):
        found=set()
        if isinstance(x,dict):
            seedctx=context or any('seed' in k.lower() for k in x)
            # Blocks often have start/count keys nested under a training_task_blocks field.
            if context and isinstance(x.get('start'),int) and isinstance(x.get('count'),int):
                if 1000000<=x['start']<100000000 and 0<x['count']<=100000:
                    found.update(range(x['start'],x['start']+x['count']))
            for k,v in x.items():found.update(seed_values(v,context or 'seed' in k.lower() or 'task_blocks' in k.lower()))
        elif isinstance(x,list):
            for v in x:found.update(seed_values(v,context))
        elif context and isinstance(x,int) and 1000000<=x<100000000:found.add(x)
        return found
    for p in sorted(paths):
        if p==SPEC:continue
        values=seed_values(json.loads(p.read_text(encoding='utf-8-sig')))
        assert not proposed&values, f'Seed reused from registered protocol {p}'
        scanned.append(str(p))
    assert t['episodes']*s['episode_slots']==8000
    updates=8000-t['warmup_physical_records']+1
    assert updates==6977 and updates//t['target_copy_every_updates']==34
    assert 9*8000==72000 and 9*updates==62793
    assert sm['episodes_per_arm']*40-sm['warmup']+1==73
    assert 73//sm['target_interval']==9
    assert sm['arms']*80==240 and sm['arms']*73==219
    assert 1-.95*min(8000/8000,1)==.050000000000000044
    assert abs((1-.95*7999/8000)-.05011875)<1e-12
    # Validate percentage-point units, not percent-relative or .01 percent.
    q=s['quality_screen']
    assert q['versus_paired_additive128']['deadline_fraction_difference_at_least']==-.01
    assert q['versus_paired_additive64']['deadline_fraction_difference_at_least']==.01
    assert (.8-.785)>=q['versus_paired_additive64']['deadline_fraction_difference_at_least']
    assert (.8-.795)<q['versus_paired_additive64']['deadline_fraction_difference_at_least']
    for name,h in s['source_sha256'].items():assert sha(ROOT/name)==h
    import torch,sys
    from emc_q import EMCQ
    sys.path.insert(0,str(ROOT/'Source'))
    from train_chapter3_ddqn import FactorizedQNetwork
    counts={}
    for name,model in [('emc',EMCQ()),('additive128',FactorizedQNetwork(533,32,5,(128,128))),('additive64',FactorizedQNetwork(533,32,5,(64,64)))]:
        counts[name]=sum(p.numel()*p.element_size() for p in model.parameters())
    assert counts['emc']==277444 and counts['emc']/counts['additive128']<=.70
    # Paired exploration ownership test: replay draws must not affect actor streams.
    streams=[np.random.default_rng(2309+100000) for _ in range(3)]
    replays=[np.random.default_rng(2309+200000) for _ in range(3)]
    for slot in range(80):
        actions=[]
        for actor,replay in zip(streams,replays):
            explore=slot<8 or actor.random()<1-.95*slot/80
            actions.append(actor.integers(0,5,32) if explore else None)
            replay.integers(0,64,4)
        for a in actions[1:]:
            if actions[0] is None:assert a is None
            else:np.testing.assert_array_equal(a,actions[0])
    out=ROOT/'validation/emc_tradeoff_spec_2026-11-04_01';out.mkdir(parents=True,exist_ok=False)
    result=dict(state='passed_specification_checks_only',spec_sha256=sha(SPEC),validator_sha256=sha(Path(__file__)),
        seed_group_counts={k:len(v) for k,v in groups.items()},prior_protocols_scanned=scanned,
        native_online_parameter_bytes=counts,updates_per_run=updates,target_copies_per_run=34,
        nine_run_training_slots=72000,nine_run_updates=62793,smoke_training_slots=240,smoke_updates=219,
        epsilon_schedule_and_quality_units_checked=True,paired_actor_replay_ownership_checked=True,
        instantiated_untrained_models_only=True,new_optimizer_updates=0,new_simulator_steps=0,
        runner_implemented=False,smoke_launched=False,development_launched=False)
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='prior_protocols_scanned'},indent=2))

if __name__=='__main__':main()
