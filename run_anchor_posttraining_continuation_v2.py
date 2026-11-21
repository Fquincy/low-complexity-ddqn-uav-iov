"""Isolated post-training continuation. Research execution remains disabled."""
import os,shutil,traceback
import run_anchor_posttraining_continuation as engineering
from run_anchor_development_v4 import *
import run_anchor_development_v4 as original
from anchor_continuation_inference_v2 import frozen_inference_gate,CONTINUATION_SOURCES
FIXTURE_SOURCE=ROOT/'validation/anchor_research_route_v2_2026-11-06_01'
RESEARCH_SOURCE=ROOT/'development_runs/terminal_anchor_development_2026-11-06_01'
CONTINUATION_RESEARCH_ENABLED=False

def audit_continuation_inputs(folder):
    folder=Path(folder);f=read(folder/'PRE_CONTINUATION_FREEZE.json');source=Path(f['source'])
    assert source.resolve() in [FIXTURE_SOURCE.resolve(),RESEARCH_SOURCE.resolve()]
    assert read(folder/'runtime_plan.json')==read(source/'runtime_plan.json')
    assert read(folder/'runtime_plan.json')['source_sha256']==f['original_source_sha256']
    for name,digest in f['input_files'].items():
        assert base.old.sha(source/name)==base.old.sha(folder/name)==digest,name
    for name,digest in f['continuation_source_sha256'].items():
        assert base.old.sha(ROOT/name)==base.old.sha(folder/'continuation_source_snapshot'/name)==digest,name
    return f

def validate_research_plan(plan):
    p=protocol();expected=p['development'];validate_plan(plan)
    seeds=expected['evaluation_seeds']
    evaluation=list(range(seeds['start'],seeds['start']+seeds['count'])) if isinstance(seeds,dict) else seeds
    checks=dict(kind='RESEARCH',fixture=False,phase='development',model_seeds=expected['model_seeds'],
        train_blocks=expected['train_blocks'],evaluation_seeds=evaluation,
        diagnostic_validation_seeds=expected['diagnostic_validation_seeds'],cfg=research_cfg(),episodes=200,
        validation_interval=50,quality_screen_authority=True,training_launched=True,
        protocol_sha256=base.old.sha(PROTOCOL),arm_order_seed=p['paired_randomization']['arm_order_rng_seed'])
    for key,value in checks.items():
        if plan.get(key)!=value:raise ValueError('frozen development plan mismatch:'+key)
    if plan['runtime_route_amendment_sha256']!=base.old.sha(ROOT/'ANCHOR_RUNTIME_ROUTE_AMENDMENT_2026-11-06.json'):
        raise ValueError('original runtime amendment changed')

def verify_evidence_manifest(path):
    m=read(path)
    for name,digest in m['files'].items():
        if base.old.sha(ROOT/name)!=digest:raise ValueError('evidence changed:'+name)
    return m

def preflight(source,fixture):
    source=Path(source).resolve()
    if fixture:return engineering.preflight(source,True)
    if source!=RESEARCH_SOURCE.resolve():raise ValueError('only original frozen development source allowed')
    for name in ['anchor_posttraining_continuation_completion_2026-11-11.json',
                 'anchor_runtime_numerical_repair_completion_2026-11-11.json',
                 'anchor_development_completion_review_manifest_2026-11-11.json']:
        verify_evidence_manifest(ROOT/'validation'/name)
    plan=read(source/'runtime_plan.json');validate_research_plan(plan)
    if any(x['stage']=='heldout_evaluation_started' for x in read(source/'events.json')):
        raise ValueError('source evaluation has already started; continuation requires review')
    if list(source.glob('*/fit_*/evaluation_*.json.gz')):raise ValueError('existing research evaluation artifacts')
    if not (source/'FAILED.json').exists():raise ValueError('original failure evidence missing')
    freeze=read(source/'ALL_FINAL_CHECKPOINTS_FREEZE.json')
    keys={f'{arm}_fit_{fit:02d}' for fit in range(5) for arm in ARMS}
    if set(freeze['checkpoints'])!=keys or freeze['count']!=85 or not freeze['before_evaluation']:
        raise ValueError('checkpoint coverage changed')
    for name,digest in plan['source_sha256'].items():
        if base.old.sha(ROOT/name)!=base.old.sha(source/'source_snapshot'/name) or base.old.sha(ROOT/name)!=digest:
            raise ValueError('original source changed:'+name)
    for fit,seed in enumerate(plan['model_seeds']):
        for arm in ARMS:
            key=f'{arm}_fit_{fit:02d}';path=source/arm/f'fit_{fit:02d}'/'final.pt'
            if base.old.sha(path)!=freeze['checkpoints'][key]:raise ValueError('checkpoint changed:'+key)
            ck=torch.load(path,map_location='cpu',weights_only=False)
            if (ck['arm'],ck['model_seed'],ck['slots'],ck['updates'],ck['copies'])!=(arm,seed,8000,6977,34):
                raise ValueError('checkpoint training identity changed:'+key)
            if ck['cfg']!=research_cfg():raise ValueError('checkpoint config changed:'+key)
            if base.old.digest(ck['online'])!=ck['online_parameter_hash'] or base.old.digest(ck['target'])!=ck['target_parameter_hash']:
                raise ValueError('checkpoint parameters changed:'+key)
            if len(list(path.parent.glob('train_*.json.gz')))!=200:raise ValueError('missing training episodes:'+key)
    return plan,freeze


def materialize_inputs(source,out,plan):
    paths=[source/'runtime_plan.json',source/'ALL_FINAL_CHECKPOINTS_FREEZE.json']
    paths+=list((source/'source_snapshot').rglob('*'))
    for arm in ARMS:
        for fit in range(len(plan['model_seeds'])):
            d=source/arm/f'fit_{fit:02d}'
            paths += [p for p in d.iterdir() if p.name in ['final.pt','optimizer.jsonl'] or p.name.startswith(('train_','validation_'))]
    digests={}
    for src in paths:
        if not src.is_file():continue
        rel=src.relative_to(source);dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        # Immutable input payloads may share storage; continuation never writes these paths.
        if rel.parts[0]=='source_snapshot' or src.name in ['runtime_plan.json','ALL_FINAL_CHECKPOINTS_FREEZE.json']:
            shutil.copy2(src,dest)
        else:
            try:os.link(src,dest)
            except OSError:shutil.copy2(src,dest)
        digests[str(rel).replace('\\','/')]=base.old.sha(src)
        assert base.old.sha(dest)==base.old.sha(src)
    return digests

def run_continuation(source,output,fixture=False):
    if not fixture and not CONTINUATION_RESEARCH_ENABLED:raise RuntimeError('Research continuation disabled without authorized entry point')
    configure();patch_training_core();p=protocol();source=Path(source).resolve();out=Path(output)
    plan,freeze=preflight(source,fixture);out.mkdir(exist_ok=False,parents=True)
    inputs=materialize_inputs(source,out,plan)
    for name in CONTINUATION_SOURCES:
        dest=out/'continuation_source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    write(out/'PRE_CONTINUATION_FREEZE.json',dict(source=str(source),input_files=inputs,
        original_source_sha256=plan['source_sha256'],continuation_source_sha256={n:base.old.sha(ROOT/n) for n in CONTINUATION_SOURCES},
        runtime_amendment_sha256=base.old.sha(ROOT/'ANCHOR_TORCH_CORE_RUNTIME_AMENDMENT_2026-11-11.json'),
        fixture=fixture,new_training_slots=0,new_optimizer_updates=0))
    audit_continuation_inputs(out)
    oldevents=read(source/'events.json');write(out/'ORIGINAL_EVENTS.json',oldevents)
    freeze_sequence=next(x['sequence'] for x in oldevents if x['stage']=='all_final_checkpoints_frozen')
    events=[x for x in oldevents if x['sequence']<=freeze_sequence]
    def event(stage,**kwargs):
        events.append(dict(sequence=len(events),stage=stage,**kwargs));write(out/'events.json',events)
    fits=[(seed,block['start'],i) for i,(seed,block) in enumerate(zip(plan['model_seeds'],plan['train_blocks']))]
    counts={}
    for seed,_,fit in fits:
        for arm in ARMS:
            ck=torch.load(out/arm/f'fit_{fit:02d}'/'final.pt',map_location='cpu',weights_only=False)
            counts[f'{arm}_fit_{fit:02d}']=dict(slots=ck['slots'],updates=ck['updates'],target_copies=ck['copies'],checkpoints=1,
                optimizer_rows=ck['updates'],model_seed=seed,train_seed_start=plan['train_blocks'][fit]['start'])
    frozen=freeze['checkpoints'];phase='development'
    try:
        event('posttraining_continuation_started',new_training_slots=0,new_optimizer_updates=0)
        event('frozen_checkpoint_inference_gate_started',checkpoint_count=len(frozen))
        inference_report=frozen_inference_gate(out,fits)
        write(out/'FROZEN_CHECKPOINT_INFERENCE.json',inference_report)
        event('frozen_checkpoint_inference_gate_complete',all_fit_cost_screens_pass=inference_report['all_fit_cost_screens_pass'])
        event('heldout_evaluation_started',seed_count=len(plan['evaluation_seeds']))
        heuristics={name:[] for name in p['comparisons']['heuristics']}
        for seed in plan['evaluation_seeds']:
            for name in heuristics:
                row,_=base.episode(None,seed,policy=name,episode_id='heldout_gate');row['model_seed']=None;row['run_kind']=plan['kind'];save_episode(out,f'heuristic_{name}_{seed}.json.gz',row);heuristics[name].append(compact_eval(row))
        evaluations={arm:[] for arm in ARMS};invariance=[]
        for fit_id,(model_seed,_,_) in enumerate(fits):
            for arm in ARMS:
                folder=out/arm/f'fit_{fit_id:02d}'
                agent,ck=load_evaluator(folder/'final.pt',arm,model_seed,plan['cfg']);before=eval_snapshot(agent)
                for seed in plan['evaluation_seeds']:
                    row,_=base.episode(agent,seed,episode_id='heldout_gate');row['model_seed']=model_seed;row['run_kind']=plan['kind'];save_episode(folder,f'evaluation_{seed}.json.gz',row);evaluations[arm].append(compact_eval(row))
                    if eval_snapshot(agent)!=before:raise AssertionError('held-out evaluation mutated weights')
                invariance.append(dict(arm=arm,fit=fit_id,before=before,after=eval_snapshot(agent)))
                del agent,ck
        comparisons=compare_all(evaluations,heuristics)
        values={arm:[value_summary_by_time([row for row in evaluations[arm] if row['model_seed']==seed])
            for seed in plan['model_seeds']] for arm in ARMS}
        bootstrap=(crossed_bootstrap(evaluations['price_both'],evaluations['additive128_unbounded'])
            if not fixture and phase=='final' else None)
        screens=quality_screens(comparisons,values,len(fits),bootstrap,phase) if not fixture else dict(evaluated=False,reason='synthetic_fixture_excluded')
        analysis=dict(run_kind=plan['kind'],comparisons=comparisons,value_stability=values,final_primary_bootstrap=bootstrap,
            quality_screens=screens,quality_claim=False,selection_performed=False,all_fits_retained=True,
            interpretation='diagnostics are empirical greedy-policy realized-return consistency, not Qstar proof')
        write(out/'comparisons_and_value_stability.json',analysis);write(out/'evaluation_invariance.json',invariance)
        for key,digest in frozen.items():
            fit=key.rsplit('_fit_',1)[1];arm=key.rsplit('_fit_',1)[0]
            path=out/arm/f'fit_{fit}'/'final.pt'
            assert base.old.sha(path)==digest
        state='completed_synthetic_full_fixture_not_research' if fixture else (phase+'_screen_passed' if aggregate_screens(screens,inference_report) else phase+'_screen_failed')
        slots=sum(x['slots'] for x in counts.values());updates=sum(x['updates'] for x in counts.values())
        result=dict(state=state,run_kind=plan['kind'],fixture=fixture,checkpoint_count=len(frozen),counts=counts,
            training_slots=slots,optimizer_updates=updates,target_copies=sum(x['target_copies'] for x in counts.values()),
            checkpoints_frozen_before_evaluation=True,heldout_evaluation_seeds=plan['evaluation_seeds'],
            frozen_checkpoint_inference_screens=inference_report['screens_per_fit'],
            frozen_checkpoint_inference_all_fit_screens_pass=inference_report['all_fit_cost_screens_pass'],
            research_training_launched=False,new_training_slots=0,new_optimizer_updates=0,policy_quality_claim=False,
            thesis_changed=False,all_failed_fits_retained=True,
            empirical_protocol_screens_passed=False if fixture else aggregate_screens(screens,inference_report))
        event('complete',training_slots=slots,optimizer_updates=updates)
        write(out/'verification.json',result)
        manifest={str(x.relative_to(out)).replace('\\','/'):base.old.sha(x) for x in out.rglob('*') if x.is_file()}
        write(out/'run_manifest.json',dict(files=manifest))
        return result
    except Exception:
        write(out/'CONTINUATION_FAILED.json',dict(error=traceback.format_exc(),fixture=fixture,new_training_slots=0,new_optimizer_updates=0))
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['integration-fixture','development','final'],required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--confirm-development-run',action='store_true')
    args=parser.parse_args()
    if args.mode!='integration-fixture':
        if args.mode=='development' and not args.confirm_development_run:parser.error('--confirm-development-run required; research continuation disabled')
        parser.error('Research continuation and final execution disabled pending implementation review')
    result=run_continuation(args.source,args.output,fixture=True)
    from audit_anchor_posttraining_continuation_v2 import audit_research
    print(json.dumps(dict(result=result,audit=audit_research(args.output)),indent=2))
if __name__=='__main__':main()
