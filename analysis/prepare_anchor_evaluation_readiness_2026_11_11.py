"""Create separately versioned continuation with guarded research-source preflight."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
mapping={'run_anchor_posttraining_continuation':'run_anchor_posttraining_continuation_v2','audit_anchor_posttraining_continuation':'audit_anchor_posttraining_continuation_v2','anchor_continuation_inference':'anchor_continuation_inference_v2'}
for old,new in mapping.items():
    s=(R/(old+'.py')).read_text(encoding='utf-8')
    for before,after in mapping.items():s=s.replace(before,after)
    if old=='anchor_continuation_inference':
        s=s.replace(" 'ANCHOR_TORCH_CORE_RUNTIME_AMENDMENT_2026-11-11.json']", " 'ANCHOR_TORCH_CORE_RUNTIME_AMENDMENT_2026-11-11.json','launch_anchor_evaluation.py',*SOURCES]")
        s=s.replace('CONTINUATION_SOURCES=[', 'CONTINUATION_SOURCES=list(dict.fromkeys([').replace("'launch_anchor_evaluation.py',*SOURCES]", "'launch_anchor_evaluation.py',*SOURCES]))")
    if old=='run_anchor_posttraining_continuation':
        s=s.replace('import os,shutil,traceback','import os,shutil,traceback\nimport run_anchor_posttraining_continuation as engineering')
        a=s.index('def preflight(source,fixture):');b=s.index('\ndef materialize_inputs(',a)
        s=s[:a]+'''def validate_research_plan(plan):
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

'''+s[b:]
        s=s.replace("    if not fixture:raise RuntimeError('Research continuation disabled pending separate implementation review')", "    if not fixture and not CONTINUATION_RESEARCH_ENABLED:raise RuntimeError('Research continuation disabled without authorized entry point')")
    p=R/(new+'.py');assert not p.exists();p.write_text(s,encoding='utf-8')
print('Created v2 continuation, inference gate and auditor; scientific post-checkpoint orchestration preserved.')
