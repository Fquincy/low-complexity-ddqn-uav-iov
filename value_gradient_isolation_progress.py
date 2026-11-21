"""Read saved progress only; never imports a model or launches a process."""
import argparse
import json
from pathlib import Path


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def snapshot(output):
    out=Path(output);root=out/'run';plan=read(root/'runtime_plan.json')
    progress=read(out/'progress.json') if (out/'progress.json').exists() else dict(stage='starting')
    checkpoint_count=len(list(root.glob('*/fit_*/final.pt')))
    deployment_count=len(list(root.glob('*/fit_*/deployment_weights.pt')))
    counts=[read(p) for p in root.glob('*/fit_*/counts.json')]
    fits=plan['fits'];policies=9*fits+1;td_per_fit=40*plan['online_episodes']-plan['warmup']
    saved=dict(demo_slots=40*len(list(root.glob('demo_*/episode_*.json'))),
        online_slots=40*len(list(root.glob('*/fit_*/train_*.json'))),
        evaluation_slots=40*len(list((root/'evaluation').rglob('episode_*.json'))),
        final_checkpoints=checkpoint_count,deployment_exports=deployment_count,completed_fit_bc_updates=sum(v['bc_updates'] for v in counts),
        completed_fit_td_updates=sum(v['updates'] for v in counts),
        completed_fit_retention_batches=sum(v['retention_batches'] for v in counts))
    saved['total_slots']=saved['demo_slots']+saved['online_slots']+saved['evaluation_slots']
    expected=dict(demo_slots=40*10*fits,online_slots=40*plan['online_episodes']*8*fits,
        evaluation_slots=40*plan['eval_episodes']*policies,final_checkpoints=9*fits,deployment_exports=9*fits,
        completed_fit_bc_updates=8*fits*plan['bc_steps'],completed_fit_td_updates=8*fits*td_per_fit,
        completed_fit_retention_batches=4*fits*td_per_fit)
    expected['total_slots']=expected['demo_slots']+expected['online_slots']+expected['evaluation_slots']
    stage=progress['stage'];audit_complete=False;archive_complete=False
    if (out/'completion.json').exists():
        audit_complete=read(out/'completion.json')['audit_passed']
        if audit_complete:stage='AUDITED_COMPLETE_ARCHIVE_NOT_VERIFIED'
    elif stage=='artifacts_complete':stage='FULL_ARTIFACT_AUDIT_RUNNING_OR_PENDING'
    if (out/'archive_verification.json').exists():
        archive_complete=read(out/'archive_verification.json').get('passed') is True
        if archive_complete and audit_complete:stage='AUDITED_AND_ARCHIVED_COMPLETE'
    if (out/'failure.txt').exists():stage='STOPPED_INSPECT_FAILURE_RECORD'
    if any(saved[k]>expected[k] for k in expected):raise ValueError('Saved counters exceed frozen plan')
    return dict(kind=plan['kind'],stage=stage,progress=progress,saved=saved,expected=expected,
        audit_complete=audit_complete,archive_receipt_present=archive_complete,
        limitation='Counters advance only at saved episodes/completed fits; receipt status does not independently rehash an archive.')


def format_text(row):
    pairs=[('demo_slots','Teacher demonstration slots'),('online_slots','Student training slots'),
        ('evaluation_slots','Evaluation slots'),('total_slots','Total simulated slots'),
        ('final_checkpoints','Final training checkpoints'),('deployment_exports','Deployment exports'),('completed_fit_bc_updates','BC updates in completed fits'),
        ('completed_fit_td_updates','TD updates in completed fits'),
        ('completed_fit_retention_batches','Retention batches in completed fits')]
    return '\n'.join([f"Kind: {row['kind']}",f"Stage: {row['stage']}"]+
        [f"{label}: {row['saved'][key]} / {row['expected'][key]}" for key,label in pairs]+[row['limitation']])+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--json',action='store_true');a=p.parse_args()
    row=snapshot(a.output);print(json.dumps(row,indent=2) if a.json else format_text(row),end='\n' if a.json else '')

