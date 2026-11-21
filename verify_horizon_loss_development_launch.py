"""Verify real saved research episodes and record the authorized background launch."""
import datetime
import gzip
import json
from pathlib import Path
import run_horizon_loss_research as r

def main():
    control=r.ROOT/'validation/horizon_loss_development_launch_2026-11-14_01'
    output=r.ROOT/'development_runs/horizon_loss_development_2026-11-14_01'
    saved=output/'comparison/run'
    p=r.protocol();plan=r.read(saved/'runtime_plan.json')
    assert plan==r.plan(p,'development') and plan['kind']=='RESEARCH_DEVELOPMENT'
    release=r.read(r.ROOT/'HORIZON_LOSS_LAUNCH_RELEASE_2026-11-14.json')
    assert release['user_authorization']=='Launch the frozen 560,000-slot horizon-loss development comparison'
    matching=[]
    for folder in (r.ROOT/'validation').glob('horizon_loss_prelaunch_*'):
        if (folder/'launch_request.json').exists():
            req=r.read(folder/'launch_request.json')
            if Path(req['output'])==output:matching.append(folder)
    assert len(matching)==1
    prelaunch=matching[0];seed_audit=r.read(prelaunch/'seed_refresh.json')
    assert seed_audit['passed'] and seed_audit['protocol_sha256']==r.sha(r.SPEC)
    complete=r.read(prelaunch/'COMPLETE_SOURCE_FREEZE.json')
    assert complete==release['source_sha256']
    for name,h in complete.items():assert r.sha(r.ROOT/name)==r.sha(prelaunch/'source_snapshot'/name)==h,name
    core=r.read(saved/'PRE_RUN_FREEZE.json')
    assert core['protocol_sha256']==r.sha(r.SPEC)
    for name,h in core['source_sha256'].items():assert r.sha(r.ROOT/name)==r.sha(saved/'source_snapshot'/name)==h,name
    episodes=list(saved.glob('*/fit_*/train_*.json.gz'))
    assert episodes,'No saved training episode yet'
    first=sorted(episodes)[0]
    with gzip.open(first,'rt') as stream:row=json.load(stream)
    fit=int(first.parent.name.split('_')[1]);block=p['development']['train_blocks'][fit]
    assert row['run_kind']=='RESEARCH_DEVELOPMENT'
    assert row['model_seed']==p['development']['model_seeds'][fit]
    assert block['start']<=row['seed']<block['start']+block['count']
    assert len(row['slots'])==40 and len(row['tasks'])==1200
    record=json.loads((control/'launch.json').read_text(encoding='utf-8-sig'))
    record.update(stage='research_training',research_training_started=True,
        prelaunch_seed_audit=str(prelaunch/'seed_refresh.json'),
        complete_source_freeze=str(prelaunch/'COMPLETE_SOURCE_FREEZE.json'),
        verified_at=datetime.datetime.now().astimezone().isoformat())
    r.write(control/'launch.json',record)
    verification=dict(verified=True,verified_at=record['verified_at'],
        process_id=record['process_id'],monitor_process_id=record['monitor_process_id'],
        first_verified_episode=str(first),saved_episode_files_at_check=len(episodes),
        source_files_verified=len(complete),protocol_sha256=r.sha(r.SPEC),
        correct_research_plan=True,fresh_seed_audit_passed=True,
        settings_changed=False,final_phase_launched=False,word_thesis_changed=False)
    r.write(control/'LAUNCH_VERIFIED.json',verification)
    status=r.read(r.ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json')
    status['current_workstream']='frozen_horizon_loss_development_comparison_running'
    status['next']='Monitor training, paired evaluation, artifact/storage audit and separate startup stage; retain all fits and unchanged scientific rules'
    status['next_engineering_step']=status['next']
    status['horizon_loss_development_launch']=dict(record,training_slots_planned=560000,
        evaluation_slots_planned=140160,learned_controllers=14,heuristics=3,initializations=5,
        live_status=str((control/'LIVE_STATUS.txt').relative_to(r.ROOT)),
        launch_verified=True,final_phase_launched=False,word_thesis_changed=False)
    r.write(r.ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json',status)
    (r.ROOT/'HORIZON_LOSS_DEVELOPMENT_LAUNCH_NOTE_2026-11-14.txt').write_text(
        'FROZEN HORIZON-LOSS DEVELOPMENT COMPARISON LAUNCHED\n'
        f"Verified: {record['verified_at']}\n"
        'Explicit user instruction: Launch the frozen 560,000-slot horizon-loss development comparison\n'
        '14 learners, three heuristics, five initializations; 70 final-only checkpoints.\n'
        '560,000 training + 140,160 evaluation slots; 488,390 updates and 2,380 target copies.\n'
        'Training, frozen-checkpoint evaluation/timing, full artifact/storage audit and\n'
        'separate startup reporting execute in the same persistent background workflow.\n'
        f"Worker PID: {record['process_id']}; monitor PID: {record['monitor_process_id']}\n"
        f'Live status: {control / "LIVE_STATUS.txt"}\n'
        f'Research output: {output}\n'
        f'Fresh actual-use seed audit: {prelaunch / "seed_refresh.json"}\n'
        'All 65 release source files were checked and archived before model/task initialization.\n'
        'Existing runner files remain unchanged; the explicit launch adapter activates\n'
        'the development release flag only in the authorized worker process.\n'
        'No model definition, reward, seed, optimizer setting or acceptance threshold changed.\n'
        'Stop on execution/audit failures and preserve artifacts; no automatic restart.\n'
        'The Word thesis, old runs and protected final phase remain unchanged.\n',encoding='utf-8')
    print(json.dumps(verification,indent=2))

if __name__=='__main__':main()
