"""Read-only research/archive verification and descriptive result extraction."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'validation/centered_linear_comparison_completion_2026-11-20_01'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    if OUT.exists(): raise RuntimeError('Preserve previous completion verification')
    binding = read(ROOT/'CENTERED_LINEAR_COMPARISON_LAUNCH_BINDING_2026-11-20.json')
    out = ROOT/binding['output']; run = out/'run'; archive = Path(binding['archive'])
    for name, digest in binding['sha256'].items():
        assert sha(ROOT/name) == digest, name
    authorization = read(out/'launch_authorization.json')
    assert authorization['authorized'] and authorization['binding_sha256'] == sha(ROOT/authorization['binding'])
    completion = read(out/'completion.json'); audit = read(out/'saved_artifact_audit.json')
    assert completion['status'] == 'DEVELOPMENT_AUDITED_COMPLETE' and completion['audit_passed'] and audit['passed']
    assert audit['simulation_slots_recorded'] == 51280
    assert audit['historical_ddqn_updates'] == 33696 and audit['historical_bc_updates'] == 3000
    assert sha(run/'manifest.json') == completion['manifest_sha256']
    manifest = read(run/'manifest.json')
    actual = {p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file() and p != run/'manifest.json'}
    assert actual == set(manifest)
    for name, digest in manifest.items(): assert sha(run/name) == digest, name
    freeze = read(run/'PRE_RUN_FREEZE.json')
    for name, digest in freeze['source_sha256'].items():
        assert sha(run/'source_snapshot'/name) == digest and sha(ROOT/name) == digest, name
    ar = read(out/'archive_verification.json')
    assert ar['passed'] and Path(ar['archive']).resolve() == archive.resolve()
    inventory = ar['sha256']
    actual_archive = {p.relative_to(archive).as_posix() for p in archive.rglob('*') if p.is_file()}
    assert actual_archive == set(inventory)|{'archive_verification.json'}
    expected = {'run/'+name for name in manifest}|{'run/manifest.json','completion.json','saved_artifact_audit.json','launch_authorization.json'}
    assert set(inventory) == expected
    for name, digest in inventory.items():
        assert sha(archive/name) == digest, name
        # Manifest already independently binds each original run artifact.
        if name.startswith('run/') and name != 'run/manifest.json':
            assert manifest[name[4:]] == digest, name
        else: assert sha(out/name) == digest, name
    assert sha(out/'archive_verification.json') == sha(archive/'archive_verification.json')
    analysis = read(run/'analysis.json'); fits = analysis['per_fit']; primary = 'centered_linear_bc_ddqn'
    arms = list(fits['0']['absolute'])
    summary = {arm:dict(mean_return=statistics.mean(row['absolute'][arm]['return_value'] for row in fits.values()),
                        mean_deadline_pct=100*statistics.mean(row['absolute'][arm]['deadline'] for row in fits.values())) for arm in arms}
    summary['teacher'] = dict(mean_return=fits['0']['teacher']['return_value'], mean_deadline_pct=100*fits['0']['teacher']['deadline'])
    policy = all(row['contrasts'][arm][metric]>=0 for row in fits.values() for arm in ['old_softplus_bc_ddqn','centered_linear_bc_only'] for metric in ['return_value','deadline'])
    policy = policy and all(statistics.mean(row['contrasts'][arm][metric] for row in fits.values())>0 for arm in ['old_softplus_bc_ddqn','centered_linear_bc_only'] for metric in ['return_value','deadline'])
    values = analysis['value_metrics']
    value = all(values[f'{primary}/{fit}']['nrmse']<values[f'old_softplus_bc_ddqn/{fit}']['nrmse'] and abs(values[f'{primary}/{fit}']['bias'])<abs(values[f'old_softplus_bc_ddqn/{fit}']['bias']) for fit in fits)
    assert (policy,value,policy and value)==(analysis['policy_screen'],analysis['value_screen'],analysis['promising'])
    assert audit['recorded_promising'] == (policy and value)
    checkpoints = list(run.glob('*/fit_*/final.pt')); assert len(checkpoints)==21
    timing = {arm:dict(median_ms_min=min(audit['timing'][f'{arm}/{i}']['median_ns']/1e6 for i in range(3)),
                      median_ms_max=max(audit['timing'][f'{arm}/{i}']['median_ns']/1e6 for i in range(3))) for arm in arms}
    record = dict(status='AUDIT_AND_ARCHIVE_INDEPENDENTLY_REVERIFIED',
                  manifest_files_rehashed=len(manifest), archive_files_rehashed=len(inventory)+1,
                  slots=51280, student_training_slots=36000, td_updates=33696, bc_updates=3000, final_checkpoints=21,
                  source_binding_verified=True, authorization_verified=True,
                  policy_screen=policy, value_screen=value, promising=policy and value,
                  mean_results=summary, per_fit=fits, value_metrics=values,
                  primary_value_distributions={str(i):audit['value_distributions'][f'{primary}/{i}'] for i in range(3)},
                  timing_median_ms_ranges=timing,teacher_timing=audit['timing']['teacher'], storage=audit['storage'],
                  evidence_sha256={name:sha(out/name) for name in ['completion.json','saved_artifact_audit.json','archive_verification.json','run/analysis.json']},
                  archive=str(archive), new_simulator_steps=0,new_optimizer_updates=0,word_thesis_changed=False,
                  limitation='Rehashed sources, saved data and independent archive; recomputed screen logic. Relies on completed full auditor for policy/target recomputation; no historical gradient replay.')
    OUT.mkdir(parents=True)
    (OUT/'verification.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:record[k] for k in ['status','manifest_files_rehashed','archive_files_rehashed','policy_screen','value_screen','promising','mean_results','timing_median_ms_ranges','storage']},indent=2))


if __name__=='__main__': main()
