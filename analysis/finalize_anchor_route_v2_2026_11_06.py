"""Record completed synthetic route integration; never launch research."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/anchor_research_route_v2_2026-11-06_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def main():
    tests=read(ROOT/'validation/anchor_research_route_v2_tests_2026-11-06.json')
    audit=read(OUT/'saved_artifact_audit.json')
    cost=read(OUT/'FROZEN_CHECKPOINT_INFERENCE.json')
    assert tests['passed'] and tests['tests']==6
    assert audit['checkpoints_checked']==34 and audit['episodes_checked']==244
    assert audit['optimizer_updates_checked']==2482
    assert cost['all_fit_cost_screens_pass']
    for name,digest in tests['source_sha256'].items():assert sha(ROOT/name)==digest,name
    # Supplement the fixture's declared sources with the prototype's transitive
    # model factory, checked against its pre-existing immutable evidence.
    prior=read(ROOT/'validation/terminal_anchor_engineering_completion_2026-11-06.json')
    factory=ROOT/'run_v3_research_gate_v2.py'
    assert sha(factory)==prior['files'][factory.name]
    supplement=OUT/'supplemental_provenance';supplement.mkdir(exist_ok=False)
    shutil.copy2(factory,supplement/factory.name)
    write(supplement/'verification.json',dict(source=factory.name,sha256=sha(factory),
        prior_manifest='validation/terminal_anchor_engineering_completion_2026-11-06.json',
        prior_manifest_sha256=sha(ROOT/'validation/terminal_anchor_engineering_completion_2026-11-06.json'),
        limitation='Supplemental post-run provenance check; not part of original pre-timing source freeze. Include directly before research launch.'))
    lines=['TERMINAL-ANCHOR RUNTIME ROUTE V2 INTEGRATION REVIEW','Date: November 6, 2026','','OUTCOME',
        'Six tests passed, including the complete two-initialization seventeen-arm',
        'synthetic workflow and independent saved-artifact auditor. The verified',
        'single-validation adapter is installed in a separately versioned runner.',
        'Original runner, protocol, amendment and previous results remain preserved.',
        '', 'VERIFICATION',
        '2,720 synthetic training slots; 2,482 updates; 306 target copies;',
        '34 final checkpoints; 244 episodes audited; 4,352 complete exported',
        'action checks; 21,760 timing rows recomputed. All initializations retained.',
        'Tests cover source amendment, route dispatch and terminal rejection,',
        'replay equivalence, prospective screen calculations, checkpoint-freeze',
        'tamper rejection and CLI/direct-API development/final launch guards.',
        'Synthetic seeds are isolated from reserved development/final task blocks.',
        '', 'COMPLETE ACTION-SELECTION COST',
        'fit | primary median ms | wide128 median ms | primary p95 ms']
    for fit in range(2):
        p=cost['metrics']['price_both'][fit];w=cost['metrics']['additive128_unbounded'][fit]
        lines.append(f"{fit} | {p['median_ms']:.6f} | {w['median_ms']:.6f} | {p['p95_ms']:.6f}")
    lines+=['Primary parameter storage: 68,264 bytes; compact64: 194,944 bytes.',
        'Storage, median and p95 screens passed in both synthetic initializations.',
        'No favorable retiming or checkpoint selection was performed.',
        'Synthetic empirical policy/value screens did not all pass. This fixture',
        'verifies implementation and auditing; it is not research quality evidence.',
        '', 'PROVENANCE LIMIT',
        'The prototype imports run_v3_research_gate_v2.py as a model factory.',
        'Its bytes match the earlier terminal-anchor engineering completion',
        'manifest and are now separately snapshotted in supplemental_provenance.',
        'This supplemental check occurred after the fixture. Before research,',
        'include this transitive source directly in the runner pre-run source',
        'freeze and verify that addition without changing scientific settings.',
        '', 'NEXT DECISION',
        'Perform a separate launch-readiness review of the frozen seventeen-arm',
        '680,000-slot protocol, runtime amendment, integrated test evidence and',
        'source coverage. Research and final modes remain disabled; no launch',
        'is authorized by this engineering result. No development/final reserved',
        'seeds were consumed. No rewards, seeds, thresholds, optimizers or model',
        'definitions were tuned from the fixture. Word thesis unchanged.',
        'Cloud/charging stations excluded; security/trust deferred.',
        '', 'FILES','run_anchor_research_route_v2.py','audit_anchor_research_route_v2.py',
        'test_anchor_research_route_v2.py',str(OUT.relative_to(ROOT))]
    report=ROOT/'TERMINAL_ANCHOR_RUNTIME_ROUTE_V2_INTEGRATION_REVIEW_2026-11-06.txt'
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    tracker=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json'
    backup=ROOT/'validation/project_status_before_anchor_route_v2_2026-11-06.json'
    assert not backup.exists();shutil.copy2(tracker,backup)
    status=read(tracker)
    status['current_workstream']='anchor_runtime_route_v2_integrated_synthetic_audit_passed_no_research'
    status['next']='separate_launch_readiness_review_including_transitive_factory_freeze_before_explicit_launch'
    status['anchor_runtime_route_v2_integration']=dict(report=report.name,tests=6,
        fixture=str(OUT.relative_to(ROOT)),engineering_cost_screens_passed=True,
        research_launched=False,final_launched=False,word_thesis_changed=False,
        supplemental_factory_provenance_verified=True,factory_in_pre_run_freeze=False)
    write(tracker,status)
    paths=[p for p in OUT.rglob('*') if p.is_file()]
    paths += [ROOT/n for n in tests['source_sha256']]
    paths += [report,Path(__file__),backup,ROOT/'validation/anchor_research_route_v2_tests_2026-11-06.json',ROOT/'analysis/integrate_anchor_runtime_route_2026_11_06.py']
    manifest=ROOT/'validation/anchor_runtime_route_v2_completion_2026-11-06.json'
    files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}
    write(manifest,dict(state='integrated_synthetic_workflow_passed_no_research',files=files))
    for name,digest in files.items():assert sha(ROOT/name)==digest
    print(json.dumps(dict(report=report.name,manifest_entries=len(files),tests=6,cost_pass=True,research_launched=False)))
if __name__=='__main__':main()
