"""Archive completed eleven-arm research preparation without launching it."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'validation/controller_v3_research_fixture_2026-11-05_04'
TESTS=ROOT/'validation/v3_research_preparation_tests_2026-11-05.json'
REPORT=ROOT/'LOW_COMPLEXITY_V3_RESEARCH_PREPARATION_REVIEW_2026-11-05.txt'
STATUS=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json'
ENGINEERING_MANIFEST=ROOT/'validation/controller_v3_runtime_compatibility_2026-11-05_01_COMPLETION_MANIFEST.json'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

tests=read(TESTS);verification=read(FIXTURE/'verification.json');audit=read(FIXTURE/'saved_artifact_audit.json')
inference=read(FIXTURE/'FROZEN_CHECKPOINT_INFERENCE.json');protocol=read(ROOT/'LOW_COMPLEXITY_V3_RESEARCH_PROTOCOL_2026-11-05.json')
assert tests['passed'] and tests['tests']==7
assert verification['state']=='completed_synthetic_full_fixture_not_research'
assert verification['training_slots']==440 and verification['optimizer_updates']==363 and verification['target_copies']==44
assert audit['state']=='passed_v3_synthetic_orchestration_audit'
assert audit['frozen_inference_audit']['complete_actions_replayed']==1408
assert audit['frozen_inference_audit']['timing_rows_recomputed']==7040
assert inference['all_fit_cost_screens_pass'] and inference['equivalence_models']==11 and inference['equivalence_actions']==1408
assert not verification['research_training_launched'] and not audit['research_quality_evidence']
for name,h in read(FIXTURE/'run_manifest.json')['files'].items():assert sha(FIXTURE/name)==h,name

prior=read(ENGINEERING_MANIFEST)
for name,h in prior['files'].items():
    path=FIXTURE/'PROJECT_STATUS_BEFORE.json' if name=='LOW_COMPLEXITY_PROJECT_STATUS.json' else ROOT/name
    assert sha(path)==h,name

status_before=FIXTURE/'PROJECT_STATUS_BEFORE.json'
status=read(STATUS);assert sha(STATUS)==sha(status_before)
status['v3_eleven_arm_research_preparation']=dict(
    state='protocol_frozen_runner_and_auditor_passed_synthetic_fixture',
    protocol='LOW_COMPLEXITY_V3_RESEARCH_PROTOCOL_2026-11-05.json',
    protocol_sha256=sha(ROOT/'LOW_COMPLEXITY_V3_RESEARCH_PROTOCOL_2026-11-05.json'),
    arms=protocol['arms'],mechanics_and_orchestration_tests=tests['tests'],
    fixture_state=verification['state'],fixture_audit=audit['state'],fixture_artifacts=str(FIXTURE.relative_to(ROOT)).replace('\\','/'),
    fixture_simulator_slots=verification['training_slots'],fixture_optimizer_updates=verification['optimizer_updates'],
    fixture_target_copies=verification['target_copies'],fixture_checkpoint_count=verification['checkpoint_count'],
    all_checkpoints_frozen_before_evaluation=True,final_weight_equivalence_models=inference['equivalence_models'],
    final_weight_complete_actions_checked=inference['equivalence_actions'],
    final_weight_timing_rows=inference['raw_rows'],
    fixture_inference_screens=inference['screens_per_fit'],
    development_training_budget=protocol['development']['training_slots'],
    development_optimizer_budget=protocol['development']['optimizer_updates'],
    final_training_budget=protocol['final']['training_slots'],final_optimizer_budget=protocol['final']['optimizer_updates'],
    development_research_launched=False,final_research_launched=False,
    policy_quality_evidence=False,return_deadline_claim_established=False,novelty_established=False,
    thesis_sufficiency_established=False,word_thesis_changed=False,
    clouds_in_active_model=False,charging_stations_in_active_model=False,security_trust_in_current_branch=False,
    review=REPORT.name)
status['current_workstream']='contribution_one_eleven_arm_protocol_and_synthetic_engineering_gate_passed_no_research_launch'
status['next']='separately_review_frozen_eleven_arm_protocol_and_fixture_then_issue_explicit_development_launch_instruction'
write(STATUS,status)

out_manifest=ROOT/'validation/v3_research_preparation_2026-11-05_COMPLETION_MANIFEST.json'
files=[p for p in FIXTURE.rglob('*') if p.is_file()]+[REPORT,TESTS,STATUS,ROOT/'LOW_COMPLEXITY_V3_RESEARCH_PROTOCOL_2026-11-05.json',
    ROOT/'run_v3_research_preparation.py',ROOT/'audit_v3_research_preparation.py',ROOT/'test_v3_research_preparation.py',
    ROOT/'controller_v3_amortized.py',ROOT/'analysis/finalize_v3_research_preparation_2026_11_05.py',
    ROOT/'LOW_COMPLEXITY_V3_RUNTIME_COMPATIBILITY_PROTOCOL_v2_2026-11-05.json']
write(out_manifest,dict(state='complete_research_preparation_synthetic_gate_passed_no_launch',
    files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files}))
print(json.dumps(dict(state='complete_research_preparation_synthetic_gate_passed_no_launch',
    tests=tests['tests'],fixture_audit=audit['state'],fixture_slots=verification['training_slots'],
    fixture_updates=verification['optimizer_updates'],fixture_target_copies=verification['target_copies'],
    checkpoints=inference['equivalence_models'],action_checks=inference['equivalence_actions'],
    timing_rows=inference['raw_rows'],development_budget=protocol['development']['training_slots'],
    development_updates=protocol['development']['optimizer_updates'],research_training_launched=False,
    thesis_changed=False,completion_manifest=out_manifest.name,manifest_files=len(files)),indent=2))
