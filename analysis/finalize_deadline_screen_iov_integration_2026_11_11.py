"""Seal paired synthetic integration evidence; no research launch."""
import hashlib,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'validation/deadline_screen_iov_fixture_2026-11-11_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def main():
    tests=read(R/'validation/deadline_screen_iov_fixture_tests_2026-11-11.json');audit=read(O/'saved_artifact_audit.json');v=read(O/'verification.json');cost=read(O/'cost.json')
    assert tests['passed'] and tests['tests']==3 and audit['checkpoints']==14
    assert audit['episodes']==56 and audit['task_records_checked']==67200 and audit['optimizer_rows_checked']==1022
    assert all(all(x[k] for k in ['storage','median','p95']) for x in cost['screens'])
    for name,digest in tests['source_sha256'].items():assert sha(R/name)==digest,name
    for name,digest in read(O/'run_manifest.json')['files'].items():assert sha(O/name)==digest,name
    for name,digest in read(R/'validation/deadline_screen_runtime_optimization_completion_2026-11-11.json')['files'].items():assert sha(R/name)==digest,name
    report=R/'DEADLINE_SCREEN_IOV_INTEGRATION_REVIEW_2026-11-11.txt'
    lines=['DEADLINE SCREEN: PAIRED UAV-IOV INTEGRATION REVIEW','Date: November11,2026','','OUTCOME',
        'The complete paired synthetic training/evaluation fixture and independent',
        'saved-artifact audit passed. Three integration tests passed. Storage and',
        'complete action-selection median/p95 gates passed in both initializations.',
        'No research experiment, reserved development/final evaluation or Word thesis',
        'edit occurred. This is engineering evidence, not a thesis policy result.',
        '', 'FIXED SYNTHETIC PLAN',
        'Protocol declared before running: seven screened model families x two',
        'initializations99811/99817, two40-slot training episodes per model.',
        'Training workload starts998110000/998170000; evaluation998190100/101.',
        'These99-prefix namespaces are isolated engineering seeds, not reserved',
        'research seeds. Real event-driven UAV-IoV simulator with original config.',
        'Fixture-only reduced warmup8,batch4,replay64,target interval8,epsilon40;',
        'Adam1e-4,gamma1,clip10. These reduced settings are not a research protocol.',
        'Only final checkpoints are retained. Training1,120slots/evaluation1,120slots,',
        '56episodes total,14checkpoints,1,022updates,126target copies.',
        'The earlier completed680,000-slot experiment was not resumed or changed.',
        '', 'TRAINING, EXPLORATION AND TARGET PATH',
        'All seven families share the identical observed-state conservative interval',
        'screen. Before EVERY training/evaluation action, the fixture verifies',
        'interval containment against exact simulator task cycles/deadlines and',
        'proves any excluded local task cannot meet its deadline on an empty local',
        'CPU. Actions are required to belong to their state-derived masks.',
        '67,200task records audited, including29,603certified-infeasible local',
        'records; no false exclusion or out-of-mask executed action was found.',
        'All four remote and motion choices remain available. No remote guarantee.',
        'Exploration consumes one epsilon uniform and32branch uniforms every slot,',
        'including warmup/greedy slots, so source RNG draws pair across all arms.',
        'The independent auditor reproduces these draws and allowed-set remapping.',
        'Different policies still generate different queue histories and actions.',
        'Replay RNG indices pair exactly across arms; terminal-anchor draws use a',
        'separate RNG. Pool membership/eviction, sampled indices, copies and finite',
        'loss/gradient/target histories are checked. Mask-aware DDQN selection uses',
        'next-state masks and the target evaluates that selected action. Terminal',
        'transitions skip next-action decoding through the verified target helper.',
        'Price/plain/compact anchored variants retain their inherited terminal loss.',
        'AdaptedBDQ retains its branchwise Huber objective; other unanchored controls',
        'use ordinary scalar DDQN Huber. The branch-loss distinction is explicit,',
        'not attributed to screening or silently changed to a mean-value loss.',
        '', 'TRAINED-WEIGHT INFERENCE AND AUDIT',
        'All14 final checkpoint hashes are frozen before evaluation. Each trained',
        'checkpoint is reconstructed into the original-PyTorch-core/native-screened',
        'runtime. This route does not use an ONNX neural-core export; do not describe',
        'this gate as an ONNX export result. All1,792 fixed synthetic observations',
        'matched eager/runtime joint actions exactly and Q within original tolerance.',
        'The28 real simulator evaluation episodes also match runtime/eager actions',
        'and saved chosen/target values. Weight hashes and train/eval mode snapshots',
        'prove evaluation invariance. Workload fingerprints match every controller',
        'within each paired task block. Full traces pass existing simulator record',
        'and saved-array consistency checks. No live research-state data are used.',
        '', 'COMPLETE ACTION COST',
        'Five fixed randomized rounds,32warmups and128calls per path/round. All',
        '14paths measured once;8,960raw timing rows independently recomputed.',
        'Checkpoint, input and source hashes and order are frozen before timing.',
        'fit | primary median ms | screened wide128 median ms | primary p95 ms']
    for fit in range(2):
        p=cost['metrics'][f'price_both_fit_{fit:02d}'];w=cost['metrics'][f'additive128_unbounded_fit_{fit:02d}']
        lines.append(f"{fit} | {p['median_ms']:.6f} | {w['median_ms']:.6f} | {p['p95_ms']:.6f}")
    lines+=['Primary FP32 parameter payload68,264bytes; compact64194,944bytes.',
        'No added trainable tensors; temporary masks/arrays and framework/process',
        'memory are not parameter payload and are not claimed to meet a device',
        'deployment memory budget. Absolute latency depends on host/load and inputs.',
        'No retiming, favorable checkpoint choice or dropping failed fits occurred.',
        '', 'THREE TESTS',
        '1. Entire synthetic training/evaluation/cost workflow and independent audit.',
        '2. Development/final CLI and direct API rejected before output creation.',
        '3. Existing-output rejection and deliberate saved-summary corruption',
        '   rejection, with original bytes restored. No re-timing in that check.',
        'All passed. Original prototype tests remain preserved as separate evidence.',
        '', 'NEXT RESEARCH-PREPARATION STAGE',
        'Freeze one prospective comparison protocol before another research run:',
        'complete model list (including the unmasked primary and strong screened',
        'controls/heuristics), architecture/anchor/loss distinctions, unchanged or',
        'explicitly amended acceptance screens, fresh disjoint development model',
        'and task blocks, final-checkpoint-only policy, full action-cost procedure',
        'and independent results auditor. Collision-check fresh reservations.',
        'Implement the research runner and test its complete reduced orchestration',
        'before a separate launch decision. This fixture has no research mode.',
        'Do not treat the fixture return/deadline outcomes as training-budget or',
        'model-selection evidence. Value calibration and useful mechanism benefit',
        'remain unproven. The diagnosed48 development tasks are consumed; final',
        'tasks remain untouched. Broader UAV-IoV literature/supervisor review of',
        'the proposed contribution remains open; masking alone is not new.',
        '', 'FILES','DEADLINE_SCREEN_IOV_FIXTURE_PROTOCOL_2026-11-11.json',
        'run_deadline_screen_iov_fixture.py','deadline_screen_iov_episode.py',
        'audit_deadline_screen_iov_fixture.py','test_deadline_screen_iov_fixture.py',
        str(O.relative_to(R)),
        '', 'Cloud/charging stations excluded. Security/trust deferred. Word thesis unchanged.']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    tracker=R/'LOW_COMPLEXITY_PROJECT_STATUS.json';backup=O/'PROJECT_STATUS_BEFORE_REVIEW.json'
    assert not backup.exists();shutil.copy2(tracker,backup);s=read(tracker)
    s['current_workstream']='deadline_screen_paired_simulator_fixture_and_artifact_audit_passed_no_research'
    s['next']='freeze_fair_fresh_seed_screened_research_comparison_and_test_runner_before_separate_launch'
    s['deadline_screen_iov_integration']=dict(report=report.name,tests=3,artifact_audit_passed=True,checkpoints=14,
        synthetic_training_slots=1120,synthetic_evaluation_slots=1120,synthetic_optimizer_updates=1022,
        task_records_checked=67200,certified_infeasible_local_records_checked=29603,engineering_action_checks=1792,
        timing_rows=8960,all_synthetic_cost_screens_passed=True,research_launched=False,word_thesis_changed=False)
    write(tracker,s)
    paths=[p for p in O.rglob('*') if p.is_file()]+[R/n for n in tests['source_sha256']]+[report,Path(__file__),R/'validation/deadline_screen_iov_fixture_tests_2026-11-11.json']
    files={str(p.relative_to(R)).replace('\\','/'):sha(p) for p in paths}
    write(R/'validation/deadline_screen_iov_integration_completion_2026-11-11.json',dict(state='paired_synthetic_training_inference_and_audit_passed_no_research',files=files))
    for name,digest in files.items():assert sha(R/name)==digest,name
    print(json.dumps(dict(report=report.name,tests=3,audit_passed=True,checkpoints=14,all_cost_screens_passed=True,research_launched=False)))
if __name__=='__main__':main()
