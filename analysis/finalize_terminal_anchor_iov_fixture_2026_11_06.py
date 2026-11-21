"""Preserve and summarize the passing simulator fixture; no new training."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/terminal_anchor_iov_fixture_2026-11-06_02'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')

def main():
    tests=read(ROOT/'validation/terminal_anchor_iov_fixture_tests_2026-11-06.json')
    assert tests['passed'] and tests['tests']==3
    for name,digest in tests['source_sha256'].items():assert sha(ROOT/name)==digest,name
    audit=read(OUT/'saved_artifact_audit.json');result=read(OUT/'verification.json')
    assert audit['state']=='passed_paired_iov_engineering_audit' and audit['episodes_checked']==54
    assert audit['export_equivalence_checks_replayed']==1536 and audit['timing_rows_recomputed']==4608
    for name,digest in read(OUT/'run_manifest.json')['files'].items():assert sha(OUT/name)==digest,name
    assert result['preserved_manifest_sha256']==sha(ROOT/'development_runs/controller_v3_development_2026-11-06_01/run_manifest.json')
    metrics=read(OUT/'inference/timings.json')['metrics']
    lines=['TERMINAL-ANCHOR PAIRED IOV INTEGRATION REVIEW','Date: November 6, 2026','',
      'OUTCOME','The paired IoV engineering fixture and saved-artifact audit passed.',
      'Three new integration tests passed. The prior eight mechanics tests remain',
      'unchanged; they were not rerun here. No research experiment was launched.',
      'These short engineering trajectories are excluded from policy-quality',
      'evidence, model selection, threshold tuning and thesis performance claims.','',
      'FROZEN FIXTURE','TERMINAL_ANCHOR_IOV_FIXTURE_PROTOCOL_2026-11-06.json',
      'Two synthetic model initializations, six correction/control variants,',
      'two real-simulator training episodes per fit,40 slots per episode:',
      '960 training simulator slots,876 optimizer updates,12 final checkpoints.',
      'Every fit receives paired task seeds and the same exploration schedule.',
      'Terminal sampling has its own random stream. All checkpoints are frozen',
      'before held-out engineering workloads. Original IoV dynamics and reward',
      'equations are reused without modifying simulator or historical model files.',
      'The48-row engineering replay wraps and terminal index eviction is checked.',
      'Shortened warmup/batch/target-copy intervals are fixture settings only.','',
      'AUDIT','The auditor checked54 episodes:24 training,24 controller evaluation,',
      'and6 heuristic episodes. It checked task/workload pairing, actor-stream',
      'pairing, transition arrays, real reward/task accounting, complete greedy',
      'evaluation actions and online/target Q, frozen checkpoint hashes, update',
      'counts, anchor-pool membership and loss arithmetic. Evaluation checks do',
      'not rerun training or reconstruct all historical gradients. The fixture',
      'itself checks weight invariance around evaluation.',
      'All sources and artifacts are covered by hashes; tampering is rejected.',
      'Development/final CLI execution and terminal deployment decoding are blocked.','',
      'DEPLOYMENT AND TIMING','All12 final models were exported with the existing ORT neural-core route',
      'and native decoder. The complete wrapper includes finite/state-shape/time',
      'grid checks and horizon scaling. All1,536 eager/export decisions agree',
      'exactly; Q agrees within atol1e-6 and rtol1e-5. The auditor independently',
      'reloads exports and replays those checks. It does not retime models.',
      'Graphs, extra arrays, fixed128 engineering inputs and ordered timing',
      'schedule freeze before the first timed call. Three timing rounds x128',
      'calls x12 paths yield4,608 rows; all statistics are recomputed from them.',
      'ORT CPU and Numba use single-thread execution. Reported costs are specific',
      'to these short-trained checkpoints, inputs, machine and runtime route.',
      'No speed superiority versus the wide reference is tested in this fixture.','',
      'arm/fit | FP32 parameter bytes | complete median ms | p95 ms | training seconds']
    for key,m in sorted(metrics.items()):
        lines.append(f"{key} | {m['parameter_bytes']} | {m['median_ms']:.5f} | {m['p95_ms']:.5f} | {result['counts'][key]['training_seconds']:.3f}")
    lines+=['','TRAINING/REPLAY COST LIMITS',
      'Training durations include replay sampling, simulator episodes and training',
      'episode saves. Per-update wall times are recorded in optimizer traces.',
      'Replay payload bytes and terminal index counts are recorded per fit.',
      'Python container overhead, interpreter baseline and peak process memory',
      'are excluded. Do not call these figures complete peak-memory measurements.',
      'Anchor variants add forward/backward work while keeping parameter storage',
      'unchanged. This fixture has no untouched old-objective control or wide128',
      'benchmark, so it does not establish the cost benefit of the new design.','',
      'PRESERVATION','Attempt _01 stopped at an export metadata reporting error (duplicate',
      'parameter-count key). Its files and failed test evidence remain preserved.',
      'Attempt _02 passed after that reporting fix; no research criteria or models',
      'were tuned from fixture performance. The original440,000-slot run and its',
      'failed performance gate remain unchanged. Final evaluation seeds untouched.','',
      'NEXT STEP','Prepare and review the full prospective research protocol and its runner/',
      'auditor integration. Include the original eleven arms, the six declared',
      'corrected arms and required heuristics, fresh collision-audited development',
      'seeds, candidate-specific value screening with explicit prospective scope,',
      'all numeric return/deadline/cost thresholds, and final-only checkpoints.',
      'Test that complete research workflow on a reduced isolated fixture, including',
      'all comparator deployment paths and failure cases, before a separate launch',
      'decision. The proposed680,000-slot experiment has NOT been enabled or run.',
      'Keep literature novelty assessment separate from implementation verification.',
      'No claim is established that terminal anchoring fixes deadline performance.','',
      'Word thesis unchanged. Cloud and charging stations excluded. Security/trust',
      'remains deferred to contribution two. No final experiment launched.','',
      'ARTIFACTS',str(OUT.relative_to(ROOT)),
      'validation/terminal_anchor_iov_fixture_tests_2026-11-06.json',
      'run_terminal_anchor_iov_fixture.py','test_terminal_anchor_iov_fixture.py']
    report=ROOT/'TERMINAL_ANCHOR_IOV_INTEGRATION_REVIEW_2026-11-06.txt'
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    tracker=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json';backup=ROOT/'validation/project_status_before_iov_anchor_fixture_2026-11-06.json'
    assert not backup.exists();shutil.copy2(tracker,backup)
    s=read(tracker);s['current_workstream']='terminal_anchor_paired_iov_integration_passed_no_research'
    s['next']='review_and_freeze_full_corrected_comparison_then_runner_auditor_fixture_before_launch'
    s['terminal_anchor_iov_fixture_2026_11_06']=dict(report=report.name,tests=3,artifact_audit=audit['state'],
        training_simulator_slots=960,optimizer_updates=876,checkpoints=12,episodes_audited=54,
        export_decisions_audited=1536,timing_rows_recomputed=4608,research_launched=False,thesis_changed=False)
    write(tracker,s)
    files=[x for x in OUT.rglob('*') if x.is_file()]+[ROOT/name for name in tests['source_sha256']]+[
        report,backup,Path(__file__),ROOT/'validation/terminal_anchor_iov_fixture_tests_2026-11-06.json',
        ROOT/'validation/terminal_anchor_iov_fixture_failed_attempt_01_tests.json']
    write(ROOT/'validation/terminal_anchor_iov_completion_2026-11-06.json',dict(state='paired_iov_engineering_passed_no_research',
        files={str(x.relative_to(ROOT)).replace('\\','/'):sha(x) for x in files}))
    print(json.dumps(dict(state='paired_iov_engineering_passed_no_research',report=report.name,
        audit=audit,primary_costs={k:v for k,v in metrics.items() if k.startswith('price_both')}),indent=2))

if __name__=='__main__':main()
