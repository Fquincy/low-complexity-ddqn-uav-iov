"""Seal prototype correctness and failed runtime gate without retiming."""
import json,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'validation/deadline_screen_engineering_2026-11-11_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def main():
    tests=read(O/'tests.json');audit=read(O/'saved_artifact_audit.json');bench=read(O/'benchmark.json')
    assert tests['passed'] and tests['tests']==7 and audit['models']==14
    assert audit['actions_replayed']==1792 and audit['timing_rows_recomputed']==8960
    for name,digest in read(O/'ARTIFACT_MANIFEST.json')['files'].items():assert sha(O/name)==digest,name
    report=R/'DEADLINE_SCREEN_PROTOTYPE_ENGINEERING_REVIEW_2026-11-11.txt'
    lines=['CONSERVATIVE DEADLINE SCREEN: PROTOTYPE ENGINEERING REVIEW','Date: November11,2026','','OUTCOME',
        'Correctness tests and saved-artifact audit passed. Runtime readiness failed:',
        'both synthetic initializations failed median<=wide128 andp95<=1ms.',
        'Do not launch research training from this gate. Preserve these timings.',
        'The next step is a declared runtime implementation optimization, with',
        'the same interval decisions, model definitions and acceptance thresholds.',
        '', 'IMPLEMENTATION',
        'deadline_screen_prototype.py provides the observed-state interval screen,',
        'shared masked greedy/exploration selection and original finite-horizon',
        'DDQN target/terminal-anchor objective. All four remote task choices and',
        'all movement choices remain available. No remote deadline guarantee is',
        'implied. The score tables are separable; eager and native selection use',
        'the same computed branch scores and original first-index tie behavior.',
        'Price/no-price/plain24, compact64, wide128 and adaptedBDQ receive exactly',
        'the same state-derived screening. Compact and plain inherited horizon/',
        'anchor configurations are retained. No new learned weights or features.',
        'The plain24 control without the horizon/anchor wrapper is also supported.',
        '', 'CONSERVATIVE NUMERICS',
        'Adjacent outward FP32 intervals contain the source normalized task',
        'cycle/deadline values. Physical scaling, local CPU division and epsilon',
        'addition use outward FP64 rounding. Exclude local only when the lower',
        'service bound exceeds the upper deadline bound plus1e-10seconds.',
        'Retain ambiguous near-boundary actions; therefore some genuinely',
        'infeasible local choices may remain when FP32 cannot certify them.',
        'No claim that every mathematically infeasible task is excluded is made.',
        'All robustly certified infeasible choices are excluded from selection.',
        'The proof assumes the frozen observation normalization and configuration;',
        'different frequencies/scales require an explicit configuration extension.',
        '', 'TEST EVIDENCE',
        'Seven tests passed.100,000 randomized exact-valued cycle/deadline fixtures',
        'verified interval containment and no false local infeasibility exclusion.',
        '3,000 tight deadline-boundary/equality perturbations remained admitted.',
        'Nonempty allowed sets, all remote/motion choices retained, invalid-state',
        'rejection, deterministic tie behavior and terminal decode/explore rejection',
        'were checked. All seven model families matched eager/native decisions',
        'and their chosen-value evaluation; parameter payload stayed unchanged.',
        'Exploration uses supplied per-branch uniforms without consuming extra RNG',
        'draws; paired experiments can remap the same source uniforms. This tests',
        'the helper, not integration with a future research runner action stream.',
        'A spy target test verifies next-state selection by the online model and',
        'evaluation of that same action by the target, with no terminal decode.',
        'Synthetic loss backward verifies finite online gradients and no target',
        'gradients; the inherited unit terminal-anchor loss is included. No optimizer',
        'step, simulator transition or research checkpoint loading occurred.',
        '', 'FIXED SYNTHETIC BENCHMARK',
        'Protocol was written before measurement. Seven families x two synthetic',
        'initializations;128 fixed synthetic observations per model. Saved14 random',
        'model states and source/payload hashes before benchmark. Five randomized',
        'rounds x128calls,32warmups per model/round. Complete runtime includes',
        'input/interval checks, original neural core, score tables, native masked',
        'selection, value/horizon handling and final action copy. No favorable',
        'retiming or fit selection. Synthetic weights are not trained policies.',
        'fit | primary median ms | wide128 median ms | primary p95 ms | parameter bytes']
    for fit in range(2):
        p=bench['metrics'][f'price_both_fit_{fit:02d}'];w=bench['metrics'][f'additive128_unbounded_fit_{fit:02d}']
        lines.append(f"{fit} | {p['median_ms']:.6f} | {w['median_ms']:.6f} | {p['p95_ms']:.6f} | {p['parameter_bytes']}")
    lines +=['Primary parameter bytes68,264; compact64194,944:storage screen passes.',
        'Primary carries20 existing buffer bytes, with no added trainable tensors.',
        'Temporary masks/score tables, framework/process memory and deployment',
        'artifact size are not included in online parameter bytes. Those quantities',
        'must not be conflated. Machine/load dependence limits absolute timings.',
        'This implementation computes full PyTorch score tables and repeats some',
        'validation/horizon work. These are optimization candidates; no profiling',
        'result here proves how much of the latency they explain.',
        '', 'AUDIT AND PRESERVATION',
        'Independent auditor reloaded14 models, replayed1,792 saved decisions and',
        'Q values, rechecked masks and source/model hashes, and recomputed8,960',
        'raw timing rows, randomized order and all screens without new timings.',
        'The first auditor invocation hit a Windows-separator lookup error in its',
        'model-payload map. Original auditor source was preserved; only lookup-key',
        'normalization was corrected. The completed rerun passed on the unchanged',
        'benchmark artifacts. Measurements and models were not regenerated.',
        'A scalar conversion warning in a test does not affect loss or gradients.',
        '', 'NEXT ENGINEERING STEP',
        'Predeclare one efficient interval/score/decoder implementation, retaining',
        'FP32 core arithmetic and conservative interval semantics. Optimize full',
        'action selection equally for compact controls; measure every required',
        'path once under a fixed protocol. Require exact action/mask equivalence,',
        'near-tie/boundary tests and unchanged model storage before timing.',
        'Do not start another research run until cost and complete reduced runner',
        'integration pass. The global value-calibration failure of the prior trained',
        'controller remains unresolved; passing synthetic tests does not fix it.',
        'Broader UAV-IoV literature review and supervisor novelty/scope discussion',
        'remain open. No claim that screening alone is a new algorithm is made.',
        '', 'SCOPE',
        'No research evaluation/training, heldout or final tasks, reward changes,',
        'seed reservation or Word thesis edits. Original failed and completed runs',
        'remain preserved. Cloud/charging stations excluded; security/trust deferred.']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    tracker=R/'LOW_COMPLEXITY_PROJECT_STATUS.json';backup=O/'PROJECT_STATUS_BEFORE_REVIEW.json'
    assert not backup.exists();shutil.copy2(tracker,backup);s=read(tracker)
    s['current_workstream']='deadline_screen_prototype_correctness_passed_synthetic_runtime_gate_failed'
    s['next']='predeclare_same_semantics_runtime_optimization_before_research_runner_or_training'
    s['deadline_screen_prototype_engineering']=dict(report=report.name,tests=7,tests_passed=True,saved_artifact_audit_passed=True,
        models=14,action_checks=1792,timing_rows=8960,parameter_storage_passed=True,median_and_p95_screens_passed=False,
        new_training_slots=0,new_simulator_slots=0,research_launched=False,word_thesis_changed=False)
    write(tracker,s)
    paths=[p for p in O.rglob('*') if p.is_file()]+[R/n for n in ['deadline_screen_prototype.py','test_deadline_screen_prototype.py','benchmark_deadline_screen_prototype.py','audit_deadline_screen_prototype.py','DEADLINE_SCREEN_ENGINEERING_PROTOCOL_2026-11-11.json']]+[report,Path(__file__)]
    files={str(p.relative_to(R)).replace('\\','/'):sha(p) for p in paths}
    write(R/'validation/deadline_screen_engineering_completion_2026-11-11.json',dict(state='correctness_passed_runtime_failed_no_research',files=files))
    for name,digest in files.items():assert sha(R/name)==digest,name
    print(json.dumps(dict(report=report.name,tests=7,audit='passed',runtime_gate='failed_both_fits',research_launched=False)))
if __name__=='__main__':main()
