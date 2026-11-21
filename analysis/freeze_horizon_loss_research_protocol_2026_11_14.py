"""Prospective design/seed freeze only. No simulator, torch or training imports."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
import freeze_deadline_screen_research_protocol_2026_11_12 as historical

NAME = 'HORIZON_LOSS_RESEARCH_PROTOCOL_2026-11-14.json'
REVIEW = 'HORIZON_LOSS_RESEARCH_PROTOCOL_REVIEW_2026-11-14.txt'
OUT = ROOT / 'validation/horizon_loss_research_protocol_freeze_2026-11-14_01'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert not (ROOT / NAME).exists() and not (ROOT / REVIEW).exists(), 'Preserve existing freeze'
    OUT.mkdir(parents=True, exist_ok=False)
    base = read(ROOT / 'DEADLINE_SCREEN_RESEARCH_PROTOCOL_2026-11-12.json')
    fixture = read(ROOT / 'HORIZON_LOSS_IOV_FIXTURE_PROTOCOL_2026-11-14.json')
    amendment = read(ROOT / 'DEADLINE_SCREEN_HORIZON_LOSS_COMPARISON_AMENDMENT_2026-11-14.json')
    p = copy.deepcopy(fixture)
    p.update(id='horizon-loss-fourteen-learner-development-v1', date='2026-11-14',
             status='frozen_research_design_pending_executable_binding_and_launch_review',
             comparison_file='DEADLINE_SCREEN_HORIZON_LOSS_COMPARISON_AMENDMENT_2026-11-14.json',
             comparison_sha256=sha(ROOT / 'DEADLINE_SCREEN_HORIZON_LOSS_COMPARISON_AMENDMENT_2026-11-14.json'),
             parent_protocol_sha256=sha(ROOT / 'DEADLINE_SCREEN_RESEARCH_PROTOCOL_2026-11-12.json'),
             engineering_fixture_protocol_sha256=sha(ROOT / 'HORIZON_LOSS_IOV_FIXTURE_PROTOCOL_2026-11-14.json'))
    p['development'] = copy.deepcopy(base['development'])
    d = p['development']
    d['model_seeds'] = [83 * 1000 + x for x in (1, 11, 17, 23, 39)]
    d['train_blocks'] = [dict(start=(830 + i) * 10**6, count=200) for i in range(5)]
    d['evaluation_seeds'] = dict(start=840 * 10**6, count=48)
    p['rng'].update(arm_order_seed=8406 * 10**5 + 1,
                    bootstrap_seed=8406 * 10**5 + 2, timing_order_seed=8436 * 10**5 + 3,
                    order='One precomputed permutation of all 70 arm/fit pairs, saved before training')
    d.update(model_fits=70, training_slots=70 * 200 * 40, training_episodes=70 * 200,
             optimizer_updates=70 * (8000 - 1024 + 1), target_copies=70 * ((8000 - 1024 + 1) // 200),
             learned_evaluation_episodes=70 * 48, heuristic_evaluation_episodes=3 * 48,
             evaluation_slots=(70 + 3) * 48 * 40)
    p['training']['loss_definitions'] = (
        'All eleven original controls retain their exact declared losses, including branchwise adapted BDQ. '
        'Only the three named normalized arms use mean Huber(Q/h_current,y/h_current) '
        'plus unchanged original-unit terminal anchor; h=40-round(40*s[-1]).')
    p['loss_correction'] = dict(
        arms=[a['id'] for a in amendment['new_arms']],
        uniform_loss='mean smooth_l1_loss(chosen_Q/current_h, detached_Bellman_target/current_h, beta=1)',
        horizon='CURRENT replay state: h=40-round(40*s[-1]); executed h in 1..40',
        anchor='Unchanged original-unit Huber(Q-r), batch32, weight1; terminal current h=1',
        acceptance_units='Original Q/return units; evaluation normalization definitions unchanged',
        diagnostic_overhead='Instrumented adapter performs additional detached next-action/target forwards for audit; report measured training cost without subtracting them',
        no_tuning_from_fixture=True)
    p['action_cost']['raw_timed_calls'] = 70 * 5 * 128
    p['action_cost']['checkpoint_order'] = 'Freeze and hash all 70 final checkpoints; reconstruct/validate/time all before policy-quality evaluation'
    p['uncertainty']['comparisons'] = 'All 16 primary and nine auxiliary contrasts; heuristics repeated by fit only for alignment, not independent models'
    p['decision']['required'] = ['engineering_and_artifact_integrity', 'references_every_fit',
        'strong_heuristics_every_fit', 'correction_mechanism', 'price_mechanism',
        'same_width_architecture', 'primary_value_every_fit', 'primary_cost_every_fit']
    p['decision']['screening_status'] = 'Retain historical screening evidence and report fresh original screened/unmasked auxiliary contrast. Correction, price and same-width mechanisms are the new primary gates; no normalized unmasked arm is invented.'
    p['authorization'] = dict(this_stage='Design freeze, source integrity and seed collision audit only; zero training/evaluation',
        research_development='Requires versioned executable binding, auditor/release checks and separate explicit user launch instruction',
        CLI='Explicit development confirmation required in CLI and API; final disabled',
        launch_authorized=False, literature='Literature novelty clearance and supervisor agreement remain outstanding')
    p['threshold_provenance'] = 'Exact prospective horizon-loss amendment thresholds and fixture comparison list retained; no smoke outcome used to change any rule. The 560000-slot budget follows 14*5*200*40, explicitly replacing the prior 11-arm budget.'
    p['saved_artifacts']['require'] += ['original_and_normalized_TD_distributions_by_current_horizon',
        'remote_deadline_misses_by_destination_upload_CPU_stage_and_episode_time',
        'loads_action_frequencies_and_UAV_movements']
    p['fixture']['status'] = 'completed_engineering_evidence_consumed_seeds_not_fresh_research'
    p['fixture']['evidence'] = 'validation/horizon_loss_iov_integration_2026-11-14_01'
    p['seed_reservation'] = dict(status='pending_local_audit', refresh_before_launch=True)
    # Audit ONLY new research reservations. Completed engineering fixture seeds
    # correctly appear in historical evidence and must not be called fresh.
    old_reserve = historical.reserve_values
    def fresh_groups(protocol):
        return {k: v for k, v in old_reserve(protocol).items()
                if k.startswith('development') or k == 'research_orders'}
    historical.reserve_values = fresh_groups
    print('Checking historical declarations and saved episode headers...', flush=True)
    try:
        audit = historical.seed_audit(p)
    finally:
        historical.reserve_values = old_reserve
    audit['reservation_scope'] = 'New development model/task/derived RNG/order streams only; consumed fixture seeds excluded from new reservation'
    write(OUT / 'seed_reservation_audit.json', audit)
    if not audit['passed']:
        raise RuntimeError('Seed audit failed; evidence preserved, no protocol published')
    p['seed_reservation'].update(status='reserved_no_execution_local_audit_passed',
        audit=str((OUT / 'seed_reservation_audit.json').relative_to(ROOT)),
        audit_sha256=sha(OUT / 'seed_reservation_audit.json'))
    # Freeze the tested implementation without editing its fixture-only guards.
    tree = ast.parse((ROOT / 'run_horizon_loss_iov_fixture.py').read_text())
    extra = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'NEW_FILES' for t in n.targets)))
    sources = set(p['definition_sha256']) | set(extra) | {
        'horizon_normalized_loss.py', 'test_horizon_normalized_loss.py',
        'HORIZON_LOSS_IOV_FIXTURE_PROTOCOL_2026-11-14.json',
        'DEADLINE_SCREEN_HORIZON_LOSS_PROSPECTIVE_BRIEF_2026-11-14.txt',
        'HORIZON_LOSS_IOV_INTEGRATION_REVIEW_2026-11-14.txt'}
    p['tested_engineering_source_sha256'] = {n: sha(ROOT / n) for n in sorted(sources)}
    for n, digest in p['definition_sha256'].items():
        assert sha(ROOT / n) == digest, n
    assert p['learned_arms'][:11] == base['learned_arms']
    assert p['comparisons'] == fixture['comparisons']
    assert p['final'] == base['final']
    assert p['scope'] == base['scope'] and p['simulator'] == base['simulator']
    assert len(p['learned_arms']) == 14 and len(p['heuristics']) == 3
    assert len(set(a['id'] for a in p['learned_arms'])) == 14
    assert len(p['comparisons']['primary_report_against_each']) == 16
    assert len(p['comparisons']['auxiliary_contrasts']) == 9
    assert len(p['comparisons']['required_reference_arms']) == 11
    assert d['training_slots'] == 560000 and d['evaluation_slots'] == 140160
    assert d['optimizer_updates'] == 488390 and d['target_copies'] == 2380
    assert p['action_cost']['raw_timed_calls'] == 44800
    for key in base['training']:
        if key != 'loss_definitions':
            assert p['training'][key] == base['training'][key], key
    write(ROOT / NAME, p)
    review = f'''HORIZON-LOSS RESEARCH COMPARISON: PROSPECTIVE FREEZE
2026-11-14

DECISION
Freeze the fourteen-learner, three-heuristic development design. Local seed
collision audit and structural/source checks passed. No research launch.
The fixture-only executable remains disabled for development and final modes.
This is a design freeze, not a claim that research execution is released.

PRIMARY AND CONTROLS
Primary: normalized_screened_price24, fixed before new research outcomes.
Retain all eleven original learner definitions and add normalized no-price24
and normalized anchored plain24. Include all-local, nearest-RSU and UAV-selective.
Plain small DDQN, screened/unmasked wide128, compact64 and adapted BDQ remain
fresh paired controls. Old consumed research outcomes are historical evidence.
All16 primary and nine auxiliary contrasts are mandatory; no winner substitution.

EXPLICIT BUDGET
Five fits per learner; 200 episodes of40 slots per fit:70 final checkpoints,
14,000 training episodes and560,000 training slots. The budget changes solely
because14 learners replace11 at the same per-fit exposure and five-fit count.
488,390 optimizer updates and2,380 target copies; final checkpoints only.
48 common evaluation episodes per fit/controller and per heuristic:
3,360 learned plus144 heuristic episodes, totaling140,160 evaluation slots.
Freeze all70 checkpoints before runtime equivalence/timing and policy evaluation.
44,800 timing readings,128 common inputs, five rounds; no favorable retiming.

SEED RESERVATIONS, NOT EXECUTION
Models: {d['model_seeds']}.
Train blocks: {d['train_blocks']}.
Evaluation: {d['evaluation_seeds']}.
Order/bootstrap/timing: {p['rng']['arm_order_seed']}, {p['rng']['bootstrap_seed']}, {p['rng']['timing_order_seed']}.
Action/replay/anchor offsets remain100000/200000/300000. Paired action draws,
task blocks and replay index schedule across arms; contents may differ by policy.
Local audit checked{audit['checked_files']} files and reserved{audit['unique_reserved_values']}
unique values. New streams are mutually disjoint, avoid prior consumed/declared
local seeds and protected historical final reservations. The completed fixture
seeds remain consumed engineering evidence, excluded from new reservations.
The audit covers local metadata/declarations/filenames and first16,384 decoded
episode-header characters. It cannot prove freshness outside its recorded scope.
Refresh actual-run seed use before launch. Protected final schedules unchanged.

FROZEN LEARNING AND ACCEPTANCE
Adam1e-4, batch128, replay50000, warmup1024, gamma1, clip10, target200 updates;
same epsilon schedule, simulator, reward, state, screen and model definitions.
Only the three specified arms divide both Q and detached y by CURRENT horizon
inside Huber. Unit-weight batch32 terminal anchoring remains in original units.
All original controls retain their exact losses, including branchwise BDQ.
Diagnostic detached forwards add measured training overhead; do not subtract it.

Eleven reference contrasts: per-fit return>=-2 and deadline fraction>=-.01.
Nearest-RSU/UAV-selective: both paired differences>=0 per fit. All-local diagnostic.
Correction, price and same-width mechanisms: both differences>=0 every fit,
equal-fit mean return gain>=2 and deadline fraction gain>=.01 for EACH mechanism.
Historical screening evidence is preserved; fresh original screening contrasts
are reported. No normalized unmasked arm or new screening novelty is assumed.
Primary original-unit value limits EVERY fit: normalized RMSE<=.25, absolute
bias<=2, near-zero/large-return fraction<=.05, terminal error p95<=2;
finite values/gradients and nonpositive primary Q. Report all arms and both
TD residual units by current horizon, terminal status and episode time.
Primary parameter payload<=half unmasked compact64; median complete action
latency<=both wide128 paths, p95<=1ms per fit. Buffers/artifact/startup separate.
Report remote misses by destination, upload/CPU stage and time, loads/actions
and movement; retain failed fits. All integrity/policy/mechanism/value/cost
requirements form one conjunction. No tuning, best fit, dropped fit or extra
budget.5,000-draw paired crossed bootstrap intervals are descriptive only.

NEXT ENGINEERING GATE
Bind a new versioned research runner/workflow/auditor to this exact protocol.
Add scale/count/source-release checks and ensure by-horizon residual and remote
stage reports are emitted/audited automatically. Preserve the tested fixture
sources; verify executable confirmation guards and source closure before a
separate explicit launch decision. No launch-ready claim is made by this freeze.
No new research simulator or optimizer step ran during this preparation.

CLAIM AND SCOPE
Engineering integration passed, but improved controller quality is unproven.
The prior research combined gate failed and remains preserved. Loss normalization
alone is established technique; novelty/literature and supervisor agreement
remain open. Contribution one remains efficient DDQN for local/UAV/RSU offloading.
Cloud, charging and security/trust excluded; Word thesis/final seeds unchanged.
'''
    (ROOT / REVIEW).write_text(review, encoding='utf-8')
    snapshot = OUT / 'source_snapshot'
    for n in sorted(sources | {NAME, REVIEW, str(Path(__file__).relative_to(ROOT))}):
        dest = snapshot / n
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / n, dest)
    write(OUT / 'verification.json', dict(passed=True, stage='design_freeze_only',
        model_comparison_and_threshold_checks=True, budget_checks=True,
        unchanged_control_and_simulator_checks=True, protected_final_unchanged=True,
        source_hashes_checked=len(p['tested_engineering_source_sha256']),
        protocol_sha256=sha(ROOT / NAME), simulator_steps=0, optimizer_updates=0,
        research_executable_bound=False, launch_authorized=False))
    status_path = ROOT / 'LOW_COMPLEXITY_PROJECT_STATUS.json'
    shutil.copy2(status_path, OUT / 'PROJECT_STATUS_BEFORE_FREEZE.json')
    status = read(status_path)
    status['current_workstream'] = 'horizon_loss_research_protocol_frozen_pending_executable_binding'
    status['next_engineering_step'] = 'Bind versioned research workflow/auditor to new freeze, audit automatic residual/remote-stage reports, test scale arithmetic and confirmation guards; separate launch decision afterwards'
    status['horizon_loss_research_freeze'] = dict(protocol=NAME, review=REVIEW,
        evidence=str(OUT.relative_to(ROOT)), training_slots=d['training_slots'],
        evaluation_slots=d['evaluation_slots'], learned_controllers=14, heuristics=3,
        initializations=5, development_seeds_reserved=True, local_seed_audit_passed=True,
        research_executable_bound=False, launch_authorized=False,
        research_launched=False, word_thesis_changed=False)
    status['horizon_normalized_loss_design'].update(research_budget_declared=True, seeds_reserved=True)
    write(status_path, status)
    write(OUT / 'manifest.json', {x.relative_to(OUT).as_posix(): sha(x)
        for x in OUT.rglob('*') if x.is_file()})
    print(json.dumps(dict(passed=True, protocol=NAME, training_slots=d['training_slots'],
                         evaluation_slots=d['evaluation_slots'], research_launched=False)), flush=True)

if __name__ == '__main__':
    main()
