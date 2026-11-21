"""Freeze design documents and audit local seed metadata; never runs a controller.

Standard library only. No imports from simulator, PyTorch or training runners.
"""
import ast
import gzip
import hashlib
import json
import operator
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = 'DEADLINE_SCREEN_RESEARCH_PROTOCOL_2026-11-12.json'
REVIEW = 'DEADLINE_SCREEN_RESEARCH_PROTOCOL_REVIEW_2026-11-12.txt'
OUT = ROOT / 'validation/deadline_screen_research_protocol_freeze_2026-11-12_01'
COMPARISON = ROOT / 'DEADLINE_SCREEN_COMPARISON_SET_FREEZE_2026-11-12.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def config_defaults(path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    constants = {n.targets[0].id: n.value.value for n in tree.body
                 if isinstance(n, ast.Assign) and len(n.targets) == 1
                 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
    def literal(node):
        if isinstance(node, ast.Name):
            return constants[node.id]
        if isinstance(node, ast.BinOp):
            operation = {ast.Div: operator.truediv, ast.Mult: operator.mul,
                         ast.Add: operator.add, ast.Sub: operator.sub}[type(node.op)]
            return operation(literal(node.left), literal(node.right))
        return ast.literal_eval(node)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Chapter3Config')
    return {n.target.id: literal(n.value) for n in cls.body
            if isinstance(n, ast.AnnAssign) and n.value is not None}


def reserve_values(p):
    groups = {}
    for phase in ('development', 'fixture'):
        x = p[phase]
        groups[phase + '_models'] = set(x['model_seeds'])
        for name, offset in p['rng']['offsets'].items():
            groups[phase + '_' + name] = {s + offset for s in x['model_seeds']}
        groups[phase + '_training'] = {s for b in x['train_blocks'] for s in range(b['start'], b['start'] + b['count'])}
        b = x['evaluation_seeds']
        groups[phase + '_evaluation'] = set(range(b['start'], b['start'] + b['count']))
    groups['research_orders'] = {p['rng'][k] for k in ('arm_order_seed', 'bootstrap_seed', 'timing_order_seed')}
    groups['fixture_orders'] = set(p['fixture']['engineering_seeds'].values())
    return groups


def historical_seed_values(value, inherited_seed=False):
    if isinstance(value, dict):
        if inherited_seed and isinstance(value.get('start'), int) and isinstance(value.get('count'), int):
            count = value['count']
            if not 0 <= count <= 1000000:
                raise ValueError('unexpected seed range length')
            yield from range(value['start'], value['start'] + count)
        for key, item in value.items():
            relevant = inherited_seed or 'seed' in key.lower() or key in ('train_blocks', 'training_starts', 'train_starts')
            yield from historical_seed_values(item, relevant)
    elif isinstance(value, list):
        for item in value:
            yield from historical_seed_values(item, inherited_seed)
    elif inherited_seed and isinstance(value, int) and not isinstance(value, bool):
        yield value


def seed_audit(p):
    groups = reserve_values(p)
    keys = list(groups)
    internal = [{'left': a, 'right': b, 'seeds': sorted(groups[a] & groups[b])}
                for i, a in enumerate(keys) for b in keys[i + 1:] if groups[a] & groups[b]]
    reserved = set().union(*groups.values())
    old = json.loads((ROOT / 'TERMINAL_ANCHOR_RESEARCH_PROTOCOL_2026-11-06.json').read_text())
    protected = set(historical_seed_values(old['final']))
    # Reserve the original final RNG streams as well as actual model/task seeds.
    protected.update(s + offset for s in old['final']['model_seeds'] for offset in p['rng']['offsets'].values())
    roots = [ROOT, ROOT.parent / 'October 31/Working/Project']
    pattern = re.compile(r'"(?:seed|model_seed|training_seed|train_seed|evaluation_seed|task_seed)"\s*:\s*(\d+)\b')
    basename_pattern = re.compile(r'(?:seed[_-]?|eval(?:uation)?[_-]|train[_-])(\d{5,})')
    counts = {'compressed_headers': 0, 'metadata_files': 0, 'text_definitions': 0, 'filenames': 0}
    collisions, errors, oversized, entries = [], [], [], []
    for base in roots:
        if not base.exists():
            errors.append({'path': str(base), 'error': 'historical root missing'})
            continue
        result = subprocess.run(['rg', '--files', '--hidden', '-g', '!**/.git/**', '-g', '!**/__pycache__/**', str(base)],
                                capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            path = Path(line)
            if path.resolve() == Path(__file__).resolve():
                continue  # This new prospective declaration is not historical use.
            rel = path.relative_to(base)
            if any(part in ('source_snapshot', 'source_freeze', 'frozen_source', 'sources', 'exports', 'continuation_source_snapshot') for part in rel.parts):
                continue
            values = {int(x) for x in basename_pattern.findall(path.name)}
            try:
                if path.name.endswith('.json.gz'):
                    with gzip.open(path, 'rt', encoding='utf-8-sig') as f:
                        prefix = f.read(16384)
                    values.update(map(int, pattern.findall(prefix)))
                    counts['compressed_headers'] += 1
                elif path.suffix.lower() == '.json' and (path.parent == base or any(t in path.name.lower() for t in
                      ('runtime', 'protocol', 'plan', 'config', 'metadata', 'counter', 'training_summary', 'seed', 'freeze', 'manifest'))):
                    if path.stat().st_size > 32 * 1024 * 1024:
                        oversized.append(str(path))
                        continue
                    content = path.read_bytes()
                    values.update(historical_seed_values(json.loads(content.decode('utf-8-sig'))))
                    counts['metadata_files'] += 1
                elif path.suffix.lower() in ('.txt', '.py') and path.parent in (base, base / 'analysis'):
                    content = path.read_bytes()
                    # Conservative literal check includes old declared reservations.
                    values.update(int(x) for x in re.findall(r'(?<![\w.])\d{5,10}(?![\w.])', content.decode('utf-8-sig')))
                    counts['text_definitions'] += 1
                elif values:
                    counts['filenames'] += 1
                else:
                    continue
                hit = sorted(values & reserved)
                if hit:
                    collisions.append({'path': str(path), 'reserved_seed_hits': hit})
                entries.append({'path': str(path), 'size': path.stat().st_size,
                                'seed_metadata_sha256': hashlib.sha256(json.dumps(sorted(values)).encode()).hexdigest()})
            except Exception as exc:
                errors.append({'path': str(path), 'error': str(exc)})
    return dict(passed=not internal and not collisions and not errors and not oversized and not (reserved & protected),
                roots=[str(x) for x in roots], counts=counts, unique_reserved_values=len(reserved),
                internal_collisions=internal, historical_collisions=collisions, errors=errors,
                oversized_metadata=oversized, protected_final_overlap=sorted(reserved & protected),
                checked_files=len(entries), checked_files_and_extracted_seed_digest=entries,
                limitations=['Local metadata schedules and first 16384 decoded characters of compressed episode headers plus filename/task seed checks.',
                             'No tensor/binary inspection, full episode-body seed search, external folders or unlogged runs.',
                             'Scheduled seeds are treated as occupied even if execution failed before consumption.',
                             'Refresh against actual saved runs before launch; a clean local audit is not universal proof of freshness.'])


def main():
    if OUT.exists() or (ROOT / NAME).exists() or (ROOT / REVIEW).exists():
        raise SystemExit('Refusing to overwrite an existing freeze')
    comparison = json.loads(COMPARISON.read_text())
    for rel, digest in comparison['definition_sha256'].items():
        if sha(ROOT / rel) != digest:
            raise ValueError('Frozen controller source changed: ' + rel)
    p = dict(id='deadline-screen-eleven-learner-development-v1', date='2026-11-12',
        status='frozen_design_not_implemented_or_launch_authorized', comparison_file=COMPARISON.name,
        comparison_sha256=sha(COMPARISON), primary=comparison['primary'], learned_arms=comparison['learned_arms'],
        heuristics=comparison['heuristics'], scope=comparison['scope'], screen=comparison['screen'],
        definition_sha256=comparison['definition_sha256'],
        simulator=dict(module='Source/chapter3_iov_env_v2.py', config='Chapter3Config() without overrides',
            defaults={**config_defaults(ROOT / 'Source/chapter3_iov_env.py'), **config_defaults(ROOT / 'Source/chapter3_iov_env_v2.py')},
            reward_changed=False, normalization_changed=False),
        training=dict(episodes_per_fit=200, slots_per_episode=40, slots_per_fit=8000, warmup=1024,
            batch_size=128, replay_capacity=50000, replay='uniform with replacement; fixed chronological insertion/eviction',
            first_update='after inserting physical slot 1024; one update each slot thereafter',
            optimizer='Adam', learning_rate=0.0001, betas=[0.9, 0.999], eps=1e-8, weight_decay=0,
            gamma=1.0, gradient_clip=10.0, huber_beta=1.0, target_interval_updates=200,
            optimizer_updates_per_fit=6977, target_copies_per_fit=34, threads=1,
            epsilon='1 during warmup, otherwise 1-.95*min(global_slot_before_action/8000,1)',
            target_initialization='exact online state copy; eval mode; requires_grad False',
            final_checkpoint_only=True, checkpoint_selection=False, research_early_stopping=False,
            anchor=dict(batch=32, weight=1.0, terminal_pool='uniform with replacement among un-evicted terminal transitions',
                        empty_pool='zero anchor loss and log event', separate_rng=True, one_optimizer_step=True),
            loss_definitions='Exactly per frozen arm: scalar Huber, scalar plus anchor, or adapted BDQ branchwise Huber. No loss substitutions.',
            training_time='Record wall time per fit, optimizer time and extra anchor sample counts; equal slots/steps do not imply equal compute'),
        rng=dict(generator='numpy.random.default_rng / PCG64; torch.manual_seed(model_seed) for initialization',
            offsets=dict(action=100000, replay=200000, anchor=300000), arm_order_seed=720600001,
            bootstrap_seed=720600002, timing_order_seed=723600003,
            action_draws='Every training slot consumes one epsilon uniform and 32 branch uniforms, even warmup/greedy; exploration ranks within ordered allowed choices',
            pairing='Same model seed, task schedule, action source draws and replay RNG schedule per fit across arms; distinct network shapes need not share tensor entries',
            replay='Same sampled indices given common buffer insertion schedule; contents differ with policy; separate terminal-anchor RNG',
            order='One precomputed permutation of all 55 arm/fit pairs, saved before training'),
        development=dict(model_seeds=[71001,71011,71017,71023,71039],
            train_blocks=[dict(start=710000000+i*1000000,count=200) for i in range(5)],
            evaluation_seeds=dict(start=720000000,count=48),
            evaluation_pairing='Same 48 episodes for every arm/fit and heuristic; heuristics run once per task, not five independent copies',
            model_fits=55, training_slots=440000, training_episodes=11000, optimizer_updates=383735,
            target_copies=1870, learned_evaluation_episodes=2640, heuristic_evaluation_episodes=144,
            evaluation_slots=111360, diagnostic_validation_seeds=[],
            policy='Frozen final online greedy policy; epsilon zero; no updates, fitting, checkpoint selection or additional validation probes'),
        comparisons=dict(primary_report_against_each=comparison['primary_policy_comparisons'],
            required_reference_arms=[a['id'] for a in comparison['learned_arms']
                                     if a['id'] not in ('screened_price24','unmasked_price24','screened_no_price24')],
            reference_each_fit=dict(return_difference_min=-2.0, deadline_fraction_difference_min=-0.01),
            strong_heuristics=['nearest_rsu','uav_selective'], heuristic_each_fit=dict(return_difference_min=0.0,deadline_fraction_difference_min=0.0),
            diagnostic_only_heuristics=['all_local'],
            mechanisms=[dict(id=key,left='screened_price24',right=arm,
                each_fit_return_difference_min=0.0,each_fit_deadline_fraction_difference_min=0.0,
                equal_fit_mean_return_difference_min=2.0,equal_fit_mean_deadline_fraction_difference_min=0.01)
                for key,arm in [('screening','unmasked_price24'),('price','screened_no_price24'),('same_width_architecture','screened_plain24_anchored')]],
            auxiliary_contrasts=comparison['mechanism_comparisons'],
            return_definition='Unrounded sum of the 40 recorded rewards per episode; larger (less negative) is better',
            deadline_definition='tasks_on_time / tasks_generated at 40-slot stop; unfinished tasks remain unsuccessful; pair by actual task fingerprint',
            aggregation='Arithmetic mean of paired episode differences within each initialization, then equal weight across five fits; no rounded-value decisions'),
        value_stability=dict(gated_arms=['screened_price24'], report_all_arms=True,
            Q_target='Record frozen target value for the same online-selected action before every executed evaluation action',
            realized_return='G_t=sum(rewards[t:40]) on that same arm trajectory; gamma1',
            normalized_rmse='sqrt(mean((Q-G)^2))/max(1,sqrt(mean(G^2)))', normalized_rmse_max=0.25,
            absolute_bias='abs(mean(Q-G))', absolute_bias_max=2.0,
            near_zero_threshold=1e-6,large_return_threshold=1.0,near_zero_large_return_fraction_max=0.05,
            terminal_p95='95th percentile of abs(Q_39-r_39) over episodes, numpy linear quantile', terminal_abs_error_p95_max=2.0,
            nonpositive_primary=True, finite_all_arms_and_optimizer_diagnostics=True,
            distributions=['mean','median','p05','p95','min','max','count'], time_bins=[[0,9],[10,19],[20,29],[30,39]],
            save=['chosen_online_Q','chosen_target_Q','realized_return_to_go','Q_minus_G','target_minus_G',
                  'Bellman_targets','TD_residuals','gradient_norm_before_clip','clipping_fraction','uniform_loss','anchor_loss',
                  'terminal_and_nonterminal_predictions','destination_counts','local_infeasibility_counts','deadline_misses_by_destination'],
            interpretation='Realized own-policy consistency under stochastic trajectories, not exact conditional optimal Q ground truth; comparator failures do not excuse primary failure'),
        action_cost=dict(backend='Unchanged original PyTorch FP32 neural cores plus native Numba decoders; one CPU thread; no ONNX conversion',
            route='New unmasked adapters must preserve existing arithmetic and match eager actions/values; no deliberately slow reference',
            common_inputs='validation/controller_redesign_engineering_2026-11-05_01/cost_gate/common_observations.npz',
            common_inputs_sha256='2acd6bf97d649df97e0084f6b61a265d3742a70421410c9ccabc6cfba1854ebb',
            observations=128,rounds=5,warmups_per_path_per_round=32,timed_calls_per_path_per_round=128,
            raw_timed_calls=35200,exact_actions=True,Q_atol=1e-6,Q_rtol=1e-5,
            measure='Input validation/conversion, screen when enabled, neural forward, decoder and final action copy; diagnostic screen logging charged if in deployed path',
            exclude='Simulator stepping, file I/O, compilation and startup; report cold startup separately',
            timing='perf_counter_ns; save every reading and fixed randomized order; pool 640 observations/path for median and linear p95; no favorable retiming',
            primary_each_fit=dict(parameter_bytes_max_fraction_of_unmasked_compact64=0.5,
                median_ms_max_fraction_of_each=['screened_wide128','unmasked_wide128'],median_ratio_max=1.0,p95_ms_max=1.0),
            storage='Confirm actual tensors; report online parameters, buffers, serialized deployment bytes, target/optimizer/replay separately. No device-memory or electrical energy claim.',
            checkpoint_order='Freeze and hash all 55 final checkpoints before any evaluation; reconstruct all runtimes, validate and time before policy-quality evaluation'),
        uncertainty=dict(stage='development continuation screen only; no final statistical claim',draws=5000,
            resampling='Paired crossed bootstrap of 5 initialization indices and 48 common task episode indices; same draws for both compared arms',
            comparisons='All 13 primary comparisons; heuristics repeated by fit only for alignment, not independent trained models',
            intervals='Two-sided descriptive percentile95%; no multiplicity-corrected superiority claim or gate on these intervals',
            units='Episodes and trained initializations, never individual time slots; five initializations limit uncertainty estimates'),
        decision=dict(required=['engineering_and_artifact_integrity','references_every_fit','strong_heuristics_every_fit',
            'screening_mechanism','price_mechanism','same_width_architecture','primary_value_every_fit','primary_cost_every_fit'],
            conjunction=True,failed_fit_policy='Retain every failed fit; no subset pass, best seed, extra budget or retrospective threshold changes',
            failures='Record which policy/value/cost/mechanism gate failed; no thesis claim promotion on partial success',
            stopping='Stop execution on nonfinite values, illegal action, mismatched workloads, invariant/audit failure; retain partial artifacts; no automatic retraining',
            infrastructure_continuation='Evaluation-only continuation of frozen checkpoints requires a separately documented source/runtime amendment and review, never silent model changes',
            success='Fresh development evidence supports reviewing a separately predeclared final confirmation; it does not itself prove thesis novelty'),
        fixture=dict(kind='SYNTHETIC_NOT_RESEARCH',model_seeds=[99921,99929],
            train_blocks=[dict(start=999210000,count=2),dict(start=999290000,count=2)],evaluation_seeds=dict(start=999390100,count=2),
            engineering_seeds=dict(observations=999390004,order=999390001,bootstrap=999390002,timing=999390003),
            episodes_per_fit=2,slots_per_episode=40,warmup=8,batch_size=4,replay_capacity=64,target_interval_updates=8,epsilon_decay=40,
            learning_rate=0.0001,gamma=1.0,gradient_clip=10.0,anchor_batch=32,anchor_weight=1.0,
            fits=22,training_slots=1760,optimizer_updates=1606,target_copies=198,
            learned_evaluation_episodes=44,heuristic_evaluation_episodes=6,evaluation_slots=2000,
            workflow='Same future orchestration/auditor; temporary isolated output; all stages and CLI/API confirmation guards tested',
            no_quality_or_selection_evidence=True, final_mode_disabled=True),
        final=dict(status='existing_reservations_protected_untouched_no_new_final_protocol',
            historical_protocol='TERMINAL_ANCHOR_RESEARCH_PROTOCOL_2026-11-06.json',
            model_seeds=[51001,51011,51017,51023,51031],
            train_blocks=[dict(start=510000000+i*1000000,count=200) for i in range(5)],
            evaluation_seeds=dict(start=520000000,count=200),launch_authorized=False,
            note='Reservations only; historical final arm counts/budgets are not silently imported into this comparison'),
        saved_artifacts=dict(require=['pre_run_source_config_protocol_freeze','complete_transitive_source_snapshot','arm_order',
            'environment_package_hardware_metadata','seed_collision_refresh','training_and_evaluation_episodes',
            'observations_and_next_observations','workload_fingerprints','actor_uniforms_and_remapped_actions',
            'replay_indices_terminal_pool_eviction_anchor_draws','optimizer_diagnostics','final_online_and_target_checkpoints',
            'all_final_checkpoints_freeze','pre_timing_freeze','equivalence_results','raw_timings',
            'evaluation_weight_mode_invariance','all_comparisons_distributions_and_screens','full_artifact_manifest','independent_audit'],
            auditor='Reconstruct counts/RNG/targets/checkpoint/runtime/action-mask semantics, summaries and decisions from artifacts; reject duplicates/missing files/tampering; use versioned independent logic',
            heuristic_adapter='Use V2 simulator and unchanged select_destinations with hover. Do not call V1 evaluate_policy; no screen-membership assertion on unmasked or heuristic actions.'),
        authorization=dict(this_stage='Design freeze and metadata audit only; zero simulator or optimizer steps',
            research_development='Requires completed runner/auditor, reduced workflow tests, live model/cost verification, environment repair and separate explicit launch instruction',
            CLI='Development must require explicit confirmation; API must enforce same authorization; final disabled',
            literature='Broader UAV-IoV full-paper review and supervisor contribution discussion remain open; screen alone not new'),
        threshold_provenance='Existing -2 return/-1 percentage point reference tolerances, primary value limits, and cost targets retained. Existing +2/+1 point mean mechanism margins now prospectively applied to screening and same-width architecture too. Same-width and screening effects are hypotheses, not expected successes inferred from smoke. All-local exclusion and requiring both wide latency references are explicit prospective rules for this new comparison only.',
        research_launched=False,new_simulator_slots=0,new_optimizer_updates=0,word_thesis_changed=False)
    common = ROOT / p['action_cost']['common_inputs']
    if sha(common) != p['action_cost']['common_inputs_sha256']:
        raise ValueError('Common engineering observations hash mismatch')
    # Audit BEFORE writing new declarations, so the protocol cannot collide with itself.
    audit = seed_audit(p)
    if not audit['passed']:
        raise SystemExit('Seed reservation audit failed; no freeze written: ' + json.dumps({k:v for k,v in audit.items() if k != 'checked_files_and_extracted_seed_digest'}))
    OUT.mkdir(parents=True)
    write(OUT / 'seed_reservation_audit.json', audit)
    p['seed_reservation'] = dict(status='reserved_no_execution_local_audit_passed',
        audit=str((OUT / 'seed_reservation_audit.json').relative_to(ROOT)),audit_sha256=sha(OUT/'seed_reservation_audit.json'),
        refresh_before_launch=True)
    write(ROOT / NAME, p)
    review = '''DEADLINE SCREEN: PROSPECTIVE DEVELOPMENT PROTOCOL FREEZE
Date: November 12, 2026

OUTCOME
The eleven-controller/three-heuristic comparison now has a prospective
development budget, fresh seed reservation, optimizer schedule and explicit
decision rules. No controller, simulator, optimizer or research run executed.
The comparison definitions and historical failed runs remain unchanged.

BUDGET
11 learned controllers x 5 initializations x 200 episodes x 40 slots:
440,000 training slots; 55 final checkpoints; 383,735 optimizer updates;
1,870 target copies. Each fit gets 8,000 slots. No best-fit selection,
intermediate research checkpoints, budget extension or early policy stopping.
Evaluation: same 48 new tasks for every fit/controller, plus each heuristic
once per task: 2,784 evaluation episodes / 111,360 simulator slots.

PAIRED SEEDS
Models: 71001,71011,71017,71023,71039.
Training: 710000000,711000000,712000000,713000000,714000000 starts,
each a consecutive 200-task block. Evaluation: 720000000..720000047.
Action/replay/anchor RNG offsets: 100000/200000/300000.
Order/bootstrap/timing: 720600001/720600002/723600003.
The local saved-run metadata/header and declaration audit found no collisions
in its checked scope. Its roots, counts, file list, errors and limitations
are preserved. This is not proof about external or unlogged runs; refresh
before launch. Protected historical final reservations are not used.

TRAINING RULES
Same existing simulator/configuration, rewards, features and model definitions.
Adam1e-4, gamma1, batch128, replay50000, warmup1024, clip10, target200 updates.
Epsilon1 during warmup, then 1-.95*min(slot_before_action/8000,1).
Each slot consumes one epsilon uniform and32 branch uniforms in every arm.
Screening applies consistently to behavior and online next-action selection.
Target evaluates that selected action; terminal target is reward only.
Anchored arms retain32 terminal samples and unit anchor weight per update.
Adapted BDQ retains branchwise Huber; no scalar-mean loss substitution.

PROSPECTIVE PASS RULES: ALL FIVE INITIALIZATIONS
Reference controls: primary paired episode mean return difference >=-2,
deadline difference >=-.01 against all eight declared reference controls.
Nearest-RSU and UAV-selective: both differences >=0 per initialization.
All-local remains a reported diagnostic floor, not an acceptance reference.

Screening, price, and same-width architecture comparisons: each fit must
have return/deadline differences >=0. Each equally weighted five-fit mean
must improve return by >=2 and deadline fraction by >=.01. The architecture
comparison changes more than price and is not a single-factor ablation.
The mechanism margins reuse the earlier price-effect standard; applying
them to screening/architecture is explicit and prospective, not evidence
that these improvements will occur. Auxiliary contrasts remain reported.

Primary value consistency per fit: normalized RMSE<=.25, absolute bias<=2,
near-zero-Q/large-return fraction<=.05, terminal abs-error p95<=2; finite
values/gradients everywhere and nonpositive primary Q. All arms' value,
target, return residual and gradient distributions must be reported by time
and terminal/nonterminal status. Realized returns are not exact Q-star labels.

Primary cost per fit: online FP32 payload<=half original compact64;
median complete action latency<=both screened and unmasked wide128;
p95<=1ms. Same fixed128 engineering observations,5rounds,32warmups and
128timed calls/path/round, all55checkpoints:35,200 raw timing readings.
Use original PyTorch FP32 cores/native decoders; no ONNX conversion or
deliberately slow baseline. Report buffers/artifact sizes/startup separately.

All required integrity, policy, mechanism, value and cost gates must pass.
No dropped fits, favorable retiming, model changes or retrospective margins.
Report 5,000-draw paired crossed-bootstrap descriptive intervals; these do
not constitute final confirmation or multiplicity-adjusted superiority.

ENGINEERING STILL REQUIRED
Repair or replace the incomplete Python environment through a versioned
environment setup; live parameter/buffer counts remain to be confirmed.
Implement the research runner and independent auditor in new files, with
unmasked runtime paths and original compact64. Unmasked/heuristic actions
must not be required to belong to the screened set. Heuristics must run in
the V2 environment using unchanged destination rules and hover, not the old
V1 evaluate_policy helper. This clarifies adapter wiring, not policy changes.

Run the same complete orchestration on isolated fixture seeds99921/99929,
two40-slot training episodes/fit. It is 1,760 training slots and2,000 evaluation
slots,1,606 updates,198copies; no quality inference or tuning. Test confirmation
guards, exact eager/runtime behavior, comparisons, freeze stages, artifact
audit, duplicate/missing files and deliberate corruption rejection.
Freeze executable sources/environment and refresh the seed audit before a
separate explicit research launch decision. This document is not a launcher.

CLAIM AND SCOPE
The goal remains a measured low-complexity DDQN improvement in UAV-IoV
offloading. A development pass only permits reviewing independent final
confirmation. Masking alone is established; broader related-work comparison
and supervisor discussion are still open. Cloud/charging excluded, security
deferred. Existing final seeds and Word thesis remain untouched.
'''
    (ROOT / REVIEW).write_text(review, encoding='utf-8')
    shutil.copy2(COMPARISON, OUT / COMPARISON.name)
    shutil.copy2(ROOT / NAME, OUT / NAME)
    shutil.copy2(ROOT / REVIEW, OUT / REVIEW)
    shutil.copy2(__file__, OUT / Path(__file__).name)
    # Structural checks are appropriate here; behavioral tests belong to implementation.
    assert p['development']['training_slots'] == 11*5*200*40
    assert p['development']['optimizer_updates'] == 55*(8000-1024+1)
    assert p['development']['target_copies'] == 55*((8000-1024+1)//200)
    assert p['development']['evaluation_slots'] == (55*48+3*48)*40
    assert p['fixture']['optimizer_updates'] == 22*(80-8+1)
    assert p['fixture']['target_copies'] == 22*((80-8+1)//8)
    assert p['fixture']['evaluation_slots'] == (22*2+3*2)*40
    assert p['action_cost']['raw_timed_calls'] == 55*5*128
    assert len(p['comparisons']['required_reference_arms']) == 8
    assert sha(COMPARISON) == p['comparison_sha256']
    for rel,digest in p['definition_sha256'].items():
        assert sha(ROOT/rel) == digest
    write(OUT/'verification.json', dict(passed=True,scope='design_structure_arithmetic_source_integrity_and_local_seed_audit',
        budget_checks=True, model_and_comparison_definitions_unchanged=True, reserved_streams_disjoint=True,
        protocol_sha256=sha(ROOT/NAME), review_sha256=sha(ROOT/REVIEW),
        simulator_steps=0, optimizer_updates=0, live_runner_tests=False, launch_authorized=False))
    status_path=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json'
    shutil.copy2(status_path,OUT/'PROJECT_STATUS_BEFORE_PROTOCOL_FREEZE.json')
    status=json.loads(status_path.read_text(encoding='utf-8-sig'))
    status['current_workstream']='deadline_screen_research_design_protocol_frozen_no_implementation_or_launch'
    status['next']='implement_versioned_runner_and_independent_auditor_repair_environment_and_test_isolated_full_workflow_before_launch_review'
    status['deadline_screen_research_protocol_freeze']=dict(protocol=NAME,review=REVIEW,
        protocol_sha256=sha(ROOT/NAME),verification=str((OUT/'verification.json').relative_to(ROOT)),
        training_slots=440000,learned_controllers=11,initializations=5,development_seeds_reserved=True,
        local_seed_audit_passed=True,runner_ready=False,research_launched=False,word_thesis_changed=False)
    write(status_path,status)
    print(json.dumps(dict(protocol=NAME,review=REVIEW,checks='passed',seed_files_checked=audit['checked_files'],
                          seed_audit_counts=audit['counts'],training_slots=440000,research_launched=False),indent=2))


if __name__ == '__main__':
    main()
