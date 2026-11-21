"""Read saved, audited summaries; write an external review, never run models."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'development_runs/assignment_price_development_2026-11-16_01'
RUN = BASE / 'run'
OUT = ROOT / 'validation/assignment_price_results_review_2026-11-19_01'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    protocol_path = ROOT / 'ASSIGNMENT_PRICE_RESEARCH_PROTOCOL_2026-11-15.json'
    p = read(protocol_path)
    a = read(RUN / 'analysis.json')
    costs = read(RUN / 'costs.json')
    opt = read(RUN / 'optimizer_distributions.json')['arms']
    completion = read(BASE / 'audited_completion/completion.json')
    audit = read(BASE / 'audited_completion/saved_artifact_audit.json')
    assert completion['status'] == 'AUDITED_COMPLETE' and audit['passed']
    assert hashlib.sha256((RUN / 'manifest.json').read_bytes()).hexdigest() == completion['run_manifest_sha256']
    primary = p['primary']
    comparisons = a['comparisons']
    rules = p['comparisons']
    failures = []
    for arm in rules['required_reference_arms'] + rules['strong_heuristics']:
        threshold = rules['reference_each_fit'] if arm in rules['required_reference_arms'] else rules['heuristic_each_fit']
        for fit, row in enumerate(comparisons[arm]['per_fit']):
            if (row['return_difference'] < threshold['return_difference_min'] or
                    row['deadline_difference'] < threshold['deadline_fraction_difference_min']):
                failures.append(dict(arm=arm, fit=fit, **row))
    v = p['value_stability']
    values = []
    for fit, row in enumerate(a['value_stability'][primary]):
        tests = {key: row[key] <= v[key + '_max'] for key in
                 ('normalized_rmse', 'absolute_bias', 'near_zero_large_return_fraction', 'terminal_abs_error_p95')}
        tests['nonpositive'] = row['positive_Q_fraction'] == 0
        values.append(dict(fit=fit, checks=tests, diagnostics=row))
    cost_rules = p['action_cost']['primary_each_fit']
    cost_rows = []
    for fit, row in enumerate(costs[primary]):
        refs = {arm: row['median_ms'] / costs[arm][fit]['median_ms']
                for arm in cost_rules['median_ms_max_fraction_of_each']}
        checks = dict(storage=row['parameter_bytes'] <= cost_rules['parameter_bytes_max_fraction_of_unmasked_compact64'] * costs['unmasked_compact64'][fit]['parameter_bytes'],
                      median=all(ratio <= cost_rules['median_ratio_max'] for ratio in refs.values()),
                      p95=row['p95_ms'] <= cost_rules['p95_ms_max'])
        assert checks == a['cost_screens'][fit]
        cost_rows.append(dict(fit=fit, checks=checks, median_ratios=refs, **row))
    gates = dict(references_every_fit=not any(x['arm'] in rules['required_reference_arms'] for x in failures),
                 strong_heuristics_every_fit=not any(x['arm'] in rules['strong_heuristics'] for x in failures))
    for mechanism in rules['mechanisms']:
        c = comparisons[mechanism['right']]
        gates[mechanism['id'] + '_mechanism'] = (
            all(x['return_difference'] >= mechanism['each_fit_return_difference_min'] and
                x['deadline_difference'] >= mechanism['each_fit_deadline_fraction_difference_min'] for x in c['per_fit']) and
            c['mean_return_difference'] >= mechanism['equal_fit_mean_return_difference_min'] and
            c['mean_deadline_difference'] >= mechanism['equal_fit_mean_deadline_fraction_difference_min'])
    gates['primary_value_every_fit'] = all(all(x['checks'].values()) for x in values)
    gates['primary_cost_every_fit'] = all(all(x['checks'].values()) for x in cost_rows)
    assert gates == a['computed_screens']
    assert all(gates.values()) == a['all_scientific_screens_passed']
    lines = [
        'ASSIGNMENT-PRICE AUDITED RESULTS AND CONTRIBUTION DECISION BRIEF',
        'Date: 2026-11-19 (Asia/Shanghai)',
        '',
        'DECISION: DO NOT ADVANCE THIS CONTROLLER TO FINAL EVALUATION OR CLAIM REVISION.',
        'This is a saved-results review, not a new experiment or a protocol amendment.',
        'The research direction remains efficient DDQN offloading in UAV-IoV.',
        'Security/trust is deferred. Cloud and charging remain excluded.',
        '',
        'EVIDENCE AND LIMITS',
        '600,000 training slots; 151,680 evaluation slots; five initializations.',
        'The saved artifact audit passed. Completion is not scientific acceptance.',
        'All seven scientific screens were independently recomputed from saved summaries',
        'and the frozen numerical rules and exactly match analysis.json.',
        'This review does not repeat the full artifact audit or archive hashing.',
        'No training, optimizer updates, simulation, export or retiming performed.',
        'No final seeds consumed; no frozen artifacts or Word thesis changed.',
        '',
        '1. WHAT IMPROVED',
        'Primary minus comparator, equal-fit mean; deadline differences in percentage points.',
        'Arm | Return difference | Deadline pp | Fits with both differences >= 0',
    ]
    for arm, c in comparisons.items():
        count = sum(x['return_difference'] >= 0 and x['deadline_difference'] >= 0 for x in c['per_fit'])
        lines.append(f"{arm} | {c['mean_return_difference']:.4f} | {100*c['mean_deadline_difference']:.4f} | {count}/5")
    lines += ['', 'These comparisons concern this simulator, training budget and development set.',
              'They do not establish superiority over well-tuned DDQN methods generally.',
              'The primary beats plain small DDQN, both wide references, compact64,',
              'nearest-RSU and UAV-selective in both outcomes in all five initializations.',
              'It loses to observed-load/radio in both outcomes in all five.',
              '', '2. EXACT PERFORMANCE FAILURES',
              'Fit indices below are 0-4, matching saved artifacts (not best-fit selection).',
              'Required learned-reference tolerance: return >= -2; deadline >= -1 pp.',
              'Strong-heuristic tolerance: both differences >= 0.']
    for x in failures:
        lines.append(f"{x['arm']} fit {x['fit']}: return {x['return_difference']:.4f}; deadline {100*x['deadline_difference']:.4f} pp")
    lines += ['', '3. MECHANISMS',
              'Each proposed mechanism must avoid regression in every initialization,',
              'and achieve mean return gain >= 2 and mean deadline gain >= 1 pp.']
    for m in rules['mechanisms']:
        c = comparisons[m['right']]
        reg = [i for i, x in enumerate(c['per_fit']) if x['return_difference'] < 0 or x['deadline_difference'] < 0]
        lines.append(f"{m['id']}: mean return {c['mean_return_difference']:.4f}; deadline {100*c['mean_deadline_difference']:.4f} pp; regressions at fits {reg}")
    lines += ['The coupling mean is dominated by fit 4: gains in the other four fits',
              'are 1.2483, -0.8759, 0.4123 and -0.3031. The mean does not prove robustness.',
              'The original screening auxiliary contrast improves deadline performance',
              'in all five fits, but return regresses at fit 4. Screening alone is',
              'therefore not evidence of a reliable joint return/deadline improvement.',
              '', '4. VALUE CALIBRATION AND TRAINING DIAGNOSTICS',
              'Fit | NRMSE (max .25) | abs bias (max 2) | terminal p95 (max 2) | clip fraction']
    for x in values:
        fit = x['fit']; d = x['diagnostics']; o = opt[primary][fit]
        lines.append(f"{fit} | {d['normalized_rmse']:.6f} | {d['absolute_bias']:.6f} | {d['terminal_abs_error_p95']:.6f} | {o['clipping_fraction']:.6f}")
        lines.append('  Mean Q-G by slot bins 0-9/10-19/20-29/30-39: ' +
                     ', '.join(f"{b['distributions']['Q_minus_G']['mean']:.4f}" for b in d['time_bins']))
    lines += ['NRMSE fails in all five fits; absolute bias fails in four.',
              'Terminal error, nonpositive values and near-zero collapse checks pass.',
              'Boundary control has improved, but full-horizon calibration remains poor.',
              'All primary optimizer updates are clipped at the frozen norm limit 10.',
              'The p95 preclip norm is about 214-245. All four main mechanism/control',
              'arms examined also have clipping fraction 1.0 in every fit.',
              'This is a diagnostic lead, not proof that clipping causes the failure.',
              'Saved target-Q is the frozen target critic on the current chosen action;',
              'it is not the Bellman target. Bellman targets and TD residuals are',
              'separately retained in optimizer distributions. Signed TD means are',
              'aggregated across training and cannot establish final convergence.',
              'Q-G measures realized own-policy consistency under stochastic rollouts,',
              'not exact conditional optimal-value error or proven divergence.',
              '', '5. DEPLOYMENT COST',
              'Fit | parameters bytes | median ms | p95 ms | median / screened wide | median / unmasked wide']
    for x in cost_rows:
        ratios = x['median_ratios']
        lines.append(f"{x['fit']} | {x['parameter_bytes']} | {x['median_ms']:.5f} | {x['p95_ms']:.5f} | {ratios['screened_wide128']:.5f} | {ratios['unmasked_wide128']:.5f}")
    lines += ['Storage passes: 68,264 parameter bytes versus 194,944 for compact64',
              '(about 65% fewer) and 422,016 for wide128 (about 84% fewer).',
              'Primary has more parameter bytes than plain24 (67,364). Buffers,',
              'serialized graphs, libraries, target, optimizer and replay are separate.',
              'The median gate fails against at least one wide reference in every fit.',
              'The absolute p95 <= 1 ms gate passes. It is not a p95 speedup claim.',
              'A faster laptop may change timing, but cannot resolve the observed',
              'performance and calibration failures. Retiming is not a rescue analysis.',
              'Three eager/export action disagreements also prevent exact deployment',
              'equivalence; policy evaluation used the frozen eager policy.',
              '', '6. SAVED DESTINATION DIAGNOSIS',
              'Destination order: local, UAV1, UAV2, RSU1, RSU2.',
              'Counts below are descriptive associations, not causal counterfactuals.']
    for arm in (primary, 'coupled_no_sweep', 'separable64', 'fixed_price_plain_sweep', 'screened_price24'):
        for fit, d in enumerate(a['value_stability'][arm]):
            lines.append(f"{arm} fit {fit}: decisions {d['destination_counts']}; not-on-time counts {d['deadline_misses_by_destination']}")
    lines += ['Primary fit 4 has 2,792 UAV1 not-on-time tasks; fit 3 has 1,177 at RSU2.',
              'The screened-price comparator fit 4 has 5,383 at UAV1 and 2,112 at RSU2.',
              'This helps locate failures and explains why one bad comparator fit',
              'can dominate a mean gain. It does not prove queue load or radio was',
              'the cause; the stored statistic counts tasks with on_time=False,',
              'including unfinished tasks, not just observed deadline-miss events.',
              '', '7. CONTRIBUTION DECISION AND NEXT BOUNDED GATE',
              'Retain the implemented DDQN controller and measured storage reduction',
              'as supported engineering results. Robust mechanism benefit, the full',
              'low-complexity claim and novelty against literature remain unproven.',
              'Do not select an alternative arm or best initialization as a successful',
              'predeclared primary after inspecting these outcomes.',
              'Before one prospective correction brief, inspect loss/gradient geometry',
              'by horizon on saved checkpoints and matched saved states, and inspect',
              'arrival/link/queue/deadline timelines for the concentrated misses.',
              'These are offline diagnostics: no optimizer steps, new task seeds or',
              'threshold changes. Test whether horizon scaling, gradient allocation',
              'or decoder ranking actually supplies evidence for a targeted correction.',
              'No confirmed root cause is established by aggregate summaries alone.',
              'Then freeze one correction and its ablations with fresh development',
              'seeds and a literature check before a separately authorized new run.',
              'Keep all historical failures and the final reservations protected.',
              '', 'SOURCES', str(protocol_path), str(RUN / 'analysis.json'),
              str(RUN / 'costs.json'), str(RUN / 'optimizer_distributions.json'),
              str(BASE / 'audited_completion/completion.json')]
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [protocol_path, RUN/'analysis.json', RUN/'costs.json', RUN/'optimizer_distributions.json',
              BASE/'audited_completion/completion.json', BASE/'audited_completion/saved_artifact_audit.json']
    result = dict(kind='POST_AUDIT_SAVED_RESULTS_REVIEW_NOT_NEW_RESEARCH',
                  protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                  input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
                  recomputed_screens=gates, saved_screens_match=True,
                  failed_reference_and_heuristic_comparisons=failures,
                  primary_value_checks=values, primary_cost=cost_rows,
                  new_training_slots=0, new_optimizer_updates=0,
                  new_evaluation_slots=0, retiming=False, final_phase_launched=False,
                  word_thesis_changed=False)
    (OUT/'review_verification.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    report = ROOT/'ASSIGNMENT_PRICE_AUDITED_RESULTS_AND_CONTRIBUTION_DECISION_2026-11-19.txt'
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path,digest in result['input_sha256'].items())
    print(json.dumps(dict(report=str(report), verification=str(OUT/'review_verification.json'),
                          saved_screens_match=True, recomputed_screens=gates)))


if __name__ == '__main__':
    main()
