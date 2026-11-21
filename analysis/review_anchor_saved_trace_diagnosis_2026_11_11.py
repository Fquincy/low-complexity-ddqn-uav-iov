"""Reconcile saved-trace diagnosis with frozen audited comparison summaries."""
import json,hashlib,math,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'development_runs/terminal_anchor_development_continuation_2026-11-11_01';D=R/'validation/anchor_saved_trace_diagnosis_2026-11-11_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def main():
    x=read(D/'diagnosis.json');analysis=read(O/'comparisons_and_value_stability.json');manifest=read(O/'run_manifest.json')['files']
    hashes=read(D/'INPUT_HASHES.json')
    for name,digest in hashes.items():assert manifest[str(Path(name).relative_to(O.relative_to(R))).replace('\\','/')]==digest,name
    rows={(r['model_seed'],r['arm']):r for r in x['per_fit']};assert len(rows)==30
    seeds=read(O/'runtime_plan.json')['model_seeds'];refs=['no_price_both','additive64_nonpositive','additive128_unbounded','nearest_rsu','uav_selective']
    for (seed,arm),row in rows.items():
        b=row['buckets'];assert b['all']['tasks']==57600
        for group in [[f'destination_{d}' for d in range(5)],[f'time_{d}' for d in range(4)],['local_possible','local_impossible']]:
            for key in ['tasks','on_time','unfinished','local_impossible']:
                assert sum(b.get(g,{}).get(key,0) for g in group)==b['all'][key]
        assert math.isclose(-sum(row['reward_cost_per_episode'].values()),row['mean_return'],abs_tol=1e-6)
        local=b.get('destination_0',{})
        assert local.get('tasks',0)-local.get('on_time',0)==local.get('local_impossible',0)
    for seed in seeds:
        for ref in refs:
            cells=[r for r in x['paired_destination_outcomes'] if r['model_seed']==seed and r['reference']==ref]
            assert sum(r['tasks'] for r in cells)==57600
            primary=rows[(seed,'price_both')];other=rows[(seed,ref)]
            assert sum(r['primary_success_reference_miss']-r['primary_miss_reference_success'] for r in cells)==primary['buckets']['all']['on_time']-other['buckets']['all']['on_time']
            expected=next(v for v in analysis['comparisons'][ref]['per_fit'] if v['seed']==seed)
            assert math.isclose(primary['mean_return']-other['mean_return'],expected['return_difference'],abs_tol=1e-8)
            assert math.isclose(primary['buckets']['all']['success_rate']-other['buckets']['all']['success_rate'],expected['deadline_difference'],abs_tol=1e-10)
    aggregated={}
    for arm in ['price_both',*refs]:
        rr=[rows[(s,arm)] for s in seeds];n=sum(r['buckets']['all']['tasks'] for r in rr)
        b={}
        for name in set(k for r in rr for k in r['buckets']):
            total=sum(r['buckets'].get(name,{}).get('tasks',0) for r in rr)
            success=sum(r['buckets'].get(name,{}).get('on_time',0) for r in rr)
            b[name]=dict(tasks=total,on_time=success,success_rate=success/total if total else None,
                local_impossible=sum(r['buckets'].get(name,{}).get('local_impossible',0) for r in rr))
        aggregated[arm]=dict(buckets=b,mean_return=sum(r['mean_return'] for r in rr)/5,
            reward_cost_per_episode={k:sum(r['reward_cost_per_episode'][k] for r in rr)/5 for k in rr[0]['reward_cost_per_episode']})
    audit=dict(state='passed_paired_count_reward_and_frozen_summary_reconciliation',per_fit_arm_rows=30,unique_input_files=len(hashes),
        matched_task_pairs=5*5*48*1200,original_manifest_hash_entries_matched=len(hashes),
        bucket_totals_reconcile=True,paired_outcome_totals_reconcile=True,all_frozen_comparison_differences_match=True,
        reward_components_reconstruct_saved_returns=True,new_simulator_slots=0,new_optimizer_updates=0)
    write(D/'verification.json',audit);write(D/'aggregate_summary.json',aggregated)
    p=aggregated['price_both']['buckets'];n=p['all']['tasks'];miss=n-p['all']['on_time'];local_bad=p['destination_0']['local_impossible']
    report=R/'TERMINAL_ANCHOR_SAVED_TRACE_DIAGNOSIS_2026-11-11.txt'
    lines=['TERMINAL-ANCHOR SAVED-TRACE DIAGNOSIS','Date: November 11, 2026','','FINDING',
        'Deadline losses concentrate on locally assigned tasks that cannot meet',
        'their deadline even with an empty local CPU queue. This is supported',
        'by saved task cycles, deadlines, destinations and outcomes. It does not',
        'prove why the learner selected those actions or that an arbitrary',
        'offloading replacement would succeed under the same system dynamics.',
        '', 'SCOPE AND CHECKS',
        'Primary, corrected no-price, compact64, wide128, nearest-RSU and',
        'UAV-selective; all five initializations, all48 consumed development',
        'evaluation tasks.1,056 unique compressed traces read; original manifest',
        'hashes matched.1,440,000 paired task comparisons checked against task',
        'identity, size, cycles, deadline and arrival. Destination/time/feasibility',
        'bucket totals, paired deadline counts and component-return reconstruction',
        'reconcile with the frozen audited per-fit comparisons.',
        'Heuristic traces are reused across five learned fits for matching; these',
        'are48 unique heuristic workloads, not240 independent heuristic episodes.',
        'Per-fit results retained; pooled task counts are descriptive and are not',
        'independent statistical replicates or post-hoc acceptance criteria.',
        '', 'TASK ALLOCATION',
        'controller | deadline success % | local allocation % | infeasible local % of all tasks']
    for arm,v in aggregated.items():
        b=v['buckets'];total=b['all']['tasks'];local=b.get('destination_0',{})
        lines.append(f"{arm} | {100*b['all']['success_rate']:.4f} | {100*local.get('tasks',0)/total:.4f} | {100*local.get('local_impossible',0)/total:.4f}")
    lines +=[f'Primary: {local_bad:,} infeasible local tasks / {miss:,} deadline failures',
        f'({100*local_bad/miss:.3f}% of failures). All primary local failures equal',
        'the analytical local-infeasibility count in these saved traces.',
        'Test: required_cycles /1.5e9 > deadline_s. This is a necessary local',
        'feasibility condition, ignoring queue delay, not a counterfactual oracle.',
        'The observed equality indicates the chosen local tasks fail from their',
        'own service requirement in these traces; it does not establish a general',
        'absence of local queue effects in other workloads.',
        'Primary RSU1/RSU2 assignments succeeded at',
        f"{100*p['destination_3']['success_rate']:.4f}% / {100*p['destination_4']['success_rate']:.4f}%.",
        'That conditional success does not prove all rejected local tasks can be',
        'offloaded together without introducing congestion/radio delays.',
        '', 'PAIRED DEADLINE ACCOUNTING',
        'Counts below match identical workload tasks under different policies.',
        'Policy-dependent queue and UAV histories differ, so they are descriptive',
        'outcome comparisons rather than isolated single-action interventions.',
        'reference | net extra primary failures on primary-local tasks | net total extra primary failures']
    for ref in refs:
        cells=[r for r in x['paired_destination_outcomes'] if r['reference']==ref]
        net=lambda subset:sum(r['primary_miss_reference_success']-r['primary_success_reference_miss'] for r in subset)
        lines.append(f"{ref} | {net([r for r in cells if r['primary_destination']==0]):,} | {net(cells):,}")
    lines +=['Other destinations partially offset local losses. Thus local-allocation',
        'differences account for more than the net deadline gap against references.',
        '', 'REWARD TRADEOFF',
        'controller | delay cost/episode | energy cost/episode | miss cost/episode | return']
    for arm,v in aggregated.items():
        k=v['reward_cost_per_episode'];lines.append(f"{arm} | {k['delay_cost']:.5f} | {k['energy_cost']:.5f} | {k['miss_cost']:.5f} | {v['mean_return']:.5f}")
    lines +=['Reward is the negative sum of these three existing weighted costs.',
        'The primary gains mean return over compact64/wide128 mainly by reducing',
        'accumulated delay cost, while incurring more deadline-miss cost. Against',
        'no-price and the strong heuristics, this tradeoff is not an improvement.',
        'The reward includes a miss penalty, but no explicit hard deadline-success',
        'constraint. A scalar return gain alone cannot support the promised joint',
        'return/deadline claim. Do not retune frozen reward weights retrospectively.',
        '', 'VALUE EVIDENCE',
        'Earlier audited primary normalized RMSE0.288-0.636 and absolute bias',
        '2.349-10.557 fail in every fit. Residual Q-minus-realized-return means',
        'are positive: predictions are overoptimistic along those trajectories.',
        'Terminal error p950.361-0.586 passes, so successful terminal anchoring',
        'has not established sufficient long-horizon policy-return calibration.',
        'The current traces localize the performance deficit but do not show',
        'whether value bias causes local selection or follows poor visitation.',
        '', 'NEXT PROSPECTIVE DESIGN STAGE',
        'Write one justified deadline-aware controller extension within the same',
        'offloading problem before another research run. A concrete candidate is',
        'a necessary-feasibility feature/gate for provably impossible local tasks,',
        'with an explicitly specified fallback when no destination is certified.',
        'It must handle radio/remote congestion, preserve DDQN selection/target',
        'semantics, and include its full action-selection cost. A local test alone',
        'does not guarantee any remote deadline or improvement.',
        'Distinguish known feasibility masking from any genuinely new mechanism:',
        'review relevant literature before asserting novelty. Compare a plain small',
        'DDQN with the same feasibility information, mechanism-removed ablation,',
        'strong references and heuristics. This prevents attributing an information',
        'or safety-screen advantage solely to architectural novelty.',
        'Keep the present run and reward fixed. Declare any objective/constraint',
        'change prospectively as a new experiment with fresh development seeds.',
        'Investigate long-horizon target calibration separately using saved data;',
        'do not assume that terminal correction or more training solves it.',
        'No new architecture is selected, no thresholds relaxed, and no retraining',
        'or reserved final evaluation is authorized by this diagnosis.',
        '', 'LIMITATIONS',
        'No off-policy simulator interventions were run. Conditional destination',
        'success rates confound allocation, workload and policy trajectories.',
        'Queue/upload wait sums are retained per destination in diagnosis.json;',
        'they describe final recorded task outcomes, not every causal queue effect.',
        'No formal uncertainty interval or new statistical acceptance test added.',
        'Security/trust deferred; cloud and charging stations excluded.',
        'Word thesis unchanged. Final evaluation remains unused.']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    tracker=R/'LOW_COMPLEXITY_PROJECT_STATUS.json';shutil.copy2(tracker,D/'PROJECT_STATUS_BEFORE_DIAGNOSIS.json');s=read(tracker)
    s['current_workstream']='anchor_saved_trace_diagnosis_local_infeasible_allocation_and_reward_tradeoff_documented'
    s['next']='prospective_deadline_aware_design_brief_literature_and_fair_ablation_before_new_training'
    s['anchor_saved_trace_diagnosis']=dict(report=report.name,unique_trace_files=1056,matched_task_pairs=1440000,
        primary_local_fraction=p['destination_0']['tasks']/n,primary_infeasible_local_fraction=local_bad/n,
        local_infeasibility_fraction_of_primary_failures=local_bad/miss,new_simulator_slots=0,new_optimizer_updates=0,final_launched=False,word_thesis_changed=False)
    write(tracker,s)
    paths=[q for q in D.iterdir() if q.is_file()]+[report,Path(__file__),R/'analysis/diagnose_anchor_saved_traces_2026_11_11.py']
    write(D/'COMPLETION_MANIFEST.json',dict(state=audit['state'],files={str(q.relative_to(R)).replace('\\','/'):sha(q) for q in paths}))
    print(json.dumps(dict(report=report.name,verification=audit,local_infeasibility_fraction_of_primary_failures=local_bad/miss)))
if __name__=='__main__':main()
