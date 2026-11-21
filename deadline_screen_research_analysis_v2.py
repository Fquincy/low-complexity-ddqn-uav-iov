"""Prospectively specified summaries and decisions; no model selection."""
import numpy as np


def distribution(values):
    x = np.asarray(values, np.float64).reshape(-1)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('empty/nonfinite diagnostic distribution')
    return dict(count=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p05=float(np.quantile(x,.05)), p95=float(np.quantile(x,.95)), min=float(x.min()), max=float(x.max()))


def values(rows):
    q=np.array([r['chosen_q'] for r in rows],np.float64)
    target=np.array([r['target_q'] for r in rows],np.float64)
    g=np.array([r['realized_return_to_go'] for r in rows],np.float64)
    e=q-g
    def part(ix):
        return {k:distribution(v[:,ix]) for k,v in [('Q',q),('target',target),('G',g),('Q_minus_G',e),('target_minus_G',target-g)]}
    destinations=np.sum([r['destination_counts'] for r in rows],axis=0)
    misses=np.sum([r['deadline_misses_by_destination'] for r in rows],axis=0)
    return dict(normalized_rmse=float(np.sqrt(np.mean(e**2))/max(1.,np.sqrt(np.mean(g**2)))),
        absolute_bias=float(abs(e.mean())),near_zero_large_return_fraction=float(((abs(q)<=1e-6)&(abs(g)>=1.)).mean()),
        terminal_abs_error_p95=float(np.quantile(abs(q[:,-1]-np.array([r['slot_rewards'][-1] for r in rows])),.95)),
        positive_Q_fraction=float((q>0).mean()),all_slots=part(slice(None)),terminal=part(slice(39,40)),
        nonterminal=part(slice(0,39)),time_bins=[dict(start=i,end=i+9,distributions=part(slice(i,i+10))) for i in range(0,40,10)],
        destination_counts=destinations.tolist(),deadline_misses_by_destination=misses.tolist(),
        certified_infeasible_local_count=sum(r['certified_infeasible_local_count'] for r in rows))


def contrast(left,right):
    if [(r['seed'],r['workload_sha256']) for r in left] != [(r['seed'],r['workload_sha256']) for r in right]:
        raise ValueError('unpaired workload or duplicate/missing episode')
    return dict(return_difference=float(np.mean([x['return_sum']-y['return_sum'] for x,y in zip(left,right)])),
                deadline_difference=float(np.mean([x['deadline_success']-y['deadline_success'] for x,y in zip(left,right)])))


def summarize(p, evaluations, heuristics, costs, fixture=False):
    primary=p['primary']; fits=len(evaluations[primary]); comp={}
    for name in p['comparisons']['primary_report_against_each']:
        per=[contrast(evaluations[primary][i],heuristics[name] if name in heuristics else evaluations[name][i]) for i in range(fits)]
        comp[name]=dict(per_fit=per,mean_return_difference=float(np.mean([x['return_difference'] for x in per])),
                       mean_deadline_difference=float(np.mean([x['deadline_difference'] for x in per])))
    auxiliary=[]
    for x in p['comparisons']['auxiliary_contrasts']:
        auxiliary.append(dict(**x,per_fit=[contrast(evaluations[x['left']][i],evaluations[x['right']][i]) for i in range(fits)]))
    vd={arm:[values(rows) for rows in fitrows] for arm,fitrows in evaluations.items()}
    rules=p['comparisons']; gates={}
    def each(names, rmin,dmin):
        return all(x['return_difference']>=rmin and x['deadline_difference']>=dmin for n in names for x in comp[n]['per_fit'])
    gates['references_every_fit']=each(rules['required_reference_arms'],rules['reference_each_fit']['return_difference_min'],rules['reference_each_fit']['deadline_fraction_difference_min'])
    gates['strong_heuristics_every_fit']=each(rules['strong_heuristics'],0.,0.)
    for m in rules['mechanisms']:
        c=comp[m['right']]
        gates[m['id']+'_mechanism']=each([m['right']],m['each_fit_return_difference_min'],m['each_fit_deadline_fraction_difference_min']) and c['mean_return_difference']>=m['equal_fit_mean_return_difference_min'] and c['mean_deadline_difference']>=m['equal_fit_mean_deadline_fraction_difference_min']
    v=p['value_stability']
    gates['primary_value_every_fit']=all(x['normalized_rmse']<=v['normalized_rmse_max'] and x['absolute_bias']<=v['absolute_bias_max']
        and x['near_zero_large_return_fraction']<=v['near_zero_large_return_fraction_max'] and x['terminal_abs_error_p95']<=v['terminal_abs_error_p95_max']
        and x['positive_Q_fraction']==0 for x in vd[primary])
    cost_rules=p['action_cost']['primary_each_fit'];cost_screens=[]
    for i in range(fits):
        x=costs[primary][i];compact=costs['unmasked_compact64'][i]
        cost_screens.append(dict(storage=x['parameter_bytes']<=cost_rules['parameter_bytes_max_fraction_of_unmasked_compact64']*compact['parameter_bytes'],
            median=all(x['median_ms']<=cost_rules['median_ratio_max']*costs[n][i]['median_ms'] for n in cost_rules['median_ms_max_fraction_of_each']),
            p95=x['p95_ms']<=cost_rules['p95_ms_max']))
    gates['primary_cost_every_fit']=all(all(x.values()) for x in cost_screens)
    # Descriptive crossed bootstrap; independently repeated by auditor.
    rng=np.random.default_rng(p['fixture']['engineering_seeds']['bootstrap'] if fixture else p['rng']['bootstrap_seed'])
    tasks=len(evaluations[primary][0]); draw_f=rng.integers(fits,size=(p['uncertainty']['draws'],fits));draw_t=rng.integers(tasks,size=(p['uncertainty']['draws'],tasks))
    intervals={}
    for n in comp:
        matrices=[np.array([[row[field]-(heuristics[n] if n in heuristics else evaluations[n][i])[j][field] for j,row in enumerate(rows)]
                           for i,rows in enumerate(evaluations[primary])],np.float64) for field in ('return_sum','deadline_success')]
        ci=[]
        for mat in matrices:
            samples=np.array([mat[f[:,None],t[None,:]].mean() for f,t in zip(draw_f,draw_t)])
            ci.append(np.quantile(samples,[.025,.975]).tolist())
        intervals[n]=dict(return_interval95=ci[0],deadline_interval95=ci[1],descriptive_only=True)
    return dict(comparisons=comp,auxiliary_contrasts=auxiliary,value_stability=vd,cost_screens=cost_screens,
        computed_screens=gates,screens_applied=not fixture,all_scientific_screens_passed=all(gates.values()) if not fixture else None,
        quality_evidence=not fixture,bootstrap=intervals)
