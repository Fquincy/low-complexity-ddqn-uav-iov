"""Descriptive paired saved-trace diagnosis; no simulation or training."""
import gzip,json,hashlib,math
from pathlib import Path
from collections import defaultdict,Counter
R=Path(__file__).resolve().parents[1];O=R/'development_runs/terminal_anchor_development_continuation_2026-11-11_01'
D=R/'validation/anchor_saved_trace_diagnosis_2026-11-11_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p,hashes):
    hashes[str(p.relative_to(R)).replace('\\','/')]=sha(p)
    with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
def add(bucket,t):
    bucket['tasks']+=1;bucket['on_time']+=bool(t['on_time']);bucket['unfinished']+=bool(t['unfinished_at_horizon'])
    bucket['queue_s']+=t['queue_delay_s'] or 0;bucket['upload_wait_s']+=t['uplink_wait_s'] or 0
    bucket['transmission_s']+=t['transmission_delay_s'] or 0
    bucket['compute_s']+=t['computation_delay_s'] or 0
    bucket['local_impossible']+=t['required_cycles']/1.5e9>t['deadline_s']+1e-10
    bucket['deadline_sum']+=t['deadline_s']
def finish(c):
    n=c['tasks'];return dict(c,success_rate=c['on_time']/n if n else None,
        local_impossible_fraction=c['local_impossible']/n if n else None,
        mean_queue_s=c['queue_s']/n if n else None,mean_upload_wait_s=c['upload_wait_s']/n if n else None)
def path(arm,fit,seed):return O/f'heuristic_{arm}_{seed}.json.gz' if arm in ['nearest_rsu','uav_selective'] else O/arm/f'fit_{fit:02d}'/f'evaluation_{seed}.json.gz'
def main():
    assert not D.exists();D.mkdir();plan=read(O/'runtime_plan.json');analysis=read(O/'comparisons_and_value_stability.json')
    hashes={};results=[];pair_results=[]
    refs=['no_price_both','additive64_nonpositive','additive128_unbounded','nearest_rsu','uav_selective']
    for fit,model_seed in enumerate(plan['model_seeds']):
        buckets=defaultdict(Counter);components=defaultdict(Counter);pair=defaultdict(Counter);returns=defaultdict(list)
        for seed in plan['evaluation_seeds']:
            primary=load(path('price_both',fit,seed),hashes)
            rows={'price_both':primary}
            for arm in refs:rows[arm]=load(path(arm,fit,seed),hashes)
            for arm,row in rows.items():
                assert row['workload_sha256']==primary['workload_sha256'] and len(row['tasks'])==1200
                returns[arm].append(row['return_sum']);seen=set()
                for t in row['tasks']:
                    assert t['task_id'] not in seen;seen.add(t['task_id']);dest=t['destination'];slot=int(t['task_id'])//30
                    assert row['slots'][slot]['action'][t['vehicle_id']]==dest
                    add(buckets[(arm,'all')],t);add(buckets[(arm,f'destination_{dest}')],t)
                    add(buckets[(arm,f'time_{slot//10}')],t)
                    add(buckets[(arm,'local_impossible' if t['required_cycles']/1.5e9>t['deadline_s']+1e-10 else 'local_possible')],t)
                assert sum(t['on_time'] for t in row['tasks'])==row['tasks_on_time']
                for s in row['slots']:
                    comp=s['reward_components'];components[arm]['delay_cost']+=.5*comp['normalized_delay_accrual']
                    components[arm]['energy_cost']+=.5*comp['normalized_uav_energy']
                    components[arm]['miss_cost']+=comp['new_deadline_misses_per_vehicle']
                reconstructed=-sum(components[arm][k] for k in ['delay_cost','energy_cost','miss_cost'])
                assert abs(reconstructed-sum(returns[arm]))<1e-5
            primary_by={t['task_id']:t for t in primary['tasks']}
            for arm in refs:
                for t in rows[arm]['tasks']:
                    p=primary_by[t['task_id']]
                    for field in ['vehicle_id','arrival_s','required_cycles','deadline_s','input_bits']:assert p[field]==t[field]
                    c=pair[(arm,p['destination'],t['destination'])];c['tasks']+=1
                    c['primary_miss_reference_success']+=not p['on_time'] and t['on_time']
                    c['primary_success_reference_miss']+=p['on_time'] and not t['on_time']
        for arm in ['price_both',*refs]:
            n=buckets[(arm,'all')]['tasks'];assert n==57600
            results.append(dict(model_seed=model_seed,arm=arm,buckets={k[1]:finish(v) for k,v in buckets.items() if k[0]==arm},
                mean_return=sum(returns[arm])/48,reward_cost_per_episode={k:v/48 for k,v in components[arm].items()}))
        for (arm,pd,rd),v in pair.items():pair_results.append(dict(model_seed=model_seed,reference=arm,primary_destination=pd,reference_destination=rd,**v))
        print(json.dumps(dict(fit=fit,model_seed=model_seed,complete=True)),flush=True)
    output=dict(kind='descriptive_saved_trace_analysis_not_causal',per_fit=results,paired_destination_outcomes=pair_results,
        primary_value_stability=analysis['value_stability']['price_both'],new_simulator_slots=0,new_optimizer_updates=0,
        local_feasibility_definition='required_cycles/1.5e9<=deadline; best-case necessary condition only, ignores queue; not a measured offload counterfactual',
        final_evaluation_opened=False)
    (D/'diagnosis.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    (D/'INPUT_HASHES.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(hashes),output=str(D))))
if __name__=='__main__':main()
