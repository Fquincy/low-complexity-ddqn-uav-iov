"""Read saved development traces only. No simulator import, checkpoint load or training."""
import gzip
import hashlib
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'validation/value_bound_development_2026-11-04_01'
OUT = ROOT / 'analysis/controller_redesign_2026-11-05'
ARMS = ['additive128_unbounded', 'additive64_unbounded', 'additive64_nonpositive', 'emc_q_unbounded', 'emc_q_nonpositive']
INPUTS = {}

def read(path):
    raw = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)

def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.fmean(xs) if xs else None

def quantile(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs: return None
    i = (len(xs)-1)*q
    j = int(i)
    return xs[j] + (xs[min(j+1,len(xs)-1)]-xs[j])*(i-j)

def summarize(rows):
    dest = defaultdict(list)
    slots = defaultdict(list)
    vehicle = defaultdict(list)
    vehicle_destinations = defaultdict(Counter)
    moves = Counter()
    components = defaultdict(list)
    energies = defaultdict(list)
    for ep in rows:
        assert len(ep['slots']) == 40 and len(ep['tasks']) == ep['tasks_generated'] == 1200
        assert sum(bool(t['on_time']) for t in ep['tasks']) == ep['tasks_on_time']
        for t in ep['tasks']:
            dest[t['destination']].append(t)
            slots[int(t['arrival_s'])].append(t)
            vehicle[t['vehicle_id']].append(t)
            vehicle_destinations[t['vehicle_id']][t['destination']] += 1
        for slot in ep['slots']:
            moves[tuple(slot['action'][-2:])] += 1
            for k,v in slot['reward_components'].items(): components[k].append(v)
        for k,v in ep['summary'].items():
            if k.endswith('_energy_j'): energies[k].append(v)
    n = sum(len(ts) for ts in dest.values())
    def task_stats(ts):
        completed = [t for t in ts if t['status']=='completed']
        return dict(count=len(ts), fraction=len(ts)/n, on_time=sum(t['on_time'] for t in ts),
                    success=mean([float(t['on_time']) for t in ts]),
                    completed=len(completed), unfinished=sum(t['unfinished_at_horizon'] for t in ts),
                    delay_s_mean=mean([t['delay_s'] for t in completed]),
                    queue_delay_s_mean=mean([t['queue_delay_s'] for t in completed]),
                    queue_delay_s_p95=quantile([t['queue_delay_s'] for t in completed],.95),
                    uplink_wait_s_mean=mean([t['uplink_wait_s'] for t in completed]),
                    deadline_excess_s_mean=mean([max(0,t['observed_elapsed_s']-t['deadline_s']) for t in ts]))
    return dict(episodes=len(rows), return_mean=mean([r['return_sum'] for r in rows]),
                deadline_mean=mean([r['deadline_success'] for r in rows]),
                destinations={str(k):task_stats(dest[k]) for k in range(5) if dest[k]},
                arrival_slots={str(k):task_stats(v) for k,v in slots.items()},
                vehicles={str(k):task_stats(v) for k,v in vehicle.items()},
                per_vehicle_destination_counts={str(k):dict(v) for k,v in vehicle_destinations.items()},
                mean_per_vehicle_modal_destination_fraction=mean([max(v.values())/sum(v.values()) for v in vehicle_destinations.values()]),
                movement_pairs={str(k):v for k,v in moves.items()},
                reward_component_slot_means={k:mean(v) for k,v in components.items()},
                energy_episode_means={k:mean(v) for k,v in energies.items()})

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    analysis = read(RUN/'development_analysis.json')
    audit = read(RUN/'saved_artifact_audit.json')
    manifest = read(RUN/'run_manifest.json')['files']
    result = dict(purpose='retrospective_exploratory_saved_trace_diagnosis_not_new_research_evaluation',
                  simulator_steps=0, optimizer_updates=0, source_run=str(RUN), models={}, heuristics={},
                  previous_audit=audit, comparisons=analysis['replicates'])
    pairings = {}
    for rep, seed in enumerate([3109,3119,3121]):
        for arm in ARMS:
            key=f'{arm}_{rep}'
            rows=read(RUN/key/'fresh.json.gz')
            fingerprints=[(r['seed'],r['workload_sha256']) for r in rows]
            assert len(rows)==48
            if rep in pairings: assert fingerprints==pairings[rep]
            else: pairings[rep]=fingerprints
            summary=summarize(rows)
            counts=read(RUN/key/'counters.json')
            training=read(RUN/key/'training_summary.json')
            validation=read(RUN/key/'validation.json')
            summary.update(model_seed=seed, counters=counts,
                train_blocks={str(start+1)+'-'+str(start+50):{
                    'return_mean':mean([r['return_sum'] for r in training[start:start+50]]),
                    'deadline_mean':mean([r['deadline_success'] for r in training[start:start+50]])}
                    for start in range(0,200,50)},
                validation={str(ep):{'return_mean':mean([r['return_sum'] for r in rr]),
                                    'deadline_mean':mean([r['deadline_success'] for r in rr])}
                            for ep,rr in validation.items()})
            # Comparing greedy Q on the fixed probes against a return of another policy is invalid.
            # Preserve probe distributions; use fresh terminal Q only against its own final reward.
            terminal_errors=[abs(r['terminal_value']-r['slot_rewards'][-1]) for r in rows]
            summary['terminal_abs_error_p95_recomputed']=quantile(terminal_errors,.95)
            summary['terminal_abs_error_mean']=mean(terminal_errors)
            relative=Path(key)/'fresh.json.gz'
            assert INPUTS[str((RUN/relative).relative_to(ROOT))]==manifest[relative.as_posix()]
            result['models'][key]=summary
            print(f'{seed} {arm}: R={summary["return_mean"]:.4f} deadline={summary["deadline_mean"]:.5f}',flush=True)
    for heuristic in ['all_local','nearest_rsu','uav_selective']:
        rows=read(RUN/f'fresh_heuristic_{heuristic}.json.gz')
        assert [(r['seed'],r['workload_sha256']) for r in rows]==pairings[0]
        result['heuristics'][heuristic]=summarize(rows)
    result['input_sha256']=INPUTS
    # All artifacts read remain byte-identical at completion.
    for rel,digest in INPUTS.items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest, rel
    result['input_hashes_unchanged_after_analysis']=True
    result['fresh_model_workload_pairing_verified']=True
    (OUT/'saved_trace_diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('Saved diagnosis; original inputs unchanged; no simulator steps or optimizer updates.',flush=True)

if __name__=='__main__': main()
