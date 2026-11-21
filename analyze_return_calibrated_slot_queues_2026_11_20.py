"""Attribute recorded delay differences; do not run alternative assignments."""
import json
from pathlib import Path
import numpy as np
import diagnose_return_calibrated_saved_traces_2026_11_20 as d


def main():
    dest=d.OUT/'within_slot_queue_evidence.json'
    if dest.exists(): raise RuntimeError('Preserve previous within-slot analysis')
    spec=d.read(d.SPEC);manifest=d.read(d.RUN/'manifest.json')
    result=dict(kind='SAVED_TASK_QUEUE_ACCOUNTING_NOT_COUNTERFACTUAL_SIMULATION',fits={})
    for fit in range(3):
        tasks={};loads={};maximum_delay={}
        for arm in [d.A,d.E,d.F]:
            eps=[d.read(d.RUN/'evaluation'/arm/str(fit)/f'episode_{i:03d}.json') for i in range(16)]
            tasks[arm]=[t for ep in eps for t in ep['records']]
            maximum_delay[arm]=max(t['delay_s'] for t in tasks[arm])
            loads[arm]={}
            for destination in [3,4]:
                counts=[];service=[]
                for ep in eps:
                    for slot in range(40):
                        part=[t for t in ep['records'] if t['arrival_s']==slot and t['destination']==destination]
                        counts.append(len(part));service.append(sum(t['required_cycles'] for t in part)/spec['simulator_defaults']['rsu_cpu_hz'][destination-3])
                loads[arm][str(destination)]=dict(tasks_per_slot=d.stats(counts),assigned_CPU_service_seconds_per_slot=d.stats(service))
        new_same=[]
        for p,e in zip(tasks[d.A],tasks[d.E]):
            assert all(p[k]==e[k] for k in ['task_id','vehicle_id','arrival_s','required_cycles','input_bits','deadline_s'])
            if not p['on_time'] and e['on_time'] and p['destination']==e['destination']==4:new_same.append((p,e))
        checks=[]
        for p,e in new_same:
            # Same RSU, fixed independent link, same task and CPU frequency:
            # changed CPU waiting accounts for the entire realized delay gap.
            diff=lambda k:p[k]-e[k]
            assert np.isclose(diff('delay_s'),diff('queue_delay_s'),rtol=1e-7,atol=1e-8)
            for k in ['transmission_delay_s','computation_delay_s','uplink_wait_s']:
                assert np.isclose(diff(k),0.,rtol=0.,atol=1e-8)
            checks.append(diff('queue_delay_s'))
        result['fits'][str(fit)]=dict(assigned_load_by_arm=loads,maximum_task_delay_s=maximum_delay,
            all_core_tasks_completed_before_next_slot=all(v<1. for v in maximum_delay.values()),
            newly_missed_same_RSU2_tasks=len(new_same),extra_CPU_queue_wait_s=d.stats(checks),
            primary_delay_s=d.stats([p['delay_s'] for p,e in new_same]),
            bc_delay_s=d.stats([e['delay_s'] for p,e in new_same]),
            identical_link_compute_and_uplink_wait_components_verified=True)
    for name,digest in d.consumed.items():
        p=d.ROOT/name
        assert d.sha(p)==digest
        if p.is_relative_to(d.RUN) and p!=d.RUN/'manifest.json':assert manifest[p.relative_to(d.RUN).as_posix()]==digest
    result.update(consumed_sha256=d.consumed,consumed_files_verified_unchanged=len(d.consumed),
        script_sha256=d.sha(Path(__file__)),new_simulator_steps=0,new_optimizer_updates=0,new_teacher_calls=0,
        limitation='Recorded queue waiting explains these same-task delay differences in realized runs. It does not show that a proposed alternative allocation or gradient change will improve future policy.')
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({fit: {k:v for k,v in row.items() if k!='assigned_load_by_arm'} for fit,row in result['fits'].items()},indent=2))


if __name__=='__main__':main()
