"""Instrumentation-only profile using saved engineering observations. No training."""
import json,time
from pathlib import Path
import numpy as np
import torch
import controller_redesign_v2 as m
import run_controller_redesign_engineering as r

ROOT=Path(__file__).resolve().parent
INPUT=ROOT/'validation/controller_redesign_engineering_2026-11-05_01/cost_gate/common_observations.npz'

def profile(output):
    r.configure();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    states=np.load(INPUT)['observations'];model=r.build('shared_congestion_cost',99001,r.fixture_cfg())['online'].eval()
    before=r.old.digest(model.state_dict());records=[]
    with torch.inference_mode():
        for s in states[:32]:model.greedy(torch.from_numpy(s)[None])
        for round_ in range(3):
            for index,state in enumerate(states):
                times=[time.perf_counter_ns()]
                x=torch.from_numpy(np.asarray(state,dtype=np.float32))[None];times.append(time.perf_counter_ns())
                tables=model.tables(x);times.append(time.perf_counter_ns())
                actions,cost,details=m.decode_cost_tables(*tables,enabled=True);times.append(time.perf_counter_ns())
                q=-cost;times.append(time.perf_counter_ns())
                action=actions[0].numpy().copy();times.append(time.perf_counter_ns())
                records.append(dict(round=round_,observation=index,**{name:times[i+1]-times[i] for i,name in enumerate(
                    ['input_conversion','tables_including_parser_network_and_cost_terms','decoder_two_prices_four_repairs','value_negation','action_conversion'])}))
    assert r.old.digest(model.state_dict())==before
    keys=['input_conversion','tables_including_parser_network_and_cost_terms','decoder_two_prices_four_repairs','value_negation','action_conversion']
    means={k:float(np.mean([x[k] for x in records]))/1e6 for k in keys};total=sum(means.values())
    # CPU operator counts provide corroborating detail; instrumented times are
    # not interpreted as a new cost-gate measurement or a trained policy result.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        with torch.inference_mode():
            for s in states[:8]:model.greedy(torch.from_numpy(s)[None])
    operators=sorted([dict(operator=x.key,calls=x.count,self_cpu_us=x.self_cpu_time_total) for x in prof.key_averages()],key=lambda x:x['self_cpu_us'],reverse=True)
    result=dict(kind='instrumentation_diagnosis_not_cost_gate',input_sha256=r.old.sha(INPUT),source_sha256={n:r.old.sha(ROOT/n) for n in r.SOURCES},
        mean_segment_ms=means,segment_mean_percent={k:v/total*100 for k,v in means.items()},operators=operators,
        research_training_slots=0,optimizer_updates=0,new_simulator_steps=0,model_seed=99001,hardware=r.hardware())
    r.write(out/'profile.json',result);r.write(out/'raw_profile.json',records)
    print(json.dumps({k:v for k,v in result.items() if k not in ['operators','source_sha256']},indent=2))
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();profile(a.output)
