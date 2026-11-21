"""Diagnostic ORT operator profile; saved inputs only, not a cost-gate rerun."""
import argparse,collections,json,time
from pathlib import Path
import numpy as np
import onnx,onnxruntime as ort
import run_controller_v3_engineering as g

ARMS=('amortized_workload_price24','plain24_nonpositive','additive128_unbounded')
def profile(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    original=g.ROOT/'validation/controller_v3_engineering_2026-11-05_01'
    source_inputs=g.read(original/'run_manifest.json')['files']
    for name,h in source_inputs.items():assert g.r.old.sha(original/name)==h
    common=np.load(original/'observations.npz')['observations'][:128]
    profiles={};records=[]
    for arm in ARMS:
        path=original/'exports'/f'{arm}_99021.onnx'
        opts=g.c.options();opts.enable_profiling=True;opts.profile_file_prefix=str(out/f'ort_{arm}')
        opts.optimized_model_filepath=str(out/f'optimized_{arm}.onnx')
        sess=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
        for state in common[:32]:g.action(sess,state)
        for round_ in range(3):
            for ix,state in enumerate(common):
                t0=time.perf_counter_ns();x=np.asarray(state,dtype=np.float32)
                if x.shape!=(533,) or not np.isfinite(x).all():raise ValueError('input')
                t1=time.perf_counter_ns();a,q=sess.run(['actions','chosen_q'],{'states':x[None]})
                t2=time.perf_counter_ns();action=a[0].copy();t3=time.perf_counter_ns()
                records.append(dict(arm=arm,round=round_,observation=ix,input_ns=t1-t0,runtime_ns=t2-t1,output_ns=t3-t2))
        raw=Path(sess.end_profiling());events=g.read(raw)
        nodes=[e for e in events if e.get('cat')=='Node' and e.get('name','').endswith('_kernel_time')]
        operators=collections.defaultdict(lambda:dict(calls=0,total_us=0.))
        for e in nodes:
            op=e.get('args',{}).get('op_name','unknown');operators[op]['calls']+=1;operators[op]['total_us']+=e['dur']
        kernels=collections.defaultdict(lambda:dict(calls=0,total_us=0.))
        for e in nodes:
            k=e['name'];kernels[k]['calls']+=1;kernels[k]['total_us']+=e['dur']
        rows=[r for r in records if r['arm']==arm]
        graph=onnx.load(out/f'optimized_{arm}.onnx')
        profiles[arm]=dict(source_graph_sha256=g.r.old.sha(path),profile_file=raw.name,
            optimized_nodes=len(graph.graph.node),optimized_ops=dict(collections.Counter(n.op_type for n in graph.graph.node)),
            instrumented_mean_segment_us={k:float(np.mean([r[k+'_ns'] for r in rows]))/1000 for k in ['input','runtime','output']},
            operators=sorted([dict(operator=k,**v) for k,v in operators.items()],key=lambda z:z['total_us'],reverse=True),
            kernels=sorted([dict(name=k,**v) for k,v in kernels.items()],key=lambda z:z['total_us'],reverse=True),
            total_instrumented_kernel_us=sum(e['dur'] for e in nodes),kernel_events=len(nodes))
    result=dict(kind='instrumented_diagnostic_not_benchmark_or_quality_evidence',profiles=profiles,
        new_simulator_steps=0,optimizer_updates=0,research_training_started=False,source_hashes={__file__:g.r.old.sha(__file__)})
    g.r.write(out/'profile.json',result);g.r.write(out/'raw_segments.json',records)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);a=parser.parse_args();profile(a.output)
