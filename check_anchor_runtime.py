"""One declared saved-weight runtime check; no simulator or optimizer calls."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import run_anchor_research_preparation as r
import anchor_runtime_single_validation as optimized

ROOT=Path(__file__).resolve().parent
SPEC=ROOT/'ANCHOR_RUNTIME_CHECK_PROTOCOL_2026-11-06.json'

def audit(out):
    out=Path(out);pre=r.read(out/'PRE_TIMING_FREEZE.json');timing=r.read(out/'timings.json')
    assert timing['freeze_sha256']==r.base.old.sha(out/'PRE_TIMING_FREEZE.json')
    assert pre['frozen_ns']<timing['first_call_ns']
    for name,digest in pre['source_hashes'].items():assert r.base.old.sha(ROOT/name)==digest,name
    for name,digest in pre['payload_hashes'].items():assert r.base.old.sha(ROOT/name)==digest,name
    raw=timing['raw'];expected=[(i,key,j) for i,keys in enumerate(pre['orders']) for key in keys for j in range(128)]
    assert [(row['round'],row['path'],row['observation']) for row in raw]==expected
    assert len(raw)==29440 and all(row['ns']>0 for row in raw)
    metrics={}
    for key in pre['orders'][0]:
        x=np.array([row['ns']/1e6 for row in raw if row['path']==key])
        metrics[key]=dict(median_ms=float(np.median(x)),p95_ms=float(np.quantile(x,.95)),calls=len(x))
    assert metrics==timing['metrics']
    gates={}
    for fit in range(2):
        p=metrics[f'price_both/{fit}/new'];wide=metrics[f'additive128_unbounded/{fit}/original']
        gates[str(fit)]=dict(storage=68264<=.5*194944,median=p['median_ms']<=wide['median_ms'],p95=p['p95_ms']<=1.)
    result=dict(state='passed_runtime_artifact_audit',cost_screens=gates,
        all_cost_screens_pass=all(all(x.values()) for x in gates.values()),timing_rows_checked=len(raw),
        training_updates=0,simulator_steps=0,research_launched=False)
    r.write(out/'saved_artifact_audit.json',result);return result

def check(out):
    r.configure();r.fused.numba.set_num_threads(1)
    out=Path(out);out.mkdir(parents=True,exist_ok=False);spec=r.read(SPEC)
    evidence=ROOT/spec['evidence'];plan=r.read(evidence/'runtime_plan.json')
    states=np.load(evidence/'frozen_checkpoint_inference/observations.npz')['observations']
    runtimes={};payloads={};equivalence=[]
    for fit,seed in enumerate(plan['model_seeds']):
        for arm in r.ARMS:
            checkpoint=evidence/arm/f'fit_{fit:02d}'/'final.pt'
            model=r.make_model(arm,seed);model.load_state_dict(torch.load(checkpoint,weights_only=False)['online']);model.eval()
            graph=evidence/'frozen_checkpoint_inference/exports'/f'{arm}_fit_{fit:02d}.onnx'
            payloads[str(checkpoint.relative_to(ROOT)).replace('\\','/')]=r.base.old.sha(checkpoint)
            payloads[str(graph.relative_to(ROOT)).replace('\\','/')]=r.base.old.sha(graph)
            session=r.compiled.session(graph)
            if arm in r.base.SHARED_ARMS:
                call=lambda x,session=session:r.compiled.compiled_values(session,x)
            else:
                extra_path=graph.with_suffix('.extra.npy');extra=np.load(extra_path)
                payloads[str(extra_path.relative_to(ROOT)).replace('\\','/')]=r.base.old.sha(extra_path)
                call=lambda x,session=session,extra=extra,arm=arm:r.runtime_values((session,extra),arm,x)
            runtimes[f'{arm}/{fit}/original']=call
            if arm in r.CORRECTED:
                new=lambda x,session=session,extra=extra,arm=arm:optimized.values((session,extra),arm,x)
                runtimes[f'{arm}/{fit}/new']=new
                runtimes[f'{arm}/{fit}/old']=runtimes.pop(f'{arm}/{fit}/original')
            with torch.inference_mode():
                for state in states:
                    a,q,_=r.greedy(model,torch.tensor(state)[None],arm)
                    b,v=call(state);np.testing.assert_array_equal(a.numpy(),b);np.testing.assert_allclose(q.numpy(),v,rtol=1e-5,atol=1e-6)
                    if arm in r.CORRECTED:
                        c,w=new(state);np.testing.assert_array_equal(b,c);np.testing.assert_array_equal(v,w)
                equivalence.append(dict(arm=arm,fit=fit,actions=128,new_route=arm in r.CORRECTED))
    # Invalid inputs must be rejected before any runtime session is touched.
    rejected=0
    for arm in r.CORRECTED:
        for bad in [np.zeros(532,np.float32),np.full(533,np.nan,np.float32)]:
            try:optimized.values(None,arm,bad)
            except ValueError:rejected+=1
            else:raise AssertionError('invalid state accepted')
        for t in [1.,-.1,1.1,.013]:
            bad=np.zeros(533,np.float32);bad[-1]=t
            try:optimized.values(None,arm,bad)
            except ValueError:rejected+=1
            else:raise AssertionError('invalid time accepted')
    r.write(out/'equivalence.json',dict(records=equivalence,invalid_inputs_rejected=rejected,
        old_new_exact_actions=1536,eager_export_actions=4352,all_old_new_values_exact=True))
    sources=['anchor_runtime_single_validation.py','check_anchor_runtime.py',SPEC.name,'run_anchor_research_preparation.py',
        'controller_v3_fused_runtime.py','compiled_controller_inference.py','terminal_anchor_prototype.py']
    rng=np.random.default_rng(spec['order_seed']);orders=[]
    for i in range(5):keys=list(runtimes);rng.shuffle(keys);orders.append(keys)
    r.write(out/'PRE_TIMING_FREEZE.json',dict(source_hashes={name:r.base.old.sha(ROOT/name) for name in sources},
        payload_hashes=payloads,orders=orders,frozen_ns=time.perf_counter_ns(),spec_sha256=r.base.old.sha(SPEC)))
    freeze=r.base.old.sha(out/'PRE_TIMING_FREEZE.json');raw=[];first=None
    for round_id,keys in enumerate(orders):
        for key in keys:
            call=runtimes[key]
            for state in states[:32]:call(state)
            for i,state in enumerate(states):
                start=time.perf_counter_ns()
                if first is None:first=start
                a,q=call(state);action=a[0].copy();elapsed=time.perf_counter_ns()-start
                assert action.shape==(32,) and np.all((action>=0)&(action<5))
                raw.append(dict(round=round_id,path=key,observation=i,ns=elapsed))
    metrics={key:dict(median_ms=float(np.median([row['ns']/1e6 for row in raw if row['path']==key])),
        p95_ms=float(np.quantile([row['ns']/1e6 for row in raw if row['path']==key],.95)),calls=640) for key in runtimes}
    r.write(out/'timings.json',dict(raw=raw,metrics=metrics,freeze_sha256=freeze,first_call_ns=first))
    return audit(out)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    print(json.dumps(audit(args.output) if args.audit else check(args.output),indent=2))
