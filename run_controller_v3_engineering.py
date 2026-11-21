"""V3 no-rollout/no-update engineering gate. Development/final modes disabled."""
import argparse,copy,json,shutil,time,traceback,warnings
from pathlib import Path
import numpy as np
import onnx,onnxruntime as ort,torch
import compiled_controller_inference as c
import run_controller_redesign_engineering as r
from controller_v3_amortized import NEW_ARMS,AmortizedPriceQ

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/'LOW_COMPLEXITY_CONTROLLER_V3_ENGINEERING_PROTOCOL_2026-11-05.json'
DESIGN=ROOT/'LOW_COMPLEXITY_CONTROLLER_V3_DESIGN_2026-11-05.txt'
ARMS=(*NEW_ARMS,*r.ARMS)
SOURCES=[*r.SOURCES,'compiled_controller_inference.py','controller_v3_amortized.py',
 'run_controller_v3_engineering.py','audit_controller_v3_engineering.py','test_controller_v3_engineering.py',PROTOCOL.name,DESIGN.name]

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def build(arm,seed):
    if arm not in ARMS:raise ValueError('unknown arm')
    if arm in NEW_ARMS:
        torch.manual_seed(seed);return AmortizedPriceQ(arm).eval()
    return r.build(arm,seed,r.fixture_cfg())['online'].eval()

def greedy(model,arm,x):return model.greedy(x) if arm in NEW_ARMS else r.greedy(model,x,arm)

class Policy(torch.nn.Module):
    def __init__(self,model,arm):
        super().__init__();self.model=model;self.arm=arm
        self.existing=c.CompletePolicy(model,arm) if arm not in NEW_ARMS else None
    def forward(self,states):
        if self.existing is not None:return self.existing(states)
        a,q,_=self.model.greedy(states);return a,q

def export(model,arm,path):
    before=r.old.digest(model.state_dict())
    torch.onnx.register_custom_op_symbolic('aten::expm1',lambda g,x:g.op('Sub',g.op('Exp',x),
        g.op('Constant',value_t=torch.tensor(1.,dtype=torch.float32))),17)
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught,torch.inference_mode():
        warnings.simplefilter('always')
        torch.onnx.export(Policy(model,arm),torch.zeros(1,533),str(path),opset_version=17,dynamo=False,
            input_names=['states'],output_names=['actions','chosen_q'],do_constant_folding=True)
    elapsed=time.perf_counter()-start;onnx.checker.check_model(onnx.load(str(path)))
    start=time.perf_counter();sess=c.session(path);startup=time.perf_counter()-start
    assert r.old.digest(model.state_dict())==before
    return sess,dict(arm=arm,parameter_hash=before,parameters=sum(p.numel() for p in model.parameters()),
        parameter_bytes=r.parameter_bytes(model),buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()),
        graph_sha256=r.old.sha(path),graph_bytes=path.stat().st_size,export_seconds=elapsed,startup_seconds=startup,
        warnings=sorted({str(w.message) for w in caught}))

def compiled_values(sess,state):
    x=np.asarray(state,dtype=np.float32)
    if x.shape!=(533,) or not np.isfinite(x).all():raise ValueError('one finite 533-element observation required')
    a,q=sess.run(['actions','chosen_q'],{'states':x[None]})
    return a,q

def action(sess,state):return compiled_values(sess,state)[0][0].copy()

def equivalent(a,q,b,v):
    if not np.isfinite(q).all() or not np.isfinite(v).all():raise AssertionError('nonfinite Q')
    np.testing.assert_array_equal(a,b);np.testing.assert_allclose(q,v,rtol=1e-5,atol=1e-6)

def screens(metrics):
    a=metrics[NEW_ARMS[0]];small=metrics['additive64_nonpositive'];wide=metrics['additive128_unbounded']
    return dict(online_storage=a['parameter_bytes']<=.5*small['parameter_bytes'],
        complete_action_median=a['median_ms']<=wide['median_ms'],complete_action_p95=a['p95_ms']<=1.)

def summarize(raw,metadata):
    result={}
    for arm in ARMS:
        rows=[z for z in raw if z['arm']==arm]
        assert len(rows)==640
        assert len({(z['round'],z['observation']) for z in rows})==640
        ns=np.array([z['nanoseconds'] for z in rows],dtype=np.float64)
        assert np.all(ns>0)
        item=metadata[f'{arm}_99021']
        result[arm]=dict(median_ms=float(np.median(ns/1e6)),p95_ms=float(np.quantile(ns/1e6,.95)),
            parameter_bytes=item['parameter_bytes'],parameters=item['parameters'],graph_bytes=item['graph_bytes'],calls=len(rows))
    return result

def freeze_inputs(out):
    # Hash previous reports, source and all previous optimization artifacts.
    prior={}
    for folder in ['validation/controller_inference_optimization_2026-11-05_01',
                   'validation/controller_redesign_engineering_2026-11-05_01','analysis/controller_redesign_2026-11-05']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file():prior[str(path.relative_to(ROOT)).replace('\\','/')]=r.old.sha(path)
    for path in ROOT.glob('*.py'):prior[path.name]=r.old.sha(path)
    for path in ROOT.glob('*.txt'):prior[path.name]=r.old.sha(path)
    r.write(out/'prior_input_hashes.json',prior)
    shutil.copy2(ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json',out/'PROJECT_STATUS_BEFORE.json')
    return prior

def run(output,tests):
    r.configure();p=read(PROTOCOL)
    assert list(NEW_ARMS)==p['new_arms'] and list(r.ARMS)==p['retained_arms']
    evidence=read(tests)
    assert evidence['passed'] and evidence['optimizer_updates']==evidence['new_simulator_steps']==0
    assert evidence['source_sha256']=={n:r.old.sha(ROOT/n) for n in SOURCES}
    assert ort.__version__=='1.31.0' and onnx.__version__=='1.23.2'
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(tests,out/'mechanics_tests.json');prior=freeze_inputs(out)
    before=dict(protocol_sha256=r.old.sha(PROTOCOL),source_sha256={n:r.old.sha(ROOT/n) for n in SOURCES},
        hardware=r.hardware(),research_training_authorized=False,pre_export_clock_ns=time.perf_counter_ns())
    r.write(out/'PRE_EXPORT_FREEZE.json',before)
    for name in SOURCES:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    assert r.old.sha(ROOT/p['observations'])==p['observations_sha256']
    common=np.load(ROOT/p['observations'])['observations']
    extra=np.random.default_rng(99025).uniform(0,1,(64,533)).astype(np.float32)
    cases=np.concatenate((common,extra));np.savez_compressed(out/'observations.npz',observations=cases)
    folder=out/'exports';folder.mkdir();metadata={};eq=[];sessions={}
    try:
        for seed in p['synthetic']['model_seeds']:
            for arm in ARMS:
                print(f'Export/equivalence {arm} seed={seed}',flush=True)
                model=build(arm,seed);path=folder/f'{arm}_{seed}.onnx';sess,meta=export(model,arm,path)
                aa=[];qq=[];bb=[];vv=[]
                with torch.inference_mode():
                    for state in cases:
                        a,q,_=greedy(model,arm,torch.from_numpy(state)[None]);b,v=compiled_values(sess,state)
                        aa.append(a[0].numpy());qq.append(float(q[0]));bb.append(b[0]);vv.append(float(v[0]))
                equivalent(aa,qq,bb,vv)
                if arm in NEW_ARMS or r.bounded(arm):assert max(vv)<=0
                reloaded=c.session(path)
                for ix in [0,64,191]:
                    first=compiled_values(sess,cases[ix]);second=compiled_values(reloaded,cases[ix])
                    np.testing.assert_array_equal(first[0],second[0]);np.testing.assert_array_equal(first[1],second[1])
                assert r.old.digest(model.state_dict())==meta['parameter_hash']
                metadata[f'{arm}_{seed}']=meta
                eq.append(dict(arm=arm,seed=seed,cases=len(cases),exact_actions=True,
                    max_abs_Q_error=float(np.max(np.abs(np.asarray(qq)-vv))),weights_unchanged=True,reload_exact=True))
                if seed==99021:sessions[arm]=sess
        r.write(out/'equivalence.json',dict(passed=True,models=eq,complete_actions=len(eq)*len(cases),
            scope='finite_synthetic_initializations_not_policy_quality_or_universal_equivalence'))
        r.write(out/'metadata.json',metadata)
        for name,h in before['source_sha256'].items():assert r.old.sha(ROOT/name)==h
        r.write(out/'PRE_TIMING_FREEZE.json',dict(protocol_sha256=before['protocol_sha256'],
            equivalence_sha256=r.old.sha(out/'equivalence.json'),graphs={k:v['graph_sha256'] for k,v in metadata.items()},
            paths=list(ARMS),rounds=5,calls=128,warmups=32,order_seed=99028,hardware=r.hardware()))
        raw=[];orders=[];rng=np.random.default_rng(99028)
        for round_ in range(5):
            order=list(ARMS);rng.shuffle(order);orders.append(order)
            for arm in order:
                for state in common[:32]:action(sessions[arm],state)
                for ix,state in enumerate(common):
                    start=time.perf_counter_ns();a=action(sessions[arm],state);elapsed=time.perf_counter_ns()-start
                    assert a.shape==(32,) and np.all((a>=0)&(a<5))
                    raw.append(dict(round=round_,arm=arm,observation=ix,nanoseconds=elapsed))
        metrics=summarize(raw,metadata);cost=screens(metrics)
        for name,h in prior.items():assert r.old.sha(ROOT/name)==h,name
        result=dict(state='engineering_cost_passed' if all(cost.values()) else 'engineering_cost_failed',
            screens=cost,metrics=metrics,timing_rows=len(raw),orders=orders,new_simulator_steps=0,optimizer_updates=0,
            research_training_started=False,policy_quality_established=False,word_thesis_changed=False,
            original_inputs_unchanged=len(prior),research_and_final_seeds_unconsumed=True,
            next_gate='prepare_frozen_research_runner_only_after_review' if all(cost.values()) else 'stop_record_failure_no_training')
        r.write(out/'raw_timings.json',raw);r.write(out/'verification.json',result)
        r.write(out/'run_manifest.json',dict(files={str(x.relative_to(out)).replace('\\','/'):r.old.sha(x) for x in out.rglob('*') if x.is_file()}))
        print(json.dumps(result,indent=2));return result
    except Exception:
        r.write(out/'FAILED.json',dict(error=traceback.format_exc(),partial_equivalence=eq,
            optimizer_updates=0,new_simulator_steps=0,research_training_started=False))
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['engineering','development','final'],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--tests',type=Path)
    parser.add_argument('--confirm-development-run',action='store_true');parser.add_argument('--confirm-final-run',action='store_true')
    a=parser.parse_args()
    if a.mode!='engineering':
        if not getattr(a,'confirm_'+a.mode+'_run'):parser.error('--confirm-'+a.mode+'-run required; research disabled')
        parser.error('Research disabled even with confirmation; a separate executable protocol is required')
    if not a.tests:parser.error('--tests mechanics evidence required')
    run(a.output,a.tests)

if __name__=='__main__':main()
