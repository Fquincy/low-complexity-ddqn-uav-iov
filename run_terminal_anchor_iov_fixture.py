"""Paired simulator engineering fixture only; no research launch interface."""
import argparse,copy,gzip,json,time
from pathlib import Path
import numpy as np
import torch
import terminal_anchor_prototype as p
import run_terminal_anchor_engineering as io
import controller_v3_fused_runtime as native
import compiled_controller_inference as compiled
from audit_controller_redesign_engineering import check_episode,verify_arrays,exclusion_check

ROOT=Path(__file__).resolve().parent
SPEC=ROOT/'TERMINAL_ANCHOR_IOV_FIXTURE_PROTOCOL_2026-11-06.json'
BASE=p.existing.base
SOURCES=[*io.SOURCES,'run_terminal_anchor_iov_fixture.py','test_terminal_anchor_iov_fixture.py',
 'controller_v3_fused_runtime.py','compiled_controller_inference.py','Source/chapter3_iov_env.py',
 'Source/chapter3_iov_env_v2.py','audit_controller_redesign_engineering.py','audit_value_bound_development.py',SPEC.name]

class Buffer(p.Replay):
    def add(self,*row):super().add(row)

def build(arm,seed):
    model=p.Critic(arm,seed);target=copy.deepcopy(model).eval().requires_grad_(False)
    return dict(arm=arm,seed=seed,online=model,target=target,opt=torch.optim.Adam(model.parameters(),lr=1e-4),
        replay=Buffer(48),action_rng=np.random.default_rng(seed+100000),replay_rng=np.random.default_rng(seed+200000),
        anchor_rng=np.random.default_rng(seed+300000),slots=0,updates=0,copies=0,actor_history=[],
        cfg=dict(warmup=8,epsilon_decay=80))

def update(agent):
    if agent['slots']<8:return None
    start=time.perf_counter_ns();batch=agent['replay'].sample(8,agent['replay_rng'])
    anchor=agent['replay'].sample(32,agent['anchor_rng'],True) if agent['online'].anchored else None
    agent['opt'].zero_grad(set_to_none=True)
    loss,info=p.objective(agent['online'],agent['target'],batch,anchor);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(agent['online'].parameters(),10,error_if_nonfinite=True)
    assert all(v.grad is None for v in agent['target'].parameters())
    agent['opt'].step();agent['updates']+=1
    copied=agent['updates']%8==0
    if copied:agent['target'].load_state_dict(agent['online'].state_dict());agent['copies']+=1
    return dict(update=agent['updates'],slot=agent['slots'],loss=float(loss.detach()),norm=float(norm),
        update_ns=time.perf_counter_ns()-start,copied=copied,indices=batch[-1].tolist(),
        anchor_indices=[] if anchor is None else anchor[-1].tolist(),terminal_pool=sorted(agent['replay'].terminal),**info)

def deployed(runtime,arm,state):
    # Complete path includes finite/shape/time checks and horizon scaling.
    x=np.asarray(state,dtype=np.float32)
    if x.shape!=(533,) or not np.isfinite(x).all():raise ValueError('finite state required')
    t=float(x[-1])*40;rounded=round(t)
    if abs(t-rounded)>1e-4 or not 0<=rounded<40:raise ValueError('nonterminal grid time required')
    a,q=native.values(runtime,p.ARMS[arm][0],x)
    return a[0],q[0]*np.float32(40-rounded if p.ARMS[arm][1] else 1.)

def fixture(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);p.existing.configure()
    spec=io.load(SPEC);assert spec['research_enabled'] is False
    preserved=io.sha(ROOT/'development_runs/controller_v3_development_2026-11-06_01/run_manifest.json')
    saved={name:getattr(BASE,name) for name in ['greedy','chosen','update']}
    BASE.greedy=lambda model,states,arm:(*model.greedy(states),{})
    BASE.chosen=lambda model,states,actions,arm:model(states,actions)
    BASE.update=update
    records={};counts={};actor_hashes={};rng=np.random.default_rng(spec['arm_order_seed']);orders=[]
    try:
        for fit,seed in enumerate(spec['model_seeds']):
            order=list(p.ARMS);rng.shuffle(order);orders.append(order)
            for arm in order:
                folder=out/arm/f'fit_{fit}';folder.mkdir(parents=True)
                agent=build(arm,seed);logs=[];start=time.perf_counter()
                for ep in range(2):
                    row,trace=BASE.episode(agent,spec['training_starts'][fit]+ep,training=True,episode_id=ep)
                    row['run_kind']='ENGINEERING_IOV_NOT_RESEARCH';BASE.save_episode(folder/f'train_{ep}.json.gz',row);logs+=trace
                counts[f'{arm}_{fit}']=dict(slots=agent['slots'],updates=agent['updates'],copies=agent['copies'],
                    training_seconds=time.perf_counter()-start,replay_rows=len(agent['replay'].rows),
                    replay_payload_bytes=sum(v.nbytes if isinstance(v,np.ndarray) else np.asarray(v).nbytes for row in agent['replay'].rows for v in row),
                    terminal_index_count=len(agent['replay'].terminal),python_container_bytes_excluded=True)
                actor_hashes[f'{arm}_{fit}']=BASE.old.digest(agent['actor_history'])
                torch.save(dict(online=agent['online'].state_dict(),target=agent['target'].state_dict()),folder/'final.pt')
                io.write(folder/'optimizer.json',logs)
        freeze={f'{arm}_{fit}':io.sha(out/arm/f'fit_{fit}'/'final.pt') for arm in p.ARMS for fit in range(2)}
        io.write(out/'ALL_CHECKPOINTS_FREEZE.json',dict(checkpoints=freeze,before_evaluation=True))
        for fit,seed in enumerate(spec['model_seeds']):
            for arm in p.ARMS:
                folder=out/arm/f'fit_{fit}';agent=build(arm,seed)
                ck=torch.load(folder/'final.pt',weights_only=True);agent['online'].load_state_dict(ck['online']);agent['target'].load_state_dict(ck['target'])
                before=BASE.old.digest((agent['online'].state_dict(),agent['target'].state_dict()))
                for task in spec['evaluation_seeds']:
                    row,_=BASE.episode(agent,task,episode_id='engineering_evaluation')
                    row['run_kind']='ENGINEERING_IOV_NOT_RESEARCH';BASE.save_episode(folder/f'evaluation_{task}.json.gz',row)
                assert before==BASE.old.digest((agent['online'].state_dict(),agent['target'].state_dict()))
                assert io.sha(folder/'final.pt')==freeze[f'{arm}_{fit}']
        for name in ['all_local','nearest_rsu','uav_selective']:
            for task in spec['evaluation_seeds']:
                row,_=BASE.episode(None,task,policy=name);row['run_kind']='ENGINEERING_IOV_NOT_RESEARCH'
                BASE.save_episode(out/f'heuristic_{name}_{task}.json.gz',row)
        inference(out,spec)
    finally:
        for name,value in saved.items():setattr(BASE,name,value)
    for name in dict.fromkeys(SOURCES):
        path=out/'source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    io.write(out/'verification.json',dict(kind='ENGINEERING_IOV_NOT_RESEARCH',counts=counts,actor_hashes=actor_hashes,
        arm_orders=orders,training_slots=960,optimizer_updates=876,preserved_manifest_sha256=preserved,
        research_launched=False,thesis_changed=False,policy_quality_evidence=False))
    assert preserved==io.sha(ROOT/'development_runs/controller_v3_development_2026-11-06_01/run_manifest.json')
    io.write(out/'run_manifest.json',dict(files={str(x.relative_to(out)).replace('\\','/'):io.sha(x) for x in out.rglob('*') if x.is_file()}))
    return audit(out)

def inference(out,spec):
    folder=out/'inference';folder.mkdir();runtimes={};meta={}
    source=ROOT/spec['observations'];assert io.sha(source)==spec['observations_sha256']
    states=np.load(source)['observations'];np.savez_compressed(folder/'observations.npz',observations=states)
    native.numba.set_num_threads(1)
    for fit,seed in enumerate(spec['model_seeds']):
        for arm in p.ARMS:
            key=f'{arm}_{fit}';model=p.Critic(arm,seed)
            model.load_state_dict(torch.load(out/arm/f'fit_{fit}'/'final.pt',weights_only=True)['online']);model.eval()
            runtime,m=native.export(model.model,model.base_arm,folder/(key+'.onnx'))
            np.save(folder/(key+'.extra.npy'),runtime[1]);runtimes[key]=runtime
            with torch.no_grad():
                for state in states:
                    a,q=model.greedy(torch.tensor(state)[None]);b,v=deployed(runtime,arm,state)
                    np.testing.assert_array_equal(a[0].numpy(),b);np.testing.assert_allclose(q.numpy(),[v],rtol=1e-5,atol=1e-6)
            assert m['parameters']==sum(v.numel() for v in model.parameters())
            meta[key]=m
    order_rng=np.random.default_rng(spec['timing_order_seed']);order=[]
    for round_id in range(3):
        keys=list(runtimes);order_rng.shuffle(keys);order.append(keys)
    payloads={x.name:io.sha(x) for x in folder.iterdir() if x.is_file()}
    io.write(folder/'PRE_TIMING_FREEZE.json',dict(payloads=payloads,orders=order,created_ns=time.perf_counter_ns()))
    frozen_hash=io.sha(folder/'PRE_TIMING_FREEZE.json');raw=[];first=None
    for round_id,keys in enumerate(order):
        for key in keys:
            arm=key.rsplit('_',1)[0]
            for state in states[:32]:deployed(runtimes[key],arm,state)
            for i,state in enumerate(states):
                start=time.perf_counter_ns()
                if first is None:first=start
                deployed(runtimes[key],arm,state);ns=time.perf_counter_ns()-start
                raw.append(dict(key=key,round=round_id,observation=i,ns=ns))
    assert io.sha(folder/'PRE_TIMING_FREEZE.json')==frozen_hash
    metrics={key:dict(median_ms=float(np.median([r['ns']/1e6 for r in raw if r['key']==key])),
        p95_ms=float(np.quantile([r['ns']/1e6 for r in raw if r['key']==key],.95)),**meta[key]) for key in runtimes}
    io.write(folder/'timings.json',dict(raw=raw,metrics=metrics,freeze_sha256=frozen_hash,first_call_ns=first))

def audit(out):
    out=Path(out);spec=io.load(out/'source_snapshot'/SPEC.name)
    files=io.load(out/'run_manifest.json')['files']
    actual={str(x.relative_to(out)).replace('\\','/') for x in out.rglob('*') if x.is_file()}
    assert actual-set(files)<={'run_manifest.json','saved_artifact_audit.json'}
    for name,digest in files.items():assert io.sha(out/name)==digest,name
    result=io.load(out/'verification.json');freeze=io.load(out/'ALL_CHECKPOINTS_FREEZE.json')
    fingerprints={};episodes=0;updates=0
    for fit,seed in enumerate(spec['model_seeds']):
        assert len({result['actor_hashes'][f'{arm}_{fit}'] for arm in p.ARMS})==1
        for arm in p.ARMS:
            folder=out/arm/f'fit_{fit}';key=f'{arm}_{fit}';assert io.sha(folder/'final.pt')==freeze['checkpoints'][key]
            ck=torch.load(folder/'final.pt',weights_only=True);model=p.Critic(arm,seed);model.load_state_dict(ck['online'])
            target=p.Critic(arm,seed);target.load_state_dict(ck['target'])
            for path in folder.glob('*.json.gz'):
                row=io.load_gzip(path) if hasattr(io,'load_gzip') else json.load(gzip.open(path,'rt',encoding='utf-8'))
                assert row['run_kind']=='ENGINEERING_IOV_NOT_RESEARCH'
                check_episode(row,BASE.Chapter3Config());states,ns=verify_arrays(folder,path.name,row);episodes+=1
                fingerprints.setdefault((fit,path.name),[]).append(row['workload_sha256'])
                if path.name.startswith('evaluation'):
                    with torch.no_grad():
                        for i,state in enumerate(states):
                            a,q=model.greedy(torch.tensor(state)[None]);tq=target(torch.tensor(state)[None],a)
                            np.testing.assert_array_equal(a[0].numpy(),row['slots'][i]['action'])
                            np.testing.assert_allclose([q.item(),tq.item()],[row['chosen_q'][i],row['target_q'][i]],rtol=1e-5,atol=1e-5)
                else:assert row['seed']==spec['training_starts'][fit]+int(path.stem.split('_')[-1].split('.')[0])
            logs=io.load(folder/'optimizer.json');assert len(logs)==73;updates+=len(logs)
            for i,row in enumerate(logs):
                assert row['update']==i+1 and row['slot']==i+8 and row['copied']==((i+1)%8==0)
                assert np.isfinite([row['loss'],row['norm'],row['update_ns']]).all()
                assert abs(row['loss']-row['uniform']-row['anchor'])<1e-3
                expected_pool={j%48 for j in range(max(0,row['slot']-48),row['slot']) if j%40==39}
                assert set(row['terminal_pool'])==expected_pool
                assert all(j in expected_pool for j in row['anchor_indices'])
                if model.anchored and expected_pool:assert len(row['anchor_indices'])==32
    assert all(len(x)==6 and len(set(x))==1 for x in fingerprints.values())
    for name in ['all_local','nearest_rsu','uav_selective']:
        for task in spec['evaluation_seeds']:
            path=out/f'heuristic_{name}_{task}.json.gz';row=json.load(gzip.open(path,'rt',encoding='utf-8'))
            check_episode(row,BASE.Chapter3Config());verify_arrays(out,path.name,row);episodes+=1
            assert row['workload_sha256']==fingerprints[(0,f'evaluation_{task}.json.gz')][0]
    folder=out/'inference';pre=io.load(folder/'PRE_TIMING_FREEZE.json');timing=io.load(folder/'timings.json')
    assert timing['freeze_sha256']==io.sha(folder/'PRE_TIMING_FREEZE.json') and pre['created_ns']<timing['first_call_ns']
    for name,digest in pre['payloads'].items():assert io.sha(folder/name)==digest
    raw=timing['raw'];expected=[(key,r,i) for r,keys in enumerate(pre['orders']) for key in keys for i in range(128)]
    assert [(x['key'],x['round'],x['observation']) for x in raw]==expected
    assert all(x['ns']>0 for x in raw)
    for key,metric in timing['metrics'].items():
        ns=[r['ns']/1e6 for r in raw if r['key']==key]
        assert metric['median_ms']==float(np.median(ns)) and metric['p95_ms']==float(np.quantile(ns,.95))
    # Independently reload exported payloads and replay complete decisions;
    # do not substitute a new timing sample for the original measurements.
    states=np.load(folder/'observations.npz')['observations'];vectors=0
    np.testing.assert_array_equal(states,np.load(ROOT/spec['observations'])['observations'])
    assert io.sha(ROOT/spec['observations'])==spec['observations_sha256']
    for fit,seed in enumerate(spec['model_seeds']):
        for arm in p.ARMS:
            key=f'{arm}_{fit}';model=p.Critic(arm,seed)
            model.load_state_dict(torch.load(out/arm/f'fit_{fit}'/'final.pt',weights_only=True)['online'])
            extra=np.load(folder/(key+'.extra.npy'));np.testing.assert_array_equal(extra,np.zeros((2,5,2),np.float32))
            session=compiled.session(folder/(key+'.onnx'))
            assert session.get_session_options().intra_op_num_threads==1
            with torch.no_grad():
                for state in states:
                    a,q=model.greedy(torch.tensor(state)[None]);b,v=deployed((session,extra),arm,state)
                    np.testing.assert_array_equal(a[0].numpy(),b);np.testing.assert_allclose(q.numpy(),[v],rtol=1e-5,atol=1e-6)
                    vectors+=1
    assert episodes==54 and updates==876
    audit_result=dict(state='passed_paired_iov_engineering_audit',episodes_checked=episodes,updates_checked=updates,
        training_simulator_slots=960,checkpoints=12,export_equivalence_checks_replayed=vectors,timing_rows_recomputed=len(raw),
        policy_quality_evidence=False,scope=exclusion_check())
    io.write(out/'saved_artifact_audit.json',audit_result);return audit_result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['fixture','development','final'],required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.mode!='fixture':parser.error('Research modes disabled')
    print(json.dumps(fixture(args.output),indent=2))
