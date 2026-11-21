"""Synthetic-only paired screened simulator training/evaluation integration."""
import argparse,copy,json,hashlib,shutil,time,traceback
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
import deadline_screen_prototype as screen
import deadline_screen_fast_runtime as fast
import deadline_screen_iov_episode as ep
R=Path(__file__).resolve().parent
SPEC=R/'DEADLINE_SCREEN_IOV_FIXTURE_PROTOCOL_2026-11-11.json'
SOURCES=['run_deadline_screen_iov_fixture.py','deadline_screen_iov_episode.py','audit_deadline_screen_iov_fixture.py','test_deadline_screen_iov_fixture.py',
 'deadline_screen_prototype.py','deadline_screen_fast_runtime.py',SPEC.name,*screen.existing.SOURCES]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def update(agent):
    cfg=agent['cfg']
    if agent['slots']<cfg['warmup']:return None
    raw=agent['replay'].sample(cfg['batch'],agent['replay_rng']);batch=(*map(torch.tensor,raw[:5]),raw[-1])
    anchor=agent['replay'].terminal_sample(32,agent['anchor_rng']) if agent['online'].anchored else None
    loss,details=screen.objective(agent['online'],agent['target'],batch,anchor)
    # Preserve the adapted BDQ branchwise Huber objective instead of silently
    # changing it to a loss only on the branch mean.
    if agent['arm']=='bdq64_uniform_adapted':
        states,actions,_,_,_,_=batch
        y=torch.tensor(details['target'],dtype=torch.float32)
        branches=agent['online'].model(states).gather(2,actions[:,:,None]).squeeze(2)
        loss=F.smooth_l1_loss(branches,y[:,None].expand_as(branches));details['uniform']=float(loss.detach())
    before=screen.existing.base.old.digest(agent['target'].state_dict())
    agent['opt'].zero_grad(set_to_none=True);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(agent['online'].parameters(),10,error_if_nonfinite=True)
    assert all(p.grad is None for p in agent['target'].parameters());agent['opt'].step();agent['updates']+=1
    assert all(torch.isfinite(p).all() for p in agent['online'].parameters())
    copied=agent['updates']%8==0
    assert screen.existing.base.old.digest(agent['target'].state_dict())==before
    if copied:agent['target'].load_state_dict(agent['online'].state_dict());agent['copies']+=1
    return dict(step=agent['updates'],physical_slots=agent['slots'],loss=float(loss.detach()),grad_norm_before_clip=float(norm),
        replay_indices=raw[-1].tolist(),anchor_indices=[] if anchor is None else anchor[-1].tolist(),
        terminal_pool=sorted(agent['replay'].terminal),terminal_samples=int(batch[4].sum()),target_copy=copied,
        uniform_loss=details['uniform'],anchor_loss=details['anchor'],empty_anchor=details['empty_anchor'],
        bellman_targets=details['target'],next_mask=screen.allowed(batch[3]).tolist())
def configure():
    screen.existing.configure();screen.existing.fused.numba.set_num_threads(1)
    ep.greedy=lambda model,x,arm:(*model.greedy(x),{})
    ep.chosen=lambda model,x,a,arm:model(x,a)
    ep.update=update

def run(output,mode='integration-fixture'):
    if mode!='integration-fixture':raise RuntimeError('Research/final execution disabled')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);configure();p=read(SPEC)
    assert p['model_seeds']==[99811,99817] and p['evaluation_seeds']==[998190100,998190101]
    for name in SOURCES:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/name,dest)
    write(out/'PRE_RUN_FREEZE.json',dict(protocol_sha256=sha(SPEC),source_sha256={n:sha(R/n) for n in SOURCES},kind=p['kind']))
    freeze={};counts={};progress=[]
    for fit,seed in enumerate(p['model_seeds']):
        for arm in p['arms']:
            online=screen.Critic(arm,seed);target=copy.deepcopy(online).eval().requires_grad_(False)
            agent=dict(arm=arm,seed=seed,online=online,target=target,opt=torch.optim.Adam(online.parameters(),lr=1e-4),
                replay=screen.existing.TerminalBuffer(64),action_rng=np.random.default_rng(seed+100000),replay_rng=np.random.default_rng(seed+200000),
                anchor_rng=np.random.default_rng(seed+300000),cfg=dict(warmup=8,batch=4,epsilon_decay=40),slots=0,updates=0,copies=0,actor_history=[])
            folder=out/arm/f'fit_{fit:02d}';folder.mkdir(parents=True);logs=[]
            for episode in range(2):
                row,items=ep.episode(agent,p['training_starts'][fit]+episode,training=True,episode_id=episode)
                row['model_seed']=seed;ep.save_episode(folder/f'train_{episode:02d}.json.gz',row);logs+=items
            (folder/'optimizer.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in logs),encoding='utf-8')
            ck=dict(arm=arm,seed=seed,online=copy.deepcopy(online.state_dict()),target=copy.deepcopy(target.state_dict()),
                slots=agent['slots'],updates=agent['updates'],copies=agent['copies'],actor_history=agent['actor_history'],
                action_rng_state=agent['action_rng'].bit_generator.state,replay_rng_state=agent['replay_rng'].bit_generator.state)
            torch.save(ck,folder/'final.pt');key=f'{arm}_fit_{fit:02d}';freeze[key]=sha(folder/'final.pt')
            counts[key]=dict(slots=agent['slots'],updates=agent['updates'],copies=agent['copies'])
            progress.append(key);write(out/'progress.json',dict(stage='training',completed_models=progress));del agent,online,target,ck
    write(out/'ALL_FINAL_CHECKPOINTS_FREEZE.json',dict(checkpoints=freeze,before_evaluation=True))
    # Every fixture checkpoint fixed before any evaluation/cost measurement.
    predictions={};runtime_models={};invariance=[]
    state_path=R/'validation/deadline_screen_engineering_2026-11-11_01/observations.npz';states=np.load(state_path)['states'];shutil.copy2(state_path,out/'engineering_observations.npz')
    for fit,seed in enumerate(p['model_seeds']):
        for arm in p['arms']:
            folder=out/arm/f'fit_{fit:02d}';ck=torch.load(folder/'final.pt',weights_only=False,map_location='cpu')
            online=screen.Critic(arm,seed).eval();online.load_state_dict(ck['online']);target=copy.deepcopy(online).eval().requires_grad_(False);target.load_state_dict(ck['target'])
            runtime=fast.Runtime(online);key=f'{arm}_fit_{fit:02d}';runtime_models[key]=runtime
            aa=[];qq=[]
            for state in states:
                with torch.inference_mode():a,q=online.greedy(torch.from_numpy(state)[None])
                b,v=runtime.values(state);np.testing.assert_array_equal(a.numpy(),b);np.testing.assert_allclose(q.numpy(),v,atol=1e-6,rtol=1e-5)
                aa.append(b[0].tolist());qq.append(float(v[0]))
            predictions[key]=dict(actions=aa,q=qq)
            agent=dict(online=online,target=target,arm=arm,runtime=runtime)
            before=screen.existing.base.old.digest((online.state_dict(),target.state_dict(),online.training,target.training))
            for task_seed in p['evaluation_seeds']:
                row,logs=ep.episode(agent,task_seed,episode_id='evaluation');assert not logs
                row['model_seed']=seed;ep.save_episode(folder/f'evaluation_{task_seed}.json.gz',row)
            after=screen.existing.base.old.digest((online.state_dict(),target.state_dict(),online.training,target.training));assert before==after
            invariance.append(dict(key=key,before=before,after=after));assert sha(folder/'final.pt')==freeze[key]
    write(out/'predictions.json',predictions);write(out/'evaluation_invariance.json',invariance)
    rng=np.random.default_rng(p['order_seed']);orders=[]
    for fit in range(2):
        for round_id in range(5):
            arms=list(p['arms']);rng.shuffle(arms);orders.append(dict(fit=fit,round=round_id,arms=arms))
    write(out/'PRE_TIMING_FREEZE.json',dict(checkpoints=freeze,orders=orders,source_sha256={n:sha(R/n) for n in SOURCES},observations_sha256=sha(out/'engineering_observations.npz')))
    raw=[]
    for order in orders:
        for arm in order['arms']:
            runtime=runtime_models[f"{arm}_fit_{order['fit']:02d}"]
            for state in states[:32]:runtime.values(state)[0][0].copy()
            for index,state in enumerate(states):
                start=time.perf_counter_ns();action=runtime.values(state)[0][0].copy();elapsed=time.perf_counter_ns()-start
                raw.append(dict(fit=order['fit'],round=order['round'],arm=arm,index=index,nanoseconds=elapsed))
    metrics={}
    for key,runtime in runtime_models.items():
        arm,fit=key.rsplit('_fit_',1);ns=np.array([r['nanoseconds'] for r in raw if r['arm']==arm and r['fit']==int(fit)],np.float64)
        metrics[key]=dict(median_ms=float(np.median(ns)/1e6),p95_ms=float(np.quantile(ns,.95)/1e6),
            parameter_bytes=sum(t.numel()*t.element_size() for t in runtime.model.parameters()))
    screens=[]
    for fit in range(2):
        pr=metrics[f'price_both_fit_{fit:02d}'];c=metrics[f'compact_both_fit_{fit:02d}'];w=metrics[f'additive128_unbounded_fit_{fit:02d}']
        screens.append(dict(fit=fit,storage=pr['parameter_bytes']<=.5*c['parameter_bytes'],median=pr['median_ms']<=w['median_ms'],p95=pr['p95_ms']<=1.))
    write(out/'raw_timings.json',raw);write(out/'cost.json',dict(metrics=metrics,screens=screens))
    assert sum(v['updates'] for v in counts.values())==1022
    result=dict(state='synthetic_fixture_completed_not_research',counts=counts,training_slots=1120,evaluation_slots=1120,optimizer_updates=1022,
        checkpoints=14,engineering_equivalence_checks=1792,timing_rows=8960,cost_screens=screens,quality_evidence=False,research_launched=False)
    write(out/'verification.json',result)
    write(out/'run_manifest.json',dict(files={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in out.rglob('*') if p.is_file()}))
    return result
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['integration-fixture','development','final'],required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.mode!='integration-fixture':parser.error('Research and final execution disabled')
    print(json.dumps(run(args.output,args.mode)))
