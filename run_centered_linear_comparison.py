"""Bound seven-arm correction orchestration; only fixture mode released."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

import centered_linear_comparison_models as variants
import heuristic_initialized_ddqn_baseline as baseline
from heuristic_pilot_teacher_adapter import build_teacher

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'Source'))
from chapter3_iov_env_v2 import Chapter3Config, Chapter3IoVEnv

SPEC = ROOT/'CENTERED_LINEAR_CORRECTION_COMPARISON_PROTOCOL_2026-11-20.json'
SPEC_SHA = '95bd2737f8700378ced695bfaf4b9a3d457a2fbe54df2f6f7f829186ac7c674f'
CONFIRM = 'LAUNCH_FROZEN_CENTERED_LINEAR_CORRECTION_COMPARISON'
DEVELOPMENT_RELEASED = False
IMPLEMENTATION = ['run_centered_linear_comparison.py','centered_linear_comparison_models.py',
                  'heuristic_pilot_teacher_adapter.py','audit_centered_linear_comparison.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def protocol():
    if sha(SPEC)!=SPEC_SHA:
        raise ValueError('frozen correction protocol hash changed')
    spec = read(SPEC)
    for name, expected in spec['definition_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('frozen definition changed: '+name)
    if json.loads(json.dumps(asdict(Chapter3Config()))) != spec['simulator_defaults']:
        raise ValueError('simulator defaults changed')
    return spec


def plan(spec, mode):
    if mode == 'development':
        return dict(mode=mode, kind='EXPLORATORY_PAIRED_DEVELOPMENT', fits=spec['fits'],
                    seeds=copy.deepcopy(spec['seeds']), demo_episodes=10, online_episodes=50,
                    eval_episodes=16, bc_steps=200, batch=64, warmup=128, replay_capacity=4096,
                    target_interval=200, timing_states=128, timing_rounds=3, timing_warmups=32)
    if mode != 'fixture':
        raise ValueError('unknown/final modes disabled')
    # New engineering-only blocks. Never replace these with reserved pilot seeds.
    return dict(mode=mode, kind='ENGINEERING_FIXTURE_NOT_RESEARCH', fits=2,
                seeds={'model_seeds':[957190001,957190011],
                       'demonstration_seed_blocks':[{'start':957200000+i*100,'count':1} for i in range(2)],
                       'online_seed_blocks':[{'start':957210000+i*100,'count':1} for i in range(2)],
                       'evaluation_seed_block':{'start':957220000,'count':1},
                       'bc_sampler_seeds':[957230001,957230011],
                       'epsilon_branch_seeds':[957240001,957240011],
                       'replay_sampler_seeds':[957250001,957250011],
                       'job_order_seed':957260001,'timing_order_seed':957260011},
                demo_episodes=1, online_episodes=1, eval_episodes=1,
                bc_steps=2, batch=4, warmup=8, replay_capacity=128,
                target_interval=8, timing_states=4, timing_rounds=1, timing_warmups=1)


def tensors_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def array_batch(rows):
    s,a,r,ns,d = zip(*rows)
    return (torch.from_numpy(np.stack(s)),torch.from_numpy(np.stack(a)),
            torch.tensor(r,dtype=torch.float32),torch.from_numpy(np.stack(ns)),
            torch.tensor(d,dtype=torch.bool))


def train_update(model, target, rows, opt):
    s,a,r,ns,d = array_batch(rows)
    prediction = model(s,a)
    selected = np.full((len(rows),32),-1,np.int64)
    target_q = torch.zeros(len(rows))
    with torch.no_grad():
        if (~d).any():
            aa = model.greedy(ns[~d])[0]
            selected[(~d).numpy()] = aa.numpy()
            target_q[~d] = target(ns[~d],aa)
        y = r+target_q
    loss = F.smooth_l1_loss(prediction,y)
    opt.zero_grad(); loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
    if (not torch.isfinite(loss) or not torch.isfinite(norm) or
            any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()) or
            any(p.grad is not None for p in target.parameters())):
        raise ValueError('nonfinite training or target gradient')
    detail = dict(loss=float(loss.detach()), gradient_norm=float(norm),
                  prediction=prediction.detach().tolist(), rewards=r.tolist(),
                  target_q=target_q.tolist(), target=y.tolist(), dones=d.tolist(),
                  next_actions=selected.tolist())
    opt.step()
    return detail


def optimizer(model, lr):
    return torch.optim.Adam(model.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0)


def run(output, mode='fixture', confirmation=None):
    # Fail closed before output creation, model construction or seed consumption.
    if mode == 'development':
        if confirmation != CONFIRM:
            raise RuntimeError('Explicit development confirmation required')
        if not DEVELOPMENT_RELEASED:
            raise RuntimeError('Development disabled pending separate readiness review and launch release')
    if mode not in ('fixture','development'):
        raise RuntimeError('Final/unknown modes disabled')
    spec = protocol(); pl = plan(spec,mode)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    run_dir=out/'run'; run_dir.mkdir()
    write(run_dir/'runtime_plan.json',pl)
    sources=set(spec['definition_sha256'])|set(IMPLEMENTATION)|{SPEC.name}
    snap=run_dir/'source_snapshot'; snap.mkdir()
    for name in sources:
        dest=snap/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    write(run_dir/'PRE_RUN_FREEZE.json',dict(protocol_sha256=sha(SPEC),
        plan_sha256=sha(run_dir/'runtime_plan.json'),source_sha256={n:sha(ROOT/n) for n in sorted(sources)},
        teacher_backend='Python reference: frozen function bodies, identity native decorators',
        environment={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
                    'platform':platform.platform(),'threads':torch.get_num_threads()}))
    events=[]
    def event(stage,**fields):
        events.append(dict(sequence=len(events),stage=stage,**fields));write(run_dir/'events.json',events)
        write(out/'progress.json',events[-1])
    teacher=build_teacher(ROOT,Chapter3Config)

    def episode(path, seed, actor, callback=None, collect_q=False):
        env=Chapter3IoVEnv(Chapter3Config(),seed=seed);s=env.reset(seed=seed)
        rows=[];qs=[];infos=[];begin=time.perf_counter_ns();actor_ns=0
        for t in range(40):
            before=time.perf_counter_ns();action=np.asarray(actor(s),dtype=np.int64);actor_ns+=time.perf_counter_ns()-before
            if action.shape!=(32,) or np.any(action<0) or np.any(action>4) or not np.take_along_axis(baseline.allowed(s[None]),action[None,:,None],2).all():
                raise ValueError('illegal policy action')
            ns,reward,done,info=env.step({'destinations':action[:30], 'uav_movement':action[30:]})
            row=(s.copy(),action.copy(),float(reward),ns.copy(),bool(done));rows.append(row)
            if collect_q:
                with torch.no_grad():qs.append(float(collect_q(torch.from_numpy(s)[None],torch.from_numpy(action)[None])[0]))
            infos.append({k:v for k,v in info.items() if k in ('reward_components','tasks_generated','tasks_completed','tasks_unfinished','deadline_success_rate')})
            if callback:callback(row,t)
            s=ns
            if done:break
        if len(rows)!=40 or not done:raise ValueError('wrong horizon')
        batch=array_batch(rows)
        np.savez_compressed(path.with_suffix('.npz'),states=batch[0].numpy(),actions=batch[1].numpy(),
            rewards=batch[2].numpy(),next_states=batch[3].numpy(),dones=batch[4].numpy(),q=np.array(qs,np.float64))
        summary=env.episode_summary()
        data=dict(seed=seed,steps=len(rows),return_value=sum(r[2] for r in rows),summary=summary,
                  records=env.export_task_records(),step_info=infos,wall_ns=time.perf_counter_ns()-begin,
                  actor_ns=actor_ns)
        write(path.with_suffix('.json'),data)
        return rows,data

    def teacher_action(s):return teacher.values(s)[0][0]

    demo={}; initial={}; initial_hash={}
    for fit in range(pl['fits']):
        event('demonstrations',fit=fit)
        folder=run_dir/f'demo_{fit}';folder.mkdir();rows=[]
        for ep in range(pl['demo_episodes']):
            rr,_=episode(folder/f'episode_{ep:03d}',pl['seeds']['demonstration_seed_blocks'][fit]['start']+ep,teacher_action)
            rows.extend(rr)
        demo[fit]=rows
        torch.manual_seed(pl['seeds']['model_seeds'][fit]);m=variants.Critic('centered_linear_bc_ddqn')
        initial[fit]=copy.deepcopy(m.state_dict());initial_hash[str(fit)]=tensors_hash(m)
    write(run_dir/'DEMONSTRATION_FREEZE.json',dict(files={p.relative_to(run_dir).as_posix():sha(p)
        for p in sorted(run_dir.glob('demo_*/*'))},initial_tensor_hashes=initial_hash))
    jobs=[(arm['id'],fit) for arm in spec['arms'] for fit in range(pl['fits'])]
    definitions={arm['id']:arm for arm in spec['arms']}
    bc_end_hashes={}
    np.random.default_rng(pl['seeds']['job_order_seed']).shuffle(jobs);write(run_dir/'job_order.json',jobs)
    ckhash={};counts={};probes={}
    for arm,fit in jobs:
        event('training',arm=arm,fit=fit)
        folder=run_dir/arm/f'fit_{fit}';folder.mkdir(parents=True)
        definition=definitions[arm]
        model=variants.Critic(arm);model.load_state_dict(initial[fit])
        if tensors_hash(model)!=initial_hash[str(fit)]:raise ValueError('initial pair mismatch')
        ds=array_batch(demo[fit]); bc_rng=np.random.default_rng(pl['seeds']['bc_sampler_seeds'][fit]);bc_log=[]
        def probe():
            with torch.no_grad():
                aa,qq,_=model.greedy(ds[0]);scalar=model.scalar_diagnostics(ds[0],aa)
                return dict(imitation_loss=float(baseline.imitation_loss(model,ds[0],ds[1])),
                            branch_agreement=float((aa==ds[1]).float().mean()),
                            joint_agreement=float((aa==ds[1]).all(1).float().mean()),actions=aa.tolist(),
                            q=qq.tolist(),scalar={k:v.tolist() for k,v in scalar.items()})
        probe_rows={'before':probe()};bc_start=time.perf_counter_ns()
        if definition['bc']:
            bc_opt=optimizer(model,.001)
            for k in range(pl['bc_steps']):
                indices=bc_rng.choice(len(demo[fit]),pl['batch'],replace=False)
                bc_opt.zero_grad();loss=baseline.imitation_loss(model,ds[0][indices],ds[1][indices])
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError('nonfinite BC')
                bc_opt.step();bc_log.append(dict(step=k,indices=indices.tolist(),loss=float(loss.detach()),gradient_norm=float(norm)))
        bc_wall=time.perf_counter_ns()-bc_start;probe_rows['after_bc']=probe();write(folder/'bc.json',bc_log)
        bc_end_hashes[f'{arm}/{fit}']=tensors_hash(model)
        target=copy.deepcopy(model).eval().requires_grad_(False) if definition['ddqn'] else None
        opt=optimizer(model,.0001) if definition['ddqn'] else None
        if opt is not None and opt.state:raise ValueError('DDQN optimizer must start empty')
        if target is not None:
            for p in target.parameters():p.grad=None
        replay=list(demo[fit]) if definition['demo_replay'] else []
        replay_ids=[f'demo:{i}' for i in range(len(replay))]
        begin_hash=hashlib.sha256(ds[0].numpy().tobytes()+ds[1].numpy().tobytes()).hexdigest() if definition['demo_replay'] else None
        choice_rng=np.random.default_rng(pl['seeds']['epsilon_branch_seeds'][fit]);sample_rng=np.random.default_rng(pl['seeds']['replay_sampler_seeds'][fit])
        slots=0;updates=0;copies=0;optlog=[];drawlog=[];online_begin=time.perf_counter_ns()
        def actor(s):
            u=float(choice_rng.random());v=choice_rng.random(32)
            eps=1. if slots<pl['warmup'] else 1-.95*min(slots/(pl['online_episodes']*40),1)
            if u<eps:
                mask=baseline.allowed(s[None])[0];a=np.array([np.flatnonzero(mask[i])[int(v[i]*mask[i].sum())] for i in range(32)],np.int64)
            else:a=baseline.deploy_actions(model,s)
            drawlog.append(dict(slot=slots,epsilon=eps,epsilon_uniform=u,branch_uniforms=v.tolist(),action=a.tolist()))
            return a
        def update(row,t):
            nonlocal slots,updates,copies
            replay.append(row);replay_ids.append(f'online:{slots}')
            if len(replay)>pl['replay_capacity']:replay.pop(0);replay_ids.pop(0)
            if slots>=pl['warmup']:
                idx=sample_rng.integers(0,len(replay),pl['batch'])
                detail=train_update(model,target,[replay[i] for i in idx],opt)
                updates+=1
                if updates%pl['target_interval']==0:target.load_state_dict(model.state_dict());copies+=1
                detail.update(update=updates,online_slot=slots,indices=idx.tolist(),transition_ids=[replay_ids[i] for i in idx],target_copy=updates%pl['target_interval']==0)
                optlog.append(detail)
            slots+=1
        for ep in range(pl['online_episodes'] if definition['ddqn'] else 0):
            episode(folder/f'train_{ep:03d}',pl['seeds']['online_seed_blocks'][fit]['start']+ep,actor,update)
        probe_rows['final']=probe();write(folder/'probes.json',probe_rows)
        key=f'{arm}/{fit}';counts[key]=dict(slots=slots,updates=updates,target_copies=copies,
            bc_updates=len(bc_log),bc_wall_ns=bc_wall,online_wall_ns=time.perf_counter_ns()-online_begin,
            initial_tensor_hash=initial_hash[str(fit)],preloaded_demo_hash=begin_hash,
            bc_end_tensor_hash=bc_end_hashes[f'{arm}/{fit}'],initial_target_copy=definition['ddqn'],
            parameter_bytes=baseline.parameter_bytes(model),buffer_bytes=sum(v.numel()*v.element_size() for v in model.buffers()))
        write(folder/'optimizer.json',optlog);write(folder/'action_draws.json',drawlog);write(folder/'counts.json',counts[key])
        if replay:
            batch=array_batch(replay);np.savez_compressed(folder/'final_replay.npz',states=batch[0].numpy(),actions=batch[1].numpy(),rewards=batch[2].numpy(),next_states=batch[3].numpy(),dones=batch[4].numpy())
        write(folder/'final_rng.json',dict(choices=choice_rng.bit_generator.state,replay=sample_rng.bit_generator.state,bc=bc_rng.bit_generator.state,transition_ids=replay_ids))
        torch.save(dict(online=model.state_dict(),target=target.state_dict() if target else None,
                        optimizer=opt.state_dict() if opt else None,counts=counts[key]),folder/'final.pt')
        ckhash[key]=sha(folder/'final.pt')
    for fit in range(pl['fits']):
        if bc_end_hashes[f'centered_linear_bc_ddqn/{fit}']!=bc_end_hashes[f'centered_linear_bc_only/{fit}']:
            raise ValueError('primary/BC-only initialization mismatch')
    write(run_dir/'CHECKPOINT_FREEZE.json',dict(checkpoints=ckhash,job_count=len(jobs),bc_end_hashes=bc_end_hashes))
    event('checkpoints_frozen')
    models={}
    for arm,fit in jobs:
        model=variants.Critic(arm).eval()
        model.load_state_dict(torch.load(run_dir/arm/f'fit_{fit}'/'final.pt',weights_only=True)['online'])
        models[f'{arm}/{fit}']=model
    evaluations={};value_metrics={}
    for key,model in list(models.items())+[('teacher',None)]:
        event('evaluation',policy=key)
        folder=run_dir/'evaluation'/key;folder.mkdir(parents=True)
        evaluations[key]=[]
        errors=[];returns=[]
        for ep in range(pl['eval_episodes']):
            _,record=episode(folder/f'episode_{ep:03d}',pl['seeds']['evaluation_seed_block']['start']+ep,
                             teacher_action if model is None else lambda s,m=model:baseline.deploy_actions(m,s),
                             collect_q=False if model is None else model)
            evaluations[key].append(dict(return_value=record['return_value'],deadline=record['summary']['deadline_success_rate']))
            if model is not None:
                saved=np.load((folder/f'episode_{ep:03d}').with_suffix('.npz'))
                g=np.cumsum(saved['rewards'][::-1],dtype=np.float64)[::-1]
                errors.extend(saved['q']-g);returns.extend(g)
        if model is not None:
            value_metrics[key]=dict(bias=float(np.mean(errors)),
                nrmse=float(np.sqrt(np.mean(np.square(errors)))/max(np.sqrt(np.mean(np.square(returns))),1e-12)))
    analysis={}
    for fit in range(pl['fits']):
        def means(key):return {m:float(np.mean([r[m] for r in evaluations[key]])) for m in ('return_value','deadline')}
        absolute={arm['id']:means(f"{arm['id']}/{fit}") for arm in spec['arms']};ref=means('teacher');primary=absolute[spec['primary']]
        analysis[str(fit)]=dict(absolute=absolute,teacher=ref,
            contrasts={arm:{m:primary[m]-row[m] for m in primary} for arm,row in absolute.items() if arm!=spec['primary']},
            minus_teacher={arm:{m:row[m]-ref[m] for m in row} for arm,row in absolute.items()})
    screens=variants.analyze(analysis,value_metrics)
    write(run_dir/'analysis.json',dict(per_fit=analysis,value_metrics=value_metrics,**screens,kind=pl['kind'],quality_claim=False))
    # Timing uses a frozen deterministic list of nonterminal evaluation states.
    states=[]
    for path in sorted((run_dir/'evaluation').rglob('*.npz')):
        states.extend(np.load(path)['states'])
    timing_rng=np.random.default_rng(pl['seeds']['timing_order_seed'])
    chosen=timing_rng.permutation(len(states))[:pl['timing_states']];timing_states=np.stack([states[i] for i in chosen])
    np.save(run_dir/'timing_states.npy',timing_states)
    paths=list(models)+['teacher'];order=[(r,k) for r in range(pl['timing_rounds']) for k in paths];timing_rng.shuffle(order)
    write(run_dir/'TIMING_FREEZE.json',dict(states_sha256=sha(run_dir/'timing_states.npy'),order=order))
    timing=[]
    for round_id,key in order:
        action=teacher_action if key=='teacher' else lambda s,m=models[key]:baseline.deploy_actions(m,s)
        for j in range(pl['timing_warmups']):action(timing_states[j%len(timing_states)])
        for i,s in enumerate(timing_states):
            before=time.perf_counter_ns();a=action(s);ns=time.perf_counter_ns()-before
            timing.append(dict(round=round_id,policy=key,state=i,ns=ns,action=a.tolist()))
    write(run_dir/'timing.json',timing)
    event('artifacts_complete')
    write(run_dir/'manifest.json',{p.relative_to(run_dir).as_posix():sha(p) for p in sorted(run_dir.rglob('*')) if p.is_file()})
    from audit_centered_linear_comparison import audit
    receipt=audit(run_dir);write(out/'saved_artifact_audit.json',receipt)
    write(out/'completion.json',dict(status='ENGINEERING_AUDITED_COMPLETE' if mode=='fixture' else 'DEVELOPMENT_AUDITED_COMPLETE',
        kind=pl['kind'],audit_passed=receipt['passed'],manifest_sha256=sha(run_dir/'manifest.json'),research_launch_authorized=False))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['fixture','development','final'],default='fixture')
    parser.add_argument('--output',required=True);parser.add_argument('--confirmation')
    args=parser.parse_args()
    print(json.dumps(run(args.output,args.mode,args.confirmation)))
