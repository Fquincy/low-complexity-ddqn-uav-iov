"""Independently recompute saved evidence; no optimizer or simulator stepping."""
import gzip,json
from pathlib import Path
import numpy as np
import torch
import run_horizon_loss_research as r
from audit_value_bound_development import check_episode
from audit_controller_redesign_engineering import verify_arrays

def demand(ok,message):
    if not ok:raise AssertionError(message)

def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)

def dist(x):
    a=np.asarray(x,np.float64).reshape(-1);demand(len(a)>0 and np.isfinite(a).all(),'bad distribution')
    return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),p05=float(np.quantile(a,.05)),p95=float(np.quantile(a,.95)),min=float(a.min()),max=float(a.max()))

def check_values(rows,saved):
    q=np.array([x['chosen_q'] for x in rows],np.float64);g=np.array([x['realized_return_to_go'] for x in rows],np.float64)
    tq=np.array([x['target_q'] for x in rows],np.float64);e=q-g
    demand(np.isfinite([q,g,tq]).all(),'nonfinite evaluation')
    demand(saved['normalized_rmse']==float(np.sqrt(np.mean(e**2))/max(1,np.sqrt(np.mean(g**2)))),'value RMSE')
    demand(saved['absolute_bias']==float(abs(e.mean())),'value bias')
    demand(saved['positive_Q_fraction']==float((q>0).mean()),'Q sign')
    demand(saved['near_zero_large_return_fraction']==float(((abs(q)<=1e-6)&(abs(g)>=1)).mean()),'near zero fraction')
    demand(saved['terminal_abs_error_p95']==float(np.quantile(abs(q[:,-1]-np.array([x['slot_rewards'][-1] for x in rows])),.95)),'terminal value')
    def distributions(ix):return {name:dist(x[:,ix]) for name,x in [('Q',q),('target',tq),('G',g),('Q_minus_G',e),('target_minus_G',tq-g)]}
    demand(saved['all_slots']==distributions(slice(None)),'all slot distribution')
    demand(saved['terminal']==distributions(slice(39,40)) and saved['nonterminal']==distributions(slice(0,39)),'terminal split')
    demand(saved['time_bins']==[dict(start=i,end=i+9,distributions=distributions(slice(i,i+10))) for i in range(0,40,10)],'time bins')
    for name in ('destination_counts','deadline_misses_by_destination'):
        demand(saved[name]==[sum(x[name][i] for x in rows) for i in range(5)],name)
    demand(saved['certified_infeasible_local_count']==sum(x['certified_infeasible_local_count'] for x in rows),'local infeasibility count')

def compact_verified(row):
    """Independent aggregation after checking the full saved episode."""
    keep={k:row[k] for k in ('seed','workload_sha256','return_sum','deadline_success','slot_rewards','chosen_q','target_q','realized_return_to_go')}
    keep['destination_counts']=[sum(t['destination']==i for t in row['tasks']) for i in range(5)]
    keep['deadline_misses_by_destination']=[sum(t['destination']==i and not t['on_time'] for t in row['tasks']) for i in range(5)]
    keep['certified_infeasible_local_count']=sum(sum(slot['excluded_local']) for slot in row['slots'])
    return keep

def pair(left,right):
    demand(len(left)==len(right),'pair length')
    demand([(x['seed'],x['workload_sha256']) for x in left]==[(x['seed'],x['workload_sha256']) for x in right],'pair workloads')
    return dict(return_difference=float(np.mean([a['return_sum']-b['return_sum'] for a,b in zip(left,right)])),
                deadline_difference=float(np.mean([a['deadline_success']-b['deadline_success'] for a,b in zip(left,right)])))

def analysis_audit(p,pl,evaluations,heuristics,costs,saved):
    primary=p['primary'];fits=len(pl['model_seeds']);expected={}
    for name in p['comparisons']['primary_report_against_each']:
        rows=[pair(evaluations[primary][i],heuristics[name] if name in heuristics else evaluations[name][i]) for i in range(fits)]
        expected[name]=dict(per_fit=rows,mean_return_difference=float(np.mean([x['return_difference'] for x in rows])),mean_deadline_difference=float(np.mean([x['deadline_difference'] for x in rows])))
    demand(saved['comparisons']==expected,'comparison recomputation')
    aux=[dict(**x,per_fit=[pair(evaluations[x['left']][i],evaluations[x['right']][i]) for i in range(fits)]) for x in p['comparisons']['auxiliary_contrasts']]
    demand(aux==saved['auxiliary_contrasts'],'auxiliary comparisons')
    demand(set(saved['value_stability'])==set(evaluations),'value arms')
    for arm,fitrows in evaluations.items():
        demand(len(saved['value_stability'][arm])==fits,'value fits')
        for i,rows in enumerate(fitrows):check_values(rows,saved['value_stability'][arm][i])
    def per(names,rm,dm):return all(x['return_difference']>=rm and x['deadline_difference']>=dm for n in names for x in expected[n]['per_fit'])
    c=p['comparisons'];v=p['value_stability'];gates={}
    gates['references_every_fit']=per(c['required_reference_arms'],c['reference_each_fit']['return_difference_min'],c['reference_each_fit']['deadline_fraction_difference_min'])
    gates['strong_heuristics_every_fit']=per(c['strong_heuristics'],0,0)
    for m in c['mechanisms']:
        x=expected[m['right']]
        gates[m['id']+'_mechanism']=per([m['right']],m['each_fit_return_difference_min'],m['each_fit_deadline_fraction_difference_min']) and x['mean_return_difference']>=m['equal_fit_mean_return_difference_min'] and x['mean_deadline_difference']>=m['equal_fit_mean_deadline_fraction_difference_min']
    gates['primary_value_every_fit']=all(x['normalized_rmse']<=v['normalized_rmse_max'] and x['absolute_bias']<=v['absolute_bias_max']
        and x['near_zero_large_return_fraction']<=v['near_zero_large_return_fraction_max'] and x['terminal_abs_error_p95']<=v['terminal_abs_error_p95_max']
        and x['positive_Q_fraction']==0 for x in saved['value_stability'][primary])
    cr=p['action_cost']['primary_each_fit'];costscreens=[]
    for i in range(fits):
        a=costs[primary][i]
        costscreens.append(dict(storage=a['parameter_bytes']<=cr['parameter_bytes_max_fraction_of_unmasked_compact64']*costs['unmasked_compact64'][i]['parameter_bytes'],
            median=all(a['median_ms']<=cr['median_ratio_max']*costs[n][i]['median_ms'] for n in cr['median_ms_max_fraction_of_each']),p95=a['p95_ms']<=cr['p95_ms_max']))
    gates['primary_cost_every_fit']=all(all(x.values()) for x in costscreens)
    demand(saved['cost_screens']==costscreens and saved['computed_screens']==gates,'acceptance screens')
    fixture=pl['mode']=='integration-fixture'
    demand(saved['quality_evidence']==(not fixture) and saved['screens_applied']==(not fixture),'fixture quality labeling')
    demand(saved['all_scientific_screens_passed']==(None if fixture else all(gates.values())),'overall decision')
    rng=np.random.default_rng(p['fixture']['engineering_seeds']['bootstrap'] if fixture else p['rng']['bootstrap_seed']);tasks=len(pl['evaluation_seeds']);draws=p['uncertainty']['draws']
    fs=rng.integers(fits,size=(draws,fits));ts=rng.integers(tasks,size=(draws,tasks))
    for n in expected:
        ci=[]
        for field in ('return_sum','deadline_success'):
            matrix=np.array([[x[field]-(heuristics[n] if n in heuristics else evaluations[n][i])[j][field] for j,x in enumerate(row)] for i,row in enumerate(evaluations[primary])],np.float64)
            samples=[np.mean(matrix[a[:,None],b[None,:]]) for a,b in zip(fs,ts)]
            ci.append(np.quantile(samples,[.025,.975]).tolist())
        demand(saved['bootstrap'][n]==dict(return_interval95=ci[0],deadline_interval95=ci[1],descriptive_only=True),'bootstrap')

def audit(output,write_report=True,report_path=None):
    out=Path(output);r.configure();p=r.protocol();pl=r.read(out/'runtime_plan.json');expected_plan=r.plan(p,pl['mode'])
    demand(pl==expected_plan,'runtime plan drift')
    manifest=r.read(out/'run_manifest.json')['files'];actual={x.relative_to(out).as_posix() for x in out.rglob('*') if x.is_file()}
    demand(actual==set(manifest)|{'run_manifest.json'}|({'saved_artifact_audit.json'} if 'saved_artifact_audit.json' in actual else set()),'unmanifested/missing files')
    for n,h in manifest.items():
        path=(out/n).resolve();demand(path.is_relative_to(out.resolve()),'manifest path escape');demand(r.sha(path)==h,'artifact hash: '+n)
    pre=r.read(out/'PRE_RUN_FREEZE.json');timing=r.read(out/'PRE_TIMING_FREEZE.json')
    demand(pre['protocol_sha256']==r.sha(r.SPEC) and pre['plan_sha256']==r.sha(out/'runtime_plan.json'),'pre-run freeze')
    demand(pre['source_sha256']==timing['source_sha256'],'source freeze mismatch')
    for n,h in pre['source_sha256'].items():demand(r.sha(r.ROOT/n)==r.sha(out/'source_snapshot'/n)==h,'source hash: '+n)
    demand(r.read(out/'environment.json')['threads']==1,'thread setting')
    events=r.read(out/'events.json');stages=[x['stage'] for x in events]
    demand([x['sequence'] for x in events]==list(range(len(events))),'event sequence')
    demand(stages[-5:]==['all_checkpoints_frozen','all_runtime_equivalence_passed','all_timing_completed','all_evaluation_completed','summaries_completed'],'freeze/runtime/evaluation stage ordering')
    defs={x['id']:x for x in p['learned_arms']};fits=len(pl['model_seeds']);jobs=[(n,i) for n in defs for i in range(fits)]
    np.random.default_rng(pl['order_seed']).shuffle(jobs)
    demand(r.read(out/'arm_order.json')==[list(x) for x in jobs],'training order')
    demand([(x['arm'],x['fit']) for x in events if x['stage']=='training']==jobs,'training event order')
    frozen=r.read(out/'ALL_FINAL_CHECKPOINTS_FREEZE.json');demand(frozen==timing['checkpoints'],'timing checkpoint freeze')
    demand(set(frozen)=={f'{a}/fit_{i:02d}/final.pt' for a,i in jobs},'missing/duplicate checkpoints')
    demand(len(list(out.rglob('*.pt')))==len(jobs),'intermediate checkpoints')
    z=np.load(out/'engineering_observations.npz');engineering=z['observations']
    demand(engineering.shape==(128,533),'engineering inputs')
    demand(r.sha(out/'engineering_observations.npz')==p['action_cost']['common_inputs_sha256']==timing['observations_sha256'],'timing inputs')
    predictions=r.read(out/'predictions.json');demand(set(predictions)=={f'{n}/{i}' for n,i in jobs},'prediction coverage')
    workloads={};evaluations={n:[[] for _ in range(fits)] for n in defs};heuristics={x['id']:[] for x in p['heuristics']}
    counts={};episode_count=task_count=update_count=action_checks=0;optimizer_summaries={n:[] for n in defs}
    def load_episode(path,seed,arm,model_seed,epid):
        nonlocal episode_count,task_count
        with gzip.open(path,'rt',encoding='utf-8') as f:row=json.load(f)
        demand((row['seed'],row['arm'],row['model_seed'],row['episode'],row['run_kind'])==(seed,arm,model_seed,epid,pl['kind']),'episode identity')
        check_episode(row,r.Chapter3Config());x,ns=verify_arrays(path.parent,path.name,row)
        demand(x.dtype==ns.dtype==np.float32,'state dtype');episode_count+=1;task_count+=1200
        np.testing.assert_allclose(row['realized_return_to_go'],np.cumsum(row['slot_rewards'][::-1])[::-1],atol=1e-9,rtol=0)
        mask=r.models.original.allowed(x)
        for t,s in enumerate(row['slots']):
            np.testing.assert_array_equal(s['excluded_local'],~mask[t,:30,0])
            for task in s['tasks_arrived']:
                i=task['vehicle_id'];lo,hi=r.models.original.intervals(np.array([x[t,8*i+6]],np.float32),500e6);dl,dh=r.models.original.intervals(np.array([x[t,8*i+7]],np.float32),.3)
                demand(lo[0]<=task['required_cycles']<=hi[0] and dl[0]<=task['deadline_s']<=dh[0],'saved interval containment')
                demand(mask[t,i,0] or task['required_cycles']/1.5e9>task['deadline_s']+1e-10,'false saved exclusion')
        return row,x,ns,mask
    for arm,fit in jobs:
        d=defs[arm];seed=pl['model_seeds'][fit];folder=out/arm/f'fit_{fit:02d}'
        demand(r.sha(folder/'final.pt')==frozen[f'{arm}/fit_{fit:02d}/final.pt'],'checkpoint hash')
        ck=torch.load(folder/'final.pt',weights_only=False,map_location='cpu');cfg=pl['cfg'];slots=cfg['episodes_per_fit']*40;updates=slots-cfg['warmup']+1
        demand((ck['arm'],ck['model_seed'],ck['definition'],ck['cfg'],ck['slots'],ck['updates'],ck['copies'])==(arm,seed,d,cfg,slots,updates,updates//cfg['target_interval_updates']),'checkpoint counters/config')
        counts[f'{arm}/fit_{fit:02d}/final.pt']=dict(slots=slots,updates=updates,copies=updates//cfg['target_interval_updates'])
        online=r.models.Critic(d,seed).eval();online.load_state_dict(ck['online']);target=r.models.Critic(d,seed).eval();target.load_state_dict(ck['target']);runtime=r.models.Runtime(online)
        demand(sum(x.numel() for x in online.parameters())==d['expected_online_parameter_count'],'live parameters')
        demand(sum(x.numel()*x.element_size() for x in online.buffers())==d['expected_buffer_bytes'],'live buffers')
        for j,state in enumerate(engineering):
            with torch.inference_mode():a,q=online.greedy(torch.from_numpy(state)[None])
            na,v=runtime.values(state);np.testing.assert_array_equal(a.numpy(),na);close(q.numpy(),v)
            demand(predictions[f'{arm}/{fit}'][j]==dict(action=na[0].tolist(),Q=float(v[0])),'saved runtime predictions');action_checks+=1
        arng=np.random.default_rng(seed+100000);rrng=np.random.default_rng(seed+200000);trng=np.random.default_rng(seed+300000)
        demand(len(ck['actor_history'])==slots,'actor history length')
        ring={};log_stream=(folder/'optimizer.jsonl').open(encoding='utf-8');log_iter=iter(log_stream)
        # Keep only numeric arrays needed for exact distributions, not nested log objects.
        logs={k:[] for k in ('grad_norm_before_clip','clipped','loss','uniform_loss','anchor_loss','bellman_targets','TD_residuals')}
        for ep in range(cfg['episodes_per_fit']):
            row,x,ns,masks=load_episode(folder/f'train_{ep:03d}.json.gz',pl['train_blocks'][fit]['start']+ep,arm,seed,ep)
            demand(not row['chosen_q'] and not row['target_q'],'training falsely labeled evaluation')
            workloads.setdefault(('train',fit,row['seed']),[]).append(row['workload_sha256'])
            for t,slot in enumerate(row['slots']):
                step=ep*40+t;u=float(arng.random());us=arng.random(32);warm=step<cfg['warmup'];epsilon=1. if warm else 1-.95*min(step/cfg['epsilon_decay'],1.)
                expected=dict(warmup=warm,epsilon=epsilon,explore=bool(warm or u<epsilon),epsilon_uniform=u,branch_uniforms=us.tolist())
                demand(ck['actor_history'][step]==expected,'action source RNG')
                action=np.array(slot['action'],np.int64)
                if d['screened']:demand(masks[t,np.arange(32),action].all(),'masked action illegal')
                if expected['explore']:
                    aa=[]
                    for branch in range(32):
                        options=np.flatnonzero(masks[t,branch]) if d['screened'] else np.arange(5)
                        aa.append(int(options[int(us[branch]*len(options))]))
                    np.testing.assert_array_equal(action,aa)
                ring[step%cfg['replay_capacity']]=(x[t],ns[t],np.float32(slot['reward']),t==39,action)
                physical=step+1
                if physical<cfg['warmup']:continue
                entry=json.loads(next(log_iter))
                for name in logs:
                    value=entry[name]
                    logs[name].append(np.asarray(value,np.float64) if isinstance(value,list) else value)
                ix=rrng.integers(0,min(cfg['replay_capacity'],physical),cfg['batch_size'])
                np.testing.assert_array_equal(ix,entry['replay_indices'])
                demand(entry['step']==physical-cfg['warmup']+1 and entry['physical_slots']==physical,'optimizer step')
                demand(entry['target_copy']==(entry['step']%cfg['target_interval_updates']==0),'target copy schedule')
                pool={i for i,v in ring.items() if v[3]};demand(entry['terminal_pool']==sorted(pool),'terminal pool eviction')
                ta=trng.choice(sorted(pool),cfg['anchor']['batch'],replace=True).tolist() if online.anchored and pool else []
                demand(entry['anchor_indices']==ta and entry['empty_anchor']==bool(online.anchored and not pool),'terminal anchor RNG')
                rewards=np.array([ring[i][2] for i in ix],np.float32);dones=np.array([ring[i][3] for i in ix])
                np.testing.assert_array_equal(entry['sample_rewards'],rewards);np.testing.assert_array_equal(entry['sample_dones'],dones.astype(np.float32))
                nact=np.array(entry['next_actions']);nv=np.array(entry['next_target_values'],np.float32)
                demand(np.all(nact[dones]==-1) and np.all(nv[dones]==0),'terminal decoding leaked')
                active=~dones
                demand(np.all((nact[active]>=0)&(nact[active]<5)),'next actions range')
                if d['screened'] and active.any():
                    nextmask=r.models.original.allowed(np.stack([ring[i][1] for i in ix[active]]))
                    demand(nextmask[np.arange(active.sum())[:,None],np.arange(32)[None,:],nact[active]].all(),'target next-state mask')
                y=rewards+nv;close(y,entry['bellman_targets']);q=np.array(entry['chosen_Q'],np.float32);close(q-y,entry['TD_residuals'])
                def huber(a,b):
                    delta=abs(np.asarray(a,np.float32)-np.asarray(b,np.float32));return float(np.where(delta<1,.5*delta**2,delta-.5).mean())
                if d['loss']=='branchwise_huber':
                    branch=np.array(entry['branch_values'],np.float32);close(branch.mean(1),q);uniform=huber(branch,y[:,None])
                elif d.get('normalized_horizon_loss',False):
                    h=np.array([40-round(float(ring[i][0][-1])*40) for i in ix],np.float32)
                    demand(np.all(h>=1),'invalid current h')
                    np.testing.assert_array_equal(entry['current_h'],h)
                    close(entry['normalized_TD_residuals'],(q-y)/h)
                    uniform=huber(q/h,y/h)
                else:uniform=huber(q,y)
                close(uniform,entry['uniform_loss'])
                demand(entry['anchor_rewards']==[float(ring[i][2]) for i in ta],'anchor rewards')
                al=huber(entry['anchor_Q'],entry['anchor_rewards']) if ta else 0.
                close(al,entry['anchor_loss']);close(entry['loss'],uniform+cfg['anchor']['weight']*al)
                demand(np.isfinite(q).all() and np.isfinite(nv).all() and np.isfinite(entry['grad_norm_before_clip']) and entry['optimizer_ns']>0,'finite optimizer history')
                demand(entry['clipped']==(entry['grad_norm_before_clip']>cfg['gradient_clip']),'clipping report')
                update_count+=1
        demand(next(log_iter,None) is None,'extra optimizer rows')
        log_stream.close()
        demand(arng.bit_generator.state==ck['action_rng_state'] and rrng.bit_generator.state==ck['replay_rng_state'] and trng.bit_generator.state==ck['anchor_rng_state'],'final RNG states')
        optimizer_summaries[arm].append(dict(fit=fit,grad_norm_before_clip=dist(logs['grad_norm_before_clip']),
            clipping_fraction=float(np.mean(logs['clipped'])),loss=dist(logs['loss']),
            uniform_loss=dist(logs['uniform_loss']),anchor_loss=dist(logs['anchor_loss']),
            Bellman_targets=dist(logs['bellman_targets']),TD_residuals=dist(logs['TD_residuals'])))
        for task_seed in pl['evaluation_seeds']:
            row,x,ns,mask=load_episode(folder/f'evaluation_{task_seed}.json.gz',task_seed,arm,seed,'evaluation')
            workloads.setdefault(('eval',task_seed),[]).append(row['workload_sha256'])
            actions=np.array([s['action'] for s in row['slots']],np.int64)
            # Reproduce the batch-one arithmetic used when recording each executed action.
            # FP32 matrix kernels may round differently for a forty-state batch.
            # No tolerance relaxation, state/action/weight changes or re-evaluation occurs.
            aa_parts=[];q_parts=[];target_parts=[]
            with torch.inference_mode():
                for slot_id in range(40):
                    tensor=torch.from_numpy(x[slot_id:slot_id+1])
                    selected,value=online.greedy(tensor)
                    target_value=target(tensor,torch.from_numpy(actions[slot_id:slot_id+1]))
                    aa_parts.append(selected);q_parts.append(value);target_parts.append(target_value)
            aa=torch.cat(aa_parts);q=torch.cat(q_parts);tq=torch.cat(target_parts)
            np.testing.assert_array_equal(aa.numpy(),actions);close(q.numpy(),row['chosen_q']);close(tq.numpy(),row['target_q'])
            for t in range(40):
                na,v=runtime.values(x[t]);np.testing.assert_array_equal(na[0],actions[t]);close(v[0],row['chosen_q'][t])
            evaluations[arm][fit].append(compact_verified(row))
    for name in heuristics:
        for seed in pl['evaluation_seeds']:
            row,x,ns,mask=load_episode(out/f'heuristic_{name}_{seed}.json.gz',seed,name,None,'evaluation')
            demand(not row['chosen_q'] and not row['target_q'],'heuristic value labels')
            for slot in row['slots']:
                # Read-only reconstruction of destination rules from logged geometry; no env.step.
                pos=np.array(slot['vehicle_positions']);rsu=np.array([[250.,500.],[750.,500.]])
                demand(slot['action'][30:]==[0,0],'heuristic must hover')
                expected=np.zeros(30,np.int64) if name=='all_local' else 3+np.argmin(np.linalg.norm(pos[:,None,:]-rsu[None,:,:],axis=2),axis=1)
                if name=='uav_selective':
                    uv=np.asarray(slot['uav_before']);distances=np.linalg.norm(pos[:,None,:]-uv[None,:,:],axis=2)
                    pairs=sorted((distances[v,u],v,u) for v in range(30) for u in range(2));used_v=set();used_u=set()
                    for _,v,u in pairs:
                        if v not in used_v and u not in used_u:expected[v]=u+1;used_v.add(v);used_u.add(u)
                np.testing.assert_array_equal(slot['action'][:30],expected)
            workloads.setdefault(('eval',seed),[]).append(row['workload_sha256']);heuristics[name].append(compact_verified(row))
    for key,hashes in workloads.items():
        count=len(defs) if key[0]=='train' else len(defs)*fits+len(heuristics)
        demand(len(hashes)==count and len(set(hashes))==1,'workload pairing')
    raw=r.read(out/'raw_timings.json');costs=r.read(out/'costs.json');orders=[];rng=np.random.default_rng(pl['timing_seed'])
    for rnd in range(5):order=list(jobs);rng.shuffle(order);orders.append([list(x) for x in order])
    demand(orders==timing['orders'],'timing order')
    demand([(x['arm'],x['fit'],x['round'],x['index']) for x in raw]==[(a,i,rnd,j) for rnd,order in enumerate(orders) for a,i in order for j in range(128)],'missing/duplicate timing rows')
    for arm in defs:
        for i in range(fits):
            ns=np.array([x['nanoseconds'] for x in raw if x['arm']==arm and x['fit']==i],np.float64);demand(len(ns)==640 and np.all(ns>0),'timings')
            x=costs[arm][i];demand(x['median_ms']==float(np.median(ns)/1e6) and x['p95_ms']==float(np.quantile(ns,.95)/1e6),'cost quantiles')
            demand(x['parameter_bytes']==defs[arm]['expected_fp32_parameter_bytes'] and x['buffer_bytes']==defs[arm]['expected_buffer_bytes'],'payload bytes')
            demand(x['checkpoint_bytes']==(out/arm/f'fit_{i:02d}'/'final.pt').stat().st_size,'checkpoint bytes')
    invariance=r.read(out/'evaluation_invariance.json');demand(len(invariance)==len(jobs) and {(x['arm'],x['fit']) for x in invariance}==set(jobs) and all(x['before']==x['after'] for x in invariance),'evaluation invariance')
    analysis_audit(p,pl,evaluations,heuristics,costs,r.read(out/'analysis.json'))
    expected_verification=dict(kind=pl['kind'],checkpoints=len(jobs),training_slots=sum(x['slots'] for x in counts.values()),optimizer_updates=update_count,
        target_copies=sum(x['copies'] for x in counts.values()),evaluation_slots=(len(jobs)+len(heuristics))*len(pl['evaluation_seeds'])*40,
        raw_timing_rows=len(raw),quality_evidence=pl['mode']!='integration-fixture',research_launched=pl['mode']=='development',counts=counts)
    demand(r.read(out/'verification.json')==expected_verification,'verification counters')
    import audit_horizon_loss_research_reports
    diagnostics=audit_horizon_loss_research_reports.audit(out,pl,p)
    result=dict(diagnostic_reports_independently_verified=diagnostics,state='passed_independent_saved_artifact_audit',kind=pl['kind'],episodes=episode_count,task_records=task_count,optimizer_rows=update_count,
        runtime_engineering_actions=action_checks,raw_timing_rows=len(raw),comparisons_and_bootstrap_recomputed=True,optimizer_distributions=optimizer_summaries,
        research_launched=pl['mode']=='development',auditor_optimizer_steps=0,auditor_simulator_steps=0)
    if write_report:r.write(Path(report_path) if report_path else out/'saved_artifact_audit.json',result)
    return result

if __name__=='__main__':
    import sys
    result=audit(sys.argv[1]);print(json.dumps({k:v for k,v in result.items() if k!='optimizer_distributions'}))
