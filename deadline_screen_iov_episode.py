"""Isolated screened simulator adapter; no research CLI."""
from run_controller_redesign_engineering import *
import deadline_screen_prototype as screen
# Explicit callbacks installed only in the isolated fixture process.
def episode(agent,seed,training=False,policy=None,episode_id=0):
    env=Chapter3IoVEnv(seed=seed);states=[];next_states=[];slots=[];q_values=[];target_values=[];logs=[]
    for t in range(40):
        state=env._observation().copy();states.append(state);work=old.workload(env);before=env.uav_positions.copy()
        details={}
        mask=screen.allowed(state[None])[0];excluded=~mask[:30,0]
        for task in work:
            i=task['vehicle_id'];c=task['required_cycles'];d=task['deadline_s']
            lo,hi=screen.intervals(np.array([state[8*i+6]],np.float32),500e6)
            dlo,dhi=screen.intervals(np.array([state[8*i+7]],np.float32),.3)
            assert lo[0]<=c<=hi[0] and dlo[0]<=d<=dhi[0]
            if excluded[i]:assert c/1.5e9>d+1e-10
        if policy:
            action=np.concatenate((old.select_destinations(env,policy),np.zeros(2,dtype=np.int64)))
        elif training:
            cfg=agent['cfg'];warm=agent['slots']<cfg['warmup']
            eps=1. if warm else 1-.95*min(agent['slots']/cfg['epsilon_decay'],1.)
            epsilon_uniform=float(agent['action_rng'].random());uniforms=agent['action_rng'].random(32)
            explore=bool(warm or epsilon_uniform<eps)
            if explore:action=screen.explore(state[None],uniforms[None])[0]
            else:
                with torch.inference_mode():action=greedy(agent['online'],torch.from_numpy(state)[None],agent['arm'])[0][0].numpy().copy()
            agent['actor_history'].append(dict(warmup=warm,epsilon=eps,explore=explore,epsilon_uniform=epsilon_uniform,branch_uniforms=uniforms.tolist()))
        else:
            with torch.inference_mode():
                tensor=torch.from_numpy(state)[None]
                at,q,details=greedy(agent['online'],tensor,agent['arm'])
                tq=chosen(agent['target'],tensor,at,agent['arm'])
                action=at[0].numpy().copy();q_values.append(float(q[0]));target_values.append(float(tq[0]))
                details={k:v[0].cpu().tolist() for k,v in details.items()}
        assert mask[np.arange(32),action].all()
        if not training and not policy:
            native_a,native_q=agent['runtime'].values(state)
            np.testing.assert_array_equal(action,native_a[0]);np.testing.assert_allclose(q.numpy(),native_q,rtol=1e-5,atol=1e-6)
        ns,reward,done,info=env.step(to_action_dict(action,30));next_states.append(ns.copy())
        if not np.isfinite(reward) or not np.isfinite(ns).all() or done!=(t==39):raise AssertionError('invalid transition')
        if training:
            agent['replay'].add(state,action,reward,ns,done);agent['slots']+=1
            item=update(agent)
            if item is not None:logs.append(item)
        after=env.uav_positions.copy()
        slots.append(dict(slot=t,action=action.tolist(),reward=reward,done=done,reward_components=info['reward_components'],
            energy_j=info['energy_j'],tasks_arrived=work,uav_before=before.tolist(),uav_after=after.tolist(),
            actual_distance_m=np.linalg.norm(after-before,axis=1).tolist(),decoder=details,excluded_local=excluded.tolist()))
    summary=env.episode_summary();rewards=[s['reward'] for s in slots]
    result=dict(seed=seed,episode=episode_id,run_kind='SYNTHETIC_NOT_RESEARCH',return_sum=sum(rewards),
        deadline_success=summary['deadline_success_rate'],tasks_generated=summary['tasks_generated'],
        tasks_on_time=summary['tasks_on_time'],slot_rewards=rewards,terminal_value=q_values[-1] if q_values else None,
        workload_sha256=old.digest([s['tasks_arrived'] for s in slots]),summary=summary,slots=slots,tasks=env.export_task_records(),
        chosen_q=q_values,target_q=target_values,realized_return_to_go=np.cumsum(rewards[::-1])[::-1].tolist(),
        observations=np.stack(states),next_observations=np.stack(next_states))
    return result,logs
