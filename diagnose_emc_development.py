"""Read-only Q, reward, optimizer and validation diagnostics for one frozen run."""
import argparse,gc,json
from pathlib import Path
import numpy as np,torch
import run_emc_tradeoff as r

RUN=r.ROOT/'validation/emc_tradeoff_development_2026-11-04_01'
OUT=r.ROOT/'validation/emc_q_value_diagnostics_2026-11-04_02'

def fraction(x):return float(np.mean(x))
def stats(x):
 x=np.asarray(x,dtype=np.float64)
 return dict(count=int(x.size),mean=float(np.mean(x)),std=float(np.std(x)),p50=float(np.quantile(x,.5)),p95=float(np.quantile(x,.95)),minimum=float(np.min(x)),maximum=float(np.max(x)))

def model_for(arm,ck):
 if arm==r.ARMS[2]:model=r.EMCQ()
 else:
  widths=(128,128) if arm==r.ARMS[0] else (64,64)
  model=r.FactorizedQNetwork(533,32,5,widths)
 model.load_state_dict(ck);return model.eval()

def read_validation(path):
 data=json.loads(path.read_text(encoding='utf-8'));result={}
 for ep,rows in data.items():
  returns=np.array([x['return_sum'] for x in rows]);deadline=np.array([x['deadline_success'] for x in rows])
  terminal=np.array([x['terminal_value'] for x in rows]);last=np.array([x['slot_rewards'][-1] for x in rows])
  result[ep]=dict(episodes=len(rows),return_mean=float(returns.mean()),return_min=float(returns.min()),return_max=float(returns.max()),
   deadline_success_mean=float(deadline.mean()),terminal_q_mean=float(terminal.mean()),last_reward_mean=float(last.mean()),
  terminal_q_positive_fraction=fraction(terminal>0),terminal_abs_error_p95=float(np.quantile(np.abs(terminal-last),.95)))
 return result

def diagnose(key):
 arm,rep=key.rsplit('_',1);rep=int(rep);folder=RUN/key
 ck=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
 online=model_for(arm,ck['online']);target=model_for(arm,ck['target']);target.requires_grad_(False)
 s=[];ns=[];actions=[];rewards=[];dones=[];train_returns=[];train_deadlines=[]
 for ep in range(1,201):
  with np.load(folder/f'train_ep{ep}.npz') as d:
   s.append(d['states'].copy());ns.append(d['next_states'].copy());actions.append(d['actions'].astype(np.int64,copy=True))
   rewards.append(d['rewards'].astype(np.float32,copy=True));dones.append(d['dones'].astype(bool,copy=True))
  summary=json.loads((folder/'training_summary.json').read_text(encoding='utf-8'))[ep-1]
  train_returns.append(summary['return_sum']);train_deadlines.append(summary['deadline_success'])
 states=torch.from_numpy(np.concatenate(s));next_states=torch.from_numpy(np.concatenate(ns))
 acts=torch.from_numpy(np.concatenate(actions));rew=np.concatenate(rewards).astype(np.float64);done=np.concatenate(dones)
 monte_carlo=np.empty_like(rew)
 for start in range(0,len(rew),40):monte_carlo[start:start+40]=np.cumsum(rew[start:start+40][::-1])[::-1]
 q_chosen=[];q_greedy=[];q_target=[]
 with torch.inference_mode():
  for start in range(0,len(rew),256):
   end=min(start+256,len(rew));st=states[start:end];nst=next_states[start:end];act=acts[start:end]
   if arm==r.ARMS[2]:
    qc=online(st,act);ga,gv=online.greedy(st)
    nna,_=online.greedy(nst);target_next=target(nst,nna)
   else:
    online_q=online(st);qc=online_q.gather(2,act[:,:,None]).squeeze(2).sum(1)
    gv=online_q.max(2).values.sum(1)
    online_next=online(nst);nna=online_next.argmax(2);target_q=target(nst)
    target_next=target_q.gather(2,nna[:,:,None]).squeeze(2).sum(1)
   q_chosen.append(qc.numpy());q_greedy.append(gv.numpy());q_target.append(target_next.numpy())
 q_chosen=np.concatenate(q_chosen);q_greedy=np.concatenate(q_greedy);q_target=np.concatenate(q_target)
 bellman_target=rew+np.where(done,0.,q_target)
 opt=[json.loads(line) for line in (folder/'optimizer.jsonl').read_text(encoding='utf-8').splitlines()]
 losses=np.array([x['loss'] for x in opt]);norms=np.array([x['grad_norm_before_clip'] for x in opt])
 chunks=np.array_split(np.arange(len(opt)),4)
 optimizer_trend=[dict(updates=len(ix),loss_mean=float(losses[ix].mean()),loss_p95=float(np.quantile(losses[ix],.95)),
   grad_norm_mean=float(norms[ix].mean()),fraction_at_or_above_clip=fraction(norms[ix]>=10)) for ix in chunks]
 actor=ck['actor_history'];explore_fraction=fraction([x['explore'] for x in actor])
 validation=read_validation(folder/'validation.json')
 fresh=[]
 import gzip
 with gzip.open(folder/'fresh.json.gz','rt',encoding='utf-8') as f:fresh=json.load(f)
 fresh_summary=dict(episode_return_mean=float(np.mean([x['return_sum'] for x in fresh])),
   deadline_success_mean=float(np.mean([x['deadline_success'] for x in fresh])),
   terminal_q_positive_fraction=fraction([x['terminal_value']>0 for x in fresh]),
   terminal_q_minus_last_reward_mean=float(np.mean([x['terminal_value']-x['slot_rewards'][-1] for x in fresh])))
 common=np.load(RUN/'common_probe_states.npz')['observations'];probe_values=[]
 with torch.inference_mode():
  for start in range(0,len(common),256):
   st=torch.from_numpy(common[start:start+256])
   probe_values.append((online.greedy(st)[1] if arm==r.ARMS[2] else online(st).max(2).values.sum(1)).numpy())
 probe_values=np.concatenate(probe_values)
 fixed_probe=dict(states=len(probe_values),greedy_q=stats(probe_values),positive_fraction=fraction(probe_values>0))
 result=dict(arm=arm,replicate=rep,model_seed=ck['model_seed'],
  action_exploration_fraction=explore_fraction,
  training_return_first50=float(np.mean(train_returns[:50])),training_return_last50=float(np.mean(train_returns[-50:])),
  training_deadline_first50=float(np.mean(train_deadlines[:50])),training_deadline_last50=float(np.mean(train_deadlines[-50:])),
  optimizer_quartiles=optimizer_trend,
  final_replay=dict(rows=len(rew),reward=stats(rew),episode_monte_carlo_return=stats(train_returns),
   return_to_go=stats(monte_carlo),q_chosen_behavior_action=stats(q_chosen),q_greedy=stats(q_greedy),
   q_double_dqn_next_target=stats(q_target),bellman_target=stats(bellman_target),
   td_residual_current_target=stats(bellman_target-q_chosen),
   positive_fractions=dict(reward=fraction(rew>0),return_to_go=fraction(monte_carlo>0),q_chosen=fraction(q_chosen>0),
    q_greedy=fraction(q_greedy>0),q_target=fraction(q_target>0),bellman_target=fraction(bellman_target>0))),
  fixed_probe=fixed_probe,scheduled_validation=validation,fresh_gate=fresh_summary)
 del ck,online,target,states,next_states,s,ns,actions,rewards,dones;gc.collect()
 return result

def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 audit=json.loads((RUN/'saved_artifact_audit.json').read_text())
 assert audit['state']=='passed_development_saved_artifact_audit'
 verification=json.loads((RUN/'verification.json').read_text());assert verification['state']=='development_screen_failed'
 OUT.mkdir(parents=True,exist_ok=False)
 results={}
 for key in verification['counters']:results[key]=diagnose(key)
 payload=dict(state='read_only_diagnostic_complete',simulator_steps=0,optimizer_updates=0,
  models=results,limits=['final replay is on-policy/exploratory data from each model run',
   'Monte Carlo return-to-go follows the realized exploratory trajectory, not an optimal continuation',
   'final checkpoints cannot show the within-training Q trajectory; scheduled terminal Q and losses provide the saved time trend',
   'positive Q compared with observed negative returns diagnoses mismatch/overestimation tendency, not its unique cause'])
 (OUT/'diagnostics.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
 compact={}
 for key,value in results.items():
  compact[key]=dict(q_greedy_mean=value['final_replay']['q_greedy']['mean'],
   mc_return_to_go_mean=value['final_replay']['return_to_go']['mean'],
   q_positive_fraction=value['final_replay']['positive_fractions']['q_greedy'],
   return_positive_fraction=value['final_replay']['positive_fractions']['return_to_go'],
   reward_positive_fraction=value['final_replay']['positive_fractions']['reward'],
   fixed_probe_q_positive_fraction=value['fixed_probe']['positive_fraction'],
   fresh_return=value['fresh_gate']['episode_return_mean'],
   validation={ep:row['return_mean'] for ep,row in value['scheduled_validation'].items()})
 print(json.dumps(dict(state=payload['state'],simulator_steps=0,optimizer_updates=0,output=str(OUT/'diagnostics.json'),models=compact),indent=2))

if __name__=='__main__':main()
