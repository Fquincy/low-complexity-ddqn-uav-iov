"""Paired additive versus SPI learning smoke; no full pilot launch."""
from pathlib import Path
import copy,json,time,hashlib,traceback
import numpy as np, torch
from torch.nn import functional as F
import sys
sys.path.insert(0,str(Path(__file__).parent/'Source'))
from chapter3_iov_env_v2 import Chapter3IoVEnv
from train_chapter3_ddqn import FactorizedQNetwork,to_action_dict
from spi_ddqn import SPIDDQN,select

ROOT=Path(__file__).resolve().parent
SEEDS=(14100000,14100001)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):
 h=hashlib.sha256();
 def walk(x):
  if isinstance(x,torch.Tensor):x=x.detach().cpu().numpy()
  if isinstance(x,np.ndarray):h.update(str((x.dtype.str,x.shape)).encode());h.update(x.tobytes())
  elif isinstance(x,dict):
   for k in sorted(x,key=str):walk(k);walk(x[k])
  elif isinstance(x,(list,tuple)):
   for z in x:walk(z)
  else:h.update(repr(x).encode())
  h.update(b'\0')
 walk(v);return h.hexdigest()

class Replay:
 def __init__(self,cap):
  self.cap=cap;self.size=0;self.pos=0;self.s=np.empty((cap,533),np.float32);self.ns=self.s.copy();self.a=np.empty((cap,32),np.int64);self.r=np.empty(cap,np.float32);self.d=np.empty(cap,np.float32)
 def add(self,s,a,r,ns,d):
  i=self.pos;self.s[i]=s;self.a[i]=a;self.r[i]=r;self.ns[i]=ns;self.d[i]=d;self.pos=(i+1)%self.cap;self.size=min(self.size+1,self.cap)
 def sample(self,n,rng):
  i=rng.integers(0,self.size,n);return self.s[i],self.a[i],self.r[i],self.ns[i],self.d[i],i

def make(arm,seed):
 torch.manual_seed(seed)
 model=FactorizedQNetwork(533,32,5,(128,128)) if arm=='additive' else SPIDDQN()
 return dict(arm=arm,online=model,target=copy.deepcopy(model).eval(),opt=torch.optim.Adam(model.parameters(),lr=1e-4),replay=Replay(64),action_rng=np.random.default_rng(seed+100000),replay_rng=np.random.default_rng(seed+200000),slots=0,updates=0,copies=0)

def choose(agent,state,eps,warmup):
 random=warmup or agent['action_rng'].random()<eps
 if random:return agent['action_rng'].integers(0,5,32,dtype=np.int64),True,1.
 with torch.inference_mode():
  if agent['arm']=='additive':return agent['online'](torch.tensor(state)[None])[0].argmax(1).numpy(),False,0.
  a,info=select(agent['online'],state);return a,False,float(info['candidate_count'])

def update(agent):
 if agent['slots']<8:return None
 s,a,r,ns,d,i=agent['replay'].sample(4,agent['replay_rng']);s=torch.tensor(s);a=torch.tensor(a);r=torch.tensor(r);ns=torch.tensor(ns);d=torch.tensor(d)
 if agent['arm']=='additive':
  q=agent['online'](s).gather(2,a[:,:,None]).squeeze(2).sum(1);n=agent['online'](ns).argmax(2);y=r+(1-d)*agent['target'](ns).gather(2,n[:,:,None]).squeeze(2).sum(1)
 else:
  q=agent['online'].joint_values(s,a);greedy=[]
  with torch.inference_mode():
   for row in ns:greedy.append(select(agent['online'],row.numpy())[0])
  na=torch.tensor(np.stack(greedy));y=r+(1-d)*agent['target'].joint_values(ns,na)
 loss=F.smooth_l1_loss(q,y);agent['opt'].zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(agent['online'].parameters(),10);agent['opt'].step();agent['updates']+=1
 if agent['updates']%8==0:agent['target'].load_state_dict(agent['online'].state_dict());agent['copies']+=1
 return dict(update=agent['updates'],loss=float(loss.detach()),sampled_indices=i.tolist())

def run(out):
 out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 agents={f'{arm}_{rep}':make(arm,1601+rep) for rep in range(2) for arm in ('additive','spi')};rows={};starts=time.perf_counter()
 for rep,seed in enumerate(SEEDS):
  worlds={k:Chapter3IoVEnv(seed=seed) for k in ('additive','spi')};history=[]
  for slot in range(80):
   if slot and slot%40==0:
    for arm in ('additive','spi'): worlds[arm].reset(seed=seed+slot//40)
   eps=1. if slot<8 else 1-.95*min(slot/80,1);actions={};infos={}
   for arm in ('additive','spi'):
    ag=agents[f'{arm}_{rep}'];state=worlds[arm]._observation();workload=[{k:t[k] for k in ('task_id','vehicle_id','arrival_s','input_bits','required_cycles','deadline_s')} for t in worlds[arm].current_tasks];a,random,beam=choose(ag,state,eps,slot<8);ns,r,d,info=worlds[arm].step(to_action_dict(a,30));ag['replay'].add(state,a,r,ns,d);ag['slots']+=1;u=update(ag);actions[arm]=a;infos[arm]=dict(reward=r,random=random,beam_candidates=beam,workload=workload,deadline=info['deadline_success_rate'],update=u)
   assert digest(infos['additive']['workload'])==digest(infos['spi']['workload'])
   history.append(dict(slot=slot,seed=seed,action_equal=np.array_equal(actions['additive'],actions['spi']),additive=infos['additive'],spi=infos['spi']))
  for arm in ('additive','spi'):
   ag=agents[f'{arm}_{rep}'];d=out/f'{arm}_{rep}';d.mkdir();(d/'history.json').write_text(json.dumps(history,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x,indent=2));torch.save({'arm':arm,'online':ag['online'].state_dict(),'target':ag['target'].state_dict(),'updates':ag['updates'],'copies':ag['copies']},d/'final.pt');rows[f'{arm}_{rep}']={'slots':ag['slots'],'updates':ag['updates'],'target_copies':ag['copies'],'mean_return':float(np.mean([x[arm]['reward'] for x in history])),'deadline_success':float(np.mean([x[arm]['deadline'] for x in history])),'beam_candidate_mean':float(np.mean([x['spi']['beam_candidates'] for x in history]))}
  
 result={'state':'complete','runs':rows,'paired_workloads':True,'total_slots':160,'total_updates':146,'new_full_pilot_launched':False,'wall_seconds':time.perf_counter()-starts,'low_complexity_claim':False}
 (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':run(Path('validation/low_complexity_smoke_2026-11-01_03'))
