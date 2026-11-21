"""Synthetic algebra/integration fixture; surrogate rewards are NOT IoV evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import terminal_anchor_prototype as p

ROOT=Path(__file__).resolve().parent
SPEC=ROOT/'TERMINAL_ANCHOR_ENGINEERING_PROTOCOL_2026-11-06.json'
SOURCES=['terminal_anchor_prototype.py','run_terminal_anchor_engineering.py','test_terminal_anchor_engineering.py',
 'controller_v3_amortized.py','run_v3_research_gate_v2.py','run_controller_redesign_engineering.py',
 'run_value_bound_tradeoff.py','Source/train_chapter3_ddqn.py',SPEC.name]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def load(path):return json.loads(Path(path).read_text())

def fixture(output):
    spec=load(SPEC);assert spec['research_enabled'] is False
    out=Path(output);out.mkdir(parents=True,exist_ok=False);p.existing.configure()
    initial_hash=sha(ROOT/'development_runs/controller_v3_development_2026-11-06_01/run_manifest.json')
    rng=np.random.default_rng(994610000)
    states=rng.uniform(0,.1,(80,533)).astype(np.float32)
    next_states=rng.uniform(0,.1,(80,533)).astype(np.float32)
    actions=rng.integers(0,5,(80,32),dtype=np.int64)
    rewards=-rng.uniform(.1,1,80).astype(np.float32)
    for i in range(80):states[i,-1]=(i%40)/40;next_states[i,-1]=(i%40+1)/40
    dones=np.array([i%40==39 for i in range(80)])
    np.savez_compressed(out/'synthetic_transitions.npz',states=states,next_states=next_states,actions=actions,rewards=rewards,dones=dones)
    logs={};freeze={}
    for arm in p.ARMS:
        online=p.Critic(arm);target=copy.deepcopy(online).eval().requires_grad_(False)
        opt=torch.optim.Adam(online.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        replay=p.Replay(48);uniform_rng=np.random.default_rng(99461+100000);anchor_rng=np.random.default_rng(99461+300000)
        rows=[];updates=0
        for i in range(80):
            replay.add((states[i],actions[i],rewards[i],next_states[i],dones[i]))
            if i<7:continue
            batch=replay.sample(8,uniform_rng);anchor=replay.sample(32,anchor_rng,True) if online.anchored else None
            opt.zero_grad(set_to_none=True);loss,detail=p.objective(online,target,batch,anchor)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(online.parameters(),10,error_if_nonfinite=True)
            assert all(v.grad is None for v in target.parameters())
            opt.step();updates+=1
            copied=updates%8==0
            if copied:target.load_state_dict(online.state_dict())
            rows.append(dict(slot=i,update=updates,loss=float(loss.detach()),gradient_norm=float(norm),
                target_copy=copied,uniform_indices=batch[-1].tolist(),anchor_indices=[] if anchor is None else anchor[-1].tolist(),
                terminal_pool=sorted(replay.terminal),**detail))
        path=out/(arm+'.pt');torch.save(dict(online=online.state_dict(),target=target.state_dict()),path)
        freeze[arm]=sha(path);logs[arm]=rows
    write(out/'training_trace.json',logs)
    write(out/'CHECKPOINT_FREEZE.json',dict(checkpoints=freeze,before_verification=True))
    # Saved synthetic held-out algebra checks only; not a policy return test.
    predictions={}
    for arm in p.ARMS:
        model=p.Critic(arm);model.load_state_dict(torch.load(out/(arm+'.pt'),weights_only=True)['online']);model.eval()
        with torch.no_grad():
            a,q=model.greedy(torch.tensor(states));terminal=torch.tensor(next_states[dones])
            assert torch.all(model(terminal,torch.tensor(actions[dones]))==0)
        predictions[arm]=dict(actions=a.tolist(),q=q.tolist(),parameters=sum(v.numel() for v in model.parameters()))
        assert sha(out/(arm+'.pt'))==freeze[arm]
    write(out/'predictions.json',predictions)
    for name in SOURCES:
        path=out/'source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    assert initial_hash==sha(ROOT/'development_runs/controller_v3_development_2026-11-06_01/run_manifest.json')
    write(out/'verification.json',dict(kind='SYNTHETIC_ALGEBRA_NOT_IOV_RESEARCH',arms=6,
        synthetic_transitions_per_arm=80,optimizer_updates=438,simulator_steps=0,
        original_run_manifest_sha256=initial_hash,policy_quality_evidence=False,thesis_changed=False))
    write(out/'manifest.json',dict(files={str(x.relative_to(out)).replace('\\','/'):sha(x) for x in out.rglob('*') if x.is_file()}))
    return audit(out)

def audit(folder):
    out=Path(folder);files=load(out/'manifest.json')['files']
    for name,digest in files.items():assert sha(out/name)==digest,name
    arrays=np.load(out/'synthetic_transitions.npz');predictions=load(out/'predictions.json');logs=load(out/'training_trace.json')
    freeze=load(out/'CHECKPOINT_FREEZE.json')['checkpoints']
    for arm in p.ARMS:
        assert sha(out/(arm+'.pt'))==freeze[arm]
        rows=logs[arm];assert len(rows)==73
        for index,row in enumerate(rows):
            assert row['update']==index+1 and row['slot']==index+7
            assert row['target_copy']==((index+1)%8==0)
            assert abs(row['loss']-row['uniform']-row['anchor'])<1e-3
            assert np.isfinite([row['loss'],row['gradient_norm']]).all()
            assert len(row['uniform_indices'])==8
            pool=row['terminal_pool']
            assert all(i in pool for i in row['anchor_indices'])
            if p.ARMS[arm][2] and pool:assert len(row['anchor_indices'])==32
            if not p.ARMS[arm][2]:assert row['anchor']==0 and not row['anchor_indices']
        model=p.Critic(arm);model.load_state_dict(torch.load(out/(arm+'.pt'),weights_only=True)['online'])
        with torch.no_grad():a,q=model.greedy(torch.tensor(arrays['states']))
        np.testing.assert_array_equal(a.numpy(),predictions[arm]['actions'])
        np.testing.assert_allclose(q.numpy(),predictions[arm]['q'],rtol=0,atol=0)
    result=dict(state='passed_synthetic_terminal_anchor_artifact_audit',checkpoints=6,updates=438,
        prediction_actions=480,manifest_entries=len(files),research_evidence=False,new_simulator_steps=0)
    write(out/'saved_artifact_audit.json',result);return result

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--mode',choices=['fixture','development','final'],required=True)
    cli.add_argument('--output',type=Path,required=True);args=cli.parse_args()
    if args.mode!='fixture':cli.error('Research modes disabled; isolated synthetic fixture only')
    print(json.dumps(fixture(args.output),indent=2))
