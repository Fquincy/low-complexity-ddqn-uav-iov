"""Read-only diagnosis of saved development export failure; no simulator calls."""
import json,sys,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
import run_anchor_development_v4 as r
from torch.nn import functional as F
OUT=ROOT/'development_runs/terminal_anchor_development_2026-11-06_01'
DEST=ROOT/'validation/anchor_development_completion_check_2026-11-11_01'
def main():
    assert not DEST.exists();DEST.mkdir()
    r.configure();r.patch_training_core();p=r.protocol();frozen=r.read(OUT/'ALL_FINAL_CHECKPOINTS_FREEZE.json')
    plan=r.read(OUT/'runtime_plan.json');source_checks=0
    for name,digest in plan['source_sha256'].items():
        assert r.base.old.sha(ROOT/name)==r.base.old.sha(OUT/'source_snapshot'/name)==digest,name
        source_checks+=1
    rows=[]
    for fit,seed in enumerate(plan['model_seeds']):
        for arm in r.ARMS:
            folder=OUT/arm/f'fit_{fit:02d}';path=folder/'final.pt'
            assert r.base.old.sha(path)==frozen['checkpoints'][f'{arm}_fit_{fit:02d}']
            ck=torch.load(path,map_location='cpu',weights_only=False)
            model=r.make_model(arm,seed);model.load_state_dict(ck['online'])
            assert r.base.old.digest(model.state_dict())==ck['online_parameter_hash']
            files=list(folder.glob('train_*.json.gz'));assert len(files)==200
            rows.append(dict(arm=arm,fit=fit,training_episodes=len(files),checkpoint_sha256=r.base.old.sha(path)))
    states=np.load(OUT/'frozen_checkpoint_inference/observations.npz')['observations']
    diagnostics=[]
    for arm in ['price_both','no_price_both']:
        model=r.make_model(arm,plan['model_seeds'][0]);ck=torch.load(OUT/arm/'fit_00/final.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['online']);model.eval()
        graph=OUT/'frozen_checkpoint_inference/exports'/f'{arm}_fit_00.onnx'
        runtime=(r.compiled.session(graph),np.load(graph.with_suffix('.extra.npy')))
        mismatches=[];max_q_error=0.;core_max_error=0.
        with torch.inference_mode():
            for index,state in enumerate(states):
                x=torch.from_numpy(state)[None];ea,eq,_=r.greedy(model,x,arm)
                na,nq=r.runtime_values(runtime,arm,state)
                raw=model.model.net(x).numpy()[0];ort=runtime[0].run(None,{'states':state[None]})[0][0]
                core_max_error=max(core_max_error,float(np.max(np.abs(raw-ort))))
                max_q_error=max(max_q_error,float(np.max(np.abs(eq.numpy()-nq))))
                for branch in np.flatnonzero(ea.numpy()[0]!=na[0]):
                    placement,motion,_,_=model.model.tables(x)
                    scores=(placement[0,branch] if branch<30 else motion[0,branch-30]).numpy()
                    start=int(branch)*5
                    # For no-price arms, isolate decoder arithmetic using the exact same raw core vector.
                    same_raw_action,_=r.fused.v3_decode(raw,state,arm=='price_both')
                    item=dict(observation=index,branch=int(branch),eager_action=int(ea[0,branch]),export_action=int(na[0,branch]),
                        eager_branch_scores=scores.tolist(),eager_branch_raw=raw[start:start+5].tolist(),
                        ort_branch_raw=ort[start:start+5].tolist(),native_action_using_eager_raw=int(same_raw_action[branch]),
                        eager_q=float(eq[0]),export_q=float(nq[0]))
                    if arm=='no_price_both':
                        divisor=np.float32(30. if branch<30 else 2.)
                        item['native_scores_using_eager_raw']=[float(np.float32(r.fused.softplus(v)/divisor)) for v in raw[start:start+5]]
                        item['native_scores_using_ort_raw']=[float(np.float32(r.fused.softplus(v)/divisor)) for v in ort[start:start+5]]
                    mismatches.append(item)
        diagnostics.append(dict(arm=arm,fit=0,observations=len(states),mismatched_branches=len(mismatches),
            mismatch_observations=len(set(x['observation'] for x in mismatches)),max_absolute_q_error=max_q_error,
            max_core_absolute_error=core_max_error,mismatches=mismatches))
    result=dict(state='training_complete_post_training_export_gate_failed',checked_at='2026-11-11',
        training_slots=len(rows)*200*40,training_episodes=len(rows)*200,checkpoints_verified=len(rows),
        source_files_verified=source_checks,training_rows=rows,saved_export_diagnostics=diagnostics,
        comparisons_available=(OUT/'comparisons_and_value_stability.json').exists(),
        final_audit_available=(OUT/'saved_artifact_audit.json').exists(),
        heldout_event_present=any(x['stage']=='heldout_evaluation_started' for x in r.read(OUT/'events.json')),
        new_simulator_steps=0,optimizer_updates=0,research_relaunched=False,thesis_changed=False)
    r.write(DEST/'diagnosis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['training_rows','saved_export_diagnostics']}))
    for d in diagnostics:print(json.dumps(d))
if __name__=='__main__':main()
