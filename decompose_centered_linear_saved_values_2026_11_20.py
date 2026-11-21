"""Supplement saved-trace diagnosis with exact fixed-state value decomposition."""
import json
import numpy as np
import torch
import diagnose_centered_linear_saved_traces_2026_11_20 as d


def main():
    output=d.OUT/'value_decomposition.json'
    if output.exists(): raise RuntimeError('Preserve prior decomposition')
    torch.set_num_threads(1)
    manifest=d.read(d.RUN/'manifest.json'); result={}
    for fit in range(3):
        states=np.concatenate([d.arrays(d.RUN/f'demo_{fit}'/f'episode_{i:03d}.npz')['states'] for i in range(10)])
        x=torch.from_numpy(states);rows={}
        for arm in [d.BC,d.PRIMARY,'uncentered_linear_bc_ddqn']:
            ck=torch.load(d.track(d.RUN/arm/f'fit_{fit}/final.pt'),weights_only=True)
            model=d.variants.Critic(arm).eval();model.load_state_dict(ck['online'])
            with torch.no_grad():
                a,q,_=model.greedy(x);p,m,v,_=model.tables(x)
                pa=-(p.gather(2,a[:,:30,None]).squeeze(2)-p.mean(2)).sum(1)
                ma=-(m.gather(2,a[:,30:,None]).squeeze(2)-m.mean(2)).sum(1)
                centered=v+pa+ma
                if arm!= 'uncentered_linear_bc_ddqn':
                    assert torch.allclose(centered,q,rtol=1e-5,atol=1e-5)
                rows[arm]=dict(value_head_mean=float(v.mean()),centered_placement_advantage_mean=float(pa.mean()),
                    centered_movement_advantage_mean=float(ma.mean()),actual_q_mean=float(q.mean()),
                    centered_formula_q_mean=float(centered.mean()),
                    movement_counts=[np.bincount(a[:,30+i].numpy(),minlength=5).tolist() for i in range(2)])
        result[str(fit)]=rows
    # Sanity check for the diagnostic's zero-terminal first-batch case and sign.
    sample=[dict(prediction=[4.,-1.],target=[5.,-2.],rewards=[-1.,-2.],target_q=[6.,0.],dones=[False,True])]
    assert d.summarize_samples(sample,'nonterminal')['mean_clipped_target_minus_prediction']==1
    assert d.summarize_samples(sample,'terminal')['mean_clipped_target_minus_prediction']==-1
    empty=[dict(prediction=[4.],target=[5.],rewards=[-1.],target_q=[6.],dones=[False])]
    assert d.summarize_samples(empty,'terminal')['mean_clipped_target_minus_prediction'] is None
    for name,digest in d.consumed.items():
        path=d.ROOT/name;assert d.sha(path)==digest
        if path.is_relative_to(d.RUN) and path!=d.RUN/'manifest.json':assert manifest[path.relative_to(d.RUN).as_posix()]==digest
    output.write_text(json.dumps(dict(kind='OFFLINE_FIXED_STATE_DECOMPOSITION',per_fit=result,
        consumed_sha256=d.consumed,source_sha256=d.sha(d.ROOT/'decompose_centered_linear_saved_values_2026_11_20.py'),
        diagnostic_helper_sanity_checks_passed=True,new_simulator_steps=0,new_optimizer_updates=0),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
