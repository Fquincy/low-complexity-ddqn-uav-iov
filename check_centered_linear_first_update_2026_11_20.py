"""Reconstruct first-update inputs from saved data and retained BC-only weights."""
import json
import numpy as np
import torch
import diagnose_centered_linear_saved_traces_2026_11_20 as d


def main():
    output=d.OUT/'first_update_decomposition.json'
    if output.exists():raise RuntimeError('Preserve previous analysis')
    torch.set_num_threads(1);manifest=d.read(d.RUN/'manifest.json');results={}
    for fit in range(3):
        folder=d.RUN/d.PRIMARY/f'fit_{fit}'
        first=d.read(folder/'optimizer.json')[0]
        rows=[];cache={}
        for identity in first['transition_ids']:
            kind,index=identity.split(':');index=int(index);ep,slot=divmod(index,40)
            path=d.RUN/f'demo_{fit}'/f'episode_{ep:03d}.npz' if kind=='demo' else folder/f'train_{ep:03d}.npz'
            if path not in cache:cache[path]=d.arrays(path)
            z=cache[path];rows.append(tuple(z[k][slot] for k in ['states','actions','rewards','next_states','dones']))
        s,a,r,ns,done=[np.stack(x) for x in zip(*rows)];mask=~done
        model=d.variants.Critic(d.PRIMARY).eval()
        ck=torch.load(d.track(d.RUN/d.BC/f'fit_{fit}/final.pt'),weights_only=True)
        model.load_state_dict(ck['online'])
        with torch.no_grad():
            st=torch.from_numpy(s);act=torch.from_numpy(a)
            prediction=model(st,act).numpy()
            current_greedy=model.greedy(st)[1].numpy()
            _,nextq,_=model.greedy(torch.from_numpy(ns[mask]));nextq=nextq.numpy()
        assert np.allclose(prediction,first['prediction'],rtol=1e-5,atol=1e-5)
        assert np.allclose(nextq,np.array(first['target_q'])[mask],rtol=1e-5,atol=1e-5)
        action_gap=current_greedy[mask]-prediction[mask]
        temporal=nextq-current_greedy[mask]
        residual=r[mask]+action_gap+temporal
        assert np.allclose(residual,(np.array(first['target'])-prediction)[mask],rtol=1e-5,atol=2e-5)
        results[str(fit)]=dict(first_update=first['update'],online_slot=first['online_slot'],
            nonterminal_samples=int(mask.sum()),reward=d.stats(r[mask]),
            same_state_greedy_minus_sampled_q=d.stats(action_gap),
            successor_greedy_minus_current_greedy_q=d.stats(temporal),
            reconstructed_target_minus_prediction=d.stats(residual),
            note='Exact first-update numerical decomposition, not evidence that higher Q represents better true return.')
    for name,digest in d.consumed.items():
        path=d.ROOT/name;assert d.sha(path)==digest
        if path.is_relative_to(d.RUN) and path!=d.RUN/'manifest.json':assert manifest[path.relative_to(d.RUN).as_posix()]==digest
    output.write_text(json.dumps(dict(kind='OFFLINE_FIRST_UPDATE_RECONSTRUCTION',per_fit=results,
        consumed_sha256=d.consumed,source_sha256=d.sha(d.ROOT/'check_centered_linear_first_update_2026_11_20.py'),
        new_simulator_steps=0,new_optimizer_updates=0),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
