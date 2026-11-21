"""Reconstruct only first saved losses/gradient paths; never apply an update."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

import return_calibrated_handoff as h
import diagnose_return_calibrated_saved_traces_2026_11_20 as d

ROOT = d.ROOT
OUT = d.OUT


def norm(gs): return float(np.sqrt(sum(float((g.double()**2).sum()) for g in gs)))


def groups(names, gs):
    return dict(total=list(gs), shared_hidden=[g for n, g in zip(names, gs) if n.startswith('student.net.0.')],
        branch_output=[g[:160] for n, g in zip(names, gs) if n.startswith('student.net.2.')],
        value_output=[g[160:] for n, g in zip(names, gs) if n.startswith('student.net.2.')])


def comparison(xs, ys):
    nx, ny = norm(xs), norm(ys)
    dot = sum(float((x.double()*y.double()).sum()) for x, y in zip(xs, ys))
    return dict(left_norm=nx, right_norm=ny, dot=dot, cosine=dot/(nx*ny) if nx*ny else None)


def main():
    target_path = OUT/'first_update_gradient_decomposition.json'
    receipt_path = OUT/'first_update_verification.json'
    if target_path.exists() or receipt_path.exists(): raise RuntimeError('Preserve previous gradient decomposition')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
    initial_receipt = d.read(OUT/'verification.json')
    assert initial_receipt['status'] == 'OFFLINE_DIAGNOSIS_COMPLETE'
    for name, digest in initial_receipt['artifacts_sha256'].items(): assert d.sha(d.track(OUT/name)) == digest
    manifest = d.read(d.RUN/'manifest.json')
    freeze = d.read(d.RUN/'CHECKPOINT_FREEZE.json')
    result = dict(kind='POSTHOC_FIRST_SAVED_UPDATE_GRADIENT_PATH_RECONSTRUCTION', fits={})
    for fit in range(3):
        initial = torch.load(d.track(d.RUN/d.E/f'fit_{fit}/final.pt'), weights_only=True)
        model = h.Critic(d.A); model.load_state_dict(initial['online'])
        assert h.tensor_hash(model) == freeze['handoff_hashes'][f'{d.A}/{fit}']
        target = h.Critic(d.A); target.load_state_dict(initial['online']); target.requires_grad_(False)
        assert h.tensor_hash(target) == h.tensor_hash(model)
        a_log = d.read(d.RUN/d.A/f'fit_{fit}/optimizer.json')[0]
        b_log = d.read(d.RUN/d.B/f'fit_{fit}/optimizer.json')[0]
        assert a_log['transition_ids'] == b_log['transition_ids']
        assert a_log['online_slot'] == b_log['online_slot'] == 128
        demo = [d.arrays(d.RUN/f'demo_{fit}'/f'episode_{i:03d}.npz') for i in range(10)]
        online = [d.arrays(d.RUN/d.A/f'fit_{fit}'/f'train_{i:03d}.npz') for i in range(4)]
        keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
        sources = {label: {k: np.concatenate([v[k] for v in data]) for k in keys}
                   for label, data in [('demo', demo), ('online', online)]}
        selected = {k: [] for k in keys}
        for transition in a_log['transition_ids']:
            source, i = transition.split(':')
            if source == 'online': assert int(i) <= 128
            for k in keys: selected[k].append(sources[source][k][int(i)])
        s, a, r, ns, done = [torch.from_numpy(np.array(selected[k])) for k in keys]
        assert s.shape == (64, 533) and a.dtype == torch.long and done.dtype == torch.bool
        pred = model(s, a)
        with torch.no_grad():
            tq = torch.zeros(64)
            selected_next = model.greedy(ns[~done])[0]
            tq[~done] = target(ns[~done], selected_next)
            y = r+tq
        assert np.allclose(pred.detach().numpy(), a_log['prediction'], rtol=1e-6, atol=1e-6)
        assert np.allclose(tq.numpy(), a_log['target_q'], rtol=1e-6, atol=1e-6)
        assert np.allclose(y.numpy(), a_log['target'], rtol=1e-6, atol=1e-6)
        idx = a_log['retention_indices']
        keep = h.imitation_loss(model, torch.from_numpy(sources['demo']['states'][idx]),
            torch.from_numpy(sources['demo']['actions'][idx]), initial['counts']['alpha'])
        td = F.smooth_l1_loss(pred, y)
        assert np.isclose(float(td.detach()), a_log['td_loss'], rtol=1e-6, atol=1e-6)
        assert np.isclose(float(keep.detach()), a_log['retention_loss'], rtol=1e-6, atol=1e-6)
        names, params = zip(*model.named_parameters())
        gt = torch.autograd.grad(td, params, retain_graph=True)
        gk = torch.autograd.grad(keep, params, retain_graph=True)
        gall = torch.autograd.grad(td+keep, params, retain_graph=True)
        assert all(torch.allclose(t+k, g, rtol=3e-5, atol=1e-5) for t, k, g in zip(gt, gk, gall))
        v = model.tables(s)[2]
        huber_slope = torch.clamp(pred.detach()-y, -1., 1.)/len(s)
        gv = torch.autograd.grad(v, params, grad_outputs=huber_slope, retain_graph=True)
        gb = torch.autograd.grad(pred-v, params, grad_outputs=huber_slope)
        assert all(torch.allclose(vg+bg, tg, rtol=3e-5, atol=1e-5) for vg, bg, tg in zip(gv, gb, gt))
        kinds = {k: groups(names, gs) for k, gs in dict(td=gt, retention=gk, total=gall,
            td_value_path=gv, td_branch_path=gb).items()}
        for group, logged in a_log['gradient_norms'].items():
            assert np.isclose(norm(kinds['total'][group]), logged, rtol=3e-5, atol=1e-5)
            assert np.isclose(norm(kinds['td'][group]), b_log['gradient_norms'][group], rtol=3e-5, atol=1e-5)
        assert norm(kinds['retention']['value_output']) == 0
        assert all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in target.parameters())
        w = model.student.net[2].weight.detach()
        result['fits'][str(fit)] = dict(initial_tensor_hash=h.tensor_hash(model),
            saved_A_and_B_first_batch_identical=True, logged_predictions_targets_losses_and_group_norms_reproduced=True,
            norm_by_loss_and_group={kind: {group: norm(vals) for group, vals in grouped.items()} for kind, grouped in kinds.items()},
            td_vs_retention={group: comparison(kinds['td'][group], kinds['retention'][group]) for group in kinds['td']},
            td_value_vs_branch_shared=comparison(kinds['td_value_path']['shared_hidden'], kinds['td_branch_path']['shared_hidden']),
            initial_value_row_weight_norm=float(w[160].norm()), initial_branch_rows_weight_norm=float(w[:160].norm()),
            global_clipping_factor=10./max(10., norm(gall)),
            parameters_unchanged_after_analysis=h.tensor_hash(model) == freeze['handoff_hashes'][f'{d.A}/{fit}'])
        assert result['fits'][str(fit)]['parameters_unchanged_after_analysis']
    for name, digest in d.consumed.items():
        p = ROOT/name
        assert d.sha(p) == digest, name
        if p.is_relative_to(d.RUN) and p != d.RUN/'manifest.json': assert manifest[p.relative_to(d.RUN).as_posix()] == digest
    result.update(consumed_sha256=d.consumed, consumed_files_verified_unchanged=len(d.consumed),
        script_sha256=d.sha(Path(__file__)), new_offline_autograd_evaluations=15,
        new_optimizer_steps=0, new_simulator_steps=0, new_teacher_calls=0,
        historical_updates_reconstructed=3, full_historical_gradient_replay=False,
        limitation='Only the first calibrated update per fit is exactly reconstructible from retained E and sampled transitions. These local derivative paths cannot establish which change would improve final policy.',
        word_thesis_changed=False)
    target_path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    receipt = dict(status='FIRST_SAVED_UPDATE_PATHS_REPRODUCED_WITHOUT_PARAMETER_UPDATE',
        artifacts_sha256={target_path.name: d.sha(target_path)}, script_sha256=result['script_sha256'],
        main_diagnosis_script_sha256=d.sha(Path(d.__file__)), consumed_files_verified_unchanged=len(d.consumed),
        new_offline_autograd_evaluations=15, new_optimizer_steps=0, new_simulator_steps=0, new_teacher_calls=0)
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['fits'], indent=2))


if __name__ == '__main__': main()
