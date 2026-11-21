"""Isolated prospective handoff mechanics. No simulator, teacher or research runner."""
import argparse
import copy
from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
from torch.nn import functional as F

import centered_linear_comparison_models as previous
import heuristic_initialized_ddqn_baseline as baseline


@dataclass(frozen=True)
class Arm:
    calibration: bool
    retention: bool
    bc: bool
    ddqn: bool
    softplus: bool = False


ARMS = {
    'calibrated_retained_ddqn': Arm(True, True, True, True),
    'calibrated_no_retention_ddqn': Arm(True, False, True, True),
    'uncalibrated_retained_ddqn': Arm(False, True, True, True),
    'uncalibrated_no_retention_ddqn': Arm(False, False, True, True),
    'calibrated_bc_only': Arm(True, False, True, False),
    'old_softplus_bc_ddqn': Arm(False, False, True, True, True),
    'plain_small_softplus_ddqn': Arm(False, False, False, True, True),
}
RIDGE_RHO = .001
SCALE_FLOOR = 1e-6
RETENTION_LAMBDA = 1.


class Critic(previous.Critic):
    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError('Unknown handoff arm')
        self.handoff_arm = arm
        family = 'old_softplus_bc_ddqn' if ARMS[arm].softplus else 'centered_linear_bc_ddqn'
        super().__init__(family)

    @property
    def definition(self):
        return ARMS[self.handoff_arm]


def tensor_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def distribution(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Nonempty finite diagnostics required')
    return dict(count=len(x), mean=float(x.mean()),
                quantiles=np.quantile(x, [0, .05, .5, .95, 1]).tolist())


def labels_valid(states, actions):
    baseline.validate_states(states, nonterminal=True)
    if (actions.device.type != 'cpu' or actions.dtype != torch.long or
            actions.shape != (len(states), 32) or torch.any(actions < 0) or torch.any(actions > 4)):
        raise ValueError('CPU int64 branch labels required')
    if not np.take_along_axis(baseline.allowed(states), actions.numpy()[:, :, None], 2).all():
        raise ValueError('Demonstration label excluded by inherited screen')


@dataclass(frozen=True)
class DemoEpisode:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor

    def validate(self):
        labels_valid(self.states, self.actions)
        if len(self.states) != 40:
            raise ValueError('Declared demonstration horizon is 40')
        if (self.rewards.device.type != 'cpu' or self.rewards.dtype != torch.float32 or
                self.rewards.shape != (40,) or not torch.isfinite(self.rewards).all() or
                torch.any(self.rewards > 0)):
            raise ValueError('Finite nonpositive FP32 demonstration rewards required')
        if (self.dones.device.type != 'cpu' or self.dones.dtype != torch.bool or
                self.dones.shape != (40,) or self.dones[:-1].any() or not self.dones[-1]):
            raise ValueError('One final demonstration terminal required')


def episode_returns(rewards):
    x = np.asarray(rewards, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite reward vector required')
    return np.cumsum(x[::-1], dtype=np.float64)[::-1].copy()


def ridge_value_coefficients(features, targets, returns):
    """FP64 standardized ridge with unpenalized intercept, per prospective brief."""
    h, t, g = (np.asarray(x, dtype=np.float64) for x in (features, targets, returns))
    if (h.ndim != 2 or h.shape[1] != 24 or len(h) == 0 or
            t.shape != (len(h),) or g.shape != t.shape or
            not all(np.isfinite(x).all() for x in (h, t, g))):
        raise ValueError('Finite [N,24] features and return/target vectors required')
    mu = h.mean(0)
    sigma = np.maximum(SCALE_FLOOR, np.sqrt(np.mean((h-mu)**2, axis=0)))
    H = (h-mu)/sigma
    sg = max(SCALE_FLOOR, float(np.sqrt(np.mean(g*g))))
    u = t/sg
    gram = H.T@H/len(h) + RIDGE_RHO*np.eye(24)
    beta = np.linalg.solve(gram, H.T@(u-u.mean())/len(h))
    w = sg*beta/sigma
    bias = sg*u.mean()-w@mu
    fp32_w, fp32_bias = w.astype(np.float32), np.float32(bias)
    expected = sg*(H@beta+u.mean())
    exported = h.astype(np.float32)@fp32_w+fp32_bias
    if (not np.isfinite(fp32_w).all() or not np.isfinite(fp32_bias) or
            not np.allclose(exported, expected, rtol=1e-5, atol=1e-4)):
        raise ValueError('HOLD: nonfinite or inaccurate FP32 value-row export')
    return fp32_w, fp32_bias, dict(
        rho=RIDGE_RHO, return_scale=sg, feature_mean=mu.tolist(),
        feature_scale=sigma.tolist(), beta=beta.tolist(), rank=int(np.linalg.matrix_rank(H)),
        regularized_condition=float(np.linalg.cond(gram)),
        max_fp32_export_error=float(np.max(np.abs(exported-expected))),
        max_equation_residual=float(np.max(np.abs(gram@beta-H.T@(u-u.mean())/len(h)))))


def imitation_loss(model, states, actions, temperature=1.):
    labels_valid(states, actions)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive finite imitation temperature required')
    p, m, _, _ = model.tables(states)
    mask = torch.from_numpy(baseline.allowed(states))
    vehicle = (-p/temperature).masked_fill(~mask[:, :30], -torch.inf)
    return (F.cross_entropy(vehicle.reshape(-1, 5), actions[:, :30].reshape(-1)) +
            F.cross_entropy((-m/temperature).reshape(-1, 5), actions[:, 30:].reshape(-1)))


def bc_loss(model, states, actions):
    if not model.definition.bc:
        raise ValueError('Plain control must not receive demonstration training')
    return imitation_loss(model, states, actions, 1.)


def probe(model, states, actions, temperature):
    with torch.no_grad():
        greedy, q, _ = model.greedy(states)
        demo_q = model(states, actions)
        p, m, v, _ = model.tables(states)
        scores = torch.cat((p, m), 1)
        mask = torch.from_numpy(baseline.allowed(states))
        ranked = scores.masked_fill(~mask, torch.inf).sort(2).values
        return dict(tensor_hash=tensor_hash(model), actions=greedy.tolist(),
                    greedy_q=q.tolist(), demonstration_q=demo_q.tolist(), value=v.tolist(),
                    all_action_ranges=(scores.max(2).values-scores.min(2).values).sum(1).tolist(),
                    margins=(ranked[:, :, 1]-ranked[:, :, 0]).tolist(),
                    imitation_loss=float(imitation_loss(model, states, actions, temperature)))


@dataclass
class Handoff:
    model: Critic
    alpha: float
    report: dict


def calibrate(model, episodes):
    """Return an independent calibrated model. Errors never partially mutate input."""
    if not model.definition.calibration:
        raise ValueError('This arm must not receive calibration')
    if len(episodes) != 10:
        raise ValueError('Exactly ten independent demonstration episodes required')
    if len({id(e) for e in episodes}) != 10:
        raise ValueError('Duplicate episode objects are not independent data')
    for e in episodes:
        e.validate()
    states = torch.cat([e.states for e in episodes])
    actions = torch.cat([e.actions for e in episodes])
    rewards = torch.cat([e.rewards for e in episodes[:8]]).numpy()
    returns = np.concatenate([episode_returns(e.rewards.numpy()) for e in episodes])
    original_hash = tensor_hash(model)
    result = copy.deepcopy(model)
    before = probe(result, states, actions, 1.)
    sr = max(SCALE_FLOOR, float(np.median(np.abs(rewards.astype(np.float64)))))
    rmax = max(before['all_action_ranges'][:320])
    alpha = sr/(sr+rmax)
    if not math.isfinite(alpha) or alpha <= 0 or np.float32(alpha) <= 0:
        raise ValueError('HOLD: alpha underflow/nonfinite')
    final = result.student.net[2]
    with torch.no_grad():
        old_w, old_b = final.weight[:160].clone(), final.bias[:160].clone()
        final.weight[:160].mul_(alpha)
        final.bias[:160].mul_(alpha)
        if (((old_w != 0) & (final.weight[:160] == 0)).any() or
                ((old_b != 0) & (final.bias[:160] == 0)).any()):
            raise ValueError('HOLD: branch coefficient underflow')
    after_scale = probe(result, states, actions, alpha)
    if before['actions'] != after_scale['actions']:
        raise ValueError('HOLD: FP32 scaling changed greedy decisions')
    if max(after_scale['all_action_ranges'][:320]) > sr+1e-5+sr*1e-5:
        raise ValueError('HOLD: finite-dataset gap bound failed')
    # A single scalar value-row update is the only additional modification.
    scaled_state = copy.deepcopy(result.state_dict())
    with torch.no_grad():
        h = result.student.net[1](result.student.net[0](states)).numpy()
        p, m, _, _ = result.tables(states)
        cost = ((p.gather(2, actions[:, :30, None]).squeeze(2)-p.mean(2)).sum(1) +
                (m.gather(2, actions[:, 30:, None]).squeeze(2)-m.mean(2)).sum(1)).numpy()
    w, bias, ridge = ridge_value_coefficients(h[:320], returns[:320]+cost[:320], returns[:320])
    with torch.no_grad():
        final.weight[160].copy_(torch.from_numpy(w))
        final.bias[160].copy_(torch.tensor(bias))
    after_fit = probe(result, states, actions, alpha)
    if before['actions'] != after_fit['actions']:
        raise ValueError('HOLD: value fit changed greedy decisions')
    for name, value in result.state_dict().items():
        expected = scaled_state[name]
        if name == 'student.net.2.weight':
            value, expected = value[:160], expected[:160]
        elif name == 'student.net.2.bias':
            value, expected = value[:160], expected[:160]
        if not torch.equal(value, expected):
            raise ValueError('Unexpected parameter change during value fit: '+name)
    if tensor_hash(model) != original_hash:
        raise ValueError('Source model mutated')
    residuals = {}
    for name, sl in [('fitting', slice(0, 320)), ('withheld_returns', slice(320, 400))]:
        residuals[name] = {}
        for stage, data in [('after_bc', before), ('after_scale', after_scale), ('after_value_fit', after_fit)]:
            error = np.asarray(data['demonstration_q'])[sl]-returns[sl]
            residuals[name][stage] = dict(error=distribution(error), rmse=float(np.sqrt(np.mean(error**2))))
    return Handoff(result, alpha, dict(
        scope='Teacher-continuation return fit; not student/optimal policy evaluation',
        fitting_episodes=list(range(8)), withheld_return_episodes=[8, 9],
        withheld_action_labels_used_by_BC_and_retention=True,
        reward_scale=sr, before_range_max=rmax, alpha=alpha, alpha_fp32=float(np.float32(alpha)),
        rho=RIDGE_RHO, ridge=ridge, residuals=residuals,
        stages=dict(after_bc=before, after_scale=after_scale, after_value_fit=after_fit),
        source_unchanged=True, new_optimizer_updates=0))


def handoff(model, episodes=None):
    if model.definition.calibration:
        if episodes is None:
            raise ValueError('Calibration requires demonstration episodes')
        return calibrate(model, episodes)
    if not model.definition.bc and episodes is not None:
        raise ValueError('Plain DDQN control cannot receive demonstration data')
    return Handoff(copy.deepcopy(model), 1., dict(calibration_applied=False, source_hash=tensor_hash(model)))


def td_setup(model):
    if not model.definition.ddqn:
        raise ValueError('BC-only arm has no TD optimizer or target')
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError('All online parameters must be trainable')
    target = copy.deepcopy(model).eval().requires_grad_(False)
    for p in target.parameters():
        p.grad = None
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001, betas=(.9, .999),
                                 eps=1e-8, weight_decay=0)
    return target, optimizer


def draw_indices(model, replay_size, replay_rng, retention_rng=None, demo_size=None, batch=64):
    if not model.definition.ddqn:
        raise ValueError('BC-only cannot draw TD replay')
    if replay_size <= 0 or batch <= 0:
        raise ValueError('Positive replay and batch sizes required')
    if model.definition.retention:
        if retention_rng is None or retention_rng is replay_rng or demo_size is None or demo_size <= 0:
            raise ValueError('Distinct retention RNG and nonempty demonstration data required')
    elif retention_rng is not None or demo_size is not None:
        raise ValueError('No-retention arm must not consume auxiliary draws')
    indices = replay_rng.integers(0, replay_size, size=batch)
    keep = retention_rng.integers(0, demo_size, size=batch) if model.definition.retention else None
    return indices, keep


def _validate_td(model, target, optimizer, batch, alpha, demonstration):
    if not model.definition.ddqn:
        raise ValueError('BC-only arm cannot update')
    if target.handoff_arm != model.handoff_arm or model is target:
        raise ValueError('Distinct same-arm target required')
    if (any(p.requires_grad or p.grad is not None for p in target.parameters()) or
            not all(p.requires_grad for p in model.parameters())):
        raise ValueError('Detached target and trainable online parameters required')
    online_storage = {p.untyped_storage().data_ptr() for p in model.parameters()}
    if any(p.untyped_storage().data_ptr() in online_storage for p in target.parameters()):
        raise ValueError('Target aliases online parameter storage')
    groups = optimizer.param_groups
    if (type(optimizer) is not torch.optim.Adam or len(groups) != 1 or
            len(groups[0]['params']) != len(list(model.parameters())) or
            {id(p) for p in groups[0]['params']} != {id(p) for p in model.parameters()} or
            any(groups[0][k] != v for k, v in dict(lr=.0001, betas=(.9, .999), eps=1e-8, weight_decay=0).items())):
        raise ValueError('Optimizer must match this online model and declared Adam settings')
    if not math.isfinite(alpha) or not 0 < alpha <= 1 or (not model.definition.calibration and alpha != 1):
        raise ValueError('Invalid fixed handoff temperature')
    s, a, r, ns, done = batch
    baseline.validate_states(s, nonterminal=True)
    labels_valid(s, a)
    baseline.validate_states(ns)
    if len(ns) != len(s):
        raise ValueError('Paired state counts required')
    if (r.device.type != 'cpu' or r.dtype != torch.float32 or r.shape != (len(s),) or
            not torch.isfinite(r).all() or done.device.type != 'cpu' or
            done.dtype != torch.bool or done.shape != r.shape):
        raise ValueError('Finite FP32 rewards and bool terminal vector required')
    if not torch.equal(done, torch.round(ns[:, -1]*40) == 40):
        raise ValueError('Terminal flags must match inherited horizon encoding')
    if model.definition.retention:
        if demonstration is None:
            raise ValueError('Retention demonstration batch required')
        labels_valid(*demonstration)
    elif demonstration is not None:
        raise ValueError('No-retention arm must not receive auxiliary batch')


def ddqn_retention_step(model, target, optimizer, batch, alpha=1., demonstration=None):
    """One detached-target, joint-loss Adam update; caller controls target-copy schedule."""
    _validate_td(model, target, optimizer, batch, alpha, demonstration)
    s, a, r, ns, done = batch
    prediction = model(s, a)
    next_actions = torch.full((len(s), 32), -1, dtype=torch.long)
    tq = torch.zeros(len(s))
    current_greedy = torch.empty(len(s))
    with torch.no_grad():
        current_greedy.copy_(model.greedy(s)[1])
        if (~done).any():
            selected = model.greedy(ns[~done])[0]
            next_actions[~done] = selected
            tq[~done] = target(ns[~done], selected)
        y = r+tq
    td = F.smooth_l1_loss(prediction, y)
    keep = imitation_loss(model, *demonstration, alpha) if model.definition.retention else td.new_zeros(())
    loss = td+RETENTION_LAMBDA*keep
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
    if (not torch.isfinite(loss) or any(not torch.isfinite(g).all() for _, g in grads) or
            any(p.grad is not None for p in target.parameters())):
        raise ValueError('HOLD: nonfinite loss/gradient or target gradient')
    squared = lambda vals: math.sqrt(sum(float(torch.sum(g.double()**2)) for g in vals))
    full_norm = squared(g for _, g in grads)
    branch_grads = [g[:160] for name, g in grads if name.startswith('student.net.2.')]
    value_grads = [g[160:] for name, g in grads if name.startswith('student.net.2.')]
    diagnostics = dict(total=full_norm, branch_output=squared(branch_grads),
                       value_output=squared(value_grads),
                       shared_hidden=squared(g for name, g in grads if name.startswith('student.net.0.')))
    clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
    if not torch.isfinite(clipped):
        raise ValueError('HOLD: nonfinite global clip norm')
    # Diagnostics refer to pre-update tensors. No optimizer replay or new rollout.
    report = dict(td_loss=float(td.detach()), retention_loss=float(keep.detach()), total_loss=float(loss.detach()),
                  prediction=prediction.detach().tolist(), rewards=r.tolist(), target_q=tq.tolist(),
                  target=y.tolist(), dones=done.tolist(), next_actions=next_actions.tolist(),
                  current_greedy_q=current_greedy.tolist(), gradient_norms=diagnostics,
                  clip_norm=float(clipped), clipped=bool(clipped > 10), alpha=alpha,
                  optimizer_steps=1, target_copy_performed=False)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('HOLD: nonfinite updated parameters')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Isolated mechanics only; research execution disabled')
    parser.add_argument('--mode', choices=['development', 'final'], required=True)
    parser.add_argument('--confirmation')
    parser.add_argument('--output')
    parser.parse_args()
    raise SystemExit('Research modes disabled: handoff mechanics only; no research runner/release')
