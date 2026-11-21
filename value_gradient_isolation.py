"""Isolated training mechanics. No research runner, simulator or teacher imports."""
import argparse
import math

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

import heuristic_initialized_ddqn_baseline as baseline
import return_calibrated_handoff as inherited

PRIMARY = 'value_gradient_isolated_ddqn'
RESCALE = 'dueling_shared_td_rescale_ddqn'
BASE = 'calibrated_retained_ddqn'
ARMS = {PRIMARY: inherited.ARMS[BASE], RESCALE: inherited.ARMS[BASE], **inherited.ARMS}
TD_RULES = {PRIMARY: 'isolate_value', RESCALE: 'rescale_shared_td'}
DEVELOPMENT_RELEASED = False
FINAL_RELEASED = False


class _TDAffine(torch.autograd.Function):
    """Original affine forward; explicitly modified FIRST-order feature gradient.

    Readout derivatives are standard. Higher-order differentiation is not
    supported. This is a training rule, not a numerical-gradient-consistent
    replacement for an ordinary affine layer.
    """
    @staticmethod
    def forward(ctx, hidden, weight, bias, rule):
        if rule not in {'isolate_value', 'rescale_shared_td'}:
            raise ValueError('Unknown TD feature-gradient rule')
        if (hidden.ndim != 2 or weight.ndim != 2 or weight.shape[0] < 2 or
                hidden.shape[0] == 0 or hidden.shape[1] != weight.shape[1] or
                bias.shape != (weight.shape[0],) or
                hidden.dtype not in {torch.float32, torch.float64} or
                any(t.device.type != 'cpu' or t.dtype != hidden.dtype for t in [hidden, weight, bias]) or
                any(not torch.isfinite(t).all() for t in [hidden, weight, bias])):
            raise ValueError('Finite, matching CPU FP32/FP64 affine tensors required')
        ctx.save_for_backward(hidden, weight)
        ctx.rule = rule
        return F.linear(hidden, weight, bias)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        hidden, weight = ctx.saved_tensors
        dh = dw = db = None
        if ctx.needs_input_grad[0]:
            if ctx.rule == 'isolate_value':
                dh = upstream[:, :-1].matmul(weight[:-1])
            else:
                dh = upstream.matmul(weight)*(1./math.sqrt(2.))
        if ctx.needs_input_grad[1]: dw = upstream.t().matmul(hidden)
        if ctx.needs_input_grad[2]: db = upstream.sum(0)
        return dh, dw, db, None


def td_affine(hidden, weight, bias, rule):
    return _TDAffine.apply(hidden, weight, bias, rule)


class Critic(inherited.Critic):
    """Unchanged deployment/BC/calibration forward; explicit routed TD method."""
    def __init__(self, arm=PRIMARY):
        if arm not in ARMS: raise ValueError('Unknown isolation comparison arm')
        super().__init__(BASE if arm in TD_RULES else arm)
        self.handoff_arm = arm

    @property
    def definition(self): return ARMS[self.handoff_arm]

    def td_prediction(self, states, actions):
        if self.handoff_arm not in TD_RULES: return self(states, actions)
        baseline.validate_states(states)
        if (actions.dtype != torch.long or actions.shape != (len(states), 32) or
                actions.device.type != 'cpu' or torch.any(actions < 0) or torch.any(actions > 4)):
            raise ValueError('CPU int64 joint actions required')
        hidden = self.student.net[1](self.student.net[0](states))
        final = self.student.net[2]
        raw = td_affine(hidden, final.weight, final.bias, TD_RULES[self.handoff_arm])
        # Preserve the original full GEMM, view/division and Q summation order.
        p = raw[:, :150].reshape(-1, 30, 5)/30
        m = raw[:, 150:160].reshape(-1, 2, 5)/2
        q = self.value(p, m, raw[:, 160], actions)
        if not torch.isfinite(q).all(): raise ValueError('Nonfinite TD prediction')
        return q


def td_setup(model):
    if not isinstance(model, Critic): raise ValueError('Isolation comparison critic required')
    return inherited.td_setup(model)


def _validate_step(model, target, optimizer, batch, alpha, demonstration):
    if not isinstance(model, Critic) or not isinstance(target, Critic):
        raise ValueError('Isolation comparison critics required')
    inherited._validate_td(model, target, optimizer, batch, alpha, demonstration)
    online_storage = {v.untyped_storage().data_ptr() for v in model.state_dict().values()}
    if any(v.untyped_storage().data_ptr() in online_storage for v in target.state_dict().values()):
        raise ValueError('Target parameter or buffer aliases online storage')


def ddqn_retention_step(model, target, optimizer, batch, alpha=1., demonstration=None):
    """One joint backward/Adam step. Routing applies ONLY to current TD values."""
    _validate_step(model, target, optimizer, batch, alpha, demonstration)
    s, a, r, ns, done = batch
    prediction = model.td_prediction(s, a)
    next_actions = torch.full((len(s), 32), -1, dtype=torch.long)
    target_q = torch.zeros(len(s))
    with torch.no_grad():
        current_greedy = model.greedy(s)[1]
        if (~done).any():
            selected = model.greedy(ns[~done])[0]
            next_actions[~done] = selected
            target_q[~done] = target(ns[~done], selected)
        y = r+target_q
    td = F.smooth_l1_loss(prediction, y)
    keep = inherited.imitation_loss(model, *demonstration, alpha) if model.definition.retention else td.new_zeros(())
    loss = td+inherited.RETENTION_LAMBDA*keep
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
    if (not torch.isfinite(loss) or any(not torch.isfinite(g).all() for _, g in grads) or
            any(p.grad is not None for p in target.parameters())):
        raise ValueError('HOLD: nonfinite loss/gradient or target gradient')
    def norm(values): return math.sqrt(sum(float(torch.sum(g.double()**2)) for g in values))
    diagnostics = dict(total=norm(g for _, g in grads),
        shared_hidden=norm(g for n, g in grads if n.startswith('student.net.0.')),
        branch_output=norm(g[:160] for n, g in grads if n.startswith('student.net.2.')),
        value_output=norm(g[160:] for n, g in grads if n.startswith('student.net.2.')))
    clip = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
    if not torch.isfinite(clip): raise ValueError('HOLD: nonfinite clip norm')
    report = dict(td_loss=float(td.detach()), retention_loss=float(keep.detach()), total_loss=float(loss.detach()),
        prediction=prediction.detach().tolist(), rewards=r.tolist(), target_q=target_q.tolist(), target=y.tolist(),
        dones=done.tolist(), next_actions=next_actions.tolist(), current_greedy_q=current_greedy.tolist(),
        gradient_norms=diagnostics, clip_norm=float(clip), clipped=bool(clip > 10), alpha=alpha,
        gradient_rule=TD_RULES.get(model.handoff_arm, 'ordinary'), optimizer_steps=1,
        target_copy_performed=False)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('HOLD: nonfinite updated parameters')
    return report


def deployment_model(model):
    """Independent, read-only ORIGINAL critic; no training rule in export class."""
    if not isinstance(model, Critic): raise ValueError('Isolation comparison critic required')
    base_arm = BASE if model.handoff_arm in TD_RULES else model.handoff_arm
    with torch.random.fork_rng(devices=[]): exported = inherited.Critic(base_arm)
    exported.load_state_dict(model.state_dict(), strict=True)
    return exported.eval().requires_grad_(False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Synthetic mechanics only; research modes disabled')
    p.add_argument('--mode', choices=['development', 'final'], required=True)
    p.add_argument('--confirmation'); p.add_argument('--output')
    p.parse_args()
    raise SystemExit('Research modes disabled: isolated gradient mechanics only; no research runner or release')
