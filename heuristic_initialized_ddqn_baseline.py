"""Training-only heuristic-guided compact DDQN baseline.

The heuristic is used to create optional demonstration labels during training.
It is never called by ``deploy_actions``. This module defines an engineering
baseline, not a new research protocol or launcher.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

import controller_v3_amortized as student_impl


def allowed(states):
    """Standalone equivalent of the inherited outward-rounded local screen.

    Kept independent of historical research runners and compilation packages.
    The fixture compares it against the original screen functions directly.
    """
    x = states.detach().cpu().numpy() if isinstance(states, Tensor) else np.asarray(states)
    if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 533 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("finite batch of 533 FP32 observations required")
    tasks = x[:, :240].reshape(-1, 30, 8)
    cycles, deadline = tasks[:, :, 6], tasks[:, :, 7]
    if np.any(cycles < 0) or np.any(cycles > 1) or np.any(deadline < 0) or np.any(deadline > 1):
        raise ValueError("task features outside normalized bounds")
    c_lo = np.nextafter(np.maximum(0., np.nextafter(cycles, np.float32(-np.inf)).astype(np.float64)) * 500e6, -np.inf)
    d_hi = np.nextafter(np.minimum(1., np.nextafter(deadline, np.float32(np.inf)).astype(np.float64)) * .300, np.inf)
    mask = np.ones((len(x), 32, 5), dtype=bool)
    mask[:, :30, 0] = ~(np.nextafter(c_lo / 1.5e9, -np.inf) > np.nextafter(d_hi + 1e-10, np.inf))
    return mask


@dataclass(frozen=True)
class BaselineSpec:
    arm: str = "plain24_nonpositive"
    teacher_name: str = "observed_load_radio_heuristic"
    teacher_training_only: bool = True
    n_vehicles: int = 30
    hidden_width: int = 24
    gamma: float = 1.0
    screen_local_actions: bool = True

    def as_dict(self):
        return asdict(self)


class TeacherInitializedCompactDDQN(nn.Module):
    """Compact student with a training-only demonstration interface."""

    def __init__(self, spec: BaselineSpec = BaselineSpec()):
        super().__init__()
        if (spec.arm != "plain24_nonpositive" or spec.n_vehicles != 30 or spec.gamma != 1.0
                or spec.hidden_width != 24 or not spec.teacher_training_only or not spec.screen_local_actions):
            raise ValueError("baseline is intentionally limited to plain24_nonpositive")
        self.spec = spec
        self.student = student_impl.AmortizedPriceQ(spec.arm, spec.n_vehicles)

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        return self.student(states, actions)

    def tables(self, states: Tensor):
        return self.student.tables(states)

    def greedy(self, states: Tensor, screened: bool | None = None):
        validate_states(states, nonterminal=True)
        placement, motion, bias, details = self.tables(states)
        use_screen = self.spec.screen_local_actions if screened is None else screened
        if use_screen:
            mask = torch.from_numpy(allowed(states)).to(device=states.device)
            placement_scores = placement.masked_fill(~mask[:, : self.student.n], float("inf"))
        else:
            placement_scores = placement
        vehicle_actions = placement_scores.argmin(2)
        movement_actions = motion.argmin(2)
        actions = torch.cat((vehicle_actions, movement_actions), 1)
        cost = bias + placement.gather(2, vehicle_actions[:, :, None]).squeeze(2).sum(1)
        cost = cost + motion.gather(2, movement_actions[:, :, None]).squeeze(2).sum(1)
        return actions, -F.softplus(cost), details


def demonstration_actions(teacher, states: np.ndarray) -> Tensor:
    """Return teacher labels without putting the teacher in deployment."""
    x = np.asarray(states)
    if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 533:
        raise ValueError("teacher states must be a [batch,533] FP32 array")
    validate_states(torch.from_numpy(x), nonterminal=True)
    labels = []
    for state in x:
        actions, _ = teacher.values(state)
        a = np.asarray(actions)
        if a.shape != (1, 32) or a.dtype != np.int64 or np.any(a < 0) or np.any(a >= 5):
            raise ValueError("invalid teacher action")
        labels.append(a[0].copy())
    labels = torch.from_numpy(np.stack(labels, axis=0))
    if not np.take_along_axis(allowed(x), labels.numpy()[:, :, None], 2).all():
        raise ValueError('teacher action excluded by the inherited screen')
    return labels


def imitation_loss(model: TeacherInitializedCompactDDQN, states: Tensor,
                   teacher_actions: Tensor, temperature: float = 1.0) -> Tensor:
    """Masked branchwise behavior loss used only for a training baseline."""
    validate_states(states, nonterminal=True)
    if not np.isfinite(temperature) or temperature <= 0 or teacher_actions.shape != (len(states), model.student.n + 2):
        raise ValueError("invalid teacher labels or temperature")
    if teacher_actions.dtype != torch.long or torch.any(teacher_actions < 0) or torch.any(teacher_actions >= 5):
        raise ValueError("teacher labels must be int64")
    placement, motion, _, _ = model.tables(states)
    mask = torch.from_numpy(allowed(states)).to(device=states.device)
    if not torch.take_along_dim(mask[:, : model.student.n], teacher_actions[:, : model.student.n, None], dim=2).all():
        raise ValueError("teacher label selects a screened-out local action")
    logits = (-placement / temperature).masked_fill(~mask[:, : model.student.n], -torch.inf)
    loss_vehicle = F.cross_entropy(logits.reshape(-1, 5), teacher_actions[:, : model.student.n].reshape(-1))
    loss_motion = F.cross_entropy((-motion / temperature).reshape(-1, 5), teacher_actions[:, model.student.n:].reshape(-1))
    return loss_vehicle + loss_motion


def validate_states(states, nonterminal=False):
    if not isinstance(states, Tensor) or states.device.type != 'cpu' or states.dtype != torch.float32:
        raise ValueError('CPU FP32 tensor required')
    allowed(states)
    t = states[:, -1] * 40
    if torch.any(t < 0) or torch.any(t > 40) or torch.any(torch.abs(t-t.round()) > 1e-4):
        raise ValueError('invalid horizon feature')
    if nonterminal and torch.any(t.round() == 40):
        raise ValueError('terminal decoding forbidden')


def ddqn_loss(model: TeacherInitializedCompactDDQN, target: TeacherInitializedCompactDDQN,
              states: Tensor, actions: Tensor, rewards: Tensor,
              next_states: Tensor, dones: Tensor) -> Tensor:
    """One detached-target DDQN loss after optional demonstration pretraining."""
    if dones.dtype not in (torch.bool, torch.uint8) or not torch.all((dones == 0) | (dones == 1)):
        raise ValueError("dones must be binary")
    validate_states(states, nonterminal=True)
    validate_states(next_states)
    if rewards.shape != (len(states),) or dones.shape != rewards.shape or not torch.isfinite(rewards).all():
        raise ValueError('invalid rewards/dones')
    if not torch.equal(next_states[:, -1].mul(40).round().eq(40), dones.bool()):
        raise ValueError('done/horizon disagreement')
    prediction = model(states, actions)
    y = rewards.detach().clone()
    active = ~dones.bool()
    if active.any():
        with torch.no_grad():
            next_actions = model.greedy(next_states[active])[0]
            y[active] += target(next_states[active], next_actions)
    return F.smooth_l1_loss(prediction, y)


@torch.no_grad()
def deploy_actions(model: TeacherInitializedCompactDDQN, state: np.ndarray) -> np.ndarray:
    """Student-only deployment path; deliberately does not call the teacher."""
    x = np.asarray(state)
    if x.dtype != np.float32 or x.shape != (533,):
        raise ValueError("deployment state must be one [533] FP32 observation")
    return model.greedy(torch.from_numpy(x)[None])[0].cpu().numpy()[0]


def parameter_bytes(model: TeacherInitializedCompactDDQN) -> int:
    return sum(p.numel() * p.element_size() for p in model.student.parameters())


def spec_record(model: TeacherInitializedCompactDDQN) -> dict:
    return {"spec": model.spec.as_dict(), "parameter_bytes": parameter_bytes(model),
            "teacher_called_by_deploy": False, "new_simulator_steps": 0,
            "new_optimizer_updates": 0}
