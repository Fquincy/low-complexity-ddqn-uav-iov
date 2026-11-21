"""Mechanics-only sequential conditional prototype; no trainer or automatic runs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from chapter3_iov_env_v2 import Chapter3Config, Chapter3IoVEnv

PHYSICAL_SIZE = 533
COMPONENTS = 32
CHOICES = 5
UNKNOWN = 5
CONTEXT_SIZE = 224
AUGMENTED_SIZE = 757
ORDER = np.array([30, 31]+list(range(30)), dtype=np.int64)
ORDER.setflags(write=False)


def _physical(value):
    a = np.asarray(value, dtype=np.float32)
    if a.shape != (PHYSICAL_SIZE,) or not np.isfinite(a).all():
        raise ValueError("finite533-value physical observation required")
    return a.copy()


def _vector(value, unknown=False):
    a = np.asarray(value)
    limit = UNKNOWN if unknown else CHOICES-1
    if a.shape != (COMPONENTS,) or not np.isfinite(a).all() or not np.all(a == np.floor(a)) or np.any((a < 0) | (a > limit)):
        raise ValueError("invalid32-component action/prefix vector")
    return a.astype(np.int64, copy=True)


def make_context(stage, prefix):
    if not isinstance(stage, (int, np.integer)) or not 0 <= stage < COMPONENTS:
        raise ValueError("stage must be0..31")
    p = _vector(prefix, unknown=True)
    known = np.zeros(COMPONENTS, dtype=bool)
    known[ORDER[:stage]] = True
    if np.any(p[known] == UNKNOWN) or np.any(p[~known] != UNKNOWN):
        raise ValueError("prefix must contain only strictly preceding choices")
    stage_hot = np.zeros(COMPONENTS, dtype=np.float32)
    stage_hot[stage] = 1.
    return np.concatenate((stage_hot, np.eye(6, dtype=np.float32)[p].ravel()))


def augmented(physical, stage, prefix):
    return np.concatenate((_physical(physical), make_context(stage, prefix)))


def active_parts(observation):
    a = np.asarray(observation, dtype=np.float32)
    if a.shape != (AUGMENTED_SIZE,) or not np.isfinite(a).all():
        raise ValueError("finite757-value augmented observation required")
    stage_hot = a[PHYSICAL_SIZE:PHYSICAL_SIZE+COMPONENTS]
    prefix_hot = a[PHYSICAL_SIZE+COMPONENTS:].reshape(COMPONENTS, 6)
    if not np.all((stage_hot == 0) | (stage_hot == 1)) or stage_hot.sum() != 1:
        raise ValueError("active stage required; absorbing observations have no action")
    if not np.all((prefix_hot == 0) | (prefix_hot == 1)) or not np.all(prefix_hot.sum(1) == 1):
        raise ValueError("invalid prefix one-hot encoding")
    stage = int(stage_hot.argmax())
    prefix = prefix_hot.argmax(1)
    expected = make_context(stage, prefix)
    if not np.array_equal(expected, a[PHYSICAL_SIZE:]):
        raise ValueError("invalid decision context")
    return a[:PHYSICAL_SIZE].copy(), a[PHYSICAL_SIZE:].copy(), stage


@dataclass(frozen=True)
class PhysicalRecord:
    observation: np.ndarray
    actions: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool
    slot: int = -1
    episode: int = -1

    def __post_init__(self):
        for name, a in (("observation", _physical(self.observation)), ("actions", _vector(self.actions)), ("next_observation", _physical(self.next_observation))):
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        if not np.isfinite(self.reward):
            raise ValueError("finite physical reward required")
        if not isinstance(self.done, (bool, np.bool_)):
            raise ValueError("boolean physical terminal flag required")
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "done", bool(self.done))


@dataclass(frozen=True)
class StageTransition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool


def reconstruct(record, stage):
    if not isinstance(stage, (int, np.integer)) or not 0 <= stage < COMPONENTS:
        raise ValueError("stage must be0..31")
    prefix = np.full(COMPONENTS, UNKNOWN, dtype=np.int64)
    prefix[ORDER[:stage]] = record.actions[ORDER[:stage]]
    state = augmented(record.observation, stage, prefix)
    choice = int(record.actions[ORDER[stage]])
    if stage < COMPONENTS-1:
        prefix[ORDER[stage]] = choice
        successor = augmented(record.observation, stage+1, prefix)
        return StageTransition(state, choice, 0., successor, False)
    successor = np.zeros(AUGMENTED_SIZE, dtype=np.float32) if record.done else augmented(record.next_observation, 0, np.full(COMPONENTS, UNKNOWN))
    return StageTransition(state, choice, record.reward, successor, record.done)


class SequentialIoVWrapper:
    """Advance physical time only at the32nd choice. Physical env stays authoritative."""
    def __init__(self, config=None, seed=None):
        self.env = Chapter3IoVEnv(config or Chapter3Config(), seed=seed)
        if self.env.num_vehicles != 30 or self.env.state_size != PHYSICAL_SIZE:
            raise ValueError("prototype supports the30-vehicle533-observation model")
        self.episode = 0
        self.last_physical_record = None
        self.reset(seed)

    @property
    def cache_token(self):
        return (id(self), self.episode, self.env.step_index)

    @property
    def done(self):
        return self._done

    def reset(self, seed=None):
        self.physical_observation = _physical(self.env.reset(seed=seed))
        self.prefix = np.full(COMPONENTS, UNKNOWN, dtype=np.int64)
        self.stage = 0
        self._done = False
        self.episode += 1
        self.last_physical_record = None
        return self.observation()

    def observation(self):
        if self._done:
            return np.zeros(AUGMENTED_SIZE, dtype=np.float32)
        return augmented(self.physical_observation, self.stage, self.prefix)

    def step(self, choice):
        if self._done:
            raise RuntimeError("episode ended; reset before requesting another choice")
        if isinstance(choice, (bool, np.bool_)) or not isinstance(choice, (int, np.integer)) or not 0 <= choice < CHOICES:
            raise ValueError("integer choice0..4 required")
        stage, slot = self.stage, self.env.step_index
        component = int(ORDER[stage])
        self.prefix[component] = int(choice)
        info = {"physical_advanced": False, "physical_slot": slot, "stage": stage, "component": component}
        if stage < COMPONENTS-1:
            self.stage += 1
            return self.observation(), 0., False, info
        vector = self.prefix.copy()
        next_physical, reward, done, physical_info = self.env.step({"destinations": vector[:30], "uav_movement": vector[30:]})
        record = PhysicalRecord(self.physical_observation, vector, reward, next_physical, done, slot, self.episode)
        self.last_physical_record = record
        self.physical_observation = _physical(next_physical)
        self.prefix.fill(UNKNOWN)
        self.stage = 0
        self._done = bool(done)
        info.update(physical_advanced=True, requested_vector=vector.copy(), physical_record=record, physical_info=physical_info)
        return self.observation(), float(reward), bool(done), info


class PhysicalReplayBuffer:
    """Ring capacity is PHYSICAL records; reconstruct prefixes without future choices."""
    def __init__(self, capacity):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("positive physical-record capacity required")
        self.capacity, self.size, self.position = capacity, 0, 0
        self.observations = np.empty((capacity, PHYSICAL_SIZE), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty((capacity, COMPONENTS), dtype=np.int8)
        self.rewards = np.empty(capacity, dtype=np.float64)
        self.dones = np.empty(capacity, dtype=bool)
        self.slots = np.empty(capacity, dtype=np.int64)
        self.episodes = np.empty(capacity, dtype=np.int64)

    def add(self, record):
        i = self.position
        self.observations[i], self.actions[i] = record.observation, record.actions
        self.next_observations[i] = record.next_observation
        self.rewards[i], self.dones[i] = record.reward, record.done
        self.slots[i], self.episodes[i] = record.slot, record.episode
        self.position = (i+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def record(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < self.size:
            raise IndexError("physical replay index outside stored records")
        return PhysicalRecord(self.observations[index], self.actions[index], self.rewards[index], self.next_observations[index], self.dones[index], int(self.slots[index]), int(self.episodes[index]))

    def sample(self, batch_size, rng):
        if self.size == 0 or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("nonempty replay and positive batch size required")
        indices = rng.integers(0, self.size, size=batch_size)
        stages = rng.integers(0, COMPONENTS, size=batch_size)
        items = [reconstruct(self.record(int(i)), int(k)) for i, k in zip(indices, stages)]
        return {"states": np.stack([s.observation for s in items]), "actions": np.array([s.action for s in items], dtype=np.int64), "rewards": np.array([s.reward for s in items], dtype=np.float32), "next_states": np.stack([s.next_observation for s in items]), "dones": np.array([s.done for s in items], dtype=np.float32), "physical_indices": indices, "stages": stages}

    @property
    def storage_bytes(self):
        return sum(a.nbytes for a in (self.observations, self.next_observations, self.actions, self.rewards, self.dones, self.slots, self.episodes))


class ConditionalQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(PHYSICAL_SIZE, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128+CONTEXT_SIZE, 32), nn.ReLU(), nn.Linear(32, CHOICES))
        self._explicit_revision = 0

    def encode(self, physical):
        return self.encoder(physical)

    def conditional(self, embedding, context):
        return self.head(torch.cat((embedding, context), dim=1))

    def forward(self, observations):
        if observations.ndim != 2 or observations.shape[1] != AUGMENTED_SIZE:
            raise ValueError("network expects batch x757 observations")
        return self.conditional(self.encode(observations[:, :PHYSICAL_SIZE]), observations[:, PHYSICAL_SIZE:])

    def notify_weights_changed(self):
        """Required for edits that bypass PyTorch mutation counters, e.g. .data writes."""
        self._explicit_revision += 1

    def weight_signature(self):
        # Native optimizer/in-place operations and load_state_dict bump _version.
        return (self._explicit_revision, tuple((id(p), p._version, str(p.device), p.dtype) for p in self.parameters()))


class InferenceCache:
    """Inference only. Never use cached embeddings in a gradient-enabled loss."""
    def __init__(self, model):
        self.model = model
        self.encoder_calls, self.head_calls = 0, 0
        self.invalidate()

    def invalidate(self):
        self._embedding = None
        self._signature = None
        self._root = None
        self._token = None

    def values(self, observation, slot_token):
        physical, context, _ = active_parts(observation)
        signature = self.model.weight_signature()
        p = next(self.model.parameters())
        with torch.inference_mode():
            if self._embedding is None or signature != self._signature or slot_token != self._token or not np.array_equal(physical, self._root):
                self._embedding = self.model.encode(torch.as_tensor(physical, dtype=p.dtype, device=p.device).unsqueeze(0))
                self._root, self._token, self._signature = physical.copy(), slot_token, signature
                self.encoder_calls += 1
            result = self.model.conditional(self._embedding, torch.as_tensor(context, dtype=p.dtype, device=p.device).unsqueeze(0))
            self.head_calls += 1
            return result[0].cpu().numpy().copy()


class SequentialActor:
    """One exploration coin per physical slot; no optimizer or learner state."""
    def __init__(self, model, seed=0):
        self.model, self.rng = model, np.random.default_rng(seed)
        self.cache = InferenceCache(model)
        self.physical_slots_started = 0
        self.exploration_slots = 0
        self.staged_choices = 0
        self.history = []
        self._token = None

    def act(self, wrapper, epsilon=0., warmup=False, explore=False):
        if wrapper.done:
            raise RuntimeError("absorbing observation has no next action")
        if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be0..1")
        token = wrapper.cache_token
        if token != self._token:
            if wrapper.stage != 0:
                raise RuntimeError("actor must join at the first stage of a slot")
            self.cache.invalidate()
            self._token, self._expected_stage = token, 0
            self._signature = self.model.weight_signature()
            self._settings = (float(epsilon), bool(warmup), bool(explore))
            random_branch = bool(explore and (warmup or self.rng.random() < epsilon))
            self._random_vector = self.rng.integers(0, CHOICES, COMPONENTS, dtype=np.int64) if random_branch else None
            self.history.append({"physical_slot": wrapper.env.step_index, "exploration_slot_counter": self.exploration_slots, "epsilon": float(epsilon), "warmup": bool(warmup), "explore": bool(explore), "random_branch": random_branch})
            self.physical_slots_started += 1
            self.exploration_slots += int(bool(explore))
        if wrapper.stage != self._expected_stage:
            raise RuntimeError("each actor choice must be followed by its wrapper step")
        if self._settings != (float(epsilon), bool(warmup), bool(explore)):
            raise RuntimeError("exploration settings must stay fixed during a slot")
        if self.model.weight_signature() != self._signature:
            raise RuntimeError("actor weights changed during a physical slot")
        if self._random_vector is not None:
            result = int(self._random_vector[ORDER[wrapper.stage]])
        else:
            result = int(self.cache.values(wrapper.observation(), token).argmax())
        self._expected_stage += 1
        self.staged_choices += 1
        if self._expected_stage == COMPONENTS:
            self.cache.invalidate()
        return result


@torch.no_grad()
def ddqn_targets(online, target, rewards, next_states, dones):
    """Gamma1 detached targets; never query an absorbing state's action values."""
    if rewards.ndim != 1 or dones.shape != rewards.shape or next_states.shape != (len(rewards), AUGMENTED_SIZE):
        raise ValueError("misaligned staged target batch")
    if not torch.all((dones == 0) | (dones == 1)):
        raise ValueError("binary terminal masks required")
    result = rewards.detach().clone()
    active = ~dones.bool()
    if active.any():
        following = next_states[active]
        selected = online(following).argmax(1, keepdim=True)
        value = target(following).gather(1, selected).squeeze(1)
        result[active] += value
    return result
