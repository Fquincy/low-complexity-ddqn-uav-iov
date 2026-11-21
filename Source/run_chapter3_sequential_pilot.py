"""Separate frozen-protocol runner. Explicit --mode smoke or development required.

Smoke is integration evidence only. No checkpoint selection, retries or resume CLI.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from chapter3_iov_env_v2 import Chapter3IoVEnv, Chapter3Config
from chapter3_sequential_conditional import (
    ORDER, UNKNOWN, PhysicalRecord, PhysicalReplayBuffer, ConditionalQNetwork,
    SequentialActor, SequentialIoVWrapper, reconstruct, ddqn_targets,
)
from train_chapter3_ddqn import FactorizedQNetwork, to_action_dict
from validate_chapter3_model import select_destinations
from benchmark_chapter3_sequential_selection import (
    SelectionFrame, SequentialSelection, AdditiveSelection, cpu_name,
)

ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "CHAPTER3_SEQUENTIAL_PILOT_SPEC_2026-11-01.json"
SPEC_HASH = "e21a6efe784cb1e8656f7c3871e4c8b27e13cf93a04893238a41ba866e040484"
ARMS = ("additive_ddqn_128_128", "conditional_cached_ddqn_128_128_head32")
REPLAY_ARRAYS = ("observations", "next_observations", "actions", "rewards", "dones", "slots", "episodes")
SOURCE_FILES = (
    "run_chapter3_sequential_pilot.py", "test_chapter3_sequential_pilot.py",
    "chapter3_sequential_conditional.py", "chapter3_iov_env.py", "chapter3_iov_env_v2.py",
    "train_chapter3_ddqn.py", "validate_chapter3_model.py",
    "benchmark_chapter3_sequential_selection.py", SPEC_PATH.name,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2, allow_nan=False), encoding="utf-8")


def write_gzip(path, value):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(jsonable(value), f, allow_nan=False)


def digest(value):
    """Canonical recursive digest; includes tensor bytes, never pickle addresses."""
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        if isinstance(v, np.ndarray):
            h.update(str((v.dtype.str, v.shape)).encode())
            h.update(v.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                visit(k)
                visit(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(str(len(v)).encode())
            for item in v:
                visit(item)
        else:
            h.update(repr(v).encode())
        h.update(b"\0")
    visit(value)
    return h.hexdigest()


def settings(mode):
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if mode not in ("smoke", "development"):
        raise ValueError("explicit smoke/development mode required")
    c = copy.deepcopy(spec["training"])
    c["mode"] = mode
    if mode == "smoke":
        s = spec["smoke"]
        for key in ("episodes", "warmup_physical_records", "batch_size", "replay_capacity_physical_records", "target_copy_every_optimizer_steps", "epsilon_decay_physical_slots"):
            c[key] = s[key]
        c.update(replicates=s["replicate_indices"], train_starts=[s["training_task_seed_start"]],
                 validation_seeds=s["validation_task_seeds"], probe_seeds=s["fixed_probe_task_seeds"],
                 fresh_seeds=list(range(s["fresh_task_seed_start"], s["fresh_task_seed_start"]+s["fresh_episodes"])),
                 diagnostic_episodes=s["diagnostic_episodes"])
    else:
        c.update(replicates=[0, 1, 2], train_starts=[x["start"] for x in spec["training_task_blocks"]],
                 validation_seeds=spec["scheduled_validation_task_seeds"], probe_seeds=spec["fixed_probe"]["task_seeds"],
                 fresh_seeds=list(range(spec["fresh_final_gate"]["task_seed_start"], spec["fresh_final_gate"]["task_seed_start"]+spec["fresh_final_gate"]["episodes"])),
                 diagnostic_episodes=spec["diagnostic_episodes"])
    c["model_seeds"] = spec["replicate_model_seeds"]
    return c


class RawActor:
    def __init__(self, model, seed):
        self.model, self.rng = model, np.random.default_rng(seed)
        self.physical_slots_started = self.exploration_slots = self.staged_choices = 0
        self.history = []

    def act(self, state, epsilon=0., warmup=False, explore=False):
        random_branch = bool(explore and (warmup or self.rng.random() < epsilon))
        self.history.append(dict(epsilon=float(epsilon), warmup=bool(warmup), explore=bool(explore), random_branch=random_branch))
        self.physical_slots_started += 1
        self.exploration_slots += int(explore)
        if random_branch:
            return self.rng.integers(0, 5, 32, dtype=np.int64)
        with torch.inference_mode():
            return self.model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))[0].argmax(1).numpy().copy()


@torch.no_grad()
def raw_targets(online, target, rewards, next_states, dones):
    if not torch.all((dones == 0) | (dones == 1)):
        raise ValueError("binary terminal masks required")
    result = rewards.detach().clone()
    active = ~dones.bool()
    if active.any():
        states = next_states[active]
        selected = online(states).argmax(2, keepdim=True)
        result[active] += target(states).gather(2, selected).squeeze(2).sum(1)
    return result


class PilotAgent:
    def __init__(self, arm, seed, config):
        if arm not in ARMS:
            raise ValueError("unsupported architecture")
        self.arm, self.seed, self.config = arm, seed, copy.deepcopy(config)
        self.sequential = arm == ARMS[1]
        torch.manual_seed(seed)
        self.online = ConditionalQNetwork() if self.sequential else FactorizedQNetwork(533, 32, 5, (128, 128))
        self.target = copy.deepcopy(self.online).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config["learning_rate"])
        self.replay = PhysicalReplayBuffer(config["replay_capacity_physical_records"])
        self.replay_rng = np.random.default_rng(seed+200000)
        self.actor = SequentialActor(self.online, seed+100000) if self.sequential else RawActor(self.online, seed+100000)
        self.physical_slots = self.updates = self.target_copies = 0

    def epsilon(self):
        if self.physical_slots < self.config["warmup_physical_records"]:
            return 1.
        return 1.-.95*min(self.physical_slots/self.config["epsilon_decay_physical_slots"], 1.)

    def sample(self):
        n = self.config["batch_size"]
        if self.sequential:
            return self.replay.sample(n, self.replay_rng)
        i = self.replay_rng.integers(0, self.replay.size, n)
        return dict(states=self.replay.observations[i], actions=self.replay.actions[i].astype(np.int64),
                    rewards=self.replay.rewards[i].astype(np.float32), next_states=self.replay.next_observations[i],
                    dones=self.replay.dones[i].astype(np.float32), physical_indices=i, stages=np.full(n, -1))

    def observe(self, record):
        # The caller can only supply a validated complete PhysicalRecord.
        if not isinstance(record, PhysicalRecord):
            raise TypeError("only complete physical records may trigger learning")
        self.replay.add(record)
        self.physical_slots += 1
        if self.physical_slots < self.config["warmup_physical_records"]:
            return None
        batch = self.sample()
        states, following = (torch.as_tensor(batch[k], dtype=torch.float32) for k in ("states", "next_states"))
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
        rewards, dones = (torch.as_tensor(batch[k], dtype=torch.float32) for k in ("rewards", "dones"))
        if self.sequential:
            chosen = self.online(states).gather(1, actions[:, None]).squeeze(1)
            targets = ddqn_targets(self.online, self.target, rewards, following, dones)
        else:
            chosen = self.online(states).gather(2, actions[:, :, None]).squeeze(2).sum(1)
            targets = raw_targets(self.online, self.target, rewards, following, dones)
        if not torch.isfinite(chosen).all() or not torch.isfinite(targets).all():
            raise FloatingPointError("nonfinite Q or target")
        loss = F.smooth_l1_loss(chosen, targets, beta=1.)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10., error_if_nonfinite=True)
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.online.parameters()):
            raise FloatingPointError("nonfinite updated weights")
        self.updates += 1
        copied = self.updates % self.config["target_copy_every_optimizer_steps"] == 0
        if copied:
            self.target.load_state_dict(self.online.state_dict())
            self.target_copies += 1
        ids = batch["physical_indices"]
        return dict(optimizer_step=self.updates, physical_slots=self.physical_slots, loss=float(loss.detach()),
                    grad_norm_before_clip=float(grad), target_copy=copied, physical_indices=ids,
                    sampled_episode_ids=self.replay.episodes[ids], sampled_slots=self.replay.slots[ids],
                    stages=batch["stages"], physical_reward_samples=int(np.sum(batch["stages"] == 31)) if self.sequential else len(ids),
                    true_terminal_samples=int(np.sum(batch["dones"])),
                    target_mean=float(targets.mean()), chosen_mean=float(chosen.detach().mean()))

    def checkpoint(self):
        r = self.replay
        return dict(format="chapter3-paired-pilot-v1", arm=self.arm, seed=self.seed, config=self.config,
                    spec_sha256=sha(SPEC_PATH), source_sha256={n: sha(ROOT/n) for n in SOURCE_FILES},
                    online=self.online.state_dict(), target=self.target.state_dict(), optimizer=self.optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), action_rng=copy.deepcopy(self.actor.rng.bit_generator.state),
                    replay_rng=copy.deepcopy(self.replay_rng.bit_generator.state),
                    physical_slots=self.physical_slots, updates=self.updates, target_copies=self.target_copies,
                    actor={k: copy.deepcopy(getattr(self.actor, k)) for k in ("physical_slots_started", "exploration_slots", "staged_choices", "history")},
                    replay_size=r.size, replay_position=r.position,
                    replay={name: getattr(r, name)[:r.size].copy() for name in REPLAY_ARRAYS})

    def restore(self, ck):
        # Verification helper only. No cross-formulation or historical-model resume.
        if (ck.get("format"), ck.get("arm"), ck.get("seed"), ck.get("config"), ck.get("spec_sha256")) != (
            "chapter3-paired-pilot-v1", self.arm, self.seed, self.config, sha(SPEC_PATH)):
            raise ValueError("incompatible checkpoint architecture/configuration/protocol")
        if ck["source_sha256"] != {n: sha(ROOT/n) for n in SOURCE_FILES}:
            raise ValueError("checkpoint source mismatch")
        self.online.load_state_dict(ck["online"])
        self.target.load_state_dict(ck["target"])
        self.optimizer.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["torch_rng"])
        self.actor.rng.bit_generator.state = ck["action_rng"]
        self.replay_rng.bit_generator.state = ck["replay_rng"]
        for k in ("physical_slots", "updates", "target_copies"):
            setattr(self, k, ck[k])
        for k, v in ck["actor"].items():
            setattr(self.actor, k, copy.deepcopy(v))
        for k, v in ck["replay"].items():
            getattr(self.replay, k)[:len(v)] = v
        self.replay.size, self.replay.position = ck["replay_size"], ck["replay_position"]
        if self.sequential:
            self.actor._token = None
            self.actor.cache.invalidate()


def workload_rows(env):
    return [[env.step_index, t["vehicle_id"], t["arrival_s"], t["input_bits"], t["required_cycles"], t["deadline_s"]]
            for t in sorted(env.current_tasks, key=lambda t: t["vehicle_id"])]


def learned_slot(agent, world, actor, training=False):
    eps = agent.epsilon() if training else 0.
    warmup = training and agent.physical_slots < agent.config["warmup_physical_records"]
    if agent.sequential:
        q0 = q31 = None
        for stage in range(32):
            # Diagnostics only during greedy evaluation, using the same cached values.
            values = actor.cache.values(world.observation(), world.cache_token) if not training and stage in (0, 31) else None
            a = actor.act(world, eps, warmup, explore=training)
            if values is not None:
                if stage == 0:
                    q0 = float(values[a])
                else:
                    q31 = float(values[a])
            _, _, _, info = world.step(a)
            assert info["physical_advanced"] == (stage == 31)
        return info["physical_record"], info["physical_info"], q0, q31
    state = world._observation()
    action = actor.act(state, eps, warmup, explore=training)
    q = None
    if not training:
        with torch.inference_mode():
            q = float(agent.online(torch.tensor(state)[None])[0].gather(1, torch.tensor(action)[:, None]).sum())
    slot = world.step_index
    nxt, reward, done, info = world.step(to_action_dict(action, 30))
    return PhysicalRecord(state, action, reward, nxt, done, slot), info, q, q


def save_episode(path, records, extra, env, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"),
                        observations=np.stack([r.observation for r in records]),
                        next_observations=np.stack([r.next_observation for r in records]),
                        actions=np.stack([r.actions for r in records]), rewards=np.array([r.reward for r in records]),
                        dones=np.array([r.done for r in records]), slots=np.array([r.slot for r in records]),
                        episodes=np.array([r.episode for r in records]),
                        workload_sha256=np.array([digest(x["workload"]) for x in extra]))
    write_gzip(path.with_suffix(".json.gz"), dict(metadata=metadata, slots=extra, tasks=env.export_task_records(), summary=env.episode_summary()))


def episode(agent, seed, path, training=False, policy=None, episode_id=0, trace=None):
    if policy is not None:
        world = env = Chapter3IoVEnv(seed=seed)
        actor = None
    elif agent.sequential:
        world = SequentialIoVWrapper(seed=seed)
        env = world.env
        actor = agent.actor if training else SequentialActor(agent.online, seed=0)
    else:
        world = env = Chapter3IoVEnv(seed=seed)
        actor = agent.actor if training else RawActor(agent.online, seed=0)
    records, extras = [], []
    q0 = q31 = None
    for slot in range(40):
        workload, before = workload_rows(env), env.uav_positions.copy()
        if policy:
            action = np.concatenate((select_destinations(env, policy), np.zeros(2, dtype=np.int64)))
            state = env._observation()
            nxt, reward, done, info = env.step(to_action_dict(action, 30))
            record = PhysicalRecord(state, action, reward, nxt, done, slot, episode_id)
        else:
            record, info, q0, q31 = learned_slot(agent, world, actor, training)
            record = PhysicalRecord(record.observation, record.actions, record.reward, record.next_observation, record.done, slot, episode_id)
        if training:
            update = agent.observe(record)
            if update is not None and trace is not None:
                trace.write(json.dumps(jsonable(update), allow_nan=False)+"\n")
                trace.flush()
        after = env.uav_positions.copy()
        extras.append(dict(slot=slot, workload=workload, uav_before=before, uav_after=after,
                           actual_distance_m=np.linalg.norm(after-before, axis=1), requested_moves=record.actions[30:],
                           reward_components=info["reward_components"], energy_j=info["energy_j"]))
        records.append(record)
    assert records[-1].done and len(records) == 40
    result = dict(seed=seed, episode=episode_id, physical_return=sum(r.reward for r in records),
                  deadline_success=env.episode_summary()["deadline_success_rate"],
                  workload_sha256=digest([x["workload"] for x in extras]),
                  terminal_reward=records[-1].reward, terminal_q_stage0=q0, terminal_q_stage31=q31)
    if path is not None:
        save_episode(path, records, extras, env, result)
    return result, records


def evaluate(agent, seeds, out, tag):
    before = digest(agent.checkpoint())
    start = time.perf_counter()
    rows = [episode(agent, seed, out/f"{tag}_{seed}")[0] for seed in seeds]
    assert digest(agent.checkpoint()) == before, "evaluation mutated training state"
    return dict(rows=rows, wall_seconds=time.perf_counter()-start, training_state_unchanged=True)


def probe_metrics(agent, records):
    forced_errors, stage0, stage31, raw_rows = [], [], [], []
    for i, r in enumerate(records):
        forced_q = None
        if agent.sequential:
            actor = SequentialActor(agent.online)
            frame = SelectionFrame()
            frame.start(r.observation)
            for stage in range(32):
                if stage in (0, 31):
                    q = actor.cache.values(frame.observation(), frame.cache_token)
                    (stage0 if stage == 0 else stage31).append(float(q.max()))
                frame.accept(actor.act(frame))
            if r.done:
                forced = reconstruct(r, 31)
                with torch.inference_mode():
                    q = float(agent.online(torch.tensor(forced.observation)[None])[0, forced.action])
                forced_q = q
                forced_errors.append(abs(q-r.reward))
        else:
            with torch.inference_mode():
                q = agent.online(torch.tensor(r.observation)[None])[0]
            stage0.append(float(q.max(1).values.sum()))
            stage31.append(stage0[-1])
            if r.done:
                chosen = float(q.gather(1, torch.tensor(r.actions.copy())[:, None]).sum())
                forced_q = chosen
                forced_errors.append(abs(chosen-r.reward))
        raw_rows.append(dict(record_index=i, slot=r.slot, done=r.done, greedy_stage0=stage0[-1],
                             greedy_stage31=stage31[-1], forced_terminal_q=forced_q,
                             forced_terminal_reward=r.reward if r.done else None))
    return dict(fixed_terminal_forced_chosen_abs_error_p95=float(np.quantile(forced_errors, .95)),
                fixed_terminal_labels=len(forced_errors), greedy_physical_states=len(records),
                fixed_positive_stage0_fraction=float(np.mean(np.array(stage0) > 0)),
                fixed_positive_stage31_fraction=float(np.mean(np.array(stage31) > 0)), raw_rows=raw_rows)


def trained_timing(agent, observations, out):
    selector = SequentialSelection(agent.online) if agent.sequential else AdditiveSelection(agent.online)
    initial = digest(agent.checkpoint())
    rows = []
    for round_id in range(2):
        for i in range(32):
            selector.select(observations[i])
        for i, obs in enumerate(observations):
            start = time.perf_counter_ns()
            action = selector.select(obs)
            elapsed = time.perf_counter_ns()-start
            assert len(action["destinations"]) == 30 and len(action["uav_movement"]) == 2
            rows.append(dict(round=round_id, observation=i, nanoseconds=elapsed))
    assert initial == digest(agent.checkpoint())
    save(out/"timing_raw.json", rows)
    ms = np.array([r["nanoseconds"] for r in rows])/1e6
    return dict(median_ms=float(np.median(ms)), p95_ms=float(np.quantile(ms, .95)), measured_calls=len(rows),
                parameter_bytes=sum(p.numel()*p.element_size() for p in agent.online.parameters()),
                replay_array_bytes=agent.replay.storage_bytes, peak_memory_measured=False,
                selection_only=True, training_state_unchanged=True)


def paired_summary(candidate, baseline, bootstrap_seed=20261108):
    assert [r["seed"] for r in candidate] == [r["seed"] for r in baseline]
    assert [r["workload_sha256"] for r in candidate] == [r["workload_sha256"] for r in baseline]
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(candidate), (2000, len(candidate)))
    answer = {}
    for metric in ("physical_return", "deadline_success"):
        delta = np.array([a[metric]-b[metric] for a, b in zip(candidate, baseline)])
        answer[metric] = dict(mean_difference=float(delta.mean()),
                              paired_episode_bootstrap95= np.quantile(delta[indices].mean(1), [.025, .975]).tolist())
    return answer


def screen(candidate, raw, heuristics, probes):
    # Fractions are on [0,1]; .02 means two percentage points.
    avg = lambda rows, name: float(np.mean([r[name] for r in rows]))
    conditions = dict(
        fresh_stage0_abs_error_p95_at_most10=float(np.quantile([abs(r["terminal_q_stage0"]-r["terminal_reward"]) for r in candidate], .95)) <= 10.,
        fresh_stage31_abs_error_p95_at_most10=float(np.quantile([abs(r["terminal_q_stage31"]-r["terminal_reward"]) for r in candidate], .95)) <= 10.,
        fixed_forced_error_p95_at_most10=probes["fixed_terminal_forced_chosen_abs_error_p95"] <= 10.,
        fixed_positive_stage0_at_most5percent=probes["fixed_positive_stage0_fraction"] <= .05,
        fixed_positive_stage31_at_most5percent=probes["fixed_positive_stage31_fraction"] <= .05,
        return_gain_at_least10=avg(candidate, "physical_return")-avg(raw, "physical_return") >= 10.,
        deadline_gain_at_least2percentage_points=avg(candidate, "deadline_success")-avg(raw, "deadline_success") >= .02-1e-12)
    for name in ("nearest_rsu", "uav_selective"):
        for metric in ("physical_return", "deadline_success"):
            conditions[f"{metric}_not_below_{name}"] = avg(candidate, metric) >= avg(heuristics[name], metric)
    return dict(conditions=conditions, all_pass=all(conditions.values()), automatic_promotion=False)


def assert_pair_initialization(c):
    for i in c["replicates"]:
        a, b = [PilotAgent(arm, c["model_seeds"][i], c) for arm in ARMS]
        for raw_index, cond_index in ((0, 0), (2, 2)):
            for name in ("weight", "bias"):
                assert torch.equal(getattr(a.online.layers[raw_index], name), getattr(b.online.encoder[cond_index], name))


def run(mode, out):
    if sha(SPEC_PATH) != SPEC_HASH:
        raise ValueError("frozen specification hash changed")
    mechanics = json.loads((ROOT/"validation/chapter3_sequential_mechanics_2026-11-01_01/verification.json").read_text())
    for name, expected in mechanics["source_sha256"].items():
        if name in ("chapter3_sequential_conditional.py", "chapter3_iov_env.py", "chapter3_iov_env_v2.py", "train_chapter3_ddqn.py"):
            assert sha(ROOT/name) == expected, f"validated source changed: {name}"
    c = settings(mode)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert_pair_initialization(c)
    out.mkdir(parents=True, exist_ok=False)
    current_agent = current_dir = None
    def status(state, **extra):
        data = dict(state=state, mode=mode, updated_at=datetime.now().astimezone().isoformat(), **extra)
        save(out/"status.json", data)
        (out/"LIVE_STATUS.txt").write_text(json.dumps(jsonable(data), indent=2), encoding="utf-8")
    started = time.perf_counter()
    try:
        snap = out/"source_snapshot"
        snap.mkdir()
        for name in SOURCE_FILES:
            shutil.copy2(ROOT/name, snap/name)
        save(out/"manifest.json", dict(settings=c, spec=json.loads(SPEC_PATH.read_text()), source_sha256={n: sha(snap/n) for n in SOURCE_FILES},
                                       environment=asdict(Chapter3Config()), cpu=cpu_name(), torch=str(torch.__version__), numpy=np.__version__,
                                       torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                                       integration_only=mode == "smoke", development_pilot_launched=mode == "development"))
        status("collecting_fixed_probe")
        fixed = []
        for i, seed in enumerate(c["probe_seeds"]):
            _, records = episode(None, seed, out/"fixed_probe"/str(seed), policy=("all_local", "nearest_rsu", "uav_selective")[i % 3])
            fixed.extend(records)
        common_obs = np.stack([fixed[i].observation for i in np.linspace(0, len(fixed)-1, 128, dtype=np.int64)])
        np.savez_compressed(out/"timing_observations.npz", observations=common_obs)
        order_rng, run_rows, checkpoints = np.random.default_rng(20261107), {}, []
        training_rows, validation_rows = {}, {}
        for rep in c["replicates"]:
            order = list(ARMS)
            order_rng.shuffle(order)
            save(out/f"pair{rep}_order.json", order)
            for arm in order:
                key = f"rep{rep}_{arm}"
                d = out/key
                d.mkdir()
                agent = current_agent = PilotAgent(arm, c["model_seeds"][rep], c)
                current_dir = d
                rows, diagnostics = [], {}
                train_wall = eval_wall = 0.
                with (d/"optimizer.jsonl").open("w", encoding="utf-8") as trace:
                    for ep in range(c["episodes"]+1):
                        if ep in c["diagnostic_episodes"]:
                            status("scheduled_validation", run=key, completed_episodes=ep)
                            diagnostics[str(ep)] = evaluate(agent, c["validation_seeds"], d/"validation", f"ep{ep}")
                            eval_wall += diagnostics[str(ep)]["wall_seconds"]
                            save(d/"diagnostics.json", diagnostics)
                        if ep == c["episodes"]:
                            break
                        status("training", run=key, episode=ep+1, physical_slots=agent.physical_slots, optimizer_steps=agent.updates)
                        start = time.perf_counter()
                        rows.append(episode(agent, c["train_starts"][rep]+ep, d/"training"/f"ep{ep+1}", True, episode_id=ep+1, trace=trace)[0])
                        train_wall += time.perf_counter()-start
                        save(d/"training_summary.json", rows)
                expected_updates = c["episodes"]*40-c["warmup_physical_records"]+1
                assert agent.physical_slots == c["episodes"]*40 and agent.updates == expected_updates
                assert agent.target_copies == expected_updates//c["target_copy_every_optimizer_steps"]
                torch.save(agent.checkpoint(), d/"final.pt")
                checkpoints.append(dict(key=key, path=str(d/"final.pt"), sha256=sha(d/"final.pt")))
                training_rows[key], validation_rows[key] = rows, diagnostics
                run_rows[key] = dict(physical_slots=agent.physical_slots, optimizer_steps=agent.updates, target_copies=agent.target_copies,
                                     training_wall_seconds=train_wall, scheduled_eval_wall_seconds=eval_wall,
                                     actor_history=agent.actor.history, action_rng=agent.actor.rng.bit_generator.state)
                save(d/"training_counters.json", run_rows[key])
                current_agent = current_dir = None
        for rep in c["replicates"]:
            ka, kb = [f"rep{rep}_{arm}" for arm in ARMS]
            assert [r["workload_sha256"] for r in training_rows[ka]] == [r["workload_sha256"] for r in training_rows[kb]]
            fields = ("epsilon", "warmup", "explore", "random_branch")
            assert [[r[f] for f in fields] for r in run_rows[ka]["actor_history"]] == [[r[f] for f in fields] for r in run_rows[kb]["actor_history"]]
            assert run_rows[ka]["action_rng"] == run_rows[kb]["action_rng"]
            for ep in c["diagnostic_episodes"]:
                assert [r["workload_sha256"] for r in validation_rows[ka][str(ep)]["rows"]] == [r["workload_sha256"] for r in validation_rows[kb][str(ep)]["rows"]]
        # All final checkpoints and hashes committed before ANY fresh rollout.
        freeze = dict(saved_at=datetime.now().astimezone().isoformat(), checkpoints=checkpoints,
                      count=len(checkpoints), expected_count=len(c["replicates"])*2)
        assert freeze["count"] == freeze["expected_count"]
        save(out/"FRESH_GATE_FREEZE.json", freeze)
        status("fresh_gate", frozen_checkpoints=len(checkpoints))
        heuristic_rows = {}
        for policy in ("all_local", "nearest_rsu", "uav_selective"):
            start = time.perf_counter()
            heuristic_rows[policy] = [episode(None, seed, out/"fresh_heuristics"/f"{policy}_{seed}", policy=policy)[0] for seed in c["fresh_seeds"]]
            save(out/f"{policy}_fresh.json", dict(rows=heuristic_rows[policy], wall_seconds=time.perf_counter()-start))
        final = {}
        for item in checkpoints:
            assert sha(item["path"]) == item["sha256"]
            ck = torch.load(item["path"], map_location="cpu", weights_only=False)
            agent = PilotAgent(ck["arm"], ck["seed"], c)
            agent.restore(ck)
            d = out/item["key"]
            status("final_diagnostics_and_timing", run=item["key"])
            start = time.perf_counter()
            fresh = evaluate(agent, c["fresh_seeds"], d/"fresh", "final")
            before = digest(agent.checkpoint())
            probes = probe_metrics(agent, fixed)
            timing = trained_timing(agent, common_obs, d)
            assert digest(agent.checkpoint()) == before
            final[item["key"]] = dict(fresh=fresh, fixed_probe=probes, timing=timing,
                                      postfreeze_diagnostic_wall_seconds=time.perf_counter()-start)
            for policy in heuristic_rows:
                paired_summary(fresh["rows"], heuristic_rows[policy])  # verifies seed/fingerprint identity
            save(d/"final_evaluation.json", final[item["key"]])
        comparisons = {}
        for rep in c["replicates"]:
            raw, candidate = [final[f"rep{rep}_{arm}"] for arm in ARMS]
            comparisons[str(rep)] = dict(paired=paired_summary(candidate["fresh"]["rows"], raw["fresh"]["rows"]),
                                          engineering_screen=screen(candidate["fresh"]["rows"], raw["fresh"]["rows"], heuristic_rows, candidate["fixed_probe"]),
                                          heuristic_comparisons={p: paired_summary(candidate["fresh"]["rows"], rows) for p, rows in heuristic_rows.items()})
        verification = dict(state="passed_integration" if mode == "smoke" else "completed_development_screen",
                            mode=mode, pairing_workload_and_action_rng_verified=True,
                            fresh_only_after_all_checkpoints_frozen=True, scheduled_evaluation_state_invariance=True,
                            fixed_and_timing_state_invariance=True, counters={k: {n: r[n] for n in ("physical_slots", "optimizer_steps", "target_copies")} for k, r in run_rows.items()},
                            comparisons=comparisons, final=final, all_final_hashes_unchanged=all(sha(x["path"]) == x["sha256"] for x in checkpoints),
                            wall_seconds=time.perf_counter()-started, thesis_claim_established=False,
                            development_pilot_launched=mode == "development", automatic_promotion=False,
                            smoke_metrics_are_learning_evidence=False)
        save(out/"verification.json", verification)
        status("complete", physical_slots=sum(r["physical_slots"] for r in run_rows.values()),
               optimizer_steps=sum(r["optimizer_steps"] for r in run_rows.values()),
               development_pilot_launched=mode == "development", thesis_claim_established=False)
        print(json.dumps(dict(output=str(out), state=verification["state"], counters=verification["counters"]), indent=2))
    except BaseException as exc:
        if current_agent is not None and current_dir is not None:
            torch.save(current_agent.checkpoint(), current_dir/"failure_state.pt")
        (out/"failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        status("failed_no_automatic_retry", error=repr(exc))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "development"), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.mode, args.output.resolve())
