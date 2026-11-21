"""Train and evaluate a factorized DDQN policy for Chapter 3 IoV MEC."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from chapter3_iov_env import Chapter3Config, Chapter3IoVEnv, LOCAL, RSU_1, UAV_1


class FactorizedQNetwork(nn.Module):
    def __init__(self, state_size: int, num_heads: int, num_actions: int, widths: tuple[int, int]):
        super().__init__()
        self.num_heads = num_heads
        self.num_actions = num_actions
        self.layers = nn.Sequential(
            nn.Linear(state_size, widths[0]),
            nn.ReLU(),
            nn.Linear(widths[0], widths[1]),
            nn.ReLU(),
            nn.Linear(widths[1], num_heads * num_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        values = self.layers(states)
        return values.view(-1, self.num_heads, self.num_actions)


class ReplayBuffer:
    def __init__(self, capacity: int, state_size: int, num_heads: int):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros((capacity, num_heads), dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        indexes = rng.integers(0, self.size, size=batch_size)
        return (
            self.states[indexes],
            self.actions[indexes],
            self.rewards[indexes],
            self.next_states[indexes],
            self.dones[indexes],
        )


class FactorizedDDQN:
    """Additive factorization of joint action values across discrete heads."""

    def __init__(
        self,
        state_size: int,
        num_heads: int,
        widths: tuple[int, int] = (128, 128),
        learning_rate: float = 1e-4,
        gamma: float = 0.95,
        batch_size: int = 128,
        replay_capacity: int = 50_000,
        target_update_steps: int = 200,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 20_000,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.num_heads = num_heads
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_steps = target_update_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.online = FactorizedQNetwork(state_size, num_heads, 5, widths).to(self.device)
        self.target = FactorizedQNetwork(state_size, num_heads, 5, widths).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(replay_capacity, state_size, num_heads)
        self.environment_steps = 0
        self.gradient_steps = 0

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.environment_steps / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        epsilon = self.epsilon if explore else 0.0
        if explore:
            self.environment_steps += 1
        if self.rng.random() < epsilon:
            return self.rng.integers(0, 5, size=self.num_heads, dtype=np.int64)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.online(state_tensor).argmax(dim=-1).squeeze(0).cpu().numpy()

    def remember(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.push(state, action, reward, next_state, done)

    def update(self) -> float | None:
        if self.memory.size < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, self.rng
        )
        states_t = torch.as_tensor(states, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, device=self.device)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device)

        q_values = self.online(states_t)
        chosen_q = q_values.gather(2, actions_t.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        with torch.no_grad():
            next_actions = self.online(next_states_t).argmax(dim=2, keepdim=True)
            next_q = self.target(next_states_t).gather(2, next_actions).squeeze(-1).sum(dim=1)
            targets = rewards_t + self.gamma * (1.0 - dones_t) * next_q

        loss = nn.functional.smooth_l1_loss(chosen_q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu())

    def save(self, path: Path) -> None:
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "environment_steps": self.environment_steps,
                "gradient_steps": self.gradient_steps,
            },
            path,
        )


def to_action_dict(actions: np.ndarray, num_vehicles: int) -> dict[str, np.ndarray]:
    return {
        "destinations": actions[:num_vehicles].astype(np.int64),
        "uav_movement": actions[num_vehicles:].astype(np.int64),
    }


def fixed_policy_actions(env: Chapter3IoVEnv, policy: str) -> dict[str, np.ndarray]:
    if policy == "ddqn":
        raise ValueError("DDQN actions require the trained agent")
    if policy == "all_local":
        destinations = np.full(env.num_vehicles, LOCAL, dtype=np.int64)
    elif policy == "nearest_uav":
        distances = np.linalg.norm(
            env.vehicle_positions[:, None, :] - env.uav_positions[None, :, :], axis=2
        )
        destinations = UAV_1 + np.argmin(distances, axis=1)
    elif policy == "nearest_rsu":
        distances = np.linalg.norm(
            env.vehicle_positions[:, None, :] - env.rsu_positions[None, :, :], axis=2
        )
        destinations = RSU_1 + np.argmin(distances, axis=1)
    elif policy == "round_robin":
        destinations = np.arange(env.num_vehicles, dtype=np.int64) % 5
    elif policy == "uav_selective":
        distances = np.linalg.norm(
            env.vehicle_positions[:, None, :] - env.rsu_positions[None, :, :], axis=2
        )
        destinations = RSU_1 + np.argmin(distances, axis=1)
        uav_distances = np.linalg.norm(
            env.vehicle_positions[:, None, :] - env.uav_positions[None, :, :], axis=2
        )
        pairs = sorted(
            (uav_distances[v, u], v, u)
            for v in range(env.num_vehicles)
            for u in range(env.num_uavs)
        )
        used_vehicles: set[int] = set()
        used_uavs: set[int] = set()
        for _, vehicle_id, uav_id in pairs:
            if vehicle_id not in used_vehicles and uav_id not in used_uavs:
                destinations[vehicle_id] = UAV_1 + uav_id
                used_vehicles.add(vehicle_id)
                used_uavs.add(uav_id)
    else:
        raise ValueError(f"unknown fixed policy: {policy}")
    return {
        "destinations": destinations,
        "uav_movement": np.zeros(env.num_uavs, dtype=np.int64),
    }


def evaluate_policy(
    config: Chapter3Config,
    policy: str,
    seed: int,
    agent: FactorizedDDQN | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = Chapter3IoVEnv(config, seed=seed)
    state = env.reset(seed=seed)
    task_rows: list[dict[str, Any]] = []
    rewards: list[float] = []
    energy_totals = {"uav_compute": 0.0, "uav_propulsion": 0.0}
    movement_counts = np.zeros((env.num_uavs, 5), dtype=np.int64)
    done = False
    while not done:
        if policy == "ddqn":
            if agent is None:
                raise ValueError("agent required for DDQN evaluation")
            action = to_action_dict(agent.act(state, explore=False), env.num_vehicles)
        else:
            action = fixed_policy_actions(env, policy)
        next_state, reward, done, info = env.step(action)
        state = next_state
        rewards.append(reward)
        for uav_id, movement in enumerate(action["uav_movement"]):
            movement_counts[uav_id, int(movement)] += 1
        energy_totals["uav_compute"] += info["energy_j"]["uav_compute"]
        energy_totals["uav_propulsion"] += info["energy_j"]["uav_propulsion"]
        task_rows.extend(
            {
                "seed": seed,
                "policy": policy,
                "uav1_movement": int(action["uav_movement"][0]),
                "uav2_movement": int(action["uav_movement"][1]),
                **record,
            }
            for record in info["task_records"]
        )
    delays = np.array([row["delay_s"] for row in task_rows])
    success = np.array([row["on_time"] for row in task_rows], dtype=np.float64)
    summary = {
        "seed": seed,
        "policy": policy,
        "tasks": len(task_rows),
        "deadline_success_rate": float(success.mean()),
        "mean_delay_s": float(delays.mean()),
        "p95_delay_s": float(np.quantile(delays, 0.95)),
        "mean_reward": float(np.mean(rewards)),
        "uav_energy_j": float(sum(energy_totals.values())),
    }
    destination_counts = np.bincount(
        [int(row["destination"]) for row in task_rows], minlength=5
    )
    for destination, count in enumerate(destination_counts):
        summary[f"destination_{destination}_count"] = int(count)
    for uav_id in range(env.num_uavs):
        for movement in range(5):
            summary[f"uav{uav_id + 1}_movement_{movement}_steps"] = int(
                movement_counts[uav_id, movement]
            )
    return summary, task_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def train_seed(args: argparse.Namespace, seed: int) -> Path:
    if not getattr(args, "allow_legacy_diagnostic", False):
        raise RuntimeError(
            "Training is gated: this trainer uses the defective v1 timeline. "
            "Use validate_chapter3_timeline_v2.py for corrected fixed-policy checks. "
            "A v2 trainer must handle delayed completions, terminal backlog and the new observation. "
            "--allow-legacy-diagnostic is for reproducing old diagnostic behavior only."
        )
    config = Chapter3Config(episode_steps=args.episode_steps)
    run_dir = Path(args.output_dir) / f"width_{args.widths[0]}_{args.widths[1]}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = Chapter3IoVEnv(config, seed=seed)
    agent = FactorizedDDQN(
        state_size=env.state_size,
        num_heads=env.num_vehicles + env.num_uavs,
        widths=tuple(args.widths),
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        target_update_steps=args.target_update_steps,
        epsilon_decay_steps=args.epsilon_decay_steps,
        seed=seed,
        device=args.device,
    )
    run_config = {
        "seed": seed,
        "model": "factorized additive DDQN",
        "torch_version": torch.__version__,
        "widths": args.widths,
        "heads": env.num_vehicles + env.num_uavs,
        "actions_per_head": 5,
        "state_size": env.state_size,
        "trainable_parameters": sum(p.numel() for p in agent.online.parameters()),
        "episodes": args.episodes,
        "episode_steps": args.episode_steps,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "replay_capacity": args.replay_capacity,
        "epsilon_decay_steps": args.epsilon_decay_steps,
        "target_update_steps": args.target_update_steps,
        "device": args.device,
        "environment": asdict(config),
        "warning": "Provisional simulation assumptions remain; this run is not thesis evidence without parameter review and repeated-seed evaluation.",
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    episode_rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    global_step = 0
    for episode in range(args.episodes):
        state = env.reset(seed=seed + episode)
        done = False
        rewards = []
        losses = []
        task_rows = []
        episode_uav_energy = 0.0
        movement_counts = np.zeros((env.num_uavs, 5), dtype=np.int64)
        while not done:
            action_vector = agent.act(state, explore=True)
            action_dict = to_action_dict(action_vector, env.num_vehicles)
            for uav_id, movement in enumerate(action_dict["uav_movement"]):
                movement_counts[uav_id, int(movement)] += 1
            next_state, reward, done, info = env.step(action_dict)
            agent.remember(state, action_vector, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            rewards.append(reward)
            task_rows.extend(
                {
                    "uav1_movement": int(action_dict["uav_movement"][0]),
                    "uav2_movement": int(action_dict["uav_movement"][1]),
                    **record,
                }
                for record in info["task_records"]
            )
            episode_uav_energy += info["energy_j"]["uav_compute"]
            episode_uav_energy += info["energy_j"]["uav_propulsion"]
            state = next_state
            global_step += 1
        episode_row = {
                "seed": seed,
                "episode": episode,
                "mean_reward": float(np.mean(rewards)),
                "deadline_success_rate": float(np.mean([x["on_time"] for x in task_rows])),
                "mean_delay_s": float(np.mean([x["delay_s"] for x in task_rows])),
                "p95_delay_s": float(np.quantile([x["delay_s"] for x in task_rows], 0.95)),
                "uav_energy_j": episode_uav_energy,
                "mean_loss": float(np.mean(losses)) if losses else None,
                "epsilon": agent.epsilon,
                "environment_steps": global_step,
            }
        for uav_id in range(env.num_uavs):
            for movement in range(5):
                episode_row[f"uav{uav_id + 1}_movement_{movement}_steps"] = int(
                    movement_counts[uav_id, movement]
                )
        destination_counts = np.bincount(
            [int(task["destination"]) for task in task_rows], minlength=5
        )
        for destination, count in enumerate(destination_counts):
            episode_row[f"destination_{destination}_tasks"] = int(count)
        episode_rows.append(episode_row)
        if (episode + 1) % args.checkpoint_every == 0 or episode + 1 == args.episodes:
            agent.save(run_dir / f"checkpoint_episode_{episode + 1}.pt")
    write_csv(run_dir / "episode_metrics.csv", episode_rows)
    agent.save(run_dir / "final_model.pt")
    (run_dir / "training_runtime_seconds.txt").write_text(
        f"{time.perf_counter() - start_time:.3f}\n", encoding="ascii"
    )

    evaluations = []
    traces: list[dict[str, Any]] = []
    policies = ("ddqn", "all_local", "nearest_rsu", "uav_selective", "round_robin")
    for eval_seed_offset in range(args.evaluation_episodes):
        eval_seed = seed + 100_000 + eval_seed_offset
        for policy in policies:
            summary, rows = evaluate_policy(config, policy, eval_seed, agent if policy == "ddqn" else None)
            evaluations.append(summary)
            traces.extend(rows)
    write_csv(run_dir / "evaluation_metrics.csv", evaluations)
    write_csv(run_dir / "evaluation_task_traces.csv", traces)
    aggregate = []
    for policy in policies:
        group = [row for row in evaluations if row["policy"] == policy]
        aggregate_row = {
                "policy": policy,
                "evaluation_episodes": len(group),
                "mean_deadline_success_rate": float(np.mean([x["deadline_success_rate"] for x in group])),
                "sd_deadline_success_rate": float(np.std([x["deadline_success_rate"] for x in group], ddof=1)) if len(group) > 1 else 0.0,
                "mean_delay_s": float(np.mean([x["mean_delay_s"] for x in group])),
                "mean_p95_delay_s": float(np.mean([x["p95_delay_s"] for x in group])),
                "mean_reward": float(np.mean([x["mean_reward"] for x in group])),
                "mean_uav_energy_j": float(np.mean([x["uav_energy_j"] for x in group])),
            }
        for destination in range(5):
            aggregate_row[f"mean_destination_{destination}_count"] = float(
                np.mean([x[f"destination_{destination}_count"] for x in group])
            )
        for uav_id in range(env.num_uavs):
            for movement in range(5):
                key = f"uav{uav_id + 1}_movement_{movement}_steps"
                aggregate_row[f"mean_{key}"] = float(np.mean([x[key] for x in group]))
        aggregate.append(aggregate_row)
    write_csv(run_dir / "evaluation_summary.csv", aggregate)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--episode-steps", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 211, 307])
    parser.add_argument("--widths", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--target-update-steps", type=int, default=200)
    parser.add_argument("--epsilon-decay-steps", type=int, default=20_000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("training_runs"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-legacy-diagnostic", action="store_true",
                        help="Explicitly permit defective v1 diagnostic runs; never thesis evidence")
    args = parser.parse_args()
    if args.episodes <= 0 or args.evaluation_episodes <= 0 or args.episode_steps <= 0:
        parser.error("episode counts and episode steps must be positive")
    if args.batch_size <= 0 or args.replay_capacity < args.batch_size:
        parser.error("replay capacity must be at least the positive batch size")
    if args.widths[0] <= 0 or args.widths[1] <= 0:
        parser.error("network widths must be positive")
    return args


def main() -> None:
    args = parse_args()
    for seed in args.seeds:
        run_dir = train_seed(args, seed)
        print(f"Completed training/evaluation seed {seed}: {run_dir}")


if __name__ == "__main__":
    main()
