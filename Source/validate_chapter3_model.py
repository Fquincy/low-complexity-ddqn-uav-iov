"""Run reproducible model-level baseline and sensitivity checks."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from chapter3_iov_env import Chapter3Config, Chapter3IoVEnv, LOCAL, RSU_1, UAV_1


SEEDS = (101, 211, 307, 401, 503)
OUTPUT = Path(__file__).parent / "validation" / "chapter3_model_validation.csv"


def select_destinations(env: Chapter3IoVEnv, policy: str) -> np.ndarray:
    if policy == "all_local":
        return np.full(env.num_vehicles, LOCAL, dtype=np.int64)
    if policy in {"nearest_uav", "nearest_uav_global_equal_share"}:
        distances = np.linalg.norm(
            env.vehicle_positions[:, None, :] - env.uav_positions[None, :, :], axis=2
        )
        return UAV_1 + np.argmin(distances, axis=1)
    if policy == "nearest_rsu":
        distances = np.linalg.norm(
            env.vehicle_positions[:, None, :] - env.rsu_positions[None, :, :], axis=2
        )
        return RSU_1 + np.argmin(distances, axis=1)
    if policy == "uav_selective":
        distances = np.linalg.norm(
            env.vehicle_positions[:, None, :] - env.rsu_positions[None, :, :], axis=2
        )
        destinations = RSU_1 + np.argmin(distances, axis=1)
        uav_distances = np.linalg.norm(
            env.vehicle_positions[:, None, :] - env.uav_positions[None, :, :], axis=2
        )
        pairs = sorted(
            (uav_distances[vehicle_id, uav_id], vehicle_id, uav_id)
            for vehicle_id in range(env.num_vehicles)
            for uav_id in range(env.num_uavs)
        )
        assigned_vehicles: set[int] = set()
        assigned_uavs: set[int] = set()
        for _, vehicle_id, uav_id in pairs:
            if vehicle_id not in assigned_vehicles and uav_id not in assigned_uavs:
                destinations[vehicle_id] = UAV_1 + uav_id
                assigned_vehicles.add(vehicle_id)
                assigned_uavs.add(uav_id)
        return destinations
    if policy == "round_robin":
        return np.arange(env.num_vehicles, dtype=np.int64) % 5
    raise ValueError(policy)


def simulate(config: Chapter3Config, policy: str, seed: int) -> dict[str, float | int | str]:
    env = Chapter3IoVEnv(config, seed=seed)
    records = []
    rewards = []
    energy_by_step = []
    reward_components = []
    miss_rates = []
    cumulative_energy = {
        "uav_compute": 0.0,
        "uav_propulsion": 0.0,
        "vehicle_compute": 0.0,
        "vehicle_uplink": 0.0,
    }
    done = False
    while not done:
        actions = {
            "destinations": select_destinations(env, policy),
            "uav_movement": np.zeros(env.num_uavs, dtype=np.int64),
        }
        _, reward, done, info = env.step(actions)
        records.extend(info["task_records"])
        rewards.append(reward)
        energy_by_step.append(dict(info["energy_j"]))
        reward_components.append(dict(info["reward_components"]))
        miss_rates.append(info["deadline_miss_rate"])
        for name, value in info["energy_j"].items():
            cumulative_energy[name] += value

    delays = np.array([record["delay_s"] for record in records])
    successes = np.array([record["on_time"] for record in records], dtype=np.float64)
    cfg = env.config
    mean_delay_component = float(np.mean([x["normalized_mean_delay"] for x in reward_components]))
    mean_uav_energy_component = float(np.mean([x["normalized_uav_energy"] for x in reward_components]))
    mean_deadline_component = float(np.mean([x["deadline_miss_rate"] for x in reward_components]))
    return {
        "policy": policy,
        "seed": seed,
        "tasks": len(records),
        "deadline_success_rate": float(successes.mean()),
        "mean_delay_s": float(delays.mean()),
        "p95_delay_s": float(np.quantile(delays, 0.95)),
        "mean_reward": float(np.mean(rewards)),
        "mean_delay_reward_component": mean_delay_component,
        "mean_uav_energy_reward_component_current": mean_uav_energy_component,
        "mean_deadline_miss_rate": float(np.mean(miss_rates)),
        "mean_delay_term_weighted": cfg.delay_reward_weight * mean_delay_component,
        "mean_uav_energy_term_weighted": cfg.uav_energy_reward_weight * mean_uav_energy_component,
        "mean_deadline_term_weighted": cfg.deadline_miss_penalty * mean_deadline_component,
        "uav_compute_energy_j": cumulative_energy["uav_compute"],
        "uav_propulsion_energy_j": cumulative_energy["uav_propulsion"],
        "vehicle_compute_energy_j": cumulative_energy["vehicle_compute"],
        "vehicle_uplink_energy_j": cumulative_energy["vehicle_uplink"],
        "uav_battery_remaining_min_j": float(env.uav_battery_j.min()),
    }


def scenarios() -> list[tuple[str, Chapter3Config]]:
    base = Chapter3Config()
    output: list[tuple[str, Chapter3Config]] = []
    for policy in ("all_local", "nearest_uav", "nearest_rsu", "uav_selective", "round_robin"):
        output.append((policy, base))
    output.append(
        (
            "nearest_uav_global_equal_share",
            replace(base, g2a_allocation_mode="global_equal_share"),
        )
    )
    for cpu_ghz in (10, 20, 40):
        for link_mbps in (100, 1000, 10000):
            output.append(
                (
                    f"rsu_cpu{cpu_ghz}_link{link_mbps}",
                    replace(
                        base,
                        rsu_cpu_hz=(cpu_ghz * 1e9, cpu_ghz * 1e9),
                        rsu_link_bps=link_mbps * 1e6,
                    ),
                )
            )
    for carrier_ghz in (0.7, 2.0, 3.5, 5.9):
        output.append(
            (f"uav_selective_carrier{carrier_ghz}", replace(base, carrier_hz=carrier_ghz * 1e9))
        )
    return output


def main() -> None:
    raise SystemExit(
        "This runner targets the defective v1 timeline and is retained for historical review. "
        "Run validate_chapter3_timeline_v2.py for the corrected simulator. "
        "The pre-v2 script is preserved in validation/pre_timeline_v2_2026-10-31."
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario, config in scenarios():
        policy = scenario if scenario in {
            "all_local", "nearest_uav", "nearest_rsu", "uav_selective", "round_robin", "nearest_uav_global_equal_share"
        } else (
            "nearest_rsu" if scenario.startswith("rsu_") else (
                "uav_selective" if scenario.startswith("uav_selective_carrier") else "nearest_uav"
            )
        )
        for seed in SEEDS:
            row = simulate(config, policy, seed)
            row["scenario"] = scenario
            row["rsu_cpu_gcycles_s"] = config.rsu_cpu_hz[0] / 1e9
            row["rsu_link_mbps"] = config.rsu_link_bps / 1e6
            row["carrier_ghz"] = config.carrier_hz / 1e9
            rows.append(row)
    fields = list(rows[0])
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} scenario-seed records to {OUTPUT}")
    print("\nScenario means across five seeds:")
    for scenario, _ in scenarios():
        group = [row for row in rows if row["scenario"] == scenario]
        print(
            f"{scenario:32s} success={np.mean([r['deadline_success_rate'] for r in group]):.3f} "
            f"delay={np.mean([r['mean_delay_s'] for r in group]):.3f}s "
            f"p95={np.mean([r['p95_delay_s'] for r in group]):.3f}s "
            f"reward={np.mean([r['mean_reward'] for r in group]):.3f} "
            f"UAV_energy={np.mean([r['uav_compute_energy_j'] + r['uav_propulsion_energy_j'] for r in group])/1000:.1f}kJ"
        )


if __name__ == "__main__":
    main()
