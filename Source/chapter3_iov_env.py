"""Standalone Chapter 3 IoV MEC environment for deterministic validation.

The environment is deliberately separate from the legacy training scripts. It
models indivisible tasks, five execution destinations, two moving UAV MEC
servers, two fixed RSUs, per-server FIFO service, deadlines, and UAV energy.
The rotary-wing power function and generic aircraft parameters follow Zeng et
al., arXiv:1804.02238 / IEEE TWC 2019; they are not calibrated to a named UAV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


LOCAL = 0
UAV_1 = 1
UAV_2 = 2
RSU_1 = 3
RSU_2 = 4
DESTINATION_COUNT = 5
HOVER, NORTH, EAST, SOUTH, WEST = range(5)


@dataclass(frozen=True)
class Chapter3Config:
    num_vehicles: int = 30
    episode_steps: int = 40
    area_m: float = 1000.0
    slot_s: float = 1.0
    task_size_bytes: tuple[float, float] = (1e6, 3e6)
    task_cycles: tuple[float, float] = (300e6, 500e6)
    deadline_s: tuple[float, float] = (0.250, 0.300)
    vehicle_cpu_hz: float = 1.5e9
    uav_cpu_hz: tuple[float, float] = (10e9, 10e9)
    rsu_cpu_hz: tuple[float, float] = (20e9, 20e9)
    rsu_cpu_is_provisional: bool = True
    rsu_link_bps: float = 1e9
    rsu_link_is_provisional: bool = True
    g2a_bandwidth_hz: float = 20e6
    g2a_allocation_mode: str = "per_link_with_interference"
    vehicle_tx_power_w: float = 1.0
    noise_dbm: float = -100.0
    carrier_hz: float = 2e9
    carrier_frequency_is_provisional: bool = True
    los_a: float = 9.61
    los_b: float = 0.16
    los_excess_loss_db: float = 1.0
    nlos_excess_loss_db: float = 20.0
    uav_altitude_m: float = 50.0
    uav_max_step_m: float = 49.0
    uav_battery_j: float = 500e3
    effective_switched_capacitance: float = 1e-28
    vehicle_effective_switched_capacitance: float = 1e-28
    vehicle_compute_energy_is_provisional: bool = True
    delay_reward_weight: float = 0.5
    uav_energy_reward_weight: float = 0.5
    uav_mass_kg: float = 100.0 / 9.81
    air_density_kg_m3: float = 1.225
    rotor_radius_m: float = 0.5
    rotor_angular_speed_rad_s: float = 400.0
    rotor_solidity: float = 0.05
    fuselage_drag_ratio: float = 0.3
    blade_drag_coefficient: float = 0.012
    induced_power_correction: float = 0.1
    deadline_miss_penalty: float = 1.0
    deadline_normalizer_s: float = 0.300

    @property
    def uav_energy_normalizer_j(self) -> float:
        return 2.0 * self.uav_battery_j / self.episode_steps


class Chapter3IoVEnv:
    """One joint system transition per slot; actions are per-task and per-UAV."""

    def __init__(self, config: Chapter3Config | None = None, seed: int | None = None):
        self.config = config or Chapter3Config()
        self.rng = np.random.default_rng(seed)
        self.area = self.config.area_m
        self.num_vehicles = self.config.num_vehicles
        self.num_uavs = 2
        self.num_rsus = 2
        self.num_servers = 4
        self.rsu_positions = np.array(
            [[0.25 * self.area, 0.5 * self.area], [0.75 * self.area, 0.5 * self.area]],
            dtype=np.float64,
        )
        self._p0, self._pi, self._v0, self._tip_speed, self._rotor_area = (
            self._rotary_wing_constants()
        )
        self.reset(seed=seed)

    @property
    def state_size(self) -> int:
        return self.num_vehicles * 9 + self.num_servers + self.num_uavs * 3 + 1

    @property
    def action_sizes(self) -> dict[str, tuple[int, ...]]:
        return {
            "destinations": (self.num_vehicles,),
            "uav_movement": (self.num_uavs,),
        }

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_index = 0
        self.time_s = 0.0
        self.vehicle_positions = self.rng.uniform(
            0.0, self.area, size=(self.num_vehicles, 2)
        )
        self.vehicle_speeds = self.rng.uniform(0.0, 33.333333, self.num_vehicles)
        self.vehicle_headings = self.rng.uniform(-np.pi, np.pi, self.num_vehicles)
        self.uav_positions = np.array(
            [[0.1 * self.area, 0.1 * self.area], [0.9 * self.area, 0.9 * self.area]],
            dtype=np.float64,
        )
        self.uav_battery_j = np.full(2, self.config.uav_battery_j, dtype=np.float64)
        self.server_free_at = np.zeros(self.num_servers, dtype=np.float64)
        self.local_free_at = np.zeros(self.num_vehicles, dtype=np.float64)
        self.tasks_processed = 0
        self.tasks_on_time = 0
        self.task_records: list[dict[str, Any]] = []
        self.last_step_records: list[dict[str, Any]] = []
        self.last_energy_j = {
            "uav_compute": 0.0,
            "uav_propulsion": 0.0,
            "vehicle_compute": 0.0,
            "vehicle_uplink": 0.0,
        }
        self.current_tasks = self._new_tasks()
        return self._observation()

    def step(self, actions: dict[str, Any]) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        destinations = np.asarray(actions["destinations"], dtype=np.int64)
        movements = np.asarray(actions["uav_movement"], dtype=np.int64)
        if destinations.shape != (self.num_vehicles,):
            raise ValueError(f"destinations must have shape ({self.num_vehicles},)")
        if movements.shape != (self.num_uavs,):
            raise ValueError("uav_movement must have shape (2,)")
        if np.any((destinations < 0) | (destinations >= DESTINATION_COUNT)):
            raise ValueError("each task must select exactly one of the five destinations")
        if np.any((movements < HOVER) | (movements > WEST)):
            raise ValueError("UAV movement actions must be in [0, 4]")
        if self.config.g2a_allocation_mode not in {
            "per_link_with_interference",
            "global_equal_share",
        }:
            raise ValueError("unsupported G2A bandwidth allocation mode")

        dt = self.config.slot_s
        propulsion_by_uav_j = self._move_uavs(movements, dt)
        propulsion_j = float(np.sum(propulsion_by_uav_j))
        task_batch = self.current_tasks
        active_uav_links = int(
            np.count_nonzero((destinations == UAV_1) | (destinations == UAV_2))
        )
        if self.config.g2a_allocation_mode == "global_equal_share":
            uav_bandwidth = self.config.g2a_bandwidth_hz / max(1, active_uav_links)
        else:
            uav_bandwidth = self.config.g2a_bandwidth_hz
        interference_by_task = self._g2a_interference_by_task(task_batch, destinations)
        task_batch = sorted(
            task_batch,
            key=lambda task: self._queue_arrival_key(
                task,
                int(destinations[int(task["vehicle_id"])]),
                uav_bandwidth,
                interference_by_task[int(task["task_id"])],
            ),
        )
        records: list[dict[str, Any]] = []
        uav_compute_j = 0.0
        vehicle_compute_j = 0.0
        vehicle_uplink_j = 0.0
        for task in task_batch:
            vehicle_id = task["vehicle_id"]
            destination = int(destinations[vehicle_id])
            delay_s, compute_energy_j, details = self._serve_task(
                task,
                destination,
                uav_bandwidth,
                interference_by_task[int(task["task_id"])],
            )
            on_time = delay_s <= task["deadline_s"]
            uav_compute_j += compute_energy_j
            vehicle_compute_j += float(details["vehicle_compute_energy_j"])
            vehicle_uplink_j += float(details["vehicle_uplink_energy_j"])
            record = {
                **task,
                "destination": destination,
                "delay_s": delay_s,
                "on_time": bool(on_time),
                "uav_compute_energy_j": compute_energy_j,
                **details,
            }
            records.append(record)
            self.task_records.append(record)

        self.uav_battery_j = np.maximum(
            0.0,
            self.uav_battery_j
            - propulsion_by_uav_j
            - self._compute_energy_by_uav(records),
        )
        self.tasks_processed += len(records)
        self.tasks_on_time += sum(record["on_time"] for record in records)
        self.last_step_records = records
        self.last_energy_j = {
            "uav_compute": uav_compute_j,
            "uav_propulsion": propulsion_j,
            "vehicle_compute": vehicle_compute_j,
            "vehicle_uplink": vehicle_uplink_j,
        }
        mean_normalized_delay = float(
            np.mean([r["delay_s"] for r in records]) / self.config.deadline_normalizer_s
        )
        miss_rate = 1.0 - float(np.mean([r["on_time"] for r in records]))
        normalized_uav_energy = (
            uav_compute_j + propulsion_j
        ) / self.config.uav_energy_normalizer_j
        reward = (
            -self.config.delay_reward_weight * mean_normalized_delay
            - self.config.uav_energy_reward_weight * normalized_uav_energy
            - self.config.deadline_miss_penalty * miss_rate
        )

        self.step_index += 1
        self.time_s += dt
        done = self.step_index >= self.config.episode_steps
        if not done:
            self._move_vehicles(dt)
            self.current_tasks = self._new_tasks()
        info = {
            "task_records": records,
            "deadline_success_rate": self.tasks_on_time / max(1, self.tasks_processed),
            "mean_delay_s": float(np.mean([r["delay_s"] for r in records])),
            "deadline_miss_rate": miss_rate,
            "energy_j": dict(self.last_energy_j),
            "reward_components": {
                "normalized_mean_delay": mean_normalized_delay,
                "normalized_uav_energy": normalized_uav_energy,
                "deadline_miss_rate": miss_rate,
            },
            "uav_battery_j": self.uav_battery_j.copy(),
            "rsu_cpu_is_provisional": self.config.rsu_cpu_is_provisional,
            "rsu_link_is_provisional": self.config.rsu_link_is_provisional,
            "carrier_frequency_is_provisional": self.config.carrier_frequency_is_provisional,
            "vehicle_compute_energy_is_provisional": self.config.vehicle_compute_energy_is_provisional,
            "g2a_allocation_mode": self.config.g2a_allocation_mode,
        }
        return self._observation(), float(reward), done, info

    def _new_tasks(self) -> list[dict[str, float | int]]:
        cfg = self.config
        sizes = self.rng.uniform(*cfg.task_size_bytes, self.num_vehicles)
        cycles = self.rng.uniform(*cfg.task_cycles, self.num_vehicles)
        deadlines = self.rng.uniform(*cfg.deadline_s, self.num_vehicles)
        return [
            {
                "task_id": self.step_index * self.num_vehicles + vehicle_id,
                "vehicle_id": vehicle_id,
                "arrival_s": self.time_s,
                "input_bits": float(size_bytes * 8.0),
                "input_bytes": float(size_bytes),
                "required_cycles": float(required_cycles),
                "deadline_s": float(deadline),
            }
            for vehicle_id, (size_bytes, required_cycles, deadline) in enumerate(
                zip(sizes, cycles, deadlines)
            )
        ]

    def _serve_task(
        self,
        task: dict[str, float | int],
        destination: int,
        bandwidth_hz: float,
        interference_w: float = 0.0,
    ) -> tuple[float, float, dict[str, float | int]]:
        cycles = float(task["required_cycles"])
        input_bits = float(task["input_bits"])
        arrival = float(task["arrival_s"])
        compute_energy_j = 0.0
        tx_delay_s = 0.0
        vehicle_uplink_energy_j = 0.0
        vehicle_compute_energy_j = 0.0
        queue_delay_s = 0.0
        server_id = -1

        if destination == LOCAL:
            vehicle_id = int(task["vehicle_id"])
            queue_delay_s = max(0.0, self.local_free_at[vehicle_id] - arrival)
            service_start = arrival + queue_delay_s
            compute_delay_s = cycles / self.config.vehicle_cpu_hz
            vehicle_compute_energy_j = (
                self.config.vehicle_effective_switched_capacitance
                * self.config.vehicle_cpu_hz**3
                * compute_delay_s
            )
            self.local_free_at[vehicle_id] = service_start + compute_delay_s
        else:
            if destination in (UAV_1, UAV_2):
                uav_id = destination - UAV_1
                rate = self._g2a_rate_bps(
                    self.vehicle_positions[int(task["vehicle_id"])],
                    self.uav_positions[uav_id],
                    bandwidth_hz,
                    interference_w,
                )
                tx_delay_s = input_bits / rate
                vehicle_uplink_energy_j = self.config.vehicle_tx_power_w * tx_delay_s
                server_id = uav_id
                cpu_hz = self.config.uav_cpu_hz[uav_id]
            else:
                rsu_id = destination - RSU_1
                rate = self.config.rsu_link_bps
                tx_delay_s = input_bits / rate
                server_id = self.num_uavs + rsu_id
                cpu_hz = self.config.rsu_cpu_hz[rsu_id]

            service_start = max(arrival + tx_delay_s, self.server_free_at[server_id])
            queue_delay_s = max(0.0, service_start - arrival - tx_delay_s)
            compute_delay_s = cycles / cpu_hz
            self.server_free_at[server_id] = service_start + compute_delay_s
            if server_id < self.num_uavs:
                compute_energy_j = (
                    self.config.effective_switched_capacitance
                    * cpu_hz**3
                    * compute_delay_s
                )

        total_delay_s = tx_delay_s + queue_delay_s + compute_delay_s
        return total_delay_s, compute_energy_j, {
            "transmission_delay_s": tx_delay_s,
            "queue_delay_s": queue_delay_s,
            "computation_delay_s": compute_delay_s,
            "server_id": server_id,
            "radio_rate_bps": rate if destination != LOCAL else 0.0,
            "vehicle_uplink_energy_j": vehicle_uplink_energy_j,
            "vehicle_compute_energy_j": vehicle_compute_energy_j,
        }

    def _queue_arrival_key(
        self,
        task: dict[str, float | int],
        destination: int,
        bandwidth_hz: float,
        interference_w: float = 0.0,
    ) -> tuple[int, float, int]:
        upload_delay_s = 0.0
        if destination in (UAV_1, UAV_2):
            uav_id = destination - UAV_1
            rate = self._g2a_rate_bps(
                self.vehicle_positions[int(task["vehicle_id"])],
                self.uav_positions[uav_id],
                bandwidth_hz,
                interference_w,
            )
            upload_delay_s = float(task["input_bits"]) / rate
        elif destination in (RSU_1, RSU_2):
            upload_delay_s = float(task["input_bits"]) / self.config.rsu_link_bps
        return destination, float(task["arrival_s"]) + upload_delay_s, int(task["task_id"])

    def _g2a_interference_by_task(
        self,
        task_batch: list[dict[str, float | int]],
        destinations: np.ndarray,
    ) -> dict[int, float]:
        interference = {int(task["task_id"]): 0.0 for task in task_batch}
        if self.config.g2a_allocation_mode != "per_link_with_interference":
            return interference
        for uav_id, destination in enumerate((UAV_1, UAV_2)):
            selected = [
                task
                for task in task_batch
                if int(destinations[int(task["vehicle_id"])]) == destination
            ]
            gains = [
                self._g2a_channel_gain(
                    self.vehicle_positions[int(task["vehicle_id"])],
                    self.uav_positions[uav_id],
                )
                for task in selected
            ]
            for idx, task in enumerate(selected):
                interference[int(task["task_id"])] = self.config.vehicle_tx_power_w * (
                    sum(gains) - gains[idx]
                )
        return interference

    def _g2a_channel_gain(self, vehicle_xy: np.ndarray, uav_xy: np.ndarray) -> float:
        cfg = self.config
        distance_m = float(
            np.sqrt(np.sum((vehicle_xy - uav_xy) ** 2) + cfg.uav_altitude_m**2)
        )
        elevation_deg = np.degrees(np.arcsin(cfg.uav_altitude_m / distance_m))
        p_los = 1.0 / (1.0 + cfg.los_a * np.exp(-cfg.los_b * (elevation_deg - cfg.los_a)))
        fspl_db = 20.0 * np.log10(4.0 * np.pi * cfg.carrier_hz / 299792458.0) + 20.0 * np.log10(distance_m)
        mean_path_loss_db = fspl_db + p_los * cfg.los_excess_loss_db + (1.0 - p_los) * cfg.nlos_excess_loss_db
        return float(10.0 ** (-mean_path_loss_db / 10.0))

    def _g2a_rate_bps(
        self,
        vehicle_xy: np.ndarray,
        uav_xy: np.ndarray,
        bandwidth_hz: float,
        interference_w: float = 0.0,
    ) -> float:
        cfg = self.config
        channel_gain = self._g2a_channel_gain(vehicle_xy, uav_xy)
        noise_w = 10.0 ** ((cfg.noise_dbm - 30.0) / 10.0)
        sinr = cfg.vehicle_tx_power_w * channel_gain / (noise_w + interference_w)
        return max(1.0, bandwidth_hz * np.log2(1.0 + sinr))

    def _move_vehicles(self, duration_s: float) -> None:
        displacement = self.vehicle_speeds[:, None] * duration_s * np.column_stack(
            (np.cos(self.vehicle_headings), np.sin(self.vehicle_headings))
        )
        self.vehicle_positions += displacement
        for axis in (0, 1):
            below = self.vehicle_positions[:, axis] < 0.0
            above = self.vehicle_positions[:, axis] > self.area
            self.vehicle_positions[below, axis] *= -1.0
            self.vehicle_positions[above, axis] = 2.0 * self.area - self.vehicle_positions[above, axis]
        self.vehicle_positions = np.clip(self.vehicle_positions, 0.0, self.area)
        self.vehicle_headings += self.rng.normal(0.0, 0.02, self.num_vehicles)

    def _move_uavs(self, actions: np.ndarray, duration_s: float) -> np.ndarray:
        directions = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]])
        propulsion_by_uav = np.zeros(self.num_uavs, dtype=np.float64)
        minimum_separation_m = 50.0
        for idx, action in enumerate(actions):
            if self.uav_battery_j[idx] <= 0.0:
                continue
            delta = directions[action] * self.config.uav_max_step_m
            new_xy = np.clip(self.uav_positions[idx] + delta, 0.0, self.area)
            other = 1 - idx
            if np.linalg.norm(new_xy - self.uav_positions[other]) < minimum_separation_m:
                new_xy = self.uav_positions[idx].copy()
            distance = float(np.linalg.norm(new_xy - self.uav_positions[idx]))
            self.uav_positions[idx] = new_xy
            speed = distance / duration_s
            propulsion_by_uav[idx] = self._propulsion_power_w(speed) * duration_s
        return propulsion_by_uav

    def _propulsion_power_w(self, speed_mps: float) -> float:
        v = float(speed_mps)
        cfg = self.config
        profile = self._p0 * (1.0 + 3.0 * v**2 / self._tip_speed**2)
        induced = self._pi * np.sqrt(
            np.sqrt(1.0 + v**4 / (4.0 * self._v0**4))
            - v**2 / (2.0 * self._v0**2)
        )
        parasite = 0.5 * cfg.fuselage_drag_ratio * cfg.air_density_kg_m3 * cfg.rotor_solidity * self._rotor_area * v**3
        return float(profile + induced + parasite)

    def _rotary_wing_constants(self) -> tuple[float, float, float, float, float]:
        cfg = self.config
        area = np.pi * cfg.rotor_radius_m**2
        tip_speed = cfg.rotor_angular_speed_rad_s * cfg.rotor_radius_m
        weight_n = cfg.uav_mass_kg * 9.81
        p0 = (
            cfg.blade_drag_coefficient
            / 8.0
            * cfg.air_density_kg_m3
            * cfg.rotor_solidity
            * area
            * cfg.rotor_angular_speed_rad_s**3
            * cfg.rotor_radius_m**3
        )
        pi = (
            (1.0 + cfg.induced_power_correction)
            * weight_n**1.5
            / np.sqrt(2.0 * cfg.air_density_kg_m3 * area)
        )
        v0 = np.sqrt(weight_n / (2.0 * cfg.air_density_kg_m3 * area))
        return float(p0), float(pi), float(v0), float(tip_speed), float(area)

    def _compute_energy_by_uav(self, records: list[dict[str, Any]]) -> np.ndarray:
        energies = np.zeros(self.num_uavs, dtype=np.float64)
        for record in records:
            server_id = int(record["server_id"])
            if 0 <= server_id < self.num_uavs:
                energies[server_id] += float(record["uav_compute_energy_j"])
        return energies

    def _observation(self) -> np.ndarray:
        cfg = self.config
        current_task_features = np.array(
            [
                [
                    task["input_bits"] / (cfg.task_size_bytes[1] * 8.0),
                    task["required_cycles"] / cfg.task_cycles[1],
                    task["deadline_s"] / cfg.deadline_s[1],
                ]
                for task in self.current_tasks
            ],
            dtype=np.float32,
        )
        vehicle_features = np.column_stack(
            (
                self.vehicle_positions / cfg.area_m,
                self.vehicle_speeds[:, None] / 33.333333,
                np.sin(self.vehicle_headings)[:, None],
                np.cos(self.vehicle_headings)[:, None],
                current_task_features,
            )
        ).astype(np.float32)
        queue_features = np.clip(
            (self.server_free_at - self.time_s) / max(cfg.deadline_s), 0.0, 10.0
        ).astype(np.float32)
        local_queue_features = np.clip(
            (self.local_free_at - self.time_s) / max(cfg.deadline_s), 0.0, 10.0
        ).astype(np.float32)
        uav_features = np.column_stack(
            (
                self.uav_positions / cfg.area_m,
                self.uav_battery_j / cfg.uav_battery_j,
            )
        ).astype(np.float32)
        return np.concatenate(
            (
                vehicle_features.ravel(),
                local_queue_features,
                queue_features,
                uav_features.ravel(),
                [self.step_index / cfg.episode_steps],
            )
        ).astype(np.float32)


if __name__ == "__main__":
    env = Chapter3IoVEnv(seed=7)
    state = env.reset(seed=7)
    actions = {
        "destinations": np.zeros(env.num_vehicles, dtype=np.int64),
        "uav_movement": np.zeros(env.num_uavs, dtype=np.int64),
    }
    _, reward, _, info = env.step(actions)
    print(f"state_size={state.size} tasks={len(info['task_records'])} reward={reward:.4f}")
