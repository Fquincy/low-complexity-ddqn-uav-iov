"""Event-driven, slotted Chapter 3 simulator; no projected task completions.

V1 is retained for historical diagnostics. Only its configuration defaults,
task generator, geometry, and propulsion helpers are reused here.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from chapter3_iov_env import (
    Chapter3Config as V1Config, Chapter3IoVEnv as V1Environment,
    LOCAL, UAV_1, UAV_2, RSU_1, RSU_2,
)

MODEL_VERSION = "chapter3-timeline-v2"
TIME_EPS = 1e-10
BITS_EPS = 1e-5
CYCLES_EPS = 1e-3


@dataclass(frozen=True)
class Chapter3Config(V1Config):
    model_version: str = MODEL_VERSION
    radio_update_interval_s: float = 0.05
    horizon_policy: str = "stop_and_report_unfinished"
    uplink_scheduler: str = "one_fifo_transmitter_per_vehicle"
    observation_model: str = "compressed_partial_observation"


class Chapter3IoVEnv(V1Environment):
    """Rates are held at most radio_update_interval_s between events.

    CPU FCFS is by actual upload completion, with task-id tie breaking.
    Vehicles have independent local CPUs and one FIFO uplink transmitter.
    Several vehicles may transmit simultaneously; unfinished uploads remain
    active interferers. RSU links retain the provisional independent fixed rate.
    """

    def __init__(self, config: Chapter3Config | None = None, seed: int | None = None):
        config = config or Chapter3Config()
        if config.model_version != MODEL_VERSION:
            raise ValueError("unexpected model version")
        if config.num_vehicles <= 0 or config.episode_steps <= 0:
            raise ValueError("positive vehicle and slot counts required")
        positive = (config.slot_s, config.radio_update_interval_s, config.vehicle_cpu_hz,
                    *config.uav_cpu_hz, *config.rsu_cpu_hz, config.rsu_link_bps,
                    config.deadline_normalizer_s, config.uav_battery_j,
                    config.g2a_bandwidth_hz)
        if any(not np.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("time intervals, capacities, scales and batteries must be positive")
        if config.horizon_policy != "stop_and_report_unfinished":
            raise ValueError("unsupported horizon policy")
        if config.uplink_scheduler != "one_fifo_transmitter_per_vehicle":
            raise ValueError("unsupported uplink scheduler")
        if config.g2a_allocation_mode not in ("per_link_with_interference", "global_equal_share"):
            raise ValueError("unsupported radio mode")
        super().__init__(config, seed)

    @property
    def state_size(self):
        # Current vehicle/task features, local workloads, radio summaries,
        # remote workload/count/oldest-age summaries, UAVs, and time.
        return 17*self.num_vehicles + 23

    def reset(self, seed=None):
        self.upload_queues = [deque() for _ in range(self.config.num_vehicles)]
        self.cpu_queues = [deque() for _ in range(self.config.num_vehicles+4)]
        self.records: list[dict[str, Any]] = []
        self.task_records = self.records
        self.tasks_generated = 0
        self.tasks_completed = 0
        self.deadline_misses = 0
        self.energy_totals = {k: 0.0 for k in (
            "uav_compute", "uav_propulsion", "vehicle_compute", "vehicle_uplink")}
        self._slot_delay_area_s = 0.0
        self._done = False
        self._completed_this_step = []
        state = super().reset(seed)
        self.task_records = self.records
        return state

    def _admit(self, task, destination):
        server = destination-1 if destination else -1
        record = {**task, "destination": int(destination), "server_id": server,
                  "remaining_bits": float(task["input_bits"]) if destination else 0.0,
                  "remaining_cycles": float(task["required_cycles"]),
                  "status": "uploading" if destination else "queued_compute",
                  "upload_start_s": None, "upload_finish_s": None,
                  "compute_start_s": None, "completion_s": None, "delay_s": None,
                  "transmission_delay_s": None, "uplink_wait_s": None,
                  "queue_delay_s": None, "computation_delay_s": None,
                  "on_time": False, "deadline_missed": False,
                  "uav_compute_energy_j": 0.0, "vehicle_compute_energy_j": 0.0,
                  "vehicle_uplink_energy_j": 0.0, "radio_active_s": 0.0,
                  "uploaded_bits": 0.0, "executed_cycles": 0.0}
        self.records.append(record)
        self.tasks_generated += 1
        if destination == LOCAL:
            record.update(upload_finish_s=float(task["arrival_s"]),
                          transmission_delay_s=0.0, uplink_wait_s=0.0)
            self.cpu_queues[int(task["vehicle_id"])].append(record)
        else:
            self.upload_queues[int(task["vehicle_id"])].append(record)

    def _cpu_frequency(self, index):
        if index < self.num_vehicles:
            return self.config.vehicle_cpu_hz
        server = index-self.num_vehicles
        if server < 2:
            return self.config.uav_cpu_hz[server] if self.uav_battery_j[server] > 0 else 0.0
        return self.config.rsu_cpu_hz[server-2]

    def _resolve_events(self):
        # Finish existing CPU jobs first. New upload arrivals at this timestamp
        # are then appended together, before any idle CPU is restarted.
        for queue in self.cpu_queues:
            while queue and queue[0]["remaining_cycles"] <= CYCLES_EPS:
                r = queue.popleft()
                r.update(status="completed", completion_s=self.time_s,
                         delay_s=self.time_s-r["arrival_s"],
                         computation_delay_s=self.time_s-r["compute_start_s"])
                r["on_time"] = r["delay_s"] <= r["deadline_s"]+TIME_EPS
                self.tasks_completed += 1
                self.tasks_on_time += int(r["on_time"])
                self._completed_this_step.append(r)
        arrivals = []
        for queue in self.upload_queues:
            if queue and queue[0]["remaining_bits"] <= BITS_EPS:
                r = queue.popleft()
                r.update(status="queued_compute", upload_finish_s=self.time_s,
                         transmission_delay_s=self.time_s-r["upload_start_s"])
                arrivals.append(r)
        for r in sorted(arrivals, key=lambda x: int(x["task_id"])):
            self.cpu_queues[self.num_vehicles+r["server_id"]].append(r)
        for i, queue in enumerate(self.cpu_queues):
            if queue and self._cpu_frequency(i) > 0:
                r = queue[0]
                if r["compute_start_s"] is None:
                    r.update(status="computing", compute_start_s=self.time_s,
                             queue_delay_s=self.time_s-r["upload_finish_s"])
        for queue in self.upload_queues:
            if queue and queue[0]["upload_start_s"] is None:
                r = queue[0]
                r.update(upload_start_s=self.time_s,
                         uplink_wait_s=self.time_s-r["arrival_s"])

    def _active_upload_rates(self):
        heads = [q[0] for q in self.upload_queues if q]
        groups = [[r for r in heads if r["destination"] == d and self.uav_battery_j[d-1] > 0]
                  for d in (UAV_1, UAV_2)]
        count = sum(map(len, groups))
        bandwidth = self.config.g2a_bandwidth_hz
        if self.config.g2a_allocation_mode == "global_equal_share":
            bandwidth /= max(1, count)
        rates = {}
        noise = 10**((self.config.noise_dbm-30)/10)
        for u, group in enumerate(groups):
            gains = [self._g2a_channel_gain(self.vehicle_positions[r["vehicle_id"]],
                                          self.uav_positions[u]) for r in group]
            for i, r in enumerate(group):
                interference = (self.config.vehicle_tx_power_w * sum(g for j,g in enumerate(gains) if j != i)
                                if self.config.g2a_allocation_mode == "per_link_with_interference" else 0.0)
                sinr = self.config.vehicle_tx_power_w*gains[i]/(noise+interference)
                rates[r["task_id"]] = float(bandwidth*np.log1p(sinr)/np.log(2.0))
        for r in heads:
            if r["destination"] in (RSU_1, RSU_2):
                rates[r["task_id"]] = self.config.rsu_link_bps
        return [(r, rates.get(r["task_id"], 0.0)) for r in heads]

    def _movement_plan(self, movements):
        start = self.uav_positions.copy()
        directions = np.array([[0,0], [0,1], [1,0], [0,-1], [-1,0]])
        target = np.clip(start+directions[movements]*self.config.uav_max_step_m, 0, self.area)
        target[self.uav_battery_j <= 0] = start[self.uav_battery_j <= 0]
        relative = start[0]-start[1]
        change = (target[0]-start[0])-(target[1]-start[1])
        fraction = float(np.clip(-np.dot(relative, change)/np.dot(change, change), 0, 1)) if np.dot(change, change) else 0
        if np.linalg.norm(relative+fraction*change) < 50.0-TIME_EPS:
            target = start.copy()  # deterministic conservative rejection
        speeds = np.linalg.norm(target-start, axis=1)/self.config.slot_s
        powers = np.array([self._propulsion_power_w(v) for v in speeds])
        return start, target, powers

    def _advance(self, end, plan):
        start_xy, target_xy, propulsion_power = plan
        slot_start = self.time_s
        while self.time_s < end-TIME_EPS:
            self._resolve_events()
            links = self._active_upload_rates()
            cpus = [(i,q[0],self._cpu_frequency(i)) for i,q in enumerate(self.cpu_queues) if q]
            uav_power = propulsion_power*(self.uav_battery_j > 0)
            for i, _, frequency in cpus:
                if self.num_vehicles <= i < self.num_vehicles+2:
                    uav_power[i-self.num_vehicles] += self.config.effective_switched_capacitance*frequency**3
            duration = min(end-self.time_s, self.config.radio_update_interval_s)
            for r, rate in links:
                if rate > 0:
                    duration = min(duration, r["remaining_bits"]/rate)
            for _, r, frequency in cpus:
                if frequency > 0:
                    duration = min(duration, r["remaining_cycles"]/frequency)
            for u in range(2):
                if uav_power[u] > 0:
                    duration = min(duration, self.uav_battery_j[u]/uav_power[u])
            if not np.isfinite(duration) or duration <= 0:
                raise RuntimeError("event engine failed to advance")
            self._slot_delay_area_s += sum(r["status"] != "completed" for r in self.records)*duration
            for r, rate in links:
                bits = min(r["remaining_bits"], rate*duration)
                r["remaining_bits"] -= bits
                r["uploaded_bits"] += bits
                if rate > 0:
                    r["radio_active_s"] += duration
                    # RSU transmit power is not specified in v1; its energy
                    # remains explicitly unmodelled, rather than fabricated.
                    energy = self.config.vehicle_tx_power_w*duration if r["destination"] in (UAV_1,UAV_2) else 0.0
                    r["vehicle_uplink_energy_j"] += energy
                    self.last_energy_j["vehicle_uplink"] += energy
            for i, r, frequency in cpus:
                cycles = min(r["remaining_cycles"], frequency*duration)
                r["remaining_cycles"] -= cycles
                r["executed_cycles"] += cycles
                if i < self.num_vehicles:
                    energy = self.config.vehicle_effective_switched_capacitance*frequency**2*cycles
                    r["vehicle_compute_energy_j"] += energy
                    self.last_energy_j["vehicle_compute"] += energy
                elif i < self.num_vehicles+2:
                    energy = self.config.effective_switched_capacitance*frequency**2*cycles
                    r["uav_compute_energy_j"] += energy
                    self.last_energy_j["uav_compute"] += energy
            alive = self.uav_battery_j > 0
            self.last_energy_j["uav_propulsion"] += float(np.sum(propulsion_power*alive)*duration)
            used = uav_power*duration
            self.uav_battery_j = np.maximum(0.0, self.uav_battery_j-used)
            self.uav_battery_j[self.uav_battery_j < 1e-8] = 0.0
            self.time_s += duration
            fraction = min(1.0, (self.time_s-slot_start)/self.config.slot_s)
            self.uav_positions[alive] = (start_xy+(target_xy-start_xy)*fraction)[alive]
        self.time_s = end
        self._resolve_events()

    def step(self, actions):
        if self._done:
            raise RuntimeError("episode has ended; reset before stepping")
        destinations = np.asarray(actions["destinations"])
        movements = np.asarray(actions["uav_movement"])
        if (destinations.shape != (self.num_vehicles,) or movements.shape != (2,)
            or not np.all(np.isfinite(destinations)) or not np.all(np.isfinite(movements))
            or np.any(destinations != np.floor(destinations)) or np.any(movements != np.floor(movements))
            or np.any((destinations < 0)|(destinations > 4)) or np.any((movements < 0)|(movements > 4))):
            raise ValueError("integer five-way destination and movement arrays required")
        self.last_energy_j = {k: 0.0 for k in self.energy_totals}
        self._completed_this_step = []
        self._slot_delay_area_s = 0.0
        for task in sorted(self.current_tasks, key=lambda r: r["task_id"]):
            self._admit(task, int(destinations[int(task["vehicle_id"])]))
        self._advance(self.time_s+self.config.slot_s, self._movement_plan(movements.astype(int)))
        new_misses = 0
        for r in self.records:
            if (not r["deadline_missed"] and not r["on_time"]
                and self.time_s >= r["arrival_s"]+r["deadline_s"]-TIME_EPS):
                r["deadline_missed"] = True
                new_misses += 1
        self.deadline_misses += new_misses
        self.tasks_processed = self.tasks_completed
        for k in self.energy_totals:
            self.energy_totals[k] += self.last_energy_j[k]
        components = {
            "normalized_delay_accrual": self._slot_delay_area_s/self.num_vehicles/self.config.deadline_normalizer_s,
            "normalized_uav_energy": (self.last_energy_j["uav_compute"]+self.last_energy_j["uav_propulsion"])/self.config.uav_energy_normalizer_j,
            "new_deadline_misses_per_vehicle": new_misses/self.num_vehicles,
        }
        reward = (-self.config.delay_reward_weight*components["normalized_delay_accrual"]
                  -self.config.uav_energy_reward_weight*components["normalized_uav_energy"]
                  -self.config.deadline_miss_penalty*components["new_deadline_misses_per_vehicle"])
        self.step_index += 1
        self._done = self.step_index >= self.config.episode_steps
        if not self._done:
            self._move_vehicles(self.config.slot_s)
            self.current_tasks = self._new_tasks()
        else:
            self.current_tasks = []
        info = {"model_version": MODEL_VERSION, "task_records": [dict(r) for r in self._completed_this_step],
                "tasks_generated": self.tasks_generated, "tasks_completed": self.tasks_completed,
                "tasks_unfinished": self.tasks_generated-self.tasks_completed,
                "deadline_success_rate": self.tasks_on_time/max(1,self.tasks_generated),
                "new_deadline_misses": new_misses, "energy_j": dict(self.last_energy_j),
                "reward_components": components, "uav_battery_j": self.uav_battery_j.copy()}
        self.last_step_records = info["task_records"]
        return self._observation(), float(reward), self._done, info

    def export_task_records(self):
        return [{**r, "unfinished_at_horizon": self._done and r["status"] != "completed",
                 "observed_elapsed_s": (r["completion_s"] if r["completion_s"] is not None else self.time_s)-r["arrival_s"]}
                for r in self.records]

    def episode_summary(self):
        completed = [r for r in self.records if r["status"] == "completed"]
        delays = [r["delay_s"] for r in completed]
        all_rows = self.export_task_records()
        return {"model_version": MODEL_VERSION, "tasks_generated": self.tasks_generated,
                "tasks_completed": self.tasks_completed,
                "tasks_unfinished": self.tasks_generated-self.tasks_completed,
                "tasks_on_time": self.tasks_on_time, "deadline_misses_observed": self.deadline_misses,
                "deadline_success_rate": self.tasks_on_time/max(1,self.tasks_generated),
                "mean_completed_delay_s": float(np.mean(delays)) if delays else None,
                "p95_completed_delay_s": float(np.quantile(delays,0.95)) if delays else None,
                "restricted_mean_elapsed_s": float(np.mean([r["observed_elapsed_s"] for r in all_rows])) if all_rows else None,
                **{k+"_energy_j": v for k,v in self.energy_totals.items()},
                "uav_battery_remaining_min_j": float(self.uav_battery_j.min())}

    def _observation(self):
        cfg = self.config
        features = np.zeros((self.num_vehicles,8))
        features[:,:2] = self.vehicle_positions/cfg.area_m
        features[:,2] = self.vehicle_speeds/33.333333
        features[:,3] = np.sin(self.vehicle_headings)
        features[:,4] = np.cos(self.vehicle_headings)
        for r in self.current_tasks:
            features[r["vehicle_id"],5:] = (r["input_bits"]/(cfg.task_size_bytes[1]*8),
                                          r["required_cycles"]/cfg.task_cycles[1],
                                          r["deadline_s"]/cfg.deadline_s[1])
        workloads = [np.log1p(sum(r["remaining_cycles"] for r in q)/f/cfg.deadline_normalizer_s)
                     for q,f in zip(self.cpu_queues,[cfg.vehicle_cpu_hz]*self.num_vehicles+list(cfg.uav_cpu_hz)+list(cfg.rsu_cpu_hz))]
        radio = np.zeros((self.num_vehicles,8))
        for v, q in enumerate(self.upload_queues):
            for r in q:
                radio[v,r["destination"]-1] += r["remaining_bits"]/(cfg.task_size_bytes[1]*8)
            radio[v,:4] = np.log1p(radio[v,:4])
            if q:
                h = q[0]
                radio[v,4:] = (h["destination"]/4, h["remaining_bits"]/(cfg.task_size_bytes[1]*8),
                               np.log1p(max(0,self.time_s-h["arrival_s"])/cfg.deadline_normalizer_s), np.log1p(len(q)))
        remote = []
        for q in self.cpu_queues[self.num_vehicles:]:
            remote.extend((np.log1p(len(q)), np.log1p(max(0,self.time_s-q[0]["arrival_s"])/cfg.deadline_normalizer_s) if q else 0,
                           float(self.uav_battery_j[len(remote)//3] > 0) if len(remote)//3 < 2 else 1.0))
        uavs = np.column_stack((self.uav_positions/cfg.area_m,self.uav_battery_j/cfg.uav_battery_j))
        state = np.concatenate((features.ravel(),workloads,radio.ravel(),remote,uavs.ravel(),[self.step_index/cfg.episode_steps])).astype(np.float32)
        if state.shape != (self.state_size,) or not np.all(np.isfinite(state)):
            raise RuntimeError("invalid observation")
        return state
