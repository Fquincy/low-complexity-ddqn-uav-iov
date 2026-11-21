"""Time full-slot untrained selection paths without stepping physical simulation."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
from types import SimpleNamespace
import time

import numpy as np
import torch

from chapter3_sequential_conditional import (
    ORDER, UNKNOWN, _physical, augmented, ConditionalQNetwork, InferenceCache,
    SequentialActor, SequentialIoVWrapper,
)
from train_chapter3_ddqn import FactorizedQNetwork, to_action_dict

ROOT = Path(__file__).resolve().parent
SPEC = ROOT/"CHAPTER3_SEQUENTIAL_TIMING_SPEC_2026-11-01.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def weight_hash(model):
    return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()


def cpu_name():
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor() or "unreported"


class SelectionFrame:
    """Policy-only adapter: prefix transitions match wrapper, no physical env.step."""
    def __init__(self):
        self.env = SimpleNamespace(step_index=0)
        self.serial = 0

    def start(self, physical):
        self.physical_observation = _physical(physical)
        self.prefix = np.full(32, UNKNOWN, dtype=np.int64)
        self.stage = 0
        self.env.step_index = int(round(float(physical[-1])*40))
        self.serial += 1

    @property
    def done(self):
        return self.stage == 32

    @property
    def cache_token(self):
        return (id(self), self.serial, self.env.step_index)

    def observation(self):
        return augmented(self.physical_observation, self.stage, self.prefix)

    def accept(self, choice):
        self.prefix[ORDER[self.stage]] = choice
        self.stage += 1
        if self.stage < 32:
            # Runtime wrapper constructs/returns this next context, even if discarded.
            self.observation()


class UncachedInference(InferenceCache):
    def values(self, observation, slot_token):
        self.invalidate()
        return super().values(observation, slot_token)


class SequentialSelection:
    def __init__(self, model, cached=True):
        self.model = model
        self.actor = SequentialActor(model, seed=0)
        if not cached:
            self.actor.cache = UncachedInference(model)
        self.frame = SelectionFrame()

    def select(self, observation):
        self.frame.start(observation)
        for _ in range(32):
            self.frame.accept(self.actor.act(self.frame, explore=False))
        return to_action_dict(self.frame.prefix.copy(), 30)


class AdditiveSelection:
    def __init__(self, model):
        self.model = model

    def select(self, observation):
        with torch.inference_mode():
            vector = self.model(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0))[0].argmax(1).numpy()
        return to_action_dict(vector, 30)


def vector(action):
    return np.concatenate((action["destinations"], action["uav_movement"]))


def validate_paths(observations, conditional, additive):
    # Independent live wrapper alignment for every stage of one selection.
    wrapper = SequentialIoVWrapper(seed=9700000)
    frame = SelectionFrame()
    frame.start(wrapper.physical_observation)
    live_actor = SequentialActor(conditional, seed=0)
    frame_actor = SequentialActor(conditional, seed=0)
    for stage in range(32):
        assert np.array_equal(wrapper.observation(), frame.observation())
        a, b = live_actor.act(wrapper), frame_actor.act(frame)
        assert a == b
        wrapper.step(a)
        frame.accept(b)
    assert np.array_equal(frame.prefix, wrapper.last_physical_record.actions)
    refs = {name: [] for name in ("additive", "sequential_cached", "sequential_uncached")}
    baseline = AdditiveSelection(additive)
    largest_error = 0.
    cached_encoders = cached_heads = uncached_encoders = uncached_heads = 0
    for i, observation in enumerate(observations):
        cached = InferenceCache(conditional)
        uncached = UncachedInference(conditional)
        pf = np.full(32, UNKNOWN, dtype=np.int64)
        cf = np.full(32, UNKNOWN, dtype=np.int64)
        for stage in range(32):
            c = cached.values(augmented(observation, stage, pf), (i, "cached"))
            u = uncached.values(augmented(observation, stage, cf), (i, "uncached"))
            largest_error = max(largest_error, float(np.max(np.abs(c-u))))
            assert np.allclose(c, u, rtol=1e-5, atol=1e-6)
            pf[ORDER[stage]], cf[ORDER[stage]] = int(c.argmax()), int(u.argmax())
        assert np.array_equal(pf, cf)
        refs["additive"].append(vector(baseline.select(observation)))
        refs["sequential_cached"].append(pf)
        refs["sequential_uncached"].append(cf)
        cached_encoders += cached.encoder_calls
        cached_heads += cached.head_calls
        uncached_encoders += uncached.encoder_calls
        uncached_heads += uncached.head_calls
    assert cached_encoders == len(observations) and cached_heads == len(observations)*32
    assert uncached_encoders == uncached_heads == len(observations)*32
    return {k: np.stack(v) for k, v in refs.items()}, {"frame_matches_live_wrapper_all32_stages": True, "cached_uncached_all_observation_values_and_actions_match": True, "max_cached_uncached_q_difference": largest_error, "reference_encoder_calls_cached": cached_encoders, "reference_encoder_calls_uncached": uncached_encoders, "reference_head_calls_each": cached_heads}


def benchmark(out):
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    gate = ROOT/"validation/chapter3_sequential_mechanics_2026-11-01_01/verification.json"
    mechanics = json.loads(gate.read_text())
    assert mechanics["state"] == "passed"
    assert sha(ROOT/"chapter3_sequential_conditional.py") == mechanics["source_sha256"]["chapter3_sequential_conditional.py"]
    source = ROOT/spec["observation_source"]
    provenance = json.loads((source.parent/"DATA_PROVENANCE.json").read_text())
    assert sha(source) == provenance["fresh_data_sha256"]
    with np.load(source) as z:
        indices = np.linspace(0, len(z["states"])-1, spec["observation_count"], dtype=np.int64)
        observations = z["states"][indices].copy()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(spec["model_seed"])
    conditional = ConditionalQNetwork().eval()
    torch.manual_seed(spec["model_seed"])
    additive = FactorizedQNetwork(533, 32, 5, (128, 128)).eval()
    initial_hashes = {"conditional": weight_hash(conditional), "additive": weight_hash(additive)}
    out.mkdir(parents=True, exist_ok=False)
    def status(state, **fields):
        data = {"state": state, "updated_at": datetime.now().astimezone().isoformat(), **fields}
        save(out/"status.json", data)
        (out/"LIVE_STATUS.txt").write_text(json.dumps(data, indent=2), encoding="utf-8")
    status("validating_paths")
    try:
        snap = out/"source_snapshot"
        snap.mkdir()
        names = [Path(__file__).name, SPEC.name, SPEC.with_suffix(".txt").name, "chapter3_sequential_conditional.py", "train_chapter3_ddqn.py", "chapter3_iov_env.py", "chapter3_iov_env_v2.py"]
        for name in names:
            shutil.copy2(ROOT/name, snap/name)
        np.savez_compressed(out/"observations.npz", observations=observations, source_indices=indices)
        torch.save({"conditional": conditional.state_dict(), "additive": additive.state_dict(), "untrained": True, "seed": spec["model_seed"]}, out/"untrained_timing_models.pt")
        refs, checks = validate_paths(observations, conditional, additive)
        np.savez_compressed(out/"reference_actions.npz", **refs)
        variants = {"additive": AdditiveSelection(additive), "sequential_cached": SequentialSelection(conditional, True), "sequential_uncached": SequentialSelection(conditional, False)}
        order_rng = np.random.default_rng(spec["order_rng_seed"])
        order, rows = [], []
        for round_id in range(spec["rounds"]):
            names_this_round = list(spec["variants"])
            order_rng.shuffle(names_this_round)
            order.append(names_this_round.copy())
            for name in names_this_round:
                status("timing", round=round_id, variant=name)
                selection = variants[name]
                for i in range(spec["warmup_full_slots_per_variant_per_round"]):
                    selection.select(observations[i % len(observations)])
                for i in range(spec["timed_full_slots_per_variant_per_round"]):
                    obs_index = i % len(observations)
                    start = time.perf_counter_ns()
                    selected = selection.select(observations[obs_index])
                    elapsed = time.perf_counter_ns()-start
                    # Checks and log accumulation occur after the stopped clock.
                    assert np.array_equal(vector(selected), refs[name][obs_index])
                    rows.append({"round": round_id, "variant": name, "call": i, "observation_index": obs_index, "nanoseconds": elapsed})
        total_per_variant = spec["rounds"]*(spec["warmup_full_slots_per_variant_per_round"]+spec["timed_full_slots_per_variant_per_round"])
        counts = {}
        for name in ("sequential_cached", "sequential_uncached"):
            actor = variants[name].actor
            expected_encoders = total_per_variant*(1 if name == "sequential_cached" else 32)
            assert actor.cache.encoder_calls == expected_encoders
            assert actor.cache.head_calls == total_per_variant*32
            assert actor.physical_slots_started == total_per_variant and actor.staged_choices == total_per_variant*32
            assert actor.exploration_slots == 0
            counts[name] = {"full_slots_including_warmup": total_per_variant, "encoder_calls": actor.cache.encoder_calls, "head_calls": actor.cache.head_calls}
        assert initial_hashes == {"conditional": weight_hash(conditional), "additive": weight_hash(additive)}
        with gzip.open(out/"timings.csv.gz", "wt", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = []
        for name in spec["variants"]:
            for round_id in ["pooled"]+list(range(spec["rounds"])):
                ns = np.array([r["nanoseconds"] for r in rows if r["variant"] == name and (round_id == "pooled" or r["round"] == round_id)], dtype=np.float64)
                summary.append({"variant": name, "round": round_id, "calls": len(ns), "median_ms": float(np.median(ns)/1e6), "p95_ms": float(np.quantile(ns, .95)/1e6), "p99_ms": float(np.quantile(ns, .99)/1e6), "minimum_ms": float(ns.min()/1e6), "maximum_ms": float(ns.max()/1e6)})
        save(out/"summary.json", summary)
        hardware = {"cpu": cpu_name(), "logical_cpus": os.cpu_count(), "platform": platform.platform(), "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "clock": vars(time.get_clock_info("perf_counter"))}
        save(out/"manifest.json", {"spec": spec, "hardware": hardware, "source_sha256": {n: sha(snap/n) for n in names}, "source_observation_sha256": sha(source), "observations_sha256": sha(out/"observations.npz"), "randomized_order": order, "model_weight_sha256": initial_hashes, "untrained_model_file_sha256": sha(out/"untrained_timing_models.pt"), "reference_action_sha256": sha(out/"reference_actions.npz"), "models_untrained": True, "all_timed_actions_match_reference": True, "path_validation": checks, "call_counts": counts, "memory_scope": {"conditional_online_parameter_bytes": sum(p.numel()*p.element_size() for p in conditional.parameters()), "additive_online_parameter_bytes": sum(p.numel()*p.element_size() for p in additive.parameters()), "cached_encoder_embedding_bytes": 128*4, "peak_process_memory_measured": False}, "neural_optimizer_updates": 0, "physical_env_step_inside_timed_interval": False, "low_complexity_or_realtime_claim": False})
        status("complete", timed_full_slots=len(rows), per_variant_measured=spec["rounds"]*spec["timed_full_slots_per_variant_per_round"], neural_optimizer_updates=0)
        print(json.dumps({"state": "complete", "output": str(out), "pooled": [r for r in summary if r["round"] == "pooled"]}, indent=2))
    except Exception as exc:
        status("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    benchmark(p.parse_args().output.resolve())
