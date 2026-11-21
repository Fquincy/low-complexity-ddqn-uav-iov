"""Resume only missing evaluations after a recorded near-tie export divergence."""
import copy
import json
from pathlib import Path
import traceback

import numpy as np
import torch

import assignment_price_bound_reports as reports
import run_assignment_price_research_bound as runner


ROOT = runner.r.ROOT
RUN = ROOT / "development_runs/assignment_price_development_2026-11-16_01/run"
AMENDMENT = ROOT / "ASSIGNMENT_PRICE_EVALUATION_NUMERICAL_AMENDMENT_2026-11-17.json"
PROTOCOL_SHA256 = "6213fe3e77946fec0b142f020577274084a795807993f140f39af0c7cbae6fef"
CONFIRM = "CONTINUE_FROZEN_ADP_EVALUATION_ONLY"


def episode(agent, seed, diagnostics):
    env = runner.r.Chapter3IoVEnv(seed=seed)
    states, next_states, slots, qs, tqs = [], [], [], [], []
    for slot in range(40):
        state = env._observation().copy()
        states.append(state)
        tasks = runner.r.models.original.existing.base.old.workload(env)
        before = env.uav_positions.copy()
        vehicles = env.vehicle_positions.copy()
        mask = runner.base.adp.allowed(state[None])[0]
        with torch.inference_mode():
            eager_action, eager_q = agent["online"].greedy(torch.from_numpy(state)[None])
            target_q = agent["target"](torch.from_numpy(state)[None], eager_action)
        native_action, native_q = agent["runtime"].values(state)
        np.testing.assert_allclose(eager_q.numpy(), native_q, rtol=1e-5, atol=2e-4)
        action = eager_action[0].numpy().copy()
        if not np.array_equal(eager_action.numpy(), native_action):
            diagnostics.append({
                "seed": seed,
                "slot": slot,
                "changed_branches": np.flatnonzero(action != native_action[0]).astype(int).tolist(),
                "eager_actions": action.tolist(),
                "native_actions": native_action[0].tolist(),
                "eager_q": float(eager_q[0]),
                "native_q": float(native_q[0]),
                "q_absolute_difference": abs(float(eager_q[0]) - float(native_q[0])),
                "state_sha256": runner.r.models.original.existing.base.old.digest(state),
            })
        if agent["definition"]["screened"]:
            assert mask[np.arange(32), action].all()
        following, reward, done, info = env.step(runner.r.to_action_dict(action, 30))
        next_states.append(following.copy())
        assert done == (slot == 39) and np.isfinite(reward) and np.isfinite(following).all()
        after = env.uav_positions.copy()
        slots.append({
            "slot": slot,
            "action": action.tolist(),
            "reward": reward,
            "done": done,
            "reward_components": info["reward_components"],
            "energy_j": info["energy_j"],
            "tasks_arrived": tasks,
            "vehicle_positions": vehicles.tolist(),
            "uav_before": before.tolist(),
            "uav_after": after.tolist(),
            "actual_distance_m": np.linalg.norm(after - before, axis=1).tolist(),
            "excluded_local": (~mask[:30, 0]).tolist(),
        })
        qs.append(float(eager_q[0]))
        tqs.append(float(target_q[0]))
    summary = env.episode_summary()
    rewards = [row["reward"] for row in slots]
    return {
        "seed": seed,
        "run_kind": "RESEARCH_DEVELOPMENT",
        "training": False,
        "policy": None,
        "return_sum": sum(rewards),
        "deadline_success": summary["deadline_success_rate"],
        "tasks_generated": summary["tasks_generated"],
        "tasks_on_time": summary["tasks_on_time"],
        "slot_rewards": rewards,
        "summary": summary,
        "slots": slots,
        "tasks": env.export_task_records(),
        "chosen_q": qs,
        "target_q": tqs,
        "realized_return_to_go": np.cumsum(rewards[::-1])[::-1].tolist(),
        "workload_sha256": runner.r.models.original.existing.base.old.digest([row["tasks_arrived"] for row in slots]),
        "observations": np.stack(states),
        "next_observations": np.stack(next_states),
    }


def preflight():
    runner.r.configure()
    amendment = runner.r.read(AMENDMENT)
    if amendment["protocol_sha256"] != PROTOCOL_SHA256 or runner.r.sha(runner.base.SPEC) != PROTOCOL_SHA256:
        raise RuntimeError("Frozen protocol changed")
    plan = runner.r.read(RUN / "runtime_plan.json")
    if plan != runner.plan("development"):
        raise RuntimeError("Frozen development plan changed")
    if (RUN / "manifest.json").exists() or (RUN / "verification.json").exists():
        raise RuntimeError("Run is already complete")
    checkpoints = runner.r.read(RUN / "ALL_CHECKPOINT_FREEZE.json")
    exports = runner.r.read(RUN / "ALL_EXPORT_FREEZE.json")
    if checkpoints != exports["checkpoints"] or len(checkpoints) != 75 or len(exports["graphs"]) != 75:
        raise RuntimeError("Frozen checkpoint/export coverage changed")
    existing = {}
    complete = 0
    for arm in runner.base.ARMS:
        for fit in range(5):
            folder = RUN / arm / f"fit_{fit:02d}"
            if runner.r.sha(folder / "final.pt") != checkpoints[f"{arm}/{fit:02d}"]:
                raise RuntimeError("Checkpoint changed")
            if runner.r.sha(folder / "final.onnx") != exports["graphs"][f"{arm}/{fit:02d}"]:
                raise RuntimeError("Export changed")
            for episode_index in range(len(plan["evaluation_seeds"])):
                json_path = folder / f"eval_{episode_index:03d}.json.gz"
                npz_path = folder / f"eval_{episode_index:03d}.npz"
                if json_path.exists() != npz_path.exists():
                    raise RuntimeError("Partial saved evaluation pair")
                if json_path.exists():
                    existing[json_path.relative_to(RUN).as_posix()] = runner.r.sha(json_path)
                    existing[npz_path.relative_to(RUN).as_posix()] = runner.r.sha(npz_path)
                    complete += 1
    if complete != 1824:
        raise RuntimeError(f"Unexpected completed learned evaluations: {complete}")
    return plan, checkpoints, exports, existing


def reconstruct_training_metadata(plan):
    counters, storage = {}, {}
    for arm in runner.base.ARMS:
        for fit in range(5):
            folder = RUN / arm / f"fit_{fit:02d}"
            checkpoint = torch.load(folder / "final.pt", weights_only=True, map_location="cpu")
            key = f"{arm}/{fit:02d}"
            first = min(path.stat().st_mtime_ns for path in folder.glob("train_*.json.gz"))
            elapsed = max(1, (folder / "final.pt").stat().st_mtime_ns - first)
            counters[key] = {"slots": checkpoint["slots"], "updates": checkpoint["updates"], "copies": checkpoint["copies"], "training_wall_ns": elapsed}
            agent = runner.build(arm, plan, fit)
            agent["online"].load_state_dict(checkpoint["online"])
            agent["target"].load_state_dict(checkpoint["target"])
            replay = np.load(folder / "replay.npz")
            size = int(replay["size"])
            optimizer = checkpoint["optimizer"]
            storage[key] = {
                "parameter_bytes": sum(t.numel() * t.element_size() for t in agent["online"].parameters()),
                "buffer_bytes": sum(t.numel() * t.element_size() for t in agent["online"].buffers()),
                "target_tensor_bytes": sum(t.numel() * t.element_size() for t in agent["target"].state_dict().values()),
                "optimizer_tensor_bytes": sum(v.numel() * v.element_size() for item in optimizer["state"].values() for v in item.values() if isinstance(v, torch.Tensor)),
                "replay_allocated_tensor_bytes": plan["cfg"]["replay_capacity"] * (2 * 533 * 4 + 32 + 8),
                "replay_populated_tensor_bytes": size * (2 * 533 * 4 + 32 + 8),
                "checkpoint_file_bytes": (folder / "final.pt").stat().st_size,
                "replay_file_bytes": (folder / "replay.npz").stat().st_size,
                "serialized_graph_bytes": (folder / "final.onnx").stat().st_size,
            }
    return counters, storage


def scan_deployment_equivalence(plan):
    action_disagreements, original_q_exceedances = [], []
    states_checked = 0
    maximum_q_absolute_difference = 0.0
    for arm in runner.base.ARMS:
        for fit in range(5):
            folder = RUN / arm / f"fit_{fit:02d}"
            agent, _ = runner.load_agent(RUN, arm, fit, plan)
            agent["runtime"] = runner.from_graph(agent, folder / "final.onnx", arm)
            for episode_index, seed in enumerate(plan["evaluation_seeds"]):
                arrays = np.load(folder / f"eval_{episode_index:03d}.npz")
                for slot, state in enumerate(arrays["states"]):
                    with torch.inference_mode():
                        eager_action, eager_q = agent["online"].greedy(torch.from_numpy(state)[None])
                    native_action, native_q = agent["runtime"].values(state)
                    np.testing.assert_allclose(eager_q.numpy(), native_q, rtol=1e-5, atol=2e-4)
                    absolute = abs(float(eager_q[0]) - float(native_q[0]))
                    maximum_q_absolute_difference = max(maximum_q_absolute_difference, absolute)
                    common = {"arm": arm, "fit": fit, "episode_index": episode_index, "seed": seed, "slot": slot,
                              "eager_q": float(eager_q[0]), "native_q": float(native_q[0]), "q_absolute_difference": absolute,
                              "state_sha256": runner.r.models.original.existing.base.old.digest(state)}
                    if not np.array_equal(eager_action.numpy(), native_action):
                        action_disagreements.append({**common,
                            "changed_branches": np.flatnonzero(eager_action.numpy()[0] != native_action[0]).astype(int).tolist(),
                            "eager_actions": eager_action.numpy()[0].tolist(), "native_actions": native_action[0].tolist()})
                    if not np.allclose(eager_q.numpy(), native_q, rtol=1e-5, atol=1e-6):
                        original_q_exceedances.append(common)
                    states_checked += 1
    return {"policy_evaluation_route": "frozen eager DDQN", "deployment_route": "frozen exported runtime",
            "states_checked": states_checked, "exact_action_disagreements": action_disagreements,
            "exact_action_disagreement_count": len(action_disagreements), "exact_action_equivalence_passed": not action_disagreements,
            "original_q_tolerance": {"rtol": 1e-5, "atol": 1e-6},
            "original_q_tolerance_exceedances": original_q_exceedances,
            "original_q_tolerance_exceedance_count": len(original_q_exceedances),
            "continuation_guard": {"rtol": 1e-5, "atol": 2e-4},
            "maximum_q_absolute_difference": maximum_q_absolute_difference}


def run(confirmation=None):
    if confirmation != CONFIRM:
        raise RuntimeError("Explicit evaluation-only continuation confirmation required")
    plan, checkpoints, exports, existing = preflight()
    runner.r.write(RUN / "PRE_EVALUATION_CONTINUATION_FREEZE_V2.json", {
        "amendment_sha256": runner.r.sha(AMENDMENT),
        "checkpoint_freeze_sha256": runner.r.sha(RUN / "ALL_CHECKPOINT_FREEZE.json"),
        "export_freeze_sha256": runner.r.sha(RUN / "ALL_EXPORT_FREEZE.json"),
        "existing_evaluation_files": existing,
        "completed_learned_episodes": 1824,
        "remaining_learned_episodes": 1776,
        "new_training_slots": 0,
        "new_optimizer_updates": 0,
    })
    events = runner.r.read(RUN / "events.json")
    def event(stage, **fields):
        events.append({"sequence": len(events), "stage": stage, **fields})
        runner.r.write(RUN / "events.json", events)
        runner.r.write(RUN / "progress.json", events[-1])
    diagnostics = []
    try:
        event("evaluation_only_continuation_v2_started", existing_learned_episodes=1824, new_training_slots=0, new_optimizer_updates=0)
        jobs = [tuple(row) for row in runner.r.read(RUN / "job_order.json")]
        for arm, fit in jobs:
            folder = RUN / arm / f"fit_{fit:02d}"
            missing = [index for index in range(len(plan["evaluation_seeds"])) if not (folder / f"eval_{index:03d}.json.gz").exists()]
            if not missing:
                continue
            event("evaluation_continuation", arm=arm, fit=fit, missing_episodes=len(missing))
            agent, checkpoint = runner.load_agent(RUN, arm, fit, plan)
            agent["runtime"] = runner.from_graph(agent, folder / "final.onnx", arm)
            before = runner.r.models.original.existing.base.old.digest(agent["online"].state_dict())
            for episode_index in missing:
                seed = plan["evaluation_seeds"][episode_index]
                episode_diagnostics = []
                row = episode(agent, seed, episode_diagnostics)
                row["run_kind"] = plan["kind"]
                runner.r.save_episode(folder / f"eval_{episode_index:03d}.json.gz", row)
                for diagnostic in episode_diagnostics:
                    diagnostics.append({"arm": arm, "fit": fit, "episode_index": episode_index, **diagnostic})
            assert before == runner.r.models.original.existing.base.old.digest(agent["online"].state_dict())
            assert runner.r.sha(folder / "final.pt") == checkpoints[f"{arm}/{fit:02d}"]
        for policy in runner.base.HEURISTICS:
            folder = RUN / "heuristics" / policy
            folder.mkdir(parents=True, exist_ok=True)
            missing = [index for index in range(len(plan["evaluation_seeds"])) if not (folder / f"eval_{index:03d}.json.gz").exists()]
            if missing:
                event("heuristic_evaluation_continuation", policy=policy, missing_episodes=len(missing))
            for episode_index in missing:
                row = runner.base.episode(None, plan["evaluation_seeds"][episode_index], policy=policy)
                row["run_kind"] = plan["kind"]
                runner.r.save_episode(folder / f"eval_{episode_index:03d}.json.gz", row)
        for name, digest in existing.items():
            if runner.r.sha(RUN / name) != digest:
                raise RuntimeError("Existing evaluation artifact changed: " + name)
        counters, storage = reconstruct_training_metadata(plan)
        runner.r.write(RUN / "counters.json", counters)
        runner.r.write(RUN / "storage.json", storage)
        equivalence = scan_deployment_equivalence(plan)
        equivalence["continuation_observed_action_disagreements"] = diagnostics
        runner.r.write(RUN / "deployment_equivalence_diagnostics.json", equivalence)
        reports.report(RUN, runner.base.protocol(), plan)
        event("complete", research_launched=True, evaluation_continuation=True, new_training_slots=0, new_optimizer_updates=0)
        runner.r.write(RUN / "manifest.json", {path.relative_to(RUN).as_posix(): runner.r.sha(path) for path in RUN.rglob("*") if path.is_file()})
        return {"status": "EVALUATION_COMPLETE", "action_disagreements": equivalence["exact_action_disagreement_count"],
                "original_q_tolerance_exceedances": equivalence["original_q_tolerance_exceedance_count"],
                "new_training_slots": 0, "new_optimizer_updates": 0}
    except Exception:
        runner.r.write(RUN / "EVALUATION_CONTINUATION_V2_FAILED.json", {"error": traceback.format_exc(), "new_training_slots": 0, "new_optimizer_updates": 0})
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--confirm-evaluation-continuation")
    args = parser.parse_args()
    if args.check_only:
        plan, checkpoints, exports, existing = preflight()
        print(json.dumps({"passed": True, "checkpoints": len(checkpoints), "exports": len(exports["graphs"]), "existing_evaluation_files": len(existing), "new_training_slots": 0}, indent=2))
    else:
        print(json.dumps(run(args.confirm_evaluation_continuation), indent=2))
