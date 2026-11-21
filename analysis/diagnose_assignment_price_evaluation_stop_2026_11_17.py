"""Reproduce the stopped evaluation without training or modifying run evidence."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import assignment_price_all_reference_deployment as deployment
import run_assignment_price_research_bound as runner


ROOT = runner.r.ROOT
RUN = ROOT / "development_runs/assignment_price_development_2026-11-16_01/run"
OUT = ROOT / "validation/assignment_price_evaluation_stop_2026-11-17_01"
ARM = "screened_wide128"
FIT = 1
EPISODE = 7


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    runner.r.configure()
    plan = runner.r.read(RUN / "runtime_plan.json")
    seed = plan["evaluation_seeds"][EPISODE]
    agent, checkpoint = runner.load_agent(RUN, ARM, FIT, plan)
    graph = RUN / ARM / f"fit_{FIT:02d}" / "final.onnx"
    agent["runtime"] = runner.from_graph(agent, graph, ARM)
    exported_core = deployment.Core(agent["online"]).eval()
    env = runner.r.Chapter3IoVEnv(seed=seed)
    records = []
    mismatch = None
    for slot in range(40):
        state = env._observation().copy()
        with torch.inference_mode():
            tensor = torch.from_numpy(state)[None]
            eager_actions, eager_q = agent["online"].greedy(tensor)
            eager_raw = exported_core(tensor).detach().numpy()
        native_actions, native_q = agent["runtime"].values(state)
        native_raw = agent["runtime"].session.run(None, {"state": state[None]})[0]
        action_equal = bool(np.array_equal(eager_actions.numpy(), native_actions))
        raw_delta = np.abs(eager_raw.astype(np.float64) - native_raw.astype(np.float64))
        row = {
            "slot": slot,
            "action_equal": action_equal,
            "eager_q": float(eager_q[0]),
            "native_q": float(native_q[0]),
            "q_absolute_difference": float(abs(float(eager_q[0]) - float(native_q[0]))),
            "raw_max_absolute_difference": float(raw_delta.max()),
        }
        records.append(row)
        if not action_equal:
            changed = np.flatnonzero(eager_actions.numpy()[0] != native_actions[0])
            branches = []
            for branch in changed:
                eager_scores = eager_raw[0, branch].astype(np.float64)
                native_scores = native_raw[0, branch].astype(np.float64)
                eager_order = np.argsort(eager_scores)[::-1]
                native_order = np.argsort(native_scores)[::-1]
                branches.append({
                    "branch": int(branch),
                    "eager_action": int(eager_actions[0, branch]),
                    "native_action": int(native_actions[0, branch]),
                    "eager_scores": eager_scores.tolist(),
                    "native_scores": native_scores.tolist(),
                    "score_differences_native_minus_eager": (native_scores - eager_scores).tolist(),
                    "eager_top_gap": float(eager_scores[eager_order[0]] - eager_scores[eager_order[1]]),
                    "native_top_gap": float(native_scores[native_order[0]] - native_scores[native_order[1]]),
                })
            mismatch = {
                "slot": slot,
                "state_sha256": runner.r.models.original.existing.base.old.digest(state),
                "changed_branches": branches,
                "eager_actions": eager_actions.numpy()[0].tolist(),
                "native_actions": native_actions[0].tolist(),
                "eager_q": float(eager_q[0]),
                "native_q": float(native_q[0]),
                "q_absolute_difference": float(abs(float(eager_q[0]) - float(native_q[0]))),
                "raw_max_absolute_difference": float(raw_delta.max()),
                "state": state.tolist(),
            }
            break
        action = native_actions[0].copy()
        env.step(runner.r.to_action_dict(action, 30))
    report = {
        "status": "REPRODUCED" if mismatch else "NOT_REPRODUCED",
        "run": str(RUN),
        "arm": ARM,
        "fit": FIT,
        "episode_index": EPISODE,
        "evaluation_seed": seed,
        "checkpoint_sha256": runner.r.sha(RUN / ARM / f"fit_{FIT:02d}" / "final.pt"),
        "graph_sha256": runner.r.sha(graph),
        "slots_examined": len(records),
        "mismatch": mismatch,
        "slot_comparisons": records,
        "new_training_slots": 0,
        "new_optimizer_updates": 0,
        "run_files_modified": False,
    }
    runner.r.write(OUT / "diagnosis.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
