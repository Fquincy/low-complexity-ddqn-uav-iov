"""Diagnose the second evaluation stop without changing research artifacts."""
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
OUT = ROOT / "validation/assignment_price_q_tolerance_stop_2026-11-17_01"
ARM, FIT, SEED = "unmasked_plain24", 2, 874000000


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    runner.r.configure()
    plan = runner.r.read(RUN / "runtime_plan.json")
    agent, _ = runner.load_agent(RUN, ARM, FIT, plan)
    graph = RUN / ARM / f"fit_{FIT:02d}/final.onnx"
    agent["runtime"] = runner.from_graph(agent, graph, ARM)
    core = deployment.Core(agent["online"]).eval()
    env = runner.r.Chapter3IoVEnv(seed=SEED)
    rows, violation = [], None
    for slot in range(40):
        state = env._observation().copy()
        with torch.inference_mode():
            tensor = torch.from_numpy(state)[None]
            eager_action, eager_q = agent["online"].greedy(tensor)
            eager_raw = core(tensor).numpy()
        native_action, native_q = agent["runtime"].values(state)
        native_raw = agent["runtime"].session.run(None, {"state": state[None]})[0]
        absolute = abs(float(eager_q[0]) - float(native_q[0]))
        relative = absolute / max(abs(float(eager_q[0])), abs(float(native_q[0])))
        action_equal = bool(np.array_equal(eager_action.numpy(), native_action))
        raw_max = float(np.max(np.abs(eager_raw.astype(np.float64) - native_raw.astype(np.float64))))
        within_original = bool(np.allclose(eager_q.numpy(), native_q, rtol=1e-5, atol=1e-6))
        row = {"slot": slot, "action_equal": action_equal, "eager_q": float(eager_q[0]), "native_q": float(native_q[0]),
               "absolute_difference": absolute, "relative_difference": relative, "raw_max_absolute_difference": raw_max,
               "within_original_tolerance": within_original}
        rows.append(row)
        if not within_original and violation is None:
            violation = {**row, "state_sha256": runner.r.models.original.existing.base.old.digest(state),
                         "eager_actions": eager_action.numpy()[0].tolist(), "native_actions": native_action[0].tolist(),
                         "raw_elements": int(eager_raw.size), "float32_unit_roundoff": float(np.finfo(np.float32).eps / 2),
                         "sum_error_bound_32_terms": float(32 * np.finfo(np.float32).eps * np.max(np.abs(eager_raw)))}
            break
        env.step(runner.r.to_action_dict(eager_action[0].numpy().copy(), 30))
    report = {"status": "REPRODUCED" if violation else "NOT_REPRODUCED", "arm": ARM, "fit": FIT, "seed": SEED,
              "checkpoint_sha256": runner.r.sha(RUN / ARM / f"fit_{FIT:02d}/final.pt"), "graph_sha256": runner.r.sha(graph),
              "violation": violation, "slots_examined": len(rows), "rows": rows,
              "new_training_slots": 0, "new_optimizer_updates": 0, "run_files_modified": False}
    runner.r.write(OUT / "diagnosis.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
