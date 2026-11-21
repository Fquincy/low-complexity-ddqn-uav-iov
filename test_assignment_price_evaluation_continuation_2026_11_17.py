import unittest
from unittest.mock import patch

import continue_assignment_price_evaluation_2026_11_17 as continuation
import run_assignment_price_research_bound as runner


class EvaluationContinuationTests(unittest.TestCase):
    def test_confirmation_precedes_preflight(self):
        with patch.object(continuation, "preflight", side_effect=AssertionError("must not run")) as gate:
            with self.assertRaisesRegex(RuntimeError, "Explicit evaluation-only continuation confirmation required"):
                continuation.run(None)
            gate.assert_not_called()

    def test_exact_failed_episode_completes_and_records_near_tie(self):
        runner.r.configure()
        plan = runner.r.read(continuation.RUN / "runtime_plan.json")
        agent, _ = runner.load_agent(continuation.RUN, "screened_wide128", 1, plan)
        agent["runtime"] = runner.from_graph(agent, continuation.RUN / "screened_wide128/fit_01/final.onnx", "screened_wide128")
        diagnostics = []
        row = continuation.episode(agent, 874000007, diagnostics)
        self.assertEqual(len(row["slots"]), 40)
        self.assertEqual(len(row["chosen_q"]), 40)
        self.assertGreaterEqual(len(diagnostics), 1)
        first = diagnostics[0]
        self.assertEqual((first["slot"], first["changed_branches"]), (26, [31]))
        self.assertEqual((first["eager_actions"][31], first["native_actions"][31]), (2, 4))
        self.assertLess(first["q_absolute_difference"], 2e-5)

    def test_second_failed_episode_completes_under_predeclared_float32_guard(self):
        runner.r.configure()
        plan = runner.r.read(continuation.RUN / "runtime_plan.json")
        agent, _ = runner.load_agent(continuation.RUN, "unmasked_plain24", 2, plan)
        agent["runtime"] = runner.from_graph(agent, continuation.RUN / "unmasked_plain24/fit_02/final.onnx", "unmasked_plain24")
        diagnostics = []
        row = continuation.episode(agent, 874000000, diagnostics)
        self.assertEqual(len(row["slots"]), 40)
        self.assertEqual(row["slots"][16]["action"], [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 3, 0, 0, 4, 0, 4, 4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
