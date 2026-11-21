import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import archive_assignment_price_run as archival
import launch_assignment_price_development_2026_11_16 as launch


class ReleaseGateTests(unittest.TestCase):
    def test_preflight_is_read_only_and_research_stays_disabled(self):
        before = {key: Path(value).exists() for key, value in launch.preflight()["paths"].items()}
        report = launch.preflight()
        after = {key: Path(value).exists() for key, value in report["paths"].items()}
        self.assertTrue(report["passed"])
        self.assertEqual(before, after)
        self.assertFalse(any(after.values()))
        self.assertFalse(launch.runner.DEVELOPMENT_RELEASED)
        self.assertFalse(report["research_launched"])

    def test_missing_confirmation_stops_before_preflight(self):
        with patch.object(launch, "preflight", side_effect=AssertionError("must not run")) as gate:
            with self.assertRaisesRegex(RuntimeError, "Explicit development confirmation required"):
                launch.launch(None)
            gate.assert_not_called()
        self.assertFalse(launch.runner.DEVELOPMENT_RELEASED)

    def test_bound_workflow_orders_run_audit_archive_and_resets_release(self):
        with tempfile.TemporaryDirectory(dir=launch.ROOT / "validation") as folder:
            root = Path(folder)
            paths = {name: str(root / name) for name in ["primary_run", "audited_completion", "prelaunch_evidence", "independent_archive"]}
            gate = {"passed": True, "paths": paths, "research_launched": False}
            calls = []

            def run(output, mode, confirmation):
                self.assertTrue(launch.runner.DEVELOPMENT_RELEASED)
                self.assertEqual(mode, "development")
                calls.append("run")
                Path(output).mkdir(parents=True)
                return Path(output)

            def complete(run_output, completion_output):
                self.assertFalse(launch.runner.DEVELOPMENT_RELEASED)
                calls.append("audit")
                Path(completion_output).mkdir(parents=True)
                return Path(completion_output)

            def archive(run_output, completion_output, archive_output):
                calls.append("archive")
                Path(archive_output).mkdir(parents=True)
                return {"archive_path": str(archive_output)}

            with patch.object(launch, "preflight", return_value=gate), patch.object(launch, "fresh_seed_audit", return_value={"passed": True}), \
                 patch.object(launch.runner, "run", side_effect=run), patch.object(launch.completion, "complete", side_effect=complete), \
                 patch.object(launch.archival, "archive", side_effect=archive):
                result = launch.launch(launch.runner.CONFIRM)
            self.assertEqual(calls, ["run", "audit", "archive"])
            self.assertEqual(result["status"], "AUDITED_AND_ARCHIVED_COMPLETE")
            self.assertFalse(launch.runner.DEVELOPMENT_RELEASED)

    def test_archive_verifies_tiny_saved_run(self):
        with tempfile.TemporaryDirectory(dir=launch.ROOT / "validation") as folder:
            root = Path(folder)
            run = root / "run"
            completion = root / "completion"
            archive = root / "archive"
            run.mkdir()
            completion.mkdir()
            (run / "manifest.json").write_text('{"example.txt":"fixture"}\n', encoding="utf-8")
            (run / "example.txt").write_bytes(b"research fixture\n")
            (completion / "completion.json").write_text(json.dumps({"status": "AUDITED_COMPLETE", "run_path": str(run.resolve())}) + "\n", encoding="utf-8")
            (completion / "saved_artifact_audit.json").write_text('{"passed":true}\n', encoding="utf-8")
            report = archival.archive(run, completion, archive)
            self.assertEqual(report["status"], "VERIFIED_INDEPENDENT_ARCHIVE")
            self.assertEqual((archive / "run/example.txt").read_bytes(), b"research fixture\n")
            self.assertTrue((archive / "ARCHIVE_VERIFICATION.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
