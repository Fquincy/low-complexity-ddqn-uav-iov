"""Audit the completed V2 run, then create the independent verified archive."""
import json
from pathlib import Path

import archive_assignment_price_run as archival
import assignment_price_audited_completion as prior
import audit_assignment_price_research_bound_v2 as audit
import run_assignment_price_research_bound as runner

ROOT = runner.r.ROOT
RUN = ROOT / "development_runs/assignment_price_development_2026-11-16_01/run"
COMPLETION = ROOT / "development_runs/assignment_price_development_2026-11-16_01/audited_completion"
ARCHIVE = Path("D:/Fitz Thesis Research Archive/assignment_price_development_2026-11-16_01")
CONFIRM = "AUDIT_AND_ARCHIVE_FROZEN_ADP_RUN"


def preflight():
    if not (RUN / "manifest.json").is_file() or not (RUN / "verification.json").is_file():
        raise RuntimeError("Completed evaluation run is missing its manifest or verification")
    if COMPLETION.exists() or ARCHIVE.exists():
        raise FileExistsError("Completion or archive output already exists")
    progress = runner.r.read(RUN / "progress.json")
    if progress.get("stage") != "complete" or progress.get("new_training_slots") != 0 or progress.get("new_optimizer_updates") != 0:
        raise RuntimeError("Evaluation continuation is not complete")
    diagnostics = runner.r.read(RUN / "deployment_equivalence_diagnostics.json")
    if diagnostics["states_checked"] != 144000:
        raise RuntimeError("Complete learned-state equivalence scan is missing")
    return {"passed": True, "run_manifest_sha256": runner.r.sha(RUN / "manifest.json"),
            "states_checked": diagnostics["states_checked"],
            "action_disagreements": diagnostics["exact_action_disagreement_count"],
            "original_q_tolerance_exceedances": diagnostics["original_q_tolerance_exceedance_count"],
            "new_training_slots": 0, "new_optimizer_updates": 0}


def complete(confirmation=None):
    if confirmation != CONFIRM:
        raise RuntimeError("Explicit audit/archive confirmation required")
    gate = preflight()
    supplemental = prior.heuristic_evidence()
    report = audit.audit(RUN)
    if not report["passed"]:
        raise RuntimeError("Saved-artifact audit failed")
    diagnostics = runner.r.read(RUN / "deployment_equivalence_diagnostics.json")
    COMPLETION.mkdir(parents=True, exist_ok=False)
    runner.r.write(COMPLETION / "saved_artifact_audit.json", report)
    runner.r.write(COMPLETION / "heuristic_cost_reference.json", supplemental)
    runner.r.write(COMPLETION / "deployment_equivalence_reference.json", {
        "source": str((RUN / "deployment_equivalence_diagnostics.json").resolve()),
        "sha256": runner.r.sha(RUN / "deployment_equivalence_diagnostics.json"),
        "states_checked": diagnostics["states_checked"],
        "exact_action_disagreement_count": diagnostics["exact_action_disagreement_count"],
        "exact_action_equivalence_passed": diagnostics["exact_action_equivalence_passed"],
        "original_q_tolerance_exceedance_count": diagnostics["original_q_tolerance_exceedance_count"],
        "maximum_q_absolute_difference": diagnostics["maximum_q_absolute_difference"],
        "interpretation": "Policy evaluation uses frozen eager DDQN actions; exported runtime discrepancies remain explicit deployment-equivalence limitations."
    })
    runner.r.write(COMPLETION / "completion.json", {
        "status": "AUDITED_COMPLETE",
        "mode": report["mode"],
        "run_manifest_sha256": runner.r.sha(RUN / "manifest.json"),
        "run_path": str(RUN.resolve()),
        "all_saved_artifacts_audited": True,
        "heuristic_cost_reported": True,
        "deployment_equivalence_reported": True,
        "deployment_exact_action_equivalence_passed": diagnostics["exact_action_equivalence_passed"],
        "new_training_slots": 0,
        "new_optimizer_updates": 0,
        "no_retiming": True,
        "word_thesis_changed": False,
        "final_phase_launched": False
    })
    runner.r.write(COMPLETION / "manifest.json", {path.relative_to(COMPLETION).as_posix(): runner.r.sha(path)
                                                    for path in COMPLETION.rglob("*") if path.is_file()})
    archive = archival.archive(RUN, COMPLETION, ARCHIVE)
    result = {"status": "AUDITED_AND_ARCHIVED_COMPLETE", "preflight": gate,
              "audit": report, "archive": {key: archive[key] for key in ["status", "archive_path", "files", "bytes"]},
              "new_training_slots": 0, "new_optimizer_updates": 0}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--confirm-audit-archive")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(preflight(), indent=2))
    else:
        complete(args.confirm_audit_archive)
