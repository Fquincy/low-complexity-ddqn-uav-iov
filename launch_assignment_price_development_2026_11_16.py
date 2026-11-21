"""Versioned launch boundary for run, audit, and independent archive."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys

import assignment_price_audited_completion as completion
import archive_assignment_price_run as archival
import run_assignment_price_research_bound as runner

ROOT = runner.r.ROOT
PLAN = ROOT / "ASSIGNMENT_PRICE_OUTPUT_RETENTION_PLAN_2026-11-16.json"
RELEASE = ROOT / "ASSIGNMENT_PRICE_DEVELOPMENT_RELEASE_2026-11-16.json"
BINDING = ROOT / "validation/assignment_price_research_binding_2026-11-15_01/ENGINEERING_BINDING_MANIFEST.json"
ENVIRONMENT = ROOT / "validation/assignment_price_launch_review_2026-11-15_01/source_environment_verification.json"
PROTOCOL_SHA256 = "6213fe3e77946fec0b142f020577274084a795807993f140f39af0c7cbae6fef"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def same_volume(left, right):
    return Path(left).resolve().drive.casefold() == Path(right).resolve().drive.casefold()


def preflight():
    runner.r.configure()
    plan = read(PLAN)
    release = read(RELEASE)
    binding = read(BINDING)
    environment = read(ENVIRONMENT)["environment"]
    if runner.r.sha(runner.base.SPEC) != PROTOCOL_SHA256 or binding["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("Frozen protocol identity changed")
    if release["status"] != "ENGINEERING_RELEASE_REQUIRES_EXPLICIT_LAUNCH_CONFIRMATION" or release["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("Versioned release record is not valid")
    for name, digest in release["launch_source_sha256"].items():
        if runner.r.sha(ROOT / name) != digest:
            raise RuntimeError("Release-bound launch source changed: " + name)
    for name, digest in binding["source_hashes"].items():
        if runner.r.sha(ROOT / name) != digest:
            raise RuntimeError("Tested source changed: " + name)
    if runner.DEVELOPMENT_RELEASED:
        raise RuntimeError("Runner release flag must be false outside the launch call")
    current = {
        "executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": runner.torch.__version__,
        "numpy": runner.np.__version__,
        "torch_path": runner.torch.__file__,
        "numpy_path": runner.np.__file__,
        "torch_threads": runner.torch.get_num_threads(),
        "interop_threads": runner.torch.get_num_interop_threads(),
    }
    for key in current:
        if current[key] != environment[key]:
            raise RuntimeError("Recorded environment changed: " + key)
    paths = {key: Path(plan[key]).resolve() for key in ["primary_run", "audited_completion", "prelaunch_evidence", "independent_archive"]}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("A declared output path already exists")
    if same_volume(paths["primary_run"], paths["independent_archive"]):
        raise RuntimeError("Independent archive must use a distinct volume")
    required = int(plan["minimum_free_bytes_per_volume"])
    primary_free = shutil.disk_usage(paths["primary_run"].anchor).free
    archive_free = shutil.disk_usage(paths["independent_archive"].anchor).free
    if primary_free < required or archive_free < required:
        raise RuntimeError("Declared primary or archive volume lacks the operational buffer")
    return {
        "passed": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_count": len(binding["source_hashes"]),
        "paths": {key: str(value) for key, value in paths.items()},
        "primary_free_bytes": primary_free,
        "archive_free_bytes": archive_free,
        "minimum_free_bytes_per_volume": required,
        "research_seeds_consumed": False,
        "research_launched": False,
    }


def walk_numbers(value, path=()):
    if isinstance(value, dict):
        if isinstance(value.get("start"), int) and isinstance(value.get("count"), int):
            yield "interval", value["start"], value["start"] + value["count"], path
        for key, item in value.items():
            yield from walk_numbers(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_numbers(item, path + (str(index),))
    elif isinstance(value, int) and not isinstance(value, bool):
        yield "point", value, value, path


def fresh_seed_audit():
    plan = runner.plan("development")
    points = plan["model_seeds"] + [seed + offset for seed in plan["model_seeds"] for offset in [100000, 200000, 300000]]
    points += [plan["order_seed"], plan["timing_seed"], plan["bootstrap_seed"]]
    intervals = [(row["start"], row["start"] + row["count"]) for row in plan["train_blocks"]]
    intervals += [(min(plan["evaluation_seeds"]), max(plan["evaluation_seeds"]) + 1)]
    files = list(ROOT.glob("*PROTOCOL*.json")) + list(ROOT.glob("*SPEC*.json")) + list(ROOT.glob("*AMENDMENT*.json")) + list(ROOT.glob("*RELEASE*.json"))
    names = {"runtime_plan.json", "plan.json", "runtime_protocol.json", "settings.json", "launch_record.json", "launch.json"}
    for directory in [ROOT / "development_runs", ROOT / "validation"]:
        for parent, dirs, found in os.walk(directory):
            dirs[:] = [name for name in dirs if name not in {"source_snapshot", "release_source_snapshot", "completion", "__pycache__", "cost_gate", "literature", "tests_first_pass", "attempt_01"}]
            files.extend(Path(parent) / name for name in found if name in names)
    collisions = []
    checked = {}
    for path in sorted(set(files)):
        if path.name == runner.base.SPEC.name:
            continue
        value = read(path)
        for kind, start, end, where in walk_numbers(value):
            if kind == "interval":
                hit = any(start <= point < end for point in points) or any(max(start, lo) < min(end, hi) for lo, hi in intervals)
            else:
                hit = start in points or any(lo <= start < hi for lo, hi in intervals)
            if hit:
                collisions.append({"file": str(path.relative_to(ROOT)), "kind": kind, "start": start, "end": end, "path": list(where)})
        checked[str(path.relative_to(ROOT))] = runner.r.sha(path)
    return {"passed": not collisions, "checked_documents": checked, "collisions": collisions, "research_seeds_consumed": False}


def launch(confirmation=None):
    if confirmation != runner.CONFIRM:
        raise RuntimeError("Explicit development confirmation required")
    gate = preflight()
    seed_audit = fresh_seed_audit()
    if not seed_audit["passed"]:
        raise RuntimeError("Fresh seed audit failed")
    paths = {key: Path(value) for key, value in gate["paths"].items()}
    paths["prelaunch_evidence"].mkdir(parents=True, exist_ok=False)
    runner.r.write(paths["prelaunch_evidence"] / "preflight.json", gate)
    runner.r.write(paths["prelaunch_evidence"] / "seed_refresh.json", seed_audit)
    for source in [PLAN, Path(__file__), ROOT / "archive_assignment_price_run.py", ROOT / "assignment_price_audited_completion.py"]:
        shutil.copy2(source, paths["prelaunch_evidence"] / source.name)
    runner.r.write(paths["prelaunch_evidence"] / "launch_request.json", {
        "confirmation": confirmation,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": runner.r.sha(ROOT / "run_assignment_price_research_bound.py"),
        "auditor_sha256": runner.r.sha(ROOT / "audit_assignment_price_research_bound_v2.py"),
        "launcher_sha256": runner.r.sha(Path(__file__)),
        "retention_plan_sha256": runner.r.sha(PLAN),
    })
    runner.DEVELOPMENT_RELEASED = True
    try:
        run_output = runner.run(paths["primary_run"], "development", confirmation)
    finally:
        runner.DEVELOPMENT_RELEASED = False
    completion_output = completion.complete(run_output, paths["audited_completion"])
    archive_record = archival.archive(run_output, completion_output, paths["independent_archive"])
    result = {"status": "AUDITED_AND_ARCHIVED_COMPLETE", "run": str(run_output), "completion": str(completion_output), "archive": archive_record["archive_path"]}
    runner.r.write(paths["prelaunch_evidence"] / "workflow_completion.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--confirm-development", choices=[runner.CONFIRM])
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(preflight(), indent=2))
    else:
        if args.confirm_development != runner.CONFIRM:
            parser.error("Explicit development confirmation required")
        print(json.dumps(launch(args.confirm_development), indent=2))
