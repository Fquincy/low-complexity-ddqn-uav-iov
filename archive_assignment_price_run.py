"""Copy an audited run to a second volume and verify every saved file."""
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root):
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in root.rglob("*") if path.is_file()
    }


def archive(run_output, completion_output, archive_output):
    run_output = Path(run_output).resolve()
    completion_output = Path(completion_output).resolve()
    archive_output = Path(archive_output).resolve()
    if archive_output.exists():
        raise FileExistsError(archive_output)
    completion = json.loads((completion_output / "completion.json").read_text(encoding="utf-8"))
    if completion["status"] != "AUDITED_COMPLETE" or Path(completion["run_path"]).resolve() != run_output:
        raise RuntimeError("Archive requires the audited completion for this exact run")
    source = {"run": inventory(run_output), "audited_completion": inventory(completion_output)}
    archive_output.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copytree(run_output, archive_output / "run")
        shutil.copytree(completion_output, archive_output / "audited_completion")
        copied = {"run": inventory(archive_output / "run"), "audited_completion": inventory(archive_output / "audited_completion")}
        if copied != source:
            raise RuntimeError("Independent archive inventory or SHA256 mismatch")
        record = {
            "status": "VERIFIED_INDEPENDENT_ARCHIVE",
            "run_path": str(run_output),
            "completion_path": str(completion_output),
            "archive_path": str(archive_output),
            "files": sum(len(group) for group in source.values()),
            "bytes": sum(row["bytes"] for group in source.values() for row in group.values()),
            "trees": source,
        }
        (archive_output / "ARCHIVE_VERIFICATION.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record
    except Exception:
        # Preserve a failed partial archive for diagnosis; never mislabel it verified.
        raise
