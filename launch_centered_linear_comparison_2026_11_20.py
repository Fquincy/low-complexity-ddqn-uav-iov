"""Explicit version-bound development release; frozen runner stays disabled on disk."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import threading
import traceback

import check_centered_linear_comparison_readiness as readiness
import run_centered_linear_comparison as runner

ROOT = Path(__file__).resolve().parent
BINDING = ROOT/'CENTERED_LINEAR_COMPARISON_LAUNCH_BINDING_2026-11-20.json'


def preflight():
    binding = runner.read(BINDING)
    if binding['authorization'] != 'USER_EXPLICIT_LAUNCH_51280_SLOT_COMPARISON':
        raise RuntimeError('Missing explicit authorization')
    for name, digest in binding['sha256'].items():
        if runner.sha(ROOT/name) != digest:
            raise RuntimeError('Launch-bound file changed: '+name)
    if Path(sys.executable).resolve() != Path(binding['python_executable']).resolve():
        raise RuntimeError('Wrong interpreter')
    if runner.DEVELOPMENT_RELEASED:
        raise RuntimeError('Default runner must remain disabled')
    if (ROOT/binding['output']).exists() or Path(binding['archive']).exists():
        raise RuntimeError('Preserve existing output/archive; no automatic retry')
    # Exclude this launch binding: it deliberately references the same protocol.
    # Its hashes contain no additional seed reservations.
    check = ROOT/'validation/centered_linear_comparison_launch_checks_2026-11-20_01'
    index = 1
    while check.exists():
        index += 1
        check = check.with_name(f'centered_linear_comparison_launch_checks_2026-11-20_{index:02d}')
    readiness.OUT = check
    readiness.main()
    return binding, check


def execute(confirmation):
    if confirmation != runner.CONFIRM:
        raise RuntimeError('Explicit development confirmation required')
    binding, check = preflight()
    out = ROOT/binding['output']
    archive = Path(binding['archive'])
    stop = threading.Event()

    def status():
        while not stop.wait(3):
            if not out.exists():
                continue
            try:
                stage = runner.read(out/'progress.json') if (out/'progress.json').exists() else {'stage':'starting'}
                run = out/'run'
                demo = len(list(run.glob('demo_*/episode_*.json')))*40
                trained = len(list(run.glob('*/fit_*/train_*.json')))*40
                evaluated = len(list((run/'evaluation').rglob('episode_*.json')))*40
                counts = [runner.read(p) for p in run.glob('*/fit_*/counts.json')]
                text = (f'Updated: {datetime.now().astimezone().isoformat()}\nStage: {stage}\n'
                        f'Teacher demonstration slots saved: {demo} / 1200\n'
                        f'Student training slots saved: {trained} / 36000\n'
                        f'Evaluation slots saved: {evaluated} / 14080\n'
                        f'Total episode slots saved: {demo+trained+evaluated} / 51280\n'
                        f'Final checkpoints: {len(list(run.glob("*/fit_*/final.pt")))} / 21\n'
                        f'TD updates in completed fits: {sum(v["updates"] for v in counts)} / 33696\n'
                        f'BC updates in completed fits: {sum(v["bc_updates"] for v in counts)} / 3000\n'
                        'Counters advance after saved episodes/fits; active work may be ahead.\n')
                (out/'LIVE_STATUS.txt').write_text(text, encoding='utf-8')
            except (OSError, ValueError):
                pass

    worker = threading.Thread(target=status, daemon=True)
    worker.start()
    try:
        runner.DEVELOPMENT_RELEASED = True
        receipt = runner.run(out, 'development', runner.CONFIRM)
        if not receipt['passed']:
            raise RuntimeError('Full artifact audit did not pass')
        stop.set(); worker.join()
        runner.write(out/'launch_authorization.json', dict(
            authorized=True, binding=BINDING.name, binding_sha256=runner.sha(BINDING),
            checks=str(check.relative_to(ROOT)), scope='frozen 51280-slot development only',
            word_thesis_changed=False, teacher_backend='Python reference'))
        # Preserve the runner's immutable completion receipt; this separate receipt
        # records the explicit authorization and verified independent archive.
        (out/'LIVE_STATUS.txt').write_text('AUDIT COMPLETE. Verifying independent D: archive.\n', encoding='utf-8')
        archive.mkdir(parents=True, exist_ok=False)
        for name in ['run', 'completion.json', 'saved_artifact_audit.json', 'launch_authorization.json']:
            source, destination = out/name, archive/name
            if source.is_dir(): shutil.copytree(source, destination)
            else: shutil.copy2(source, destination)
        inventory = {p.relative_to(archive).as_posix():runner.sha(p) for p in archive.rglob('*') if p.is_file()}
        expected = {p.relative_to(out).as_posix() for p in (out/'run').rglob('*') if p.is_file()}
        expected.update(['completion.json','saved_artifact_audit.json','launch_authorization.json'])
        if set(inventory) != expected:
            raise RuntimeError('Archive inventory mismatch')
        for name, digest in inventory.items():
            if runner.sha(out/name) != digest:
                raise RuntimeError('Archive hash mismatch: '+name)
        runner.write(out/'archive_verification.json', dict(passed=True, archive=str(archive), files=len(inventory), sha256=inventory))
        shutil.copy2(out/'archive_verification.json', archive/'archive_verification.json')
        if runner.sha(out/'archive_verification.json') != runner.sha(archive/'archive_verification.json'):
            raise RuntimeError('Archive receipt mismatch')
        (out/'LIVE_STATUS.txt').write_text('AUDITED_AND_ARCHIVED_COMPLETE\n51280 / 51280 total simulated slots; 36000 / 36000 student training slots.\nAll 21 final checkpoints retained. Scientific interpretation pending.\n', encoding='utf-8')
    except BaseException:
        stop.set(); worker.join()
        if out.exists():
            (out/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
            (out/'LIVE_STATUS.txt').write_text('STOPPED: inspect failure.txt and stderr.log. No automatic retry.\n', encoding='utf-8')
        raise
    finally:
        runner.DEVELOPMENT_RELEASED = False
        stop.set(); worker.join()
    return dict(status='AUDITED_AND_ARCHIVED_COMPLETE', output=str(out), archive=str(archive))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--confirmation')
    args = parser.parse_args()
    if args.check_only:
        binding, checks = preflight()
        print(json.dumps(dict(status='PREFLIGHT_PASSED', output=binding['output'], checks=str(checks))))
    else:
        print(json.dumps(execute(args.confirmation)))
