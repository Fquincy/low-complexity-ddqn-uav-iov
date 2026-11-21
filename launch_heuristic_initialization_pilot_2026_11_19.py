"""Explicit, version-bound pilot launch; default research runner stays closed."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

import check_heuristic_initialization_pilot_readiness as readiness
import run_heuristic_initialization_pilot as runner

ROOT=Path(__file__).resolve().parent
RELEASE=ROOT/'HEURISTIC_INITIALIZATION_PILOT_LAUNCH_BINDING_2026-11-19.json'
CHECKS=ROOT/'validation/heuristic_initialization_pilot_launch_checks_2026-11-19_01'


def preflight():
    release=runner.read(RELEASE)
    if release['authorization']!='USER_EXPLICIT_LAUNCH_17680_SLOT_PILOT':raise RuntimeError('No explicit launch binding')
    for name,digest in release['sha256'].items():
        if runner.sha(ROOT/name)!=digest:raise RuntimeError('Launch-bound source changed: '+name)
    if runner.PILOT_RELEASED:raise RuntimeError('Default runner must stay disabled')
    if Path(sys.executable).resolve()!=Path(release['python_executable']).resolve():raise RuntimeError('Wrong interpreter')
    # Reuse the tested read-only checks, preserving the previous readiness report.
    check_path=CHECKS
    index=1
    while check_path.exists():
        index+=1
        check_path=CHECKS.with_name(CHECKS.name.rsplit('_',1)[0]+f'_{index:02d}')
    readiness.OUT=check_path
    readiness.main()
    return release


def execute(confirmation):
    if confirmation!=runner.CONFIRM:raise RuntimeError('Explicit pilot confirmation required')
    release=preflight();out=ROOT/release['output'];archive=Path(release['archive'])
    if out.exists() or archive.exists():raise RuntimeError('Output or archive already exists; do not overwrite')
    stop=threading.Event()
    def status():
        while not stop.wait(3):
            if not out.exists():continue
            try:
                stage=runner.read(out/'progress.json') if (out/'progress.json').exists() else {'stage':'starting'}
                demo=len(list((out/'run').glob('demo_*/episode_*.json')))*40
                trained=len(list((out/'run').glob('*/fit_*/train_*.json')))*40
                evaluated=len(list((out/'run'/'evaluation').rglob('episode_*.json')))*40
                counts=[runner.read(p) for p in (out/'run').glob('*/fit_*/counts.json')]
                text=(f"Updated: {datetime.now().astimezone().isoformat()}\n"
                      f"Stage: {stage}\nTeacher demonstration slots saved: {demo} / 1200\n"
                      f"Student training slots saved: {trained} / 12000\n"
                      f"Evaluation slots saved: {evaluated} / 4480\n"
                      f"Total episode slots saved: {demo+trained+evaluated} / 17680\n"
                      f"Final checkpoints: {len(list((out/'run').glob('*/fit_*/final.pt')))} / 6\n"
                      f"DDQN updates in completed fits: {sum(v['updates'] for v in counts)} / 11232\n"
                      "Counters advance after saved episodes/fits; active episode work may be ahead.\n")
                (out/'LIVE_STATUS.txt').write_text(text,encoding='utf-8')
            except (OSError,ValueError):pass
    worker=threading.Thread(target=status,daemon=True);worker.start()
    try:
        runner.PILOT_RELEASED=True
        receipt=runner.run(out,'pilot',runner.CONFIRM)
    except BaseException:
        stop.set();worker.join();out.mkdir(parents=True,exist_ok=True)
        (out/'LIVE_STATUS.txt').write_text('STOPPED: inspect stderr.log and failure.txt. No automatic retry.\n',encoding='utf-8')
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        runner.PILOT_RELEASED=False
        stop.set();worker.join()
    runner.write(out/'launch_authorization.json',dict(binding=RELEASE.name,sha256=runner.sha(RELEASE),
        authorized=True,scope='frozen exploratory pilot only',teacher_backend='Python reference',word_thesis_changed=False))
    (out/'LIVE_STATUS.txt').write_text('PILOT AUDIT COMPLETE. Creating independent D: archive; scientific interpretation pending.\n',encoding='utf-8')
    # Archive immutable research artifacts; exclude active process log/status files.
    archive.mkdir(parents=True,exist_ok=False)
    selected=['run','completion.json','saved_artifact_audit.json','launch_authorization.json']
    for name in selected:
        source=out/name;destination=archive/name
        if source.is_dir():shutil.copytree(source,destination)
        else:shutil.copy2(source,destination)
    inventory={p.relative_to(archive).as_posix():runner.sha(p) for p in archive.rglob('*') if p.is_file()}
    for name,digest in inventory.items():
        if runner.sha(out/name)!=digest:raise RuntimeError('Archive mismatch: '+name)
    runner.write(out/'archive_verification.json',dict(passed=True,archive=str(archive),files=len(inventory),sha256=inventory))
    shutil.copy2(out/'archive_verification.json',archive/'archive_verification.json')
    (out/'LIVE_STATUS.txt').write_text('AUDITED_AND_ARCHIVED_COMPLETE\n17680 / 17680 simulated slots; 12000 / 12000 student training slots.\nAll fits retained. Read analysis.json and audit before interpreting the scientific screen.\n',encoding='utf-8')
    return dict(status='AUDITED_AND_ARCHIVED_COMPLETE',output=str(out),archive=str(archive))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');p.add_argument('--confirmation')
    args=p.parse_args()
    if args.check_only:print(json.dumps(preflight()))
    else:print(json.dumps(execute(args.confirmation)))
