"""Read-only technical launch decision; never enables or invokes training."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import run_horizon_loss_research as r

OUT=r.ROOT/'validation/horizon_loss_launch_review_2026-11-14_01'
BIND=r.ROOT/'validation/horizon_loss_research_binding_2026-11-14_01'
REPORT='HORIZON_LOSS_LAUNCH_READINESS_REVIEW_2026-11-14.txt'

def main():
    assert not (r.ROOT/REPORT).exists(), 'Preserve earlier review'
    OUT.mkdir(parents=True,exist_ok=True)
    p=r.protocol()
    release=r.read(BIND/'ENGINEERING_RELEASE_MANIFEST.json')
    for name,h in release['source_sha256'].items():
        assert r.sha(r.ROOT/name)==r.sha(BIND/'release_source_snapshot'/name)==h,name
    for name,h in release['files'].items():assert r.sha(BIND/name)==h,name
    assert release['protocol_sha256']==r.sha(r.SPEC)
    tests=r.read(BIND/'tests.json');assert tests['passed'] and tests['tests']==4
    integrity=r.read(BIND/'workflow/workflow_verification.json')
    assert integrity['artifact_audit_passed'] and integrity['deployment_storage_audit']['passed']
    assert integrity['quality_evidence'] is False and integrity['research_launched'] is False
    assert r.read(BIND/'release_review_verification.json')['passed']
    assert r.read(BIND/'startup_fixture/startup_verification.json')['passed']
    saved=r.read(BIND/'workflow/run/environment.json')
    # The original recorder serialized a None metadata name to JSON key "null".
    # Apply the same serialization before comparing; do not change packages.
    packages=json.loads(json.dumps({d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))
    assert packages==saved['packages']
    assert sys.version==saved['python'] and sys.executable==saved['executable']
    assert str(r.torch.__version__)==saved['torch'] and str(r.np.__version__)==saved['numpy']
    assert str(r.torch.__file__)==saved['torch_path']
    assert str(r.models.fast.decoder.numba.__file__)==saved['numba_path']
    assert r.DEVELOPMENT_RELEASED is False and p['authorization']['launch_authorized'] is False
    seeds=r.read(OUT/'seed_refresh.json')
    assert seeds['passed'] and seeds['protocol_sha256']==r.sha(r.SPEC)
    assert not seeds['development_collisions'] and not seeds['final_episode_seed_hits'] and not seeds['errors']
    assert seeds['development_final_sets_disjoint']
    declaration=r.read(r.ROOT/p['seed_reservation']['audit'])
    assert declaration['passed'] and r.sha(r.ROOT/p['seed_reservation']['audit'])==p['seed_reservation']['audit_sha256']
    r.write(OUT/'source_environment_verification.json',dict(passed=True,
        source_files=len(release['source_sha256']),artifact_files=len(release['files']),
        package_distributions=len(packages),python_executable=sys.executable,
        python_and_packages_match=True,runtime_module_paths_match=True,
        package_key_comparison='Same JSON serialization as original recorder; unnamed distribution key becomes null string',
        protocol_sha256=r.sha(r.SPEC),source_release_enabled=False,research_launched=False))
    d=p['development']
    decision=dict(technical_ready=True,recommendation='Proceed with frozen development comparison after separate explicit user launch instruction',
        protocol_sha256=r.sha(r.SPEC),entrypoint='horizon_loss_research_complete_workflow.py',
        required_confirmation=r.CONFIRM,training_slots=d['training_slots'],evaluation_slots=d['evaluation_slots'],
        learned_controllers=14,heuristics=3,initializations=5,final_checkpoints=70,
        actual_use_seed_audit_files=seeds['files_checked'],source_files_verified=len(release['source_sha256']),
        core_tests_passed=4,technical_blockers=[],launch_authorized=False,source_release_enabled=False,
        research_launched=False,new_simulator_steps=0,new_optimizer_updates=0,
        final_phase_authorized=False,word_thesis_changed=False,
        launch_requirements=['Separate explicit launch instruction',
            'Recheck release source hashes and protocol before enabling the existing development guard',
            'Archive the complete 63-file reviewed release before any model/task initialization',
            'Refresh actual-use seed audit again if any intervening run or seed reservation is introduced',
            'Use the recorded Python environment and exact frozen protocol; final execution remains disabled'])
    r.write(OUT/'launch_decision.json',decision)
    review=f'''HORIZON-LOSS DEVELOPMENT: SEPARATE LAUNCH-READINESS DECISION
2026-11-14

DECISION
Technically ready. Recommend proceeding with the frozen 560,000-slot
development comparison after a separate explicit launch instruction. No
technical blocker was found in this review. The source release remains
disabled; no research training, evaluation or final phase was launched.

WHAT WAS CHECKED
All{len(release['source_sha256'])} reviewed source files match the current files and archived release
copies. All{len(release['files'])} frozen evidence artifacts match their manifest. The protocol hash
is {r.sha(r.SPEC)}.
The exact Python version/executable, serialized package inventory, PyTorch and
NumPy versions, and actual PyTorch/Numba runtime module paths match the tested
environment. An initial direct package-dictionary comparison differed because
JSON turns an unnamed metadata key into the string null; applying identical
serialization yields an exact match. No package or runtime was changed.

The four core tests, complete paired fixture, independent artifact audit,
deployment-storage audit, diagnostic reconstruction, corruption rejection and
confirmation/release guards passed. Separate startup reporting passed for all
28 saved fixture checkpoints. The outer workflow's added completion stage was
exercised on those already audited checkpoints without repeating training.
Fixture results remain engineering evidence; quality_evidence is false.

The fresh actual-use audit examined{seeds['files_checked']} records:
{seeds['compressed_headers']} compressed episode headers and{seeds['actual_runtime_metadata']} runtime metadata files.
No development-reservation collision, protected-final episode seed hit, read
error or development/final overlap was found. The earlier declaration audit
also remains valid. The new actual-use scan excludes archived source copies,
including release_source_snapshot, so declarations are not misreported as
execution. Scope is the four recorded local run roots, bounded episode-header
prefixes and runtime schedules; this is not proof about external/unlogged runs.

FIXED EXPERIMENT
Primary: normalized_screened_price24.14 learners plus three heuristics;
five paired initializations;200 episodes of40 slots per learner/fit.
70 final-only checkpoints;560,000 training slots;488,390 optimizer updates;
2,380 target copies.48 shared evaluation tasks give140,160 evaluation slots.
The complete simulator budget is700,160 slots. The full audit should cover
17,504 episodes and21,004,800 task records.8,960 engineering action checks,
44,800 steady timing readings and70 separate startup records are expected.

All16 primary and nine auxiliary contrasts remain required. Controls, rewards,
seeds, optimizer settings, architecture and thresholds were not changed.
Original-unit value stability, policy/deadline, correction/price/architecture,
cost and integrity gates must all pass. No best fit or favorable retiming.
All models retain final checkpoints, including scientifically failed fits.

LIMITS OF THIS APPROVAL
This approves technical readiness, not controller success or thesis novelty.
The earlier failed research result stays preserved. Loss normalization remains
a hypothesis and has not yet been judged on the fresh development evidence.
Routine normalization alone is not a novelty claim. Literature positioning and
supervisor agreement remain separate from this engineering decision.
Startup readings include full training-checkpoint reconstruction, fresh Python
imports and a first action with existing caches. They are not cold-machine or
minimal deployment-bundle startup measurements and have no acceptance gate.

EXECUTION BOUNDARY
The reviewed entry point is horizon_loss_research_complete_workflow.py.
Its underlying runner requires {r.CONFIRM} and a reviewed source release.
Keep the development guard closed until the explicit launch instruction.
On authorized launch, verify and archive the complete63-file reviewed release
before any model/task initialization, use the recorded Python executable,
and retain live progress/error logs. Activate only the existing development
release guard; do not change the frozen scientific design. If other runs or
reservations occur first, refresh the seed audit. Final execution stays disabled.

To authorize the next action, the explicit instruction is:
Launch the frozen 560,000-slot horizon-loss development comparison.

The Word thesis, protected final seeds and previous successful/failed runs
remain unchanged. Cloud/charging remain excluded; security/trust is deferred.

EVIDENCE
validation/horizon_loss_launch_review_2026-11-14_01/
  launch_decision.json
  source_environment_verification.json
  seed_refresh.json
The prior binding review and ENGINEERING_RELEASE_MANIFEST.json remain preserved.
'''
    (r.ROOT/REPORT).write_text(review,encoding='utf-8')
    for path in (r.ROOT/REPORT,Path(__file__),r.ROOT/'analysis/audit_horizon_loss_launch_seeds_2026_11_14.py'):
        shutil.copy2(path,OUT/path.name)
    status_path=r.ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json'
    shutil.copy2(status_path,OUT/'PROJECT_STATUS_BEFORE_LAUNCH_REVIEW.json')
    status=r.read(status_path)
    status['current_workstream']='horizon_loss_launch_readiness_passed_awaiting_separate_explicit_instruction'
    status['next']='Separate explicit launch instruction for frozen 560000-slot horizon-loss development comparison; keep final phase disabled'
    status['next_engineering_step']=status['next']
    status['horizon_loss_launch_review']=dict(review=REPORT,evidence=str(OUT.relative_to(r.ROOT)),
        technical_ready=True,seed_refresh_passed=True,source_and_environment_verified=True,
        technical_blockers=[],launch_authorized=False,source_release_enabled=False,
        research_launched=False,word_thesis_changed=False)
    r.write(status_path,status)
    r.write(OUT/'manifest.json',{x.relative_to(OUT).as_posix():r.sha(x) for x in OUT.rglob('*') if x.is_file()})
    print(json.dumps(decision,indent=2))

if __name__=='__main__':main()
