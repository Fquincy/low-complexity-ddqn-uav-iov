"""Run and retain isolated synthetic mechanics evidence; research remains closed."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import traceback
import unittest

import numpy as np
import torch

import return_calibrated_handoff as old
import test_value_gradient_isolation as tests
import value_gradient_isolation as vgi

ROOT=Path(__file__).resolve().parent
DEFAULT=ROOT/'validation/value_gradient_isolation_mechanics_2026-11-20_01'
SOURCES=['value_gradient_isolation.py','test_value_gradient_isolation.py',Path(__file__).name,
         'test_return_calibrated_handoff.py']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def run(output):
    out=Path(output).resolve()
    if not out.is_relative_to(ROOT/'validation'):raise ValueError('Synthetic evidence must stay in validation')
    if out.exists():raise RuntimeError('Preserve previous evidence; use a fresh output directory')
    binding=read(ROOT/'RETURN_CALIBRATED_COMPARISON_LAUNCH_BINDING_2026-11-20.json')
    for name,digest in binding['sha256'].items():assert sha(ROOT/name)==digest,name
    design=read(ROOT/'VALUE_GRADIENT_ISOLATION_DESIGN_STATUS_2026-11-20.json')
    brief_receipt=ROOT/'validation/value_gradient_isolation_brief_2026-11-20_01/verification.json'
    receipt=read(brief_receipt)
    assert sha(ROOT/design['brief'])==design['brief_sha256']==receipt['brief_sha256']
    assert sha(ROOT/'VALUE_GRADIENT_ISOLATION_DESIGN_STATUS_2026-11-20.json')==receipt['design_status_sha256']
    assert not design['research_protocol_frozen'] and not vgi.DEVELOPMENT_RELEASED and not vgi.FINAL_RELEASED
    out.mkdir(parents=True)
    sources={name:sha(ROOT/name) for name in SOURCES}
    for name in SOURCES:
        p=out/'source_snapshot'/name;p.parent.mkdir(exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    write(out/'input_bindings.json',dict(kind='ENGINEERING_SYNTHETIC_NOT_RESEARCH',
        frozen_launch_source_sha256=binding['sha256'],source_sha256=sources,
        design_status_sha256=sha(ROOT/'VALUE_GRADIENT_ISOLATION_DESIGN_STATUS_2026-11-20.json'),
        design_verification_sha256=sha(brief_receipt),brief_sha256=design['brief_sha256'],
        fixture='Deterministic modular/sine arrays; all initial weights overwritten inside forked ambient RNG; no seed selected',
        research_seed_values_used=False,simulator_imported=False,teacher_imported=False))
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    rng_before=torch.get_rng_state().clone();numpy_before=np.random.get_state()
    log=io.StringIO();case_records=[]
    class Result(unittest.TextTestResult):
        def addSuccess(self,test):
            super().addSuccess(test);case_records.append(dict(test=test.id(),passed=True));print('PASS '+test._testMethodName,flush=True)
        def addFailure(self,test,err):
            super().addFailure(test,err);case_records.append(dict(test=test.id(),passed=False));print('FAIL '+test._testMethodName,flush=True)
        def addError(self,test,err):
            super().addError(test,err);case_records.append(dict(test=test.id(),passed=False));print('ERROR '+test._testMethodName,flush=True)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(tests.Mechanics)
    result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=Result).run(suite)
    (out/'tests.txt').write_text(log.getvalue(),encoding='utf-8')
    write(out/'test_results.json',dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        skipped=len(result.skipped),passed=result.wasSuccessful(),cases=case_records))
    passed=result.wasSuccessful();exports={};artifact_error=None
    if passed:
        try:
            data=tests.fixtures.batch();examples=tests.demo()
            np.savez_compressed(out/'synthetic_inputs.npz',states=data[0].numpy(),actions=data[1].numpy(),
                rewards=data[2].numpy(),next_states=data[3].numpy(),dones=data[4].numpy(),
                retention_states=examples[0].numpy(),retention_labels=examples[1].numpy())
            for arm in [vgi.PRIMARY,vgi.RESCALE]:
                handoff=tests.calibrate(tests.model(arm));model=handoff.model
                target,optimizer=vgi.td_setup(model);target_hash=old.tensor_hash(target)
                handle=tests.counted(optimizer,'artifact_TD_optimizer_steps')
                report=vgi.ddqn_retention_step(model,target,optimizer,data,handoff.alpha,examples);handle.remove()
                assert old.tensor_hash(target)==target_hash
                deployed=vgi.deployment_model(model)
                # Serialize weights only; target/replay/optimizer remain in a distinct artifact.
                buffer=io.BytesIO();torch.save(deployed.state_dict(),buffer)
                deployed_path=out/(arm+'_deployment_weights.pt')
                with deployed_path.open('xb') as f:f.write(buffer.getvalue())
                reload=tests.fixtures.model();reload.load_state_dict(torch.load(deployed_path,weights_only=True))
                for n in [1,8,40]:
                    s=tests.fixtures.states(n)
                    for left,right in zip(model.greedy(s)[:2],reload.greedy(s)[:2]):assert torch.equal(left,right)
                    assert torch.equal(model(s,reload.greedy(s)[0]),reload(s,reload.greedy(s)[0]))
                checkpoint=out/(arm+'_SYNTHETIC_training_checkpoint.pt')
                torch.save(dict(kind='ENGINEERING_SYNTHETIC_NOT_RESEARCH',online=model.state_dict(),
                    target=target.state_dict(),optimizer=optimizer.state_dict(),optimizer_steps=1),checkpoint)
                write(out/(arm+'_calibration.json'),handoff.report)
                write(out/(arm+'_step.json'),report)
                exports[arm]=dict(deployment_class=type(deployed).__module__+'.'+type(deployed).__name__,
                    parameter_count=sum(p.numel() for p in model.parameters()),parameter_bytes=baseline_bytes(model),
                    buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()),
                    deployment_file_bytes=deployed_path.stat().st_size,training_checkpoint_bytes=checkpoint.stat().st_size,
                    same_original_state_dict=True,exact_action_Q_parity=True,
                    target_unchanged=True,rule=report['gradient_rule'])
        except Exception:
            passed=False;artifact_error=traceback.format_exc();(out/'artifact_failure.txt').write_text(artifact_error,encoding='utf-8')
    numpy_after=np.random.get_state()
    rng_unchanged=torch.equal(rng_before,torch.get_rng_state()) and numpy_before[0]==numpy_after[0] and np.array_equal(numpy_before[1],numpy_after[1]) and numpy_before[2:]==numpy_after[2:]
    assert rng_unchanged,'Synthetic fixture changed ambient Torch/NumPy RNG state'
    for name,digest in binding['sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in sources.items():assert sha(ROOT/name)==sha(out/'source_snapshot'/name)==digest,name
    loaded=list(sys.modules)
    assert not any(name.endswith(('chapter3_iov_env','chapter3_iov_env_v2','heuristic_pilot_teacher_adapter')) for name in loaded)
    record=dict(status='ISOLATED_MECHANICS_PASSED_RESEARCH_DISABLED' if passed else 'ENGINEERING_FAILURE_PRESERVED_NOT_RESEARCH',
        passed=passed,kind='ENGINEERING_SYNTHETIC_NOT_RESEARCH',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),artifact_failure=artifact_error,
        synthetic_counts=tests.COUNTS,synthetic_optimizer_steps=sum(v for k,v in tests.COUNTS.items() if k.endswith('optimizer_steps')),
        exports=exports,source_sha256=sources,frozen_historical_sources_unchanged=True,
        ambient_torch_numpy_rng_unchanged=rng_unchanged,reserved_research_seeds_used=False,
        research_protocol_frozen=False,research_runner_implemented=False,results_auditor_implemented=False,
        development_release_enabled=False,final_release_enabled=False,
        new_research_optimizer_updates=0,new_simulator_steps=0,new_teacher_calls=0,
        actual_controller_mechanics_tested=True,scientific_quality_established=False,novelty_established=False,
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),threads=torch.get_num_threads()),
        limitation='Synthetic first-order autograd, optimizer and original eager-PyTorch deployment export checks only; no learning-quality result or ONNX/native/UAV validation.',
        research_launch_authorized=False,final_phase_launched=False,word_thesis_changed=False)
    write(out/'verification.json',record)
    write(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps(record,indent=2))
    return passed


def baseline_bytes(model):return sum(p.numel()*p.element_size() for p in model.parameters())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Synthetic mechanics only; no research modes')
    parser.add_argument('--output',type=Path,default=DEFAULT)
    sys.exit(0 if run(parser.parse_args().output) else 1)
