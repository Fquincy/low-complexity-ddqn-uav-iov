"""Generate new continuation auditor/inference helper; preserve frozen predecessors."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
s=(R/'run_anchor_development_v4.py').read_text(encoding='utf-8')
a=s.index('def frozen_inference_gate(');b=s.index('\ndef eval_snapshot(',a)
f=s[a:b].replace('fused.export(', 'export_original_core(').replace("'native_core_and_decoder'", "'original_torch_core_native_decoder'")
f=f.replace("backend='ORT CPU FP32 sequential one-thread; native fastmath and parallel disabled'", "backend='Original PyTorch FP32 cores one-thread plus native decoder; four coupled families full-policy ORT CPU FP32'")
f=f.replace("frozen_checkpoint_sha256=frozen['checkpoints'],", "runtime_amendment_sha256=base.old.sha(ROOT/'ANCHOR_TORCH_CORE_RUNTIME_AMENDMENT_2026-11-11.json'),\n        continuation_source_sha256={n:base.old.sha(ROOT/n) for n in CONTINUATION_SOURCES},\n        frozen_checkpoint_sha256=frozen['checkpoints'],")
path=R/'anchor_continuation_inference.py';assert not path.exists()
path.write_text('''"""Complete saved-checkpoint inference gate for the declared original-core route."""
from run_anchor_development_v4 import *
from anchor_original_torch_core_runtime import TorchCoreSession
CONTINUATION_SOURCES=['run_anchor_posttraining_continuation.py','audit_anchor_posttraining_continuation.py',
 'anchor_continuation_inference.py','anchor_original_torch_core_runtime.py',
 'ANCHOR_TORCH_CORE_RUNTIME_AMENDMENT_2026-11-11.json']
def export_original_core(model,arm,path):
    (_,extra),meta=fused.export(model,arm,path)
    return (TorchCoreSession(model,arm),extra),meta

'''+f,encoding='utf-8')
s=(R/'audit_anchor_development_v4.py').read_text(encoding='utf-8')
s=s.replace('import run_anchor_development_v4 as runner','import run_anchor_posttraining_continuation as runner\nfrom anchor_original_torch_core_runtime import TorchCoreSession')
s=s.replace("'native_core_and_decoder'", "'original_torch_core_native_decoder'")
s=s.replace("session=compiled.session(path);mode='native'", "session=TorchCoreSession(model.model if arm in runner.CORRECTED else model,runner.anchor.ARMS[arm][0] if arm in runner.CORRECTED else arm);mode='native'")
a=s.index('            opts=session.get_session_options()');b=s.index('            sessions[arm]=',a)
block=s[a:b];s=s[:a]+"            if mode=='complete':\n"+''.join('    '+line+'\n' for line in block.splitlines())+s[b:]
s=s.replace("    assert set(plan['source_sha256'])==set(runner.SOURCES)","    runner.audit_continuation_inputs(folder)\n    assert set(plan['source_sha256'])==set(runner.read(folder/'PRE_CONTINUATION_FREEZE.json')['original_source_sha256'])")
s=s.replace("    assert verification['research_training_launched']==(not synthetic)","    assert verification['research_training_launched'] is False\n    assert verification['new_training_slots']==verification['new_optimizer_updates']==0")
s=s.replace("'passed_v3_synthetic_shared_research_audit' if synthetic else 'passed_v3_research_saved_artifact_audit'", "'passed_synthetic_posttraining_continuation_audit' if synthetic else 'passed_posttraining_continuation_full_audit'")
s=s.replace("        checkpoints_checked=len(fit_hashes)","        new_training_slots=0,new_optimizer_updates=0,checkpoints_checked=len(fit_hashes)")
s=s.replace("    audit_timing_freeze(folder)","    assert pre['runtime_amendment_sha256']==base.old.sha(runner.ROOT/'ANCHOR_TORCH_CORE_RUNTIME_AMENDMENT_2026-11-11.json')\n    assert pre['continuation_source_sha256']==runner.read(folder.parent/'PRE_CONTINUATION_FREEZE.json')['continuation_source_sha256']\n    audit_timing_freeze(folder)")
path=R/'audit_anchor_posttraining_continuation.py';assert not path.exists();path.write_text(s,encoding='utf-8')
print('Created new inference gate and full continuation auditor.')
