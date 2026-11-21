"""Declare a distinct same-weight arithmetic route, before equivalence or timing."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
P=R/'ANCHOR_TORCH_CORE_RUNTIME_AMENDMENT_2026-11-11.json';assert not P.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((R/'ANCHOR_RUNTIME_NUMERICAL_CORRECTION_PROTOCOL_2026-11-11.json').read_text())
x=dict(old)
x.update(id='anchor-original-pytorch-core-native-decoder-v1',
    hypothesis='Original PyTorch neural-core arithmetic removes cross-backend near-tie changes. Retain unchanged native decoders. Complete cost impact unknown.',
    candidate='Every native-decoder family uses original PyTorch CPU FP32 neural core with one thread and original weights; coupled full-policy ORT families retain original route.',
    baseline='Original enable-all ORT core and existing native decoder; same exported graphs and original full-policy ORT families.',
    preceding_failed_candidate='validation/anchor_numerical_correction_2026-11-11_01',
    preceding_manifest_sha256=sha(R/'validation/anchor_numerical_correction_2026-11-11_01/COMPLETION_MANIFEST.json'),
    route_fairness='Apply core replacement to all native-decoder families, including compact64 and wide128 references, BDQ and EMC, not selectively to the failing comparator.',
    deployment_storage='Existing online FP32 tensor payload definition retained; include all core parameters and required auxiliary buffers; Python/PyTorch runtime dependency and process memory not claimed lightweight.',
    no_model_selection=True,reference_policy_unchanged=True,
    latency='Complete original input validation, original neural core, native decoder, horizon scaling and final action copy. Single five-round benchmark; no retiming or fit selection.',
    no_heldout_evaluation=True,continuation_not_authorized_by_equivalence=True)
P.write_text(json.dumps(x,indent=2),encoding='utf-8');print(P.name)
