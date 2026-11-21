"""Predeclare one backend-only correction before testing saved weights."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
P=R/'ANCHOR_RUNTIME_NUMERICAL_CORRECTION_PROTOCOL_2026-11-11.json';assert not P.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
x=dict(id='anchor-runtime-disable-graph-optimization-v1',date='2026-11-11',kind='SAVED_WEIGHT_ENGINEERING_NOT_RESEARCH',
 hypothesis='Disabling ORT graph optimization may remove fusion-related rounding changes at near ties; outcome unknown.',
 candidate='ORT CPU FP32 sequential one thread, graph optimization disabled; original exported graphs and native decoder unchanged.',
 scientific_protocol_sha256=sha(R/'TERMINAL_ANCHOR_RESEARCH_PROTOCOL_2026-11-06.json'),
 checkpoint_freeze_sha256=sha(R/'development_runs/terminal_anchor_development_2026-11-06_01/ALL_FINAL_CHECKPOINTS_FREEZE.json'),
 observations_sha256='2acd6bf97d649df97e0084f6b61a265d3742a70421410c9ccabc6cfba1854ebb',
 models=85,observations_per_model=128,exact_joint_actions_required=True,Q_atol=1e-6,Q_rtol=1e-5,
 input='Only frozen final checkpoints and existing fixed engineering observations; no simulator or held-out episodes.',
 reference='Original batch-one PyTorch greedy implementation used in training; no reference changes.',
 baseline='Original enable-all ORT route on identical graphs, report all mismatches without stopping at first.',
 localization='For price/plain families additionally replay original PyTorch neural-core output through original native decoder. Diagnostic only, not a second accepted route.',
 regression='Known no_price_both fit0 observation110 branch18: baseline must reproduce destination0 vs reference4; candidate must choose reference4.',
 value_and_action_failures='Report every model and mismatch; exact action criterion not weakened for near ties.',
 costs='Only if every candidate model passes, one five-round benchmark with 128 calls and32 warmups, original order seed623600003; no timing retries.',
 unchanged=['model weights','network architecture','reward','DDQN update','optimizer','seeds','screen thresholds','native action decoder'],
 no_retraining=True,no_heldout_evaluation=True,final_disabled=True,word_thesis_changed=False,
 success_next='Review correction and same-weight cost evidence before implementing a separate posttraining continuation.',
 failure_next='Preserve failed candidate. No timing or heldout evaluation; diagnose numerical route separately. No automatic route selection.')
P.write_text(json.dumps(x,indent=2),encoding='utf-8');print(P.name)
