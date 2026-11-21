"""Read-only saved-evidence review plus isolated gradient checks; no fitting."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from torch.nn import functional as F
from spi_ddqn import SPIDDQN

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(1)
    out=ROOT/'analysis/spi_objective_review_2026-11-04'
    out.mkdir(parents=True,exist_ok=False)
    smoke=ROOT/'validation/low_complexity_smoke_2026-11-01_03'
    beam_path=ROOT/'validation/spi_beam_coverage_2026-11-01_01/verification.json'
    beam=json.loads(beam_path.read_text())
    with np.load(ROOT/'observations.npz') as z:states=torch.tensor(z['observations'][:4])
    evidence={}
    inputs=[ROOT/'spi_ddqn.py',ROOT/'run_low_complexity_smoke.py',ROOT/'audit_spi_beam_coverage.py',beam_path]
    for rep in (0,1):
        ckpath=smoke/f'spi_{rep}/final.pt';inputs.append(ckpath)
        ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        online=SPIDDQN();online.load_state_dict(ck['online'])
        target=SPIDDQN();target.load_state_dict(ck['target'])
        # Reproduce the loss graph only, with disjoint synthetic legal actions.
        # No optimizer.step and no simulator calls; checkpoints never changed.
        actions=torch.arange(128).reshape(4,32)%5
        q=online.joint_values(states,actions)
        y=-torch.ones(4)+target.joint_values(states,actions)
        F.smooth_l1_loss(q,y).backward()
        assert all(p.grad is None for p in online.component_head.parameters())
        assert any(p.grad is not None and torch.any(p.grad!=0) for p in target.parameters())
        # Model init seeds recorded by the actual smoke loop.
        torch.manual_seed(1601+rep);initial=SPIDDQN()
        head_equal=all(torch.equal(a,b) for a,b in zip(initial.component_head.parameters(),online.component_head.parameters()))
        assert head_equal
        rows=beam['results'][f'spi_{rep}']['rows']
        gaps=np.array([r['full_best_score']-r['base_best_score'] for r in rows])
        # Old field was forced zero; replace only in new evidence, not old files.
        evidence[f'spi_{rep}']=dict(generator_head_unchanged_from_initialization=head_equal,
            generator_head_gradients_absent=True,target_gradients_present_in_original_loss_graph=True,
            mean_wider_minus_base_score=float(gaps.mean()),p95_wider_minus_base_score=float(np.quantile(gaps,.95)),
            max_wider_minus_base_score=float(gaps.max()),min_wider_minus_base_score=float(gaps.min()),
            old_gap_field_is_forced_placeholder=all(r['base_full_score_gap']==0 for r in rows),
            all_greedy_candidates_count8=all(r['base_candidates']==8 for r in rows))
    verification=json.loads((smoke/'verification.json').read_text())
    assert sum(r['slots'] for r in verification['runs'].values())==320
    assert sum(r['updates'] for r in verification['runs'].values())==292
    result=dict(state='passed_read_only_objective_audit',date='2026-11-04',results=evidence,
        actual_training_slots_all_four_runs=320,actual_updates_all_four_runs=292,
        reported_totals160_and146_omit_one_arm=True,
        beam128_is_not_exhaustive_5power32=True,new_optimizer_updates=0,new_simulator_steps=0,
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},script_sha256=sha(Path(__file__)))
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
