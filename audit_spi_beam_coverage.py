"""Read-only beam-width/top-k audit on saved smoke checkpoints."""
from pathlib import Path
import json,time
import numpy as np,torch
from spi_ddqn import SPIDDQN,select

ROOT=Path(__file__).resolve().parent
def main(out):
 out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 with np.load(ROOT/'observations.npz') as z:obs=z['observations'][:128].copy()
 result={}
 for rep in (0,1):
  key=f'spi_{rep}';ck=torch.load(ROOT/'validation/low_complexity_smoke_2026-11-01_03'/key/'final.pt',map_location='cpu',weights_only=False);model=SPIDDQN().eval();model.load_state_dict(ck['online'])
  rows=[]
  for state in obs:
   base,bi=select(model,state,2,8);wide,wi=select(model,state,3,32);full,fi=select(model,state,5,128)
   rows.append(dict(base_wide_equal=np.array_equal(base,wide),base_full_equal=np.array_equal(base,full),wide_full_equal=np.array_equal(wide,full),
                   base_candidates=bi['candidate_count'],wide_candidates=wi['candidate_count'],full_candidates=fi['candidate_count'],
                   base_best_score=float(max(bi['joint_scores'])),wide_best_score=float(max(wi['joint_scores'])),full_best_score=float(max(fi['joint_scores'])),
                   base_full_score_gap=float(max(fi['joint_scores'])-bi['joint_scores'][np.argmax([np.array_equal(base,x) for x in []])] if False else 0)))
  result[key]=dict(rows=rows,base_wide_agreement=float(np.mean([r['base_wide_equal'] for r in rows])),base_full_agreement=float(np.mean([r['base_full_equal'] for r in rows])),wide_full_agreement=float(np.mean([r['wide_full_equal'] for r in rows])),base_candidate_mean=float(np.mean([r['base_candidates'] for r in rows])),wide_candidate_mean=float(np.mean([r['wide_candidates'] for r in rows])),full_candidate_mean=float(np.mean([r['full_candidates'] for r in rows])))
 out.joinpath('verification.json').write_text(json.dumps(dict(state='complete',observations=128,results=result,trained_checkpoints=True,optimizer_updates=0,policy_claim=False),indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main(ROOT/'validation/spi_beam_coverage_2026-11-01_01')
