"""Design reservation/algebra checks only. Never imports a controller or simulator."""
import hashlib
import json
from pathlib import Path

import numpy as np
from freeze_heuristic_initialization_pilot_plan_2026_11_19 import ranges

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation/return_calibrated_handoff_brief_2026-11-20_01'
STATUS=ROOT/'RETURN_CALIBRATED_DDQN_HANDOFF_DESIGN_STATUS_2026-11-20.json'
BRIEF=ROOT/'RETURN_CALIBRATED_DDQN_HANDOFF_DESIGN_BRIEF_2026-11-20.txt'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write_new(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def algebra_checks():
    # Explicit deterministic fixtures, no research seeds or model/optimizer.
    z=((np.arange(12*32*5).reshape(12,32,5)%29)-14)/7.
    z[0,0]=[1.,1.,2.,3.,4.]
    masks=np.ones(z.shape,bool);masks[::2,:30,0]=False
    sr=.3;rmax=np.max(np.ptp(z,axis=2).sum(1));alpha=sr/(sr+rmax)
    scaled=(z.astype(np.float32)*np.float32(alpha)).astype(np.float32)
    a=np.where(masks,z,np.inf).argmin(2);aa=np.where(masks,scaled,np.inf).argmin(2)
    assert alpha>0 and np.array_equal(a,aa)
    assert np.ptp(scaled,axis=2).sum(1).max()<=sr+1e-6
    # One explicit unmasked equal-cost tie must retain first-index choice.
    assert z[0,0].argmin()==scaled[0,0].argmin()==0
    def ce(scores,temp):
        logits=np.where(masks,-scores/temp,-np.inf)
        m=logits.max(2,keepdims=True)
        lp=logits-m-np.log(np.exp(logits-m).sum(2,keepdims=True))
        return -np.take_along_axis(lp,a[:,:,None],2).mean()
    assert np.isclose(ce(z,1),ce(scaled,alpha),rtol=1e-6,atol=1e-6)
    # Positive semi-definite/rank-deficient features and constant columns.
    h=(np.arange(40*24).reshape(40,24)%17)/11.
    h[:,0]=1.;h[:,1]=h[:,2]
    t=np.linspace(-9,-1,40);sg=np.sqrt(np.mean(t*t))
    mu=h.mean(0);sigma=np.maximum(1e-6,np.sqrt(np.mean((h-mu)**2,0)))
    H=(h-mu)/sigma;u=t/sg;centered=u-u.mean();rho=1e-3
    beta=np.linalg.solve(H.T@H/len(h)+rho*np.eye(24),H.T@centered/len(h))
    aug=np.concatenate([H/np.sqrt(len(h)),np.sqrt(rho)*np.eye(24)])
    rhs=np.concatenate([centered/np.sqrt(len(h)),np.zeros(24)])
    independent=np.linalg.lstsq(aug,rhs,rcond=None)[0]
    assert np.allclose(beta,independent,atol=1e-10,rtol=1e-9)
    w=sg*beta/sigma;b=sg*u.mean()-w@mu
    expected=sg*(H@beta+u.mean());actual=h@w+b
    assert np.allclose(actual,expected,atol=1e-10,rtol=1e-9)
    export=h.astype(np.float32)@w.astype(np.float32)+np.float32(b)
    assert np.allclose(export,actual,atol=1e-5,rtol=1e-5)
    assert np.isfinite(w).all() and np.isfinite(b)
    reward=np.array([-1.,-2.]);done=np.array([False,True]);bootstrap=np.array([3.,999.])
    target=reward+np.where(done,0,bootstrap)
    assert np.array_equal(target,np.array([2.,-2.]))
    return dict(passed=True,scope='Independent array algebra sanity checks, not model or learning validation',
        checks=['positive FP32 scaling/action/tie preservation and finite-dataset range bound',
                'masked imitation loss invariant when temperature rescales',
                'ridge solve matches independent augmented least squares with deficient features',
                'folded value coefficients and FP32 export reproduce regression',
                'terminal bootstrap arithmetic'],
        synthetic_alpha=float(alpha),ridge_rank=int(np.linalg.matrix_rank(H)),
        ridge_max_coefficient=float(np.abs(w).max()),fp32_export_max_error=float(np.abs(export-actual).max()))


def main():
    if STATUS.exists() or (OUT/'verification.json').exists():raise RuntimeError('Preserve existing design evidence')
    diag=ROOT/'validation/centered_linear_saved_trace_diagnosis_2026-11-20_01'
    dv=read(diag/'verification.json');assert dv['passed']
    for name,h in dv['evidence_sha256'].items():assert sha(diag/name)==h,name
    for name,h in dv['diagnostic_source_sha256'].items():assert sha(ROOT/name)==h,name
    freeze=read(ROOT/'development_runs/centered_linear_correction_comparison_2026-11-20_01/run/PRE_RUN_FREEZE.json')
    for name,h in freeze['source_sha256'].items():assert sha(ROOT/name)==h,name
    seeds=dict(model_seeds=[968280001+10*i for i in range(3)],
        demonstration_seed_blocks=[dict(start=968290000+1000*i,count=10) for i in range(3)],
        online_seed_blocks=[dict(start=968300000+1000*i,count=50) for i in range(3)],
        evaluation_seed_block=dict(start=968310000,count=16),
        bc_sampler_seeds=[968320001+10*i for i in range(3)],
        epsilon_branch_seeds=[968330001+10*i for i in range(3)],
        replay_sampler_seeds=[968340001+10*i for i in range(3)],
        retention_sampler_seeds=[968350001+10*i for i in range(3)],
        job_order_seed=968360001,timing_order_seed=968360011)
    proposed=ranges(seeds);scanned=[];existing=[]
    paths=sorted(ROOT.glob('*.json'))+sorted((ROOT/'validation').glob('*/run/runtime_plan.json'))+sorted((ROOT/'development_runs').glob('*/run/runtime_plan.json'))
    for p in paths:
        v=read(p);scanned.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
        existing.extend((a,b,str(p.relative_to(ROOT))+label) for a,b,label in ranges(v))
    collisions=[]
    for i,(a,b,label) in enumerate(proposed):
        for x,y,old in existing+proposed[:i]:
            if max(a,x)<=min(b,y):collisions.append(dict(proposed=label,existing=old))
    assert not collisions,collisions
    algebra=algebra_checks()
    literature=read(OUT/'literature_retrieval.json')
    assert len(literature['results'])==4
    for row in literature['results']:
        assert row['status']=='retrieved' and row['abstract']
        assert sha(OUT/(row['name']+'.html'))==row['sha256']
    arms=[dict(id='calibrated_retained_ddqn',calibration=True,retention=True,bc=True,ddqn=True,role='primary'),
          dict(id='calibrated_no_retention_ddqn',calibration=True,retention=False,bc=True,ddqn=True,role='retention_ablation'),
          dict(id='uncalibrated_retained_ddqn',calibration=False,retention=True,bc=True,ddqn=True,role='calibration_ablation'),
          dict(id='uncalibrated_no_retention_ddqn',calibration=False,retention=False,bc=True,ddqn=True,role='previous_failed_centered'),
          dict(id='calibrated_bc_only',calibration=True,retention=False,bc=True,ddqn=False,role='no_TD_control'),
          dict(id='old_softplus_bc_ddqn',calibration=False,retention=False,bc=True,ddqn=True,role='previous_baseline'),
          dict(id='plain_small_softplus_ddqn',calibration=False,retention=False,bc=False,ddqn=True,role='ordinary_small_DDQN')]
    assert sum(a['bc'] for a in arms)*3*200==3600
    assert sum(a['ddqn'] for a in arms)*3*(2000-128)==33696
    assert 3*400+6*3*2000+(7*3+1)*16*40==51280
    status=dict(status='PROSPECTIVE_BRIEF_COMPLETE_ISOLATED_MECHANICS_NEXT',
        brief=BRIEF.name,brief_sha256=sha(BRIEF),research_protocol_frozen=False,
        controller_implemented=False,novelty_established=False,
        seeds=seeds,seed_reservations_only=True,arms=arms,
        prospective_constants=dict(gap_scale='S_r/(S_r+R_max)',reward_scale_floor=1e-6,ridge_rho=.001,retention_lambda=1.,calibration_fit_demo_episodes=8,calibration_check_demo_episodes=2),
        proposed_totals=dict(simulated_slots=51280,student_training_slots=36000,evaluation_slots=14080,demo_slots=1200,bc_updates=3600,td_updates=33696,retention_batches=11232,ridge_solves=9,final_checkpoints=21),
        seed_audit=dict(documents=len(scanned),collisions=0,coverage='Root JSON plus validation and development_runs */run/runtime_plan.json; not exhaustive historical log scan'),
        next_gate='isolated synthetic mechanics for calibration and retention; no reserved seeds',
        research_launch_authorized=False,launched=False,final_seeds_consumed=False,word_thesis_changed=False)
    write_new(STATUS,status)
    verification=dict(status='DESIGN_ALGEBRA_AND_SEED_RESERVATIONS_VERIFIED_NOT_RESEARCH_RELEASE',
        algebra=algebra,seed_documents=scanned,seed_collisions=collisions,
        design_status_sha256=sha(STATUS),brief_sha256=sha(BRIEF),preparation_script_sha256=sha(Path(__file__)),
        literature_sha256=sha(OUT/'literature_retrieval.json'),frozen_historical_sources_unchanged=True,
        new_simulator_steps=0,new_optimizer_updates=0,new_teacher_calls=0,
        actual_controller_tested=False,novelty_established=False,word_thesis_changed=False)
    write_new(OUT/'verification.json',verification)
    print(json.dumps(dict(status=status['status'],documents_scanned=len(scanned),seed_collisions=len(collisions),algebra=algebra),indent=2))


if __name__=='__main__':main()
