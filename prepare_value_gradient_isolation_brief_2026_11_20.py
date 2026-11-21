"""Verify design algebra, evidence, reservations and counts; no controller imports."""
import hashlib
import itertools
import json
from pathlib import Path
import re
import numpy as np

from freeze_heuristic_initialization_pilot_plan_2026_11_19 import ranges

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation/value_gradient_isolation_brief_2026-11-20_01'
BRIEF=ROOT/'VALUE_GRADIENT_ISOLATION_DESIGN_BRIEF_2026-11-20.txt'
STATUS=ROOT/'VALUE_GRADIENT_ISOLATION_DESIGN_STATUS_2026-11-20.json'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def algebra_checks():
    # Small deterministic arrays. H is the toy hidden feature variable.
    H=(np.arange(12).reshape(4,3)+1)/13.
    W=((np.arange(21).reshape(7,3)%11)-5)/17.
    bias=np.linspace(-.2,.3,7)
    target=np.array([-.9,-.5,-1.2,-.7])
    actions=np.array([[0,1,0],[1,0,1],[0,0,1],[1,1,0]])
    labels=1-actions;onehot=np.eye(2)[actions];labelhot=np.eye(2)[labels]
    scale=np.array([.5,.5,1.])[None,:,None];tau=.6
    raw=H@W.T+bias;z=raw[:,:6].reshape(4,3,2)*scale
    centered=(onehot*z).sum((1,2))-z.mean(2).sum(1)
    q=raw[:,6]-centered;up=np.zeros((4,7))
    slope=np.clip(q-target,-1,1)/4
    up[:,:6]=(slope[:,None,None]*(.5-onehot)*scale).reshape(4,6)
    up[:,6]=slope
    logits=-z/tau;prob=np.exp(logits-logits.max(2,keepdims=True));prob/=prob.sum(2,keepdims=True)
    keep=np.zeros_like(up);keep[:,:6]=((labelhot-prob)*scale/(4*3*tau)).reshape(4,6)
    expected_hidden=up[:,:6]@W[:6]+keep@W
    ordinary_hidden=(up+keep)@W
    expected_weight=(up+keep).T@H;expected_bias=(up+keep).sum(0)

    def loss(h,w,b,mode):
        td_h=H+(h-H)/np.sqrt(2) if mode=='rescale' else h
        row=td_h@w.T+b
        if mode=='isolate': row[:,6]=H@w[6]+b[6]
        zz=row[:,:6].reshape(4,3,2)*scale
        pred=row[:,6]-((onehot*zz).sum((1,2))-zz.mean(2).sum(1))
        delta=np.abs(pred-target);td=np.where(delta<1,.5*delta**2,delta-.5).mean()
        kk=(h@w.T+b)[:,:6].reshape(4,3,2)*scale
        ll=-kk/tau;maximum=ll.max(2,keepdims=True)
        logsum=maximum[:,:,0]+np.log(np.exp(ll-maximum).sum(2))
        imitation=(logsum-(labelhot*ll).sum(2)).mean()
        return td+imitation

    def finite_difference(x,fn):
        answer=np.zeros_like(x);eps=1e-6
        for idx in np.ndindex(x.shape):
            p=x.copy();m=x.copy();p[idx]+=eps;m[idx]-=eps
            answer[idx]=(fn(p)-fn(m))/(2*eps)
        return answer

    values=[loss(H,W,bias,mode) for mode in ['ordinary','isolate','rescale']]
    assert values[0]==values[1]==values[2]
    assert np.allclose(finite_difference(H,lambda v:loss(v,W,bias,'isolate')),expected_hidden,rtol=1e-6,atol=1e-8)
    assert np.allclose(finite_difference(W,lambda v:loss(H,v,bias,'isolate')),expected_weight,rtol=1e-6,atol=1e-8)
    assert np.allclose(finite_difference(bias,lambda v:loss(H,W,v,'isolate')),expected_bias,rtol=1e-6,atol=1e-8)
    assert np.allclose(ordinary_hidden-expected_hidden,up[:,6,None]*W[6],rtol=1e-12,atol=1e-12)
    rescaled=(up@W)/np.sqrt(2)+keep@W
    assert np.allclose(finite_difference(H,lambda v:loss(v,W,bias,'rescale')),rescaled,rtol=1e-6,atol=1e-8)
    # Pure V upstream must still update V weights/bias but not the features.
    pure_v=np.zeros_like(up);pure_v[:,6]=[.1,.2,.3,.4]
    assert not np.any(pure_v[:,:6]@W[:6]) and np.linalg.norm(pure_v.T@H)>0
    assert not np.any(keep[:,6]) and np.linalg.norm(keep@W)>0
    # Exact masked additive maximum, including a deterministic equal-score tie.
    toy=np.array([[0.,0.],[2.,1.],[0.,3.]])
    mask=np.array([[True,True],[False,True],[True,True]])
    greedy=np.where(mask,toy,np.inf).argmin(1)
    candidates=[a for a in itertools.product(range(2),repeat=3) if all(mask[b,a[b]] for b in range(3))]
    cost=lambda a:sum(toy[b,a[b]]-toy[b].mean() for b in range(3))
    assert tuple(greedy)==min(candidates,key=cost) and greedy[0]==0
    # Queued task toy: action-score optimality alone is not deadline feasibility.
    assert np.count_nonzero(np.array([.2,.4])<=.3)==1
    assert np.count_nonzero(np.array([.25,.25])<=.3)==2
    architecture={str(w):dict(parameters=(533+1)*w+(w+1)*161,
        FP32_parameter_bytes=4*((533+1)*w+(w+1)*161),dense_weight_products=533*w+w*161) for w in [24,64,128]}
    assert [architecture[str(w)]['parameters'] for w in [24,64,128]]==[16841,44641,89121]
    assert [architecture[str(w)]['dense_weight_products'] for w in [24,64,128]]==[16656,44416,88832]
    return dict(passed=True,scope='Independent deterministic NumPy algebra and finite differences, not controller tests or learning evidence',
        checks=['identical forward loss at matched tensors for all three gradient rules',
            'isolated hidden gradient matches finite differences with value features held constant',
            'readout weight and bias gradients preserved',
            'ordinary-minus-isolated hidden gradient equals exactly the V path',
            'adapted rescale changes TD hidden gradient only; auxiliary gradients preserved',
            'pure V hidden block, live V readout and no V imitation gradient',
            'masked additive maximum and deterministic tie by enumeration',
            'within-slot queue counterexample and exact architecture arithmetic'],
        architecture=architecture,toy_loss=values[0],toy_removed_value_hidden_gradient_norm=float(np.linalg.norm(ordinary_hidden-expected_hidden)))


def main():
    if STATUS.exists() or (OUT/'verification.json').exists():raise RuntimeError('Preserve previous brief evidence')
    diagnosis=ROOT/'validation/return_calibrated_saved_trace_diagnosis_2026-11-20_01'
    receipt=read(diagnosis/'review_verification.json')
    assert receipt['status']=='OFFLINE_DIAGNOSIS_REVIEW_VERIFIED_NO_RESEARCH_UPDATES'
    for name,digest in receipt['artifacts_sha256'].items():assert sha(diagnosis/name)==digest,name
    for name,digest in receipt['source_and_report_sha256'].items():assert sha(ROOT/name)==digest,name
    old=read(ROOT/'RETURN_CALIBRATED_HANDOFF_COMPARISON_PROTOCOL_2026-11-20.json')
    assert sha(ROOT/'RETURN_CALIBRATED_HANDOFF_COMPARISON_PROTOCOL_2026-11-20.json')=='1602eda36acad77475f0c734fde10448daa2f1b96243c00d3f6f257870a7bee1'
    binding=read(ROOT/'RETURN_CALIBRATED_COMPARISON_LAUNCH_BINDING_2026-11-20.json')
    for name,digest in binding['sha256'].items():assert sha(ROOT/name)==digest,name
    literature=read(OUT/'literature_retrieval.json')
    assert len(literature['results'])==11
    for row in literature['results']:
        assert row['status']=='retrieved',row['name']
        assert sha(OUT/row['file'])==row['sha256']
        if row['kind']=='abstract':assert row['abstract'] and row['title']
    for row in read(OUT/'offloading_overlap_refresh.json'):
        assert row['status']=='retrieved' and sha(OUT/row['file'])==row['sha256']
    dueling=(OUT/'dueling_fulltext.html').read_text(encoding='utf-8')
    decoupled=(OUT/'decoupled_representation_fulltext.html').read_text(encoding='utf-8')
    assert 'rescale the combined gradient' in re.sub('<[^>]+>',' ',dueling)
    assert 'fully detached' in re.sub('<[^>]+>',' ',decoupled)
    assert sha(ROOT/'collect_value_gradient_isolation_literature_2026_11_20.py')==literature['collector_sha256']
    seeds=dict(model_seeds=[978480001+10*i for i in range(3)],
        demonstration_seed_blocks=[dict(start=978490000+1000*i,count=10) for i in range(3)],
        online_seed_blocks=[dict(start=978500000+1000*i,count=50) for i in range(3)],
        evaluation_seed_block=dict(start=978510000,count=16),
        bc_sampler_seeds=[978520001+10*i for i in range(3)],
        epsilon_branch_seeds=[978530001+10*i for i in range(3)],
        replay_sampler_seeds=[978540001+10*i for i in range(3)],
        retention_sampler_seeds=[978550001+10*i for i in range(3)],
        job_order_seed=978560001,timing_order_seed=978560011)
    scanned=[];existing=[]
    paths=sorted(ROOT.glob('*.json'))+sorted((ROOT/'validation').glob('*/run/runtime_plan.json'))+sorted((ROOT/'development_runs').glob('*/run/runtime_plan.json'))
    for path in paths:
        data=read(path);scanned.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path)))
        existing.extend((a,b,path.relative_to(ROOT).as_posix()+label) for a,b,label in ranges(data))
    proposed=ranges(seeds);collisions=[]
    for i,(lo,hi,label) in enumerate(proposed):
        for a,b,old_label in existing+proposed[:i]:
            if max(lo,a)<=min(hi,b):collisions.append(dict(proposed=label,existing=old_label))
    assert not collisions,collisions
    arms=[dict(id='value_gradient_isolated_ddqn',calibration=True,retention=True,bc=True,ddqn=True,
               role='prospective_primary',TD_shared_rule='branch path only'),
          dict(id='dueling_shared_td_rescale_ddqn',calibration=True,retention=True,bc=True,ddqn=True,
               role='adapted_known_gradient_rule_control',TD_shared_rule='(value+branch)/sqrt(2)')]
    arms.extend(old['arms'])
    assert len(arms)==9
    totals=dict(demonstration_slots=3*10*40,online_slots=sum(a['ddqn'] for a in arms)*3*2000,
        evaluation_slots=(len(arms)*3+1)*16*40,bc_updates=sum(a['bc'] for a in arms)*3*200,
        ddqn_updates=sum(a['ddqn'] for a in arms)*3*1872,
        retention_batches=sum(a['retention'] for a in arms)*3*1872,
        ridge_solves=sum(a['calibration'] for a in arms)*3,final_checkpoints=len(arms)*3,
        timing_calls=(len(arms)*3+1)*3*128,timing_warmups=(len(arms)*3+1)*3*32)
    totals['simulated_slots']=sum(totals[k] for k in ['demonstration_slots','online_slots','evaluation_slots'])
    assert (totals['simulated_slots'],totals['bc_updates'],totals['ddqn_updates'],totals['retention_batches'],totals['ridge_solves'],totals['final_checkpoints'])==(67120,4800,44928,22464,15,27)
    algebra=algebra_checks()
    status=dict(status='PROSPECTIVE_DESIGN_AND_LITERATURE_CHECK_COMPLETE_MECHANICS_NEXT',
        brief=BRIEF.name,brief_sha256=sha(BRIEF),inherited_protocol='RETURN_CALIBRATED_HANDOFF_COMPARISON_PROTOCOL_2026-11-20.json',
        inherited_protocol_sha256=sha(ROOT/'RETURN_CALIBRATED_HANDOFF_COMPARISON_PROTOCOL_2026-11-20.json'),
        change='Block only V-to-hidden TD gradient; preserve forward function, trainable readout, branch TD and imitation gradients',
        primary='value_gradient_isolated_ddqn',arms=arms,seeds=seeds,seed_reservations_only=True,
        proposed_totals=totals,paired_policy_controls=['calibrated_retained_ddqn','dueling_shared_td_rescale_ddqn','calibrated_bc_only','old_softplus_bc_ddqn'],
        prospective_policy_rule='nonnegative return and deadline contrast every fit, positive equal-fit means in both for every required control',
        prospective_value_control='uncalibrated_no_retention_ddqn',
        prospective_value_rule='own-policy NRMSE and absolute Q-G bias strictly smaller every fit',
        broader_comparison_required=['64/128 capacity DDQN references','adapted BDQ','specified full DQfD-style reference','ordinary small DDQN','BC-only','old baseline','observed-load/radio','all-local','nearest-RSU','UAV-selective'],
        research_protocol_frozen=False,actual_controller_implemented=False,model_tested=False,novelty_established=False,
        seed_audit=dict(documents=len(scanned),collisions=0,coverage='Root JSON and validation/development_runs */run/runtime_plan.json; not a full historical text/log scan'),
        next_gate='isolated synthetic routing/rescaling helpers, autograd/reference/forward/export checks; no research seeds or simulator',
        new_simulator_steps=0,new_optimizer_updates=0,new_teacher_calls=0,research_launch_authorized=False,
        launched=False,final_seeds_consumed=False,word_thesis_changed=False)
    write(STATUS,status)
    verification=dict(status='DESIGN_EVIDENCE_ALGEBRA_COUNTS_AND_SEED_RESERVATIONS_VERIFIED_NOT_RESEARCH_RELEASE',
        algebra=algebra,seed_documents=scanned,seed_collisions=collisions,source_binding_unchanged=True,
        diagnosis_receipt_sha256=sha(diagnosis/'review_verification.json'),brief_sha256=sha(BRIEF),
        design_status_sha256=sha(STATUS),preparation_script_sha256=sha(Path(__file__)),
        literature_retrieval_sha256=sha(OUT/'literature_retrieval.json'),offloading_overlap_refresh_sha256=sha(OUT/'offloading_overlap_refresh.json'),
        actual_controller_implemented=False,actual_model_tested=False,novelty_established=False,
        new_simulator_steps=0,new_optimizer_updates=0,new_teacher_calls=0,new_autograd_evaluations=0,
        research_launch_authorized=False,word_thesis_changed=False)
    write(OUT/'verification.json',verification)
    write(OUT/'manifest.json',{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}|
        {p.name:sha(p) for p in [BRIEF,STATUS,Path(__file__),ROOT/'collect_value_gradient_isolation_literature_2026_11_20.py']})
    print(json.dumps(dict(status=status['status'],seed_documents_checked=len(scanned),seed_collisions=collisions,
        proposed_totals=totals,algebra=algebra),indent=2))


if __name__=='__main__':main()
