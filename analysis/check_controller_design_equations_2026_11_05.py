"""Synthetic algebra checks only; not an implemented neural controller or simulator run."""
import itertools
import json
import math
import random
from pathlib import Path

OUT=Path(__file__).resolve().parent/'controller_redesign_2026-11-05'
rng=random.Random(99002)  # Synthetic algebra fixture; not any research seed.

def score(a,u,w,L,beta):
    n=len(a)
    A=[sum(w[i][d] for i in range(n) if a[i]==d) for d in range(5)]
    return sum(u[i][d] for i,d in enumerate(a))+sum(beta[d]*((L[d]+A[d])**2-L[d]**2)/(4*n) for d in range(1,5))

def price(i,d,a,w,L,beta):
    if d==0:return 0.
    H=L[d]+sum(w[k][d] for k in range(len(a)) if k!=i and a[k]==d)
    return beta[d]*(2*H*w[i][d]+w[i][d]**2)/(4*len(a))

def decode(u,w,L,beta,repair_rounds=0):
    a=tuple(min(range(5),key=lambda d:u[i][d]) for i in range(len(u)))
    candidates=[a]
    for _ in range(2):
        a=tuple(min(range(5),key=lambda d:u[i][d]+price(i,d,a,w,L,beta)) for i in range(len(u)))
        candidates.append(a)
    values=[score(a,u,w,L,beta) for a in candidates]
    j=min(range(3),key=lambda k:values[k])
    selected=candidates[j]
    current=values[j]
    for _ in range(repair_rounds):
        choices=[]
        for i in range(len(u)):
            old=selected[i]
            for d in range(5):
                delta=u[i][d]-u[i][old]+price(i,d,selected,w,L,beta)-price(i,old,selected,w,L,beta)
                moved=list(selected);moved[i]=d
                exact=score(moved,u,w,L,beta)-current
                assert math.isclose(delta,exact,rel_tol=1e-10,abs_tol=1e-10)
                choices.append((delta,i,d))
        delta,i,d=min(choices)
        if delta < -1e-6:
            moved=list(selected);moved[i]=d
            if score(moved,u,w,L,beta)<current-1e-6:
                selected=tuple(moved)
        new_cost=score(selected,u,w,L,beta)
        assert new_cost <= current+1e-12
        current=new_cost
    return selected,current,values[0]

gaps=[];repaired_gaps=[];checks=0
for fixture in range(40):
    n=4
    u=[[rng.uniform(.01,1)/n for _ in range(5)] for _ in range(n)]
    w=[[0.]+[rng.uniform(.02,1) for _ in range(4)] for _ in range(n)]
    L=[0.]+[rng.uniform(0,2) for _ in range(4)]
    beta=[0.]+[rng.uniform(.01,2) for _ in range(4)]
    a,value,initial=decode(u,w,L,beta)
    optimum=min(score(b,u,w,L,beta) for b in itertools.product(range(5),repeat=n))
    assert value>=-1e-12 and value<=initial+1e-12 and value>=optimum-1e-12
    gaps.append(value-optimum)
    repaired,rv,_=decode(u,w,L,beta,repair_rounds=4)
    assert optimum-1e-12<=rv<=value+1e-12
    repaired_gaps.append(rv-optimum)
    for i in range(n):
        for d in range(5):
            local=list(a);local[i]=0
            varied=list(a);varied[i]=d
            actual=score(varied,u,w,L,beta)-score(local,u,w,L,beta)
            expected=u[i][d]-u[i][0]+price(i,d,a,w,L,beta)
            assert math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10)
            checks+=1
    za,zv,_=decode(u,w,L,[0.]*5)
    exact=sum(min(row) for row in u)
    assert math.isclose(zv,exact,rel_tol=1e-10,abs_tol=1e-10)

u=[[10.,0.,0.,20.,20.],[10.,0.,0.,20.,20.]]
w=[[0.,1.,1.,1.,1.],[0.,1.,1.,1.,1.]]
L=[0.]*5;beta=[0.,1.,1.,1.,1.]
_,symmetric_cost,_=decode(u,w,L,beta)
symmetric_optimum=min(score(a,u,w,L,beta) for a in itertools.product(range(5),repeat=2))
assert symmetric_cost>symmetric_optimum+.1
_,repaired_symmetric,_=decode(u,w,L,beta,repair_rounds=4)
assert math.isclose(repaired_symmetric,symmetric_optimum,abs_tol=1e-12)

result=dict(status='passed',kind='synthetic_equation_checks_not_research_or_runner_tests',synthetic_seed=99002,
            physical_simulator_steps=0,optimizer_updates=0,fixtures=40,marginal_cost_identities=checks,
            retained_candidate_dominance=True,zero_penalty_exactness=True,
            exact_fixture_fraction=sum(g<1e-10 for g in gaps)/len(gaps),
            maximum_fixture_cost_gap=max(gaps),mean_fixture_cost_gap=sum(gaps)/len(gaps),
            symmetric_counterexample=dict(decoder_cost=symmetric_cost,exhaustive_cost=symmetric_optimum,
                                          gap=symmetric_cost-symmetric_optimum,meaning='synchronous_choices_can_oscillate_together_and_miss_balanced_assignment'),
            v2_four_repairs=dict(symmetric_cost=repaired_symmetric,mean_random_fixture_gap=sum(repaired_gaps)/len(repaired_gaps),
                                 maximum_random_fixture_gap=max(repaired_gaps),nonincreasing_full_cost_verified=True,
                                 exact_single_move_delta_verified=True,guarantee='does_not_guarantee_global_optimality'),
            conclusion='price_decoder_is_approximate_nonzero_optimality_gaps_expected_no_policy_evidence')
OUT.mkdir(exist_ok=True)
(OUT/'synthetic_equation_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
