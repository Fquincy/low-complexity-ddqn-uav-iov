"""Exact rational specification checks, not a trained model or simulator prototype."""
from fractions import Fraction as F
from itertools import product
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation/assignment_price_design_algebra_2026-11-15_01'
BRIEF='ASSIGNMENT_DEPENDENT_PRICE_DESIGN_BRIEF_2026-11-15.txt'

def potential(a,w,q,beta,eta,g):
    n=len(a);d=len(q);result=sum(g[i][a[i]] for i in range(n))
    for j in range(d):
        values=[w[i][j] for i in range(n) if a[i]==j]
        W=sum(values,F(0));S=sum((x*x for x in values),F(0))
        result+=beta[j]*(2*q[j]*W+eta[j]*W*W+(1-eta[j])*S)/(4*n)
    return result

def expanded(a,w,q,beta,eta,g):
    n=len(a);answer=sum(g[i][a[i]] for i in range(n))
    for d in range(len(q)):
        ix=[i for i,x in enumerate(a) if x==d]
        self_terms=sum((2*q[d]*w[i][d]+w[i][d]**2 for i in ix),F(0))
        cross=sum((2*eta[d]*w[i][d]*w[j][d] for pos,i in enumerate(ix) for j in ix[pos+1:]),F(0))
        answer+=beta[d]*(self_terms+cross)/(4*n)
    return answer

def decode(w,q,beta,eta,g,allowed,order):
    n=len(w);D=len(q)
    old=lambda i,d:g[i][d]+beta[d]*(2*(q[d]+eta[d]*sum(w[j][d] for j in range(n) if j!=i))*w[i][d]+w[i][d]**2)/(4*n)
    a=[min(allowed[i],key=lambda d:(old(i,d),d)) for i in range(n)]
    initial=list(a);costs=[potential(a,w,q,beta,eta,g)];evaluations=0
    for i in order:
        def insert(d):
            others=sum(w[j][d] for j in range(n) if j!=i and a[j]==d)
            return g[i][d]+beta[d]*(2*(q[d]+eta[d]*others)*w[i][d]+w[i][d]**2)/(4*n)
        scores={d:insert(d) for d in allowed[i]};evaluations+=len(scores)
        best=min(scores.values());choice=a[i] if scores[a[i]]==best else min(d for d in scores if scores[d]==best)
        a[i]=choice;costs.append(potential(a,w,q,beta,eta,g));assert costs[-1]<=costs[-2]
    return a,initial,costs,evaluations

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    checks=0;insertions=0;cases=[]
    # Deterministic synthetic algebra inputs, no model/task/RNG reservations.
    for case in range(12):
        n,D=3,3
        w=[[F(0)]+[F((i+1)*(j+1)+case%3,20) for j in range(1,D)] for i in range(n)]
        q=[F(0),F(case%4,7),F((case+1)%4,9)]
        beta=[F(0),F(case%3+1,4),F((case+1)%3+1,4)];eta=[F(1,3)]*3
        g=[[F((i+2*j+case)%7,30) for j in range(D)] for i in range(n)]
        allowed=[[1,2] if (case+i)%3==0 else [0,1,2] for i in range(n)]
        assignments=list(product(*allowed))
        for a in assignments:
            value=potential(a,w,q,beta,eta,g);assert value==expanded(a,w,q,beta,eta,g);assert value>=0;checks+=1
            for i in range(n):
                old=a[i]
                def insertion(d):
                    others=sum(w[j][d] for j in range(n) if j!=i and a[j]==d)
                    return g[i][d]+beta[d]*(2*(q[d]+eta[d]*others)*w[i][d]+w[i][d]**2)/(4*n)
                for d in allowed[i]:
                    b=list(a);b[i]=d
                    assert potential(b,w,q,beta,eta,g)-value==insertion(d)-insertion(old)
                    insertions+=1
        result,initial,costs,count=decode(w,q,beta,eta,g,allowed,list(reversed(range(n))))
        optimum=min(potential(a,w,q,beta,eta,g) for a in assignments)
        assert costs[-1]>=optimum and count==sum(map(len,allowed)) and len(costs)==n+1
        cases.append(dict(case=case,assignments=len(assignments),nonincrease=True,candidate_evaluations=count,
            final_cost=str(costs[-1]),exact_optimum=str(optimum)))
    # A coordinate method can remain suboptimal: requires a two-task swap.
    weights=[F(x,100) for x in [8,7,6,5]]
    w=[[F(0),v,v] for v in weights];q=[F(0)]*3;beta=[F(0),F(1),F(1)];eta=[F(1,3)]*3
    g=[[F(1,100)]*3 for _ in range(4)];allowed=[[1,2]]*4
    result,initial,costs,count=decode(w,q,beta,eta,g,allowed,list(range(4)))
    possibilities=list(product(*allowed));best=min(possibilities,key=lambda a:potential(a,w,q,beta,eta,g))
    optimum=potential(best,w,q,beta,eta,g);assert costs[-1]>optimum
    # With no remote penalty the best placement tables remain unchanged.
    zero=[F(0)]*3;out,_,_,_=decode(w,q,zero,eta,g,allowed,list(range(4)))
    assert out==[1]*4
    report=dict(passed=True,exact_potential_assignments=checks,exact_marginal_differences=insertions,
        deterministic_cases=cases,suboptimal_counterexample=dict(weights=list(map(str,weights)),
            initial=initial,one_pass=result,final_cost=str(costs[-1]),better_assignment=best,
            exact_optimum=str(optimum),gap=str(costs[-1]-optimum)),
        scope='Exact rational algebra only. Reference enumeration is deliberately slow and is not the bounded production decoder or a model/gradient/runtime test.',
        simulator_steps=0,optimizer_updates=0,model_initializations=0,research_seeds_reserved=False)
    (OUT/'algebra_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    for name in [BRIEF,Path(__file__).name]:shutil.copy2(ROOT/name,OUT/name)
    decision=dict(date='2026-11-15',candidate='assignment-dependent price DDQN',stage='prospective_design_and_exact_algebra_only',
        critic_change='assignment-dependent workload potential',decoder='one fixed coordinate pass from expected-load allocation',
        network_parameters_unchanged=True,network_parameter_bytes=68264,buffer_bytes=20,
        neural_precision='FP32',scalar_precision='FP64 explicitly proposed; matched new controls required',
        decoding_bound='O(ND+N log N)',global_optimality_guarantee=False,deadline_guarantee=False,
        novelty_established=False,implementation_complete=False,research_budget_declared=False,
        training_authorized=False,final_authorized=False,word_thesis_changed=False,
        next='Isolated algebra/model/decoder prototype, gradient and adversarial numerical tests, then cost check; no simulator or research training')
    (OUT/'design_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    status_path=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json';shutil.copy2(status_path,OUT/'PROJECT_STATUS_BEFORE_DESIGN.json')
    status=json.loads(status_path.read_text(encoding='utf-8-sig'))
    status['current_workstream']='assignment_dependent_price_design_frozen_algebra_checked_prototype_pending'
    status['next']=decision['next'];status['next_engineering_step']=decision['next']
    status['assignment_dependent_price_design']=dict(brief=BRIEF,evidence=str(OUT.relative_to(ROOT)),
        exact_algebra_checks_passed=True,suboptimal_counterexample_preserved=True,
        implementation_complete=False,novelty_established=False,research_launched=False,word_thesis_changed=False)
    status_path.write_text(json.dumps(status,indent=2)+'\n')
    manifest={x.relative_to(OUT).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in OUT.rglob('*') if x.is_file()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(passed=True,assignments=checks,marginal_checks=insertions,
                         suboptimal_example_gap=str(costs[-1]-optimum),new_training=False)))

if __name__=='__main__':main()
