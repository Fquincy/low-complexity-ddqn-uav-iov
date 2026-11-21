"""Record prospective design and bounded literature review; no code/model changes."""
import json,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'analysis/deadline_design_literature_2026-11-11'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def main():
    retrieval=json.loads((D/'retrieval.json').read_text());assert all(r['downloaded'] and r['abstract'] and r['title'] for r in retrieval)
    brief=R/'DEADLINE_AWARE_DDQN_PROSPECTIVE_DESIGN_BRIEF_2026-11-11.txt';assert not brief.exists()
    brief.write_text('''DEADLINE-AWARE COMPACT DDQN: PROSPECTIVE DESIGN BRIEF
Date: November 11, 2026
Status: candidate design; no implementation, frozen research protocol or launch.
Working name: feasibility-screened compact workload-price DDQN (FS-DDQN).

1. QUESTION AND EVIDENCE
Can a compact DDQN with conservative local-feasibility screening avoid the
observed impossible-local choices while retaining return, deadline performance,
value calibration, storage and complete action latency against fair references?
Saved primary traces assigned66.57% of tasks locally;29.64% of all tasks were
provably infeasible locally. They account for96.30% of deadline failures.
This localizes a failure pattern, not its learning cause or the effect of any
replacement action. The last run failed scientific gates and remains failed.

2. SCOPE AND MINIMAL CHANGE
Keep the current UAV/RSU/local offloading problem, environment,40-slot horizon,
533 observed features, reward weights, task dynamics and security-free scope.
No cloud or charging stations. Do not import private simulator state or future
arrivals into the controller. Existing raw observation already contains task
cycles, deadlines and CPU workloads. The first candidate adds a necessary local
feasibility screen; it does not add a new reward, Lagrangian learner, supervised
label or terminal correction at the same time. Keeping one intervention helps
attribute any benefit. The prior horizon/terminal-anchor mechanisms are inherited
and must be named as existing experimental components rather than new discoveries.

3. OBSERVED-STATE FEASIBILITY EQUATIONS
For vehicle i, C_i is current task cycles, delta_i its deadline, f_local=1.5e9Hz.
Necessary empty-queue local service time: L_i = C_i/f_local.
A local action cannot meet its deadline when L_i > delta_i + TIME_EPS,
where TIME_EPS=1e-10 matches the simulator. Queueing can only increase delay.
This deliberately ignores local queue workload to obtain a conservative minimal
screen. A queue-aware screen would be a later separately declared extension.

Controller uses observed features z_C=s[8*i+6], z_delta=s[8*i+7], with scales
C_max=500e6 cycles and delta_max=.300seconds, from the frozen environment config.
FP32 normalized observations do not reconstruct exact task values near a boundary.
Do NOT compare point reconstructions and claim an exact infeasibility proof.
Construct outward intervals around z_C and z_delta using adjacent FP32 values
(nextafter toward -infinity/+infinity), clip to the valid nonnegative ranges,
and multiply/divide with outward rounding in FP64 using the declared scales.
Let C_lo be the lower cycle bound and delta_hi the upper deadline bound.
Exclude local ONLY if lower_round(C_lo/f_local) > upper_round(delta_hi+TIME_EPS).
Retain local whenever numerical uncertainty overlaps the feasibility boundary.
An engineering gate must establish interval containment against exact task
records, randomized valid values and boundary adversarial fixtures before use.
No tuning epsilon, threshold learned from success labels, or per-fit exception.

m_i(0;s)=0 only for this robustly certified infeasible local case; otherwise1.
m_i(d;s)=1 for d=1..4 (the two UAVs and two RSUs).
Remote actions are NOT certified deadline-feasible. There is always at least one
retained action because all four remotes remain available. The fallback if none
is certified feasible is ordinary learned selection among retained remotes;
log that no guarantee exists. Do not invent an oracle fallback or task drop.
Movement branches keep all five actions. No new unavailable action is introduced.
If future failures force masking every remote destination, that is outside this
candidate and requires a new explicit no-action/drop/fallback semantics review.

4. CRITIC AND ACTION DECODER
Reuse the compact24 hidden-unit workload-price critic. For N=30 vehicles,D=5
choices and two motion branches, let g_id(s)>=0 be learned normalized task cost
plus the existing queue/workload-price marginal term. Let v_jb(s)>=0 be movement
cost and b(s)>=0 the bias. With h(s)=40-round(40*s[-1]), define
Q_theta(s,a) = -h(s)[b(s) + sum_i g_i,a_i(s) + sum_j v_j,a_j(s)].
The existing price term is beta_d[2(L_d+eta_d*sum_(k!=i)w_kd)w_id+w_id^2]/120;
local beta_0=0, beta_d=softplus(raw_beta_d), eta=softmax(raw_eta).
All task costs for the whole joint decision are computed once from observed
state, as in the original separable critic. Screening adds no new weights.
Choose a_i*=argmin_(d:m_i(d;s)=1) g_id(s), with first-index tie behavior;
choose motion by its original argmin. Cost arithmetic and Q evaluation use the
original accepted PyTorch-core/native-decoder route, with screen included.
The decoder is exact for THIS separable masked critic because its allowed set
is a product of branch sets and h>0. This does not solve the environment's true
coupled queue/resource optimization or imply an optimal offloading policy.
At h=0, return Q=0 and never decode an action.

5. CONSISTENT MASKED DDQN UPDATE AND EXPLORATION
For each nonterminal transition:
a_next = argmax_(a in A_m(s_next)) Q_online(s_next,a)
y = r + gamma*Q_target(s_next,a_next); gamma=1.
For a terminal transition y=r; do not decode the next state.
Loss retains the original uniform DDQN Huber plus unit-weight terminal-anchor
Huber on32 sampled terminal transitions, with its separate RNG and one optimizer
step. Replay stores the unchanged observed states and joint actions. Recompute
the screen from next-state observations; never substitute current-state masks.
The target network evaluates the online-selected allowed action without a second
argmax. Keep epsilon schedule, uniform replay, Adam and target copies unchanged
when a later protocol is frozen. During exploration, sample each branch uniformly
from its retained choices; use the same declared source random uniforms across
screened comparisons and document action remapping. Report changed visitation.
The unmasked ablation intentionally has a different allowed action set; it
cannot share identical realized behavior, only declared workload/RNG schedules.

6. EXPECTED COST AND LIMITS
Original primary online parameter payload68,264bytes remains unchanged if no
network features/tensors are added; this must be verified in implementation.
Screen cost O(N), at most30 interval predicates, independent of network width.
Feature storage can be O(N) masks or a fixed32x5 temporary action mask. Decoder
remains O((N+2)D). Neural forward work is O(dH+H*(5N+20)), d=17N+23,H=24.
Inference is O(N) for fixed H,D, with substantial constants to measure. Generic
dense joint-action output would grow exponentially, but it is not the sole fair
baseline: compact additive and BDQ controls also have linear branch outputs.
Report complete latency including input checks, interval screening, forward,
decoder, fallback logging overhead and final action copy. Online tensor payload,
serialized deployment artifact bytes and framework/process memory are different
quantities. Retain the existing parameter-byte gate and report auxiliary buffer
bytes separately; do not claim device feasibility from parameter bytes alone.

7. PROSPECTIVE FAIR COMPARISON SET
Candidate must compete with screened controls, not only unmasked predecessors:
A. screened compact-price24 with inherited horizon and terminal anchor (primary).
B. same candidate without screening (direct screening ablation).
C. screened compact-no-price24, otherwise identical (price mechanism ablation).
D. screened plain24 with identical horizon/anchor (same-size architecture control).
E. screened compact64 with identical horizon/anchor (width/capacity control).
F. screened plain24 under ordinary finite-horizon DDQN without horizon scaling or
   extra terminal-anchor loss (plain small DDQN learning-control baseline).
G. screened wide128 original reference (stronger capacity reference).
H. screened adapted BDQ64 (known branching reference).
I. unmasked compact64 and wide128 historical architecture definitions, rerun fresh
   if included in the frozen research protocol; do not substitute old seed fits.
J. nearest-RSU, UAV-selective and all-local heuristics, explicitly unchanged.
All-local may knowingly violate feasibility and is a diagnostic baseline, not
an example of certified screening. Heuristics use no extra future/task labels.
Before research, specify exact screened BDQ/tie/value normalization and model
parameter counts, require mask-aware target selection for every screened learner,
and choose the complete fixed arm list once. No default research budget or seeds
are authorized by this design. Do not quietly inherit17 arms/680,000slots.

8. ACCEPTANCE AND FRESHNESS
Retain conservative existing standards prospectively: primary vs required strong
references return difference>=-2 and deadline difference>=-1percentage point
in each of five fits; against strong heuristics require nondegradation in both.
Primary value normalized RMSE<=.25,bias<=2,nearzero-large-return fraction<=.05,
terminal p95 absolute error<=2 in each fit. Existing cost targets: online payload
<=50% compact64,median latency<=wide128 andp95<=1ms for each fit on the declared
host. Exact eager/runtime actions and existing Q tolerance remain required.
Any mechanism claim needs its corresponding positive ablation evidence, not only
beating an unmasked weak baseline. Exact mechanism and screening minimum effects
must be written before launch; do not invent a favorable margin after results.
The diagnosed48 development evaluation seeds and610xx model seeds are consumed.
Reserve and collision-audit fresh development task/model blocks in a later
protocol; leave historical final510xx/520xx blocks untouched. Repeated design
cycles remain development, not independent final confirmation. Use all fits.
Current global value-calibration failure remains an unresolved gate. Screening
is not assumed to fix it; no thesis claim proceeds merely because local choices
are eliminated. Do not add another loss intervention without a new brief.

9. NOVELTY POSITION AND REVIEW STATUS
Masking, necessary deadline checks, branching critics, Double-Q targets and
finite-horizon conditioning are established components. This brief DOES NOT
establish a new algorithmic invention. A defensible candidate claim, if later
supported, would be a measured implementation improvement integrating conservative
observed-state feasibility with a compact workload-price DDQN in this specific
UAV-IoV model, outperforming equally informed compact alternatives at lower cost.
Whether that is sufficiently distinct and substantial requires broader domain
literature comparison, successful ablations and supervisor agreement. If screened
plain DDQN matches the candidate, do not claim price architecture novelty.

10. NEXT ENGINEERING GATE
Implement only the interval screen, masked decoder/DDQN target path and synthetic
fixtures. Verify exact-record containment, no false infeasibility exclusion at
boundaries, nonempty sets, all infeasible-local decisions removed, exploration
membership, next-state/terminal handling, eager/runtime ties, zero added parameters,
training RNG accounting and complete cost on fixed engineering inputs. Give
screened baselines the same inputs. No research training/evaluation or Word thesis
edits in that gate. A broader UAV-IoV literature check and supervisor discussion
remain necessary before promoting this candidate to a thesis novelty claim.
''',encoding='utf-8')
    lit=R/'DEADLINE_AWARE_DDQN_LITERATURE_POSITION_2026-11-11.txt'
    lines=['BOUNDED LITERATURE POSITION CHECK','Date: November11,2026','','SCOPE',
        'Retrieved arXiv landing-page metadata and abstracts for four known relevant',
        'works. HTML evidence and URLs are saved under',str(D.relative_to(R)),
        'Existing local extracted DDQN/branching texts were also inspected.',
        'This is a limited component-overlap check, NOT an exhaustive UAV-IoV',
        'related-work search, full-paper verification or novelty clearance. No',
        'full-text theoretical conclusion is inferred from a landing-page abstract.','']
    notes={
        '2006.14171':'Invalid action masking is established; this paper concerns policy-gradient algorithms. Its abstract does not prove masked-DDQN convergence or deadline guarantees.',
        '1711.08946':'Shared branching architectures with linear output growth and BDQ are established. Linear decoder scaling alone is not a new contribution.',
        '1509.06461':'Separating online action selection and target evaluation is established Double DQN. It does not guarantee calibration in this simulator.',
        '1705.10528':'Reward-plus-constraint formulations are established. This proposal is not CPO and inherits no CPO constraint-satisfaction guarantee.'}
    for row in retrieval:
        key=row['url'].rsplit('/',1)[-1];lines += [row['title'],row['url'],notes[key],'']
    lines +=['OPEN DOMAIN WORK',
        'Search UAV-assisted vehicular offloading papers for deadline-feasibility',
        'masking, queue-aware action screening, branching DDQN and model-based',
        'local execution lower bounds. Read closest full papers, compare state',
        'information, joint-action handling, guarantee assumptions, objective,',
        'cost measurement and ablation standards. Do not claim no prior art.',
        'The project already contains narrower IoV paper extracts; those are not',
        'a complete review of this new masking/feasibility design.',
        '', 'PROVISIONAL CLAIM',
        'A measured implementation improvement in the approved offloading problem',
        'is a research hypothesis. The simple local test itself is not a novelty.',
        'Department/supervisor approval and empirical success remain unresolved.']
    lit.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    spec=R/'DEADLINE_AWARE_DDQN_DESIGN_CANDIDATE_2026-11-11.json'
    write(spec,dict(status='prospective_design_only_not_frozen_research_protocol',brief=brief.name,literature_position=lit.name,
        primary='screened_compact_price24_horizon_terminal_anchor',change='conservative_interval_local_infeasibility_screen',
        mask_remote_actions=False,remote_deadline_guarantee=False,reward_changed=False,weights_changed=False,
        architecture_changed=False,observed_features_changed=False,seeds_reserved=False,research_budget_declared=False,
        novelty_established=False,full_domain_literature_complete=False,new_training_slots=0,new_simulator_slots=0,word_thesis_changed=False))
    tracker=R/'LOW_COMPLEXITY_PROJECT_STATUS.json';backup=D/'PROJECT_STATUS_BEFORE_DESIGN.json';assert not backup.exists();shutil.copy2(tracker,backup)
    status=json.loads(tracker.read_text(encoding='utf-8'));status['current_workstream']='deadline_aware_candidate_brief_declared_no_implementation_or_research'
    status['next']='implement_conservative_interval_screen_masked_ddqn_and_equal_information_baselines_on_synthetic_fixtures'
    status['deadline_aware_design_candidate']=dict(brief=brief.name,literature=lit.name,novelty_established=False,
        full_domain_review_pending=True,research_launched=False,seeds_reserved=False,word_thesis_changed=False)
    write(tracker,status)
    paths=[p for p in D.iterdir() if p.is_file()]+[brief,lit,spec,Path(__file__)]
    write(D/'COMPLETION_MANIFEST.json',dict(state='prospective_design_and_component_literature_review_recorded',files={str(p.relative_to(R)).replace('\\','/'):sha(p) for p in paths}))
    print(json.dumps(dict(brief=brief.name,literature=lit.name,research_launched=False,novelty_established=False)))
if __name__=='__main__':main()
