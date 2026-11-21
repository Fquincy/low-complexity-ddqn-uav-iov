"""Generate a prospective design record from saved traces; never run a learner."""
import hashlib
import json
import re
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'analysis/controller_redesign_2026-11-05'
D = json.loads((OUT/'saved_trace_diagnosis.json').read_text(encoding='utf-8'))
DATE = '2026-11-05'
PREFIX = 'LOW_COMPLEXITY_CONTROLLER_REDESIGN_'

def write(name, text):
    path=ROOT/(PREFIX+name+'_'+DATE+'.txt')
    if path.exists(): raise FileExistsError(path)
    path.write_text(text.strip()+'\n',encoding='utf-8')
    return path

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

rows=[]
for key,m in D['models'].items():
    rows.append(f"{m['model_seed']} | {key.rsplit('_',1)[0]:26s} | {m['return_mean']:9.4f} | {100*m['deadline_mean']:7.3f}%")
bad=D['models']['emc_q_nonpositive_1']
misses=sum(t['count']-t['on_time'] for t in bad['destinations'].values())
rsu2=bad['destinations']['4']
diagnosis=f'''
SAVED-TRACE CONTROLLER DIAGNOSIS
Date: {DATE}
Status: retrospective exploratory diagnosis; no new evaluation or training.

READ THIS FIRST
The completed value-bound run and its failed aggregate screen are preserved.
This document proposes explanations to test, not a successful new controller.
The Word thesis and existing source, checkpoints and frozen protocol are unchanged.

1. EVIDENCE AND METHOD
Run: validation/value_bound_development_2026-11-04_01/
Independent existing saved-artifact audit: {D['previous_audit']['state']}.
This analysis reads all 15 saved fresh evaluation traces (48 episodes/model),
all training summaries, scheduled-validation summaries and model counters,
plus the three heuristic traces. Each learned trace contains 57,600 tasks.
All learned models and heuristics have identical ordered fresh workload
fingerprints. Task totals and on-time counts were independently recomputed.
The 15 fresh trace hashes match the original run manifest. All inputs to this
analysis were rehashed at completion and remained unchanged. This is a focused
trace check, not a second full audit of all original artifacts.
New simulator steps: 0. Optimizer updates: 0. Checkpoints loaded: 0.
Reproduction: python analysis/diagnose_value_bound_saved_2026_11_05.py
Machine-readable results and exact input hashes:
analysis/controller_redesign_2026-11-05/saved_trace_diagnosis.json

2. FINAL POLICY RESULTS: ALL INITIALIZATIONS
Model seed | arm                        | mean return | on-time tasks
{chr(10).join(rows)}
Heuristics, same 48 workloads:
'''+''.join(f"{k}: return {m['return_mean']:.4f}; deadline {100*m['deadline_mean']:.3f}%.\n" for k,m in D['heuristics'].items())+f'''
The compact bounded additive model also varies across initializations:
78.437%, 85.595% and 76.203% deadline success. It is not an already solved,
reliable teacher. Do not assume distillation from it fixes the problem.

3. WHERE THE UNSUCCESSFUL PRIMARY INITIALIZATION LOSES DEADLINES
Seed 3119 bounded EMC, over 57,600 generated tasks:
'''+''.join(f"Destination {k}: {t['count']} tasks ({100*t['fraction']:.3f}%); on-time {100*t['success']:.3f}%; mean completed CPU queue wait {t['queue_delay_s_mean']:.6f} s.\n" for k,t in bad['destinations'].items())+f'''
Destination mapping: 0 local; 1 UAV1; 2 UAV2; 3 RSU1; 4 RSU2.
Total missed tasks: {misses}. Of these, {rsu2['count']-rsu2['on_time']} occurred
at RSU2 ({100*(rsu2['count']-rsu2['on_time'])/misses:.3f}% of all misses).
The same policy sends 62.542% of tasks to RSU2 and only 25.547% to RSU1.
The compact bounded additive comparator sends 46.439% and 51.759% to the two
RSUs and reaches 85.595% deadline success, versus the primary's 75.115%.
Successful bounded EMC seeds distribute about 37.5-41.3% to each RSU and reach
about 90.4-90.5% success. These are selected-destination conditional statistics,
not randomized causal estimates of which destination is intrinsically better.

The loss is present throughout the episode: seed 3119 success is about 74.7%
for tasks arriving in slot 0, 75.0% in slot 10, 75.3% in slot 20 and 75.1%
in slot 39. It is not only unfinished work at the final horizon.
Its scheduled-validation success is 82.7% at episode 100, 76.2% at 150 and
74.317% at 200. This shows regression before the fresh gate, not simply a
newly encountered fresh-workload problem. The final-only checkpoint rule stays;
episode 100 must not be retrospectively selected as the research result.

Across all bounded models, each vehicle keeps its modal destination on roughly
94-96% of these saved evaluation decisions. This suggests limited adaptation
and possible vehicle-index specialization, but is not itself proof of a bug:
the current workload distribution may favor stable routes. A causal test must
compare a shared representation and an explicit load mechanism prospectively.

4. VALUE CALIBRATION: WHY THE PREVIOUS PASS IS LIMITED
All bounded primary models pass the old nonpositive-sign and terminal-error
screens. However, their fixed-probe greedy-Q medians are near zero: negative
zero, about -4.74e-24 and about -5.01e-32. The bounded additive probes are also
near zero. These fixed probes come from heuristic trajectories, so they are
not the models' own on-policy state distribution and cannot be compared
directly to another policy's return as a calibration error.
The primary fresh-terminal absolute-error p95 values are 0.506, 0.683 and
0.512. Their corresponding final-slot rewards are themselves small negative
values. A near-zero predictor can pass an absolute threshold of 10; this test
does not establish accurate values over the rest of a 40-slot episode.

The existing outer transform Q=-softplus(C_raw) has derivative
dQ/dC_raw=-sigmoid(C_raw). Very negative raw scores attenuate gradients and can
underflow to zero in finite precision. This is a mathematical risk consistent
with the probes; the saved analysis does not establish it caused the seed's
policy regression. Correct raw-score action selection can still rank actions
even when displayed transformed values are nearly tied at zero.
Follow-up needs own-trajectory chosen Q and realized return-to-go at every
slot, value/target/TD distributions, and a prospective transform ablation.

5. COST AND MODEL LIMITATIONS
Online FP32 bytes: additive128 422,016; additive64 194,944; EMC 277,444.
The bounded EMC is smaller than the wide model but larger and slower than
the compact additive model. Shrinking stored weights alone has not established
lower complete action-selection cost or reliable policy quality.
The current observation is compressed/partially observed, not a demonstrated
complete Markov state. RSU CPU capacities and the fixed 1 Gbit/s RSU link are
explicit provisional assumptions; RSU preference must be interpreted in that
setting. The reward uses delay, missed deadlines and UAV energy. It is not a
measurement of all system or controller electrical energy. Security is absent.

6. DESIGN RESPONSE AND LIMITS
Test one candidate with shared vehicle processing, explicit bounded-work
server-load coordination and a sum of nonnegative cost contributions rather
than an outer transform over an unrestricted joint sum. Three separate
ablations isolate these changes. No causal attribution or novelty is claimed
from this retrospective diagnosis. All consumed seeds remain development data.
The design and comparison files specify the next prospective tests.
'''

brief=r'''
CONTROLLER DESIGN BRIEF - VERSION 1
Date: 2026-11-05
Status: prospective design specification; not implemented, trained or validated.
Working name: shared-entity congestion-cost DDQN (descriptive, not a novelty claim).

1. PURPOSE AND CLAIM BOUNDARY
Implement a smaller centralized action-value controller for the existing UAV-IoV
offloading simulator, then test whether it preserves return/deadline performance
at lower complete action-selection cost. The candidate is a DDQN-style learner
with a bounded approximate joint-action decoder; it does NOT have exact joint
argmax and must never be described as exact DDQN optimization.
The proposed contribution is the tested combination of resource-aware value
factorization, shared entity processing and a bounded-work decoder. The field's
known DDQN, parameter sharing, pooling, and congestion-pricing ideas are credited.
No first-in-literature claim is established by this design stage.

2. SCOPE DECISION FOR THIS ENGINEERING BRANCH
Keep local vehicle CPUs, two UAV servers, two RSUs and five UAV movements.
No cloud, charging, trust, attacks or security-aware policy is introduced.
Keep the current simulator, physical transition, task distribution, reward,
observation and 40-slot horizon for controlled controller comparisons.
Describe the controller as a centralized simulated scheduler; no UAV-board
deployment, embedded inference or controller power measurement is demonstrated.
This decision supports an efficiency/offloading chapter. It cannot substantiate
a security-aware thesis title or contribution. Supervisor discussion must settle
whether security belongs elsewhere or needs a separately specified extension.
No administrative approval or thesis-title change is assumed here.

3. INFORMATION AND EXACT OBSERVATION MAPPING
The existing vector has 17N+23 entries, N=30. Parse, without changing values:
  features: 8N (position x,y; speed; sin/cos heading; task bits/cycles/deadline)
  CPU workloads: N local plus 4 remote log1p-normalized workload entries
  radio summaries: 8N
  remote queue/count/age/availability summaries: 12
  UAV position/battery summaries: 6
  normalized time: 1.
For vehicle i, x_i has its 8 features, 1 local CPU workload and 8 radio entries
(17). Global g has 4 remote workloads, 12 queue entries, 6 UAV entries and time
(23). Capacities/normalizers are public configuration constants available to
every comparator. No full hidden queue, future arrival, task completion, or
simulator peek is allowed. Preserve the same partial-observation limitation.

4. NETWORK: ALL DIMENSIONS FIXED FOR VERSION 1
All Linear layers include biases; hidden activations are ReLU.
  z_i = phi(x_i), shared MLP 17 -> 32 -> 16.
  pooled = mean_i z_i.
  h = rho(concat(pooled,g)), MLP 39 -> 32 -> 16.
  l_i = shared MLP(concat(z_i,h)), 32 -> 16 -> 5.
  m = mobility MLP(h), 16 -> 16 -> 10, reshaped [2,5].
  [b,v_1,...,v_4] = Linear(h), 16 -> 5.
No attention, recurrence, vehicle-ID embedding, learned per-vehicle head, or
repeated neural forward pass per decoder iteration is added.
Use framework-default Linear initialization, with identical paired seeds and
documented construction order. No outcome-driven initialization selection.

5. VALUE FUNCTION
Let sp(x)=log(1+exp(x)), evaluated with numerically stable softplus.
Vehicle destination d_i is in {0,...,4}; UAV movement b_j in {0,...,4}.
For remote destination d=1,...,4:
  L_d = expm1(existing remote normalized log-workload entry).
  w_i,d = task_required_cycles_i / (CPU_frequency_d * deadline_normalizer_s).
Both L and w have units of service time / deadline_normalizer_s.
Reconstruct current task cycles from the existing feature and fixed normalizer.
Let A_d(a)=sum_i w_i,d * 1[d_i=d], and beta_d(s)=sp(v_d).
Define a nonnegative incremental congestion term
  P(s,a)=sum_{d=1}^4 beta_d(s)/(4N) * [(L_d+A_d(a))^2-L_d^2].
The subtraction removes pre-existing queue cost independent of the new action;
L_d,A_d>=0 make P>=0. This is a learned inductive bias in the Q approximation,
not a change to the simulator reward or an exact queue-delay equation.
  C_theta(s,a)=sp(b) + mean_i sp(l_i,d_i)
               + (1/2)sum_{j=1}^2 sp(m_j,b_j) + P(s,a).
  Q_theta(s,a)=-C_theta(s,a) <= 0.
The cost form places softplus on individual contributions, avoiding cancellation
inside one outer joint softplus. Individual terms can still saturate: improved
calibration or learning stability must be measured, not assumed.
It cannot represent every joint Q-function. In particular, at a fixed state,
mobility and placement are separable; the architecture does not represent the
current EMC model's direct same-slot motion-placement interactions. Compare
against EMC and reject the design if this restriction materially hurts quality.

6. BOUNDED-WORK APPROXIMATE DECODER D_theta(s)
Compute all neural outputs once. Let u_i,d=sp(l_i,d)/N.
Movement choices independently minimize sp(m_j,b)/2.
  a^(0): each destination minimizes u_i,d; ties choose lowest legal index.
Perform exactly two synchronous price-refinement passes, k=0,1:
  A_d^(k)=sum_i w_i,d*1[a_i^(k)=d].
  H_i,d^(k)=L_d+A_d^(k)-w_i,d*1[a_i^(k)=d].
  p_i,d^(k)=beta_d/(4N) * [2*H_i,d^(k)*w_i,d + w_i,d^2].
  Set p_i,0=0 (independent local CPUs).
  a_i^(k+1)=argmin_d [u_i,d+p_i,d^(k)] simultaneously for all i.
Score a^(0),a^(1),a^(2) with the FULL C above and retain the minimum;
ties choose earliest candidate. Keep the selected independent movements.
The output costs no more under this model than a^(0), because a^(0) is retained.
This does NOT guarantee improvement in real return or deadline performance,
nor exact global minimization. Synchronous passes can oscillate or preserve an
unbalanced solution, especially with symmetric costs. Record candidate agreement,
load splits and reduced-fixture exhaustive optimality gaps. No beam search,
full environment transition, hidden queue simulation or adaptive retry occurs.
Keep all five destinations legal; do not invent a deadline-feasibility oracle.

7. DDQN-STYLE LEARNING
For replay tuple (s,a,r,s',done), define a'=D_theta(s') under no_grad.
  y = r                              if done
      r + gamma*Q_target(s',a')       otherwise.
  loss = mean Huber_beta1(Q_theta(s,a)-stop_gradient(y)).
Use the SAME decoder for behavior and online successor selection; evaluate the
target network at that complete selected action. Target parameters have no
gradient. Because D is approximate, this is an approximate-greedy DDQN update;
no exact Bellman-optimality backup, convergence guarantee or optimality theorem
is claimed. Zero terminal bootstrap is essential; gamma remains 1.0.
Use uniform replay and the inherited optimizer schedule without reward changes.
Terminal next states must not invoke a decoder or a target forward call.

8. EXPECTED COST - ANALYTICAL, NOT A BENCHMARK
Parameter accounting including biases:
  phi 1,104; rho 1,808; shared placement head 613;
  mobility head 442; bias/load head 85. TOTAL 4,052 parameters.
FP32 online tensors: 16,208 bytes. Target: another 16,208 bytes.
Dense Linear multiply-accumulates at N=30: 51,696 per state, excluding nonlinear
operations, pooling, feature parsing and all decoder work.
Existing flat64: 48,736 parameters / 194,944 bytes / 48,448 Linear MACs.
Existing flat128: 105,504 parameters / 422,016 bytes / 105,088 Linear MACs.
Thus the proposal is much smaller in stored weights, but its neural MAC count
is about 6.7% ABOVE flat64. Parameter sharing is not automatically faster.
Decoder work is O(K*N*D) with K=2,D=5 plus three full score calculations;
workspace O(N*D+N*16), no 5^(N+2) table. Widths/server count are fixed here.
Parameter storage is independent of N, but work and activation storage grow
with N; performance at unseen vehicle counts is not established.
The same replay allocation remains about 215 MB at the current capacity;
do not translate online-weight savings into a similar total training-RAM saving.
Complete measured inference includes parsing/conversion, pooling, every neural
module, price passes, full candidate scoring and final action conversion.

9. REQUIRED IMPLEMENTATION GATE BEFORE LEARNING
Implement in new files only after design review. Check observation parsing,
units, shapes, legal actions, nonpositive finite Q, known-cost examples,
correct fixed candidate scoring, target detach, terminal handling and gradients
through every intended module. Test decoder against exhaustive small fixtures;
report gaps honestly and verify retained-candidate dominance. Check permutation
behavior with ties called out; do not assume the simulator's ID tie breaking is
fully permutation invariant. Check exact parameter/MAC arithmetic against code.
Run a complete reduced runner+auditor fixture with nonresearch seeds and no
long-run launch. Benchmark every comparator on the same machine/backend.
If the predeclared runtime/storage gate fails, stop before training and record
that failure. Do not silently change two passes or choose the best timing run.

10. LITERATURE CHECK AND NOVELTY LIMIT
[1] Van Hasselt, Guez and Silver, Deep Reinforcement Learning with Double
Q-learning, AAAI 2016; arXiv:1509.06461. Source of separated online selection
and target evaluation. Neither DDQN nor its two networks is a new contribution.
[2] Tavakoli, Pardo and Kormushev, Action Branching Architectures for Deep
Reinforcement Learning, AAAI 2018; arXiv:1711.08946, especially equations 1,6,7.
Already uses shared decisions and action branches with linear output growth.
Our shared per-vehicle weights plus explicit action-dependent congestion cost
and three-candidate approximate decoder differ from independent branch greedy
selection; replacing heads alone is not novelty. A BDQ adaptation is required.
[3] Zaheer et al., Deep Sets, NeurIPS 2017; arXiv:1703.06114.
Shared entity encoders and symmetric pooling are established. No claim to have
invented permutation-invariant pooling or variable-size network processing.
[4] Rashid et al., QMIX: Monotonic Value Function Factorisation for Deep
Multi-Agent Reinforcement Learning, ICML 2018, PMLR 80:4295-4304.
Already coordinates factored values with a monotonic mixer and tractable greedy
consistency. Here the controller is centralized with explicit assignment-load
coupling and an approximate decoder; do not claim QMIX's exact greedy property.
[5] Hao et al., Joint Task Offloading, Resource Allocation, and Trajectory Design
for Multi-UAV Cooperative Edge Computing With Task Priority, IEEE TMC 23(9),
8649-8663, 2024; DOI 10.1109/TMC.2024.3350078. Relevant workload/model reference,
not a lightweight DDQN baseline. Its cloud/cooperation/hybrid-action system is
different; our task, cycle and deadline adoption is an adaptation, not reproduction.
[6] Xu et al., Energy-Efficient Multi-Agent Deep Reinforcement Learning Task
Offloading and Resource Allocation for UAV Edge Computing, Sensors 25,3403,
2025; DOI 10.3390/s25113403. Uses MATD3-TORA and joint UAV-MEC resource decisions.
It establishes that UAV-MEC coordination/latency-energy optimization is already
studied. It does not by itself validate the proposed DDQN architecture.

The full texts of [1]-[4] were retrieved from arXiv/PMLR primary sites; [5]-[6]
were read from the user's local full papers. Copies/extractions are under
analysis/controller_redesign_2026-11-05/references/. This is a bounded comparison
with these six papers, not an exhaustive recent UAV-IoV novelty search. Before
calling the mechanism novel in a thesis or publication, perform a targeted
recent search for shared-parameter, congestion-aware, branching and compressed
DDQN offloading, and compare the exact equations/decoder to the closest methods.
Neither a narrower network nor a successful experiment alone establishes novelty.

11. SUPERVISOR DISCUSSION POINT
Proposed extension within UAV-IoV computational offloading: implement a compact
shared controller with explicit resource-load coordination and measure its full
policy/cost tradeoff against plain small DDQN, a wider DDQN, EMC and adapted BDQ.
Ask whether this engineering extension is within the approved topic and what
evidence is required for the degree. Agree where security is handled. The title,
the old conference paper and historical experiments must not be rewritten to
imply that this new architecture was previously implemented or validated.
'''

ARMS=['shared_congestion_cost','shared_no_congestion','untied_congestion_cost','shared_joint_bound','additive64_nonpositive','additive128_unbounded','emc_q_nonpositive','bdq64_uniform_adapted']
spec=dict(id='shared-congestion-cost-ddqn-design-v1',date=DATE,
          status='prospective_design_only_implementation_and_launch_not_authorized_by_this_file',
          research_training_launched=False,thesis_modified=False,primary_arm=ARMS[0],arms=ARMS,
          simulator='Source/chapter3_iov_env_v2.py',security=False,
          network=dict(entity=[17,32,16],context=[39,32,16],placement=[32,16,5],mobility=[16,16,10],global_head=[16,5],expected_parameters=4052,expected_linear_macs_N30=51696),
          decoder=dict(price_passes=2,full_scored_candidates=3,exact_joint_argmax=False),
          training=dict(episodes=200,slots_per_episode=40,slots_per_model=8000,updates_per_model=6977,gamma=1.0,optimizer='Adam',learning_rate=0.0001,betas=[.9,.999],eps=1e-8,weight_decay=0,batch_size=128,replay_capacity=50000,warmup=1024,gradient_clip=10,target_interval_updates=200,target_copies_per_model=34,loss='Huber_beta1',checkpoint='final_only',epsilon_decay_slots=8000,cpu_threads=1,cpu_interop_threads=1),
          development=dict(model_seeds=[41001,41011,41017,41023,41039],train_blocks=[dict(start=410000000+i*1000000,count=200) for i in range(5)],validation=list(range(421000000,421000005)),probes=list(range(422000000,422000008)),gate=list(range(420000000,420000048))),
          final=dict(model_seeds=[51001,51011,51017,51023,51031],train_blocks=[dict(start=510000000+i*1000000,count=200) for i in range(5)],gate=list(range(520000000,520000200)),status='reserved_unconsumed_no_model_selection'),
          integration_smoke=dict(model_seed=43001,train=[430000000,430000001],evaluation=[430000100,430000101],purpose='integration_only_no_quality_selection'),
          fixture=dict(model_seed=99001,train=[990000000],evaluation=[990000100],purpose='synthetic_only'),
          rng=dict(action_offset=100000,replay_offset=200000,arm_order=420600001,bootstrap=420600002,timing_order=420600003,final_bootstrap=520600002),
          cost_gate=dict(online_bytes_ratio_vs_additive64_at_most=.5,complete_action_median_ratio_vs_additive128_at_most=1.0,complete_action_p95_ms_at_most=1.0,scope='same_named_CPU_backend_not_embedded_deployment'),
          development_quality=dict(all5_initializations_required=True,return_margin_vs_each_learned_reference=-2.,deadline_fraction_margin_vs_each_learned_reference=-.01,heuristic_return_margin=0.,heuristic_deadline_margin=0.,congestion_ablation_mean_return_improvement_at_least=2.,congestion_ablation_mean_deadline_improvement_at_least=.01,congestion_ablation_each_fit_both_nonnegative=True),
          value_diagnostics=dict(own_trajectory_all_slots=True,chosen_Q_nonpositive=True,finite_values=True,realized_return_normalized_RMSE_at_most=.25,realized_return_abs_bias_at_most=2.,near_zero_Q_large_return_fraction_at_most=.05,near_zero_threshold=1e-6,large_return_threshold=1.,terminal_abs_error_p95_at_most=2.,interpretation='empirical_greedy_policy_return_consistency_not_Qstar_proof'),
          uncertainty=dict(final_primary_comparator='additive128_unbounded',paired_crossed_model_and_episode_bootstrap_draws=5000,co_primary_metrics=['episode_return','deadline_fraction'],each_one_sided_lower_confidence=.975,return_margin=-2.,deadline_fraction_margin=-.01,secondary_comparisons='exploratory_report_all_no_formal_multiple_comparison_claim',training_initializations=5),
          authorization=dict(implementation='separate_next_gate',research_launch='separate_explicit_instruction_required',final='separate_locked_model_review_and_launch_required'))

# Seed-range collisions are checked against historical source/config/text, not
# against newly authored files in this design record. Search every numeric literal
# in readable project files, including old summaries; this is conservative.
reserved=set()
def collect(v):
    if isinstance(v,dict):
        if 'start' in v and 'count' in v:
            reserved.update(range(v['start'],v['start']+v['count']))
        for x in v.values():collect(x)
    elif isinstance(v,list):
        for x in v:collect(x)
    elif isinstance(v,int) and not isinstance(v,bool) and v>=41000:reserved.add(v)
for k in ['development','final','integration_smoke','fixture']:collect(spec[k])
for k,v in spec['rng'].items():
    if not k.endswith('_offset'):reserved.add(v)
for phase in ['development','final']:
    for n in spec[phase]['model_seeds']:
        reserved.update([n+100000,n+200000])
collision=[]; scanned=0
base=ROOT.parent
excluded={'site-packages','.git','__pycache__','.venv','venv','venv311','venvName','node_modules'}
history_paths=[]
for directory,dirs,files in os.walk(base):
    dirs[:]=[d for d in dirs if d not in excluded]
    history_paths.extend(Path(directory)/name for name in files)
for p in history_paths:
    if not p.is_file() or p.suffix.lower() not in {'.json','.txt','.py','.csv','.yaml','.yml'}: continue
    if any(part in excluded for part in p.parts) or p.stat().st_size>8_000_000:continue
    if OUT in p.parents or p==Path(__file__).resolve() or p.name.startswith(PREFIX):continue
    nums={int(n) for n in re.findall(r'(?<![\w.])\d+(?![\w.])',p.read_text(encoding='utf-8',errors='ignore')) if len(n)<=10}
    overlap=reserved&nums
    if overlap:collision.append(dict(path=str(p),numbers=sorted(overlap)))
    scanned+=1
seed_check=dict(status='passed' if not collision else 'collision_detected',files_scanned=scanned,
                scope='numeric_literals_readable_source_config_text_csv_up_to8MB_excludes_binary_compressed_files_and_new_design_outputs',
                caveat='not_a_proof_against_unlogged_or_external_runs; runtime_manifest_disjointness_check_required_before_launch',collisions=collision)
(OUT/'seed_reservation_check.json').write_text(json.dumps(seed_check,indent=2),encoding='utf-8')
if collision:raise ValueError(str(collision)[:1200])

protocol=r'''
PROSPECTIVE CONTROLLER COMPARISON PROTOCOL - VERSION 1
Date: 2026-11-05
Status: design-stage predeclaration, not an executable frozen research runner.
No new learning, simulator evaluation, thesis editing or research launch occurred.
The earlier November 4 protocol and its results are NOT changed by this plan.

1. HYPOTHESES AND PRIMARY OUTCOME
Primary candidate: shared_congestion_cost, exactly as the design brief specifies.
Hypothesis: weight sharing reduces online storage, while explicit assignment-load
coupling preserves return/deadline performance at lower complete decision cost.
Separately test whether termwise nonnegative costs improve value consistency.
Smaller storage, stable Q, good policy and faster inference are different claims.
All four must be evaluated and all predeclared failures reported.

2. EIGHT LEARNED ARMS: NO POST-HOC ADDITION/SELECTION
A shared_congestion_cost: full proposed 4,052-parameter controller, two passes.
B shared_no_congestion: same network and termwise cost, beta forced to zero;
  decoder reduces to independent minima. Keep/load-head parameters in declared
  storage accounting (zero-gradient masked outputs intentional). Isolates the
  congestion mechanism and its decoder as a combined treatment.
C untied_congestion_cost: per-vehicle phi AND placement networks independently
  parameterized; same dimensions, pooling, cost, two passes and schedule.
  Expected 53,845 parameters. Tests sharing, not a claim that removing sharing
  is a competitive new baseline. Do not add vehicle information to either arm.
D shared_joint_bound: same shared network/beta/decoder budget as A, but
  Q=-softplus(b+mean_i l_i,d_i+mean_j m_j,b_j+P). Decoder uses raw l/N in
  the same price passes and scores the complete RAW inner cost; minimizes it
  before applying the decreasing transform. Tests the cost parameterization.
E additive64_nonpositive: existing 64-64 additive DDQN with its joint outer
  negative-softplus transform and correct raw argmin. Plain compact reference.
F additive128_unbounded: original 128-128 additive DDQN. Primary cost/quality
  reference; old results on different seeds are not substituted for new fits.
G emc_q_nonpositive: existing bounded EMC with its exact raw-min decoder.
  Stronger motion-placement representation comparator, with mixed prior results.
H bdq64_uniform_adapted: explicitly adapted BDQ, not the original paper's full
  implementation or a reproduction of published performance. Trunk 533->64->64;
  ReLU, output V (64->1) plus 32x5 advantages (64->160); 48,801 parameters.
  Q_j=V+A_j-mean_d A_j,d. Action per branch argmax. Target y=r+gamma*(1/32)
  sum_j Q_target,j(s',argmax Q_online,j(s')). Mean per-branch Huber_beta1 loss;
  shared-trunk gradient scaled 1/33 as in the paper's gradient-rescaling rule.
  Uniform replay, Huber instead of paper MSE, widths, schedule and simulator are
  disclosed adaptations for this paired comparison; do not call it original BDQ.
  Joint-value diagnostic = branch mean (not sum). No nonpositive constraint.
Heuristics: existing all_local, nearest_rsu and uav_selective, unchanged.
Optional untrained forced-no-correction A decoder is a diagnostic only, clearly
separated from B, which is independently trained without congestion.

3. FIXED TRAINING CONDITIONS AND ACCOUNTING
All arms: 200 episodes x40 physical slots=8,000 slots/model; final checkpoint
only; uniform replay capacity50,000, warmup1,024, batch128. One update after
each new record starting at1,024 =>6,977 updates/model; target copy every200
updates =>34 copies. Adam lr1e-4, betas(.9,.999), eps1e-8, weight_decay0;
gamma1; Huber beta1; global gradient clip10 (BDQ trunk rescale before clip).
All hidden layers ReLU. Exploration: epsilon1 during warmup, otherwise
1-.95*min(t_before_decision/8000,1). No random action space reduction.
CPU intra/inter-op threads1; final-only checkpoints; no early stopping based
on rewards, validation-selected weights, seed removal or budget extension.
Paired exogenous tasks and exploration coin/uniform joint vector per replicate;
separate replay RNG. Gradient/representation differences remain part of each arm.
Actual arm source hashes, package versions and full hardware/backend must be
frozen in a new executable manifest after the implementation gate, before run.

4. RESERVED SEEDS AND BUDGET
Development model seeds: 41001,41011,41017,41023,41039.
Paired training blocks, respectively: 410000000,411000000,412000000,
413000000,414000000, each 200 consecutive seeds.
Validation:421000000..421000004 at episodes0,50,100,150,200, diagnostics only.
Common timing/probe workloads:422000000..422000007.
Development evaluation:420000000..420000047 (48 workloads), after ALL40 final
checkpoints are saved and hashed. No model changes after viewing this gate.
Development budget:8 arms x5 fits x8,000=320,000 physical training slots,
279,080 optimizer updates,1,360 target copies.

Untouched final phase: new model seeds51001,51011,51017,51023,51031;
training blocks510000000,511000000,512000000,513000000,514000000, each200.
Final evaluation:520000000..520000199 (200 workloads). Same eight arms, budget
and source definitions; no validation or selection on final tasks. Train these
only after development passes and a separate final launch is authorized.
Both phases combined:640,000 training slots,558,160 updates,2,720 copies.
This is a prospective research budget, NOT authority to start it.

Integration-only smoke: model43001; train430000000..001; eval430000100..101.
Synthetic fixture: model99001; train990000000; eval990000100; outputs marked
SYNTHETIC_NOT_RESEARCH and stored temporarily. Neither is quality evidence.
Action/replay RNG offsets100000/200000. Order420600001; bootstrap420600002;
timing-order420600003. Final bootstrap520600002.
seed_reservation_check.json records a conservative historical literal scan.
Compressed/binary/unlogged external runs were not exhaustively searched.
Before any launch, validate actual frozen run manifests and seed ranges for
all workloads/model/RNG streams. A collision blocks launch and requires a
documented protocol amendment BEFORE using any new seed. No seed was consumed
by the saved-trace diagnosis or design checks.

5. MECHANICS/COST GATE BEFORE ANY RESEARCH TRAINING
Unit checks plus full reduced orchestration/auditor fixture must pass.
No quality inference or selection from the integration smoke.
Predeclared cost screens for A on the same CPU/backend as references:
  online FP32 parameter bytes <=50% of E;
  complete-action median latency <=100% of F;
  complete-action p95 <=1.0 ms.
Timing:128 common observations, five rounds,32 warmups/path/round,128 timed
calls/path/round, predeclared randomized order; save every nanosecond reading.
Include parsing, conversions, pooling, all heads, load passes, scoring and
action conversion; exclude simulator stepping/file I/O. All arms same dtype,
threads, backend and device. No optimized candidate versus deliberately slow
reference. Log compilation/startup separately if any compilation is used;
backend is chosen before timing and then fixed. Do not benchmark until the
implementation gate; no timing-derived architecture choice in this stage.
Report target, optimizer, replay, activation and serialized checkpoint sizes
separately; FP32 tensor bytes are not file size or peak process memory.
If A misses the runtime gate, stop before research training. Smaller weights
with slower inference cannot be rebranded as a faster controller. A later
memory/runtime tradeoff study would need a new explicit protocol.

6. DEVELOPMENT QUALITY SCREEN - ALL FIVE FITS
For EACH A initialization, compared on paired episodes with each of E,F,G,H:
  mean episode return difference >=-2.0;
  deadline-success fraction difference >=-.01 (one percentage point).
Against each declared heuristic:both mean differences >=0 for every fit.
Mechanism check vs B:each fit has both differences>=0, and across five fitted
pairs the equally weighted mean return improvement>=2 and deadline improvement
>=.01. Report C and D comparisons completely; do not cherry-pick whichever
ablation happens to favor the story. Passing the development screen is an
engineering continuation decision, not statistical proof of novelty.
Thresholds are prospective design choices motivated by meaningful return,
deadline and cost tolerances, informed by completed development diagnosis.
They are NOT independently established universal standards or chosen from a
new smoke. No change to the previous experiment's thresholds is made.

7. VALUE STABILITY AND OWN-POLICY RETURN CONSISTENCY
Save chosen online Q and target Q at every state along each fresh greedy
trajectory, before action execution. Compute G_t=sum_{k=t}^{39} r_k for the
same trajectory. For each fit, across all its evaluation episodes/slots:
  chosen online/target values and all optimizer diagnostics finite;
  A chosen Q<=0; report zero/positive fractions for every arm;
  sqrt(mean((Q-G)^2))/max(1,sqrt(mean(G^2))) <=.25;
  abs(mean(Q-G)) <=2.0;
  fraction with abs(Q)<=1e-6 AND abs(G)>=1 <=.05;
  final-slot abs(Q-r) p95 <=2.0.
Report all-slot and per-time-bin Q,target,TD residual,gradient norms/clipping,
return residuals, mean/median/p05/p95/extrema. Bootstrap episode summaries,
not individual time slots. These measure consistency with realized returns of
the evaluated greedy policy; single realized returns have stochastic noise and
are not exact conditional Q-star ground truth. No sign-only calibration claim.
Fixed heuristic-state probes remain useful coverage/stability diagnostics;
do not compare their Q to a different policy's realized trajectory as truth.

8. FINAL CONFIRMATION AND UNCERTAINTY
Freeze all model definitions/settings before final fitting. Freeze all40 final
checkpoints and hashes before evaluating final tasks. No fallback to successful
seeds, repeated final gates, best checkpoint or favorable workload subsets.
Primary final quality comparator:F. Co-primary metrics:return and deadline
fraction, margins -2 and-.01. Report per-fit paired differences, plus5,000-draw
paired crossed bootstrap:resample the five model pairs and the200 common task
episode indices, sharing sampled indices across the compared arms. Use one-sided
97.5% lower limits for each of the two co-primary metrics; require each lower
limit>=its margin. Also require every fit's development-style point screens
and the predeclared cost/value screens. Secondary comparisons/ablations remain
fully reported; no uncorrected many-comparison statistical superiority claim.
Five initializations give limited training-variability evidence; do not treat
the1,000 fit-episode combinations as1,000 independent trained controllers.
Final failure means the claim is not established; preserve it without retuning.

9. MODELING ASSUMPTIONS AND FINAL CLAIM SCOPE
Keep the existing physical model for controlled controller development. Before
thesis results, justify or explicitly declare RSU CPU/link/radio assumptions
and the partial observation, finite horizon, FIFO service and UAV energy scope.
Reserve a separate robustness extension in the final report:RSU CPU10/20/40
Gcycles/s crossed with RSU link100Mbit/s/1Gbit/s (six settings), all other
constants unchanged. Evaluate the already frozen final policies and heuristics
on the same final workload seeds in every setting with zero updates. This is
out-of-distribution sensitivity (default setting repeated for invariance), not
retraining at each setting or validation of a realistic vehicle-to-RSU channel.
Claim default-setting performance only if sensitivity is weak; report where
the controller loses quality. No robustness model selection is permitted.
Default model assumptions may be retained as simulation assumptions if clearly
declared; do not call the2024 source an exact reproduction of this IoV system.
No attack/security contribution or controller electrical-energy saving is tested.
No automated rewriting of the thesis or the published conference paper.

10. DECISION AND STOPPING RULES
Design complete != implementation ready != training authorized != thesis claim.
Next gate:review the equations/decoder and scope, then implement A-H and their
tests in new files, confirm mechanics/cost, run reduced synthetic orchestration
and an independent auditor, and preserve a new hashed executable protocol.
An explicit launch instruction is required for either research phase.
Stop on nonfinite values, mismatched workloads, illegal actions or failed audit;
keep partial outputs; no silent restart. All failed designs stay in the record.
If design changes after this declaration, save v2 with reasons before training;
keep v1 and its hash. Do not reuse consumed final workloads as untouched evidence.
'''

paths=[write('DIAGNOSIS',diagnosis),write('DESIGN_BRIEF',brief),write('COMPARISON_PROTOCOL',protocol)]
protocol_path=OUT/'prospective_comparison_v1.json'
if protocol_path.exists(): raise FileExistsError(protocol_path)
protocol_path.write_text(json.dumps(spec,indent=2),encoding='utf-8')
protected=[ROOT/'LOW_COMPLEXITY_VALUE_BOUND_PROTOCOL_2026-11-04.json',ROOT/'run_value_bound_tradeoff.py',ROOT/'emc_q.py',ROOT/'Source/chapter3_iov_env_v2.py']
protection={str(p.relative_to(ROOT)):sha(p) for p in protected}
outputs={str(p.relative_to(ROOT)):sha(p) for p in paths+[protocol_path]}
assert spec['training']['slots_per_model']==spec['training']['episodes']*spec['training']['slots_per_episode']
assert spec['training']['updates_per_model']==8000-1024+1
assert 8*5*6977==279080
assert sum([(17*32+32)+(32*16+16),(39*32+32)+(32*16+16),(32*16+16)+(16*5+5),(16*16+16)+(16*10+10),16*5+5])==4052
assert 30*(17*32+32*16)+39*32+32*16+30*(32*16+16*5)+16*16+16*10+16*5==51696
assert 4052+29*(1104+613)==53845
assert 48736+65==48801
dev=set(spec['development']['gate']); final=set(spec['final']['gate'])
assert not dev&final
for v in D['input_sha256']:
    assert sha(ROOT/v)==D['input_sha256'][v],v
assert protection['LOW_COMPLEXITY_VALUE_BOUND_PROTOCOL_2026-11-04.json']=='844a0c5987aaa871e8f2540bcc8e0c7864f902fdaf23817e4324b93c17c40734'
record=dict(status='design_files_written_readback_checks_passed',research_launch=False,new_simulator_steps=0,new_optimizer_updates=0,
            original_protocol_preserved=True,diagnostic_inputs_unchanged=True,protected_file_sha256=protection,
            outputs_sha256=outputs,seed_reservation=seed_check,implementation_status='not_implemented',
            literature_scope='six_full_papers_bounded_comparison_not_exhaustive_novelty_search')
(OUT/'design_stage_verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
for p in paths:
    text=p.read_text(encoding='utf-8');assert len(text)>1000 and '\ufffd' not in text
    print(p.name,len(text),'characters',sha(p))
print('PASS: arithmetic, preserved input hashes, prospective seed scan, output readback; no research launch.')
