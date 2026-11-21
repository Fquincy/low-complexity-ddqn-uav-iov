"""Cross-check offline probes against saved evidence and write the diagnosis."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import assignment_price_prototype as adp

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'development_runs/assignment_price_development_2026-11-16_01/run'
OUT = ROOT/'validation/assignment_price_offline_diagnosis_2026-11-19_01'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    torch.set_num_threads(1)
    p = read(ROOT/'ASSIGNMENT_PRICE_RESEARCH_PROTOCOL_2026-11-15.json')
    d = read(OUT/'diagnostics.json'); a = read(RUN/'analysis.json')
    assert len(d['gradients']) == 80 and len(d['task_timelines']) == 5
    for name, sha in p['definition_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha, name
    for name, sha in d['input_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
    # Exercise numerical reproduction and the terminal Bellman identity on saved states.
    for fit in range(5):
        model = adp.Critic('coupled_sweep',p['development']['model_seeds'][fit]).eval()
        checkpoint = torch.load(RUN/'coupled_sweep'/f'fit_{fit:02d}'/'final.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint['online'])
        for ep in d['plan']['episodes']:
            import gzip
            folder=RUN/'coupled_sweep'/f'fit_{fit:02d}'
            with gzip.open(folder/f'eval_{ep:03d}.json.gz','rt') as stream: row=json.load(stream)
            with np.load(folder/f'eval_{ep:03d}.npz') as z:
                s=torch.from_numpy(z['states']); ns=torch.from_numpy(z['next_states'])
            actions=torch.tensor([x['action'] for x in row['slots']],dtype=torch.int64)
            with torch.no_grad():
                # Evaluation used singleton neural forwards. Batched BLAS can
                # differ by a few FP32 ulps; reproduce the saved route exactly.
                q=np.array([model(s[i:i+1],actions[i:i+1]).item() for i in range(40)])
                np.testing.assert_allclose(q,row['chosen_q'],rtol=1e-10,atol=1e-10)
                y=adp.targets(model,model,torch.tensor([row['slot_rewards'][-1]],dtype=torch.float64),ns[-1:],torch.tensor([True]))
                assert float(y[0]) == row['slot_rewards'][-1]
                assert model(ns[-1:],actions[-1:]).item() == 0
    for t in d['task_timelines']:
        fit=t['fit']; counts=t['counts']
        assert counts['tasks']==57600
        assert counts['primary_not_on_time']==sum(a['value_stability']['coupled_sweep'][fit]['deadline_misses_by_destination'])
        assert counts['primary_not_on_time']==counts['primary_miss_heuristic_success']+counts['both_miss']
        assert counts['heuristic_not_on_time']==counts['primary_success_heuristic_miss']+counts['both_miss']
        ours=t['reward_component_sums_primary']; theirs=t['reward_component_sums_heuristic']
        delta=-(.5*(ours['normalized_delay_accrual']-theirs['normalized_delay_accrual'])+
                .5*(ours['normalized_uav_energy']-theirs['normalized_uav_energy'])+
                ours['new_deadline_misses_per_vehicle']-theirs['new_deadline_misses_per_vehicle'])/48
        assert np.isclose(delta,a['comparisons']['observed_load_radio']['per_fit'][fit]['return_difference'])
    lines=[
        'ASSIGNMENT-PRICE OFFLINE GRADIENT AND TASK-TIMELINE DIAGNOSIS',
        'Date: 2026-11-19 (Asia/Shanghai)', '',
        'DECISION: RADIO/ASSIGNMENT PHYSICS IS THE MAIN POLICY-DESIGN LEAD;',
        'GRADIENT CONFLICT IS A SEPARATE CALIBRATION LEAD, NOT A PROVEN ROOT CAUSE.',
        '', 'SCOPE AND VERIFICATION',
        'Four checkpoint arms, all five initializations, 80 gradient batches.',
        'Gradient states/actions come from primary episodes 0,12,24,36: four',
        'fixed, evenly spaced saved episodes, with ten slots per horizon bin.',
        'Each arm sees identical primary states and executed primary actions.',
        'Other arms are therefore probed off-policy; this is not their own-policy',
        'gradient distribution or a reconstruction of past replay sampling.',
        'All 48 primary evaluation episodes per fit are used for task diagnosis.',
        'All workload/task identities are matched against saved heuristic traces.',
        '329 consumed inputs were hashed before and after diagnosis: unchanged.',
        'Frozen model-definition hashes verified against the protocol.',
        'All 800 primary checkpoint predictions reproduce saved chosen Q.',
        'Terminal Bellman targets equal saved terminal rewards; Q at h=0 is zero.',
        'Task miss counts reproduce saved analysis; reward-component differences',
        'reproduce all five saved primary-minus-heuristic return differences.',
        'No optimizer steps, task generation, simulator rollout, exports or timing.',
        '', '1. GRADIENT GEOMETRY',
        'The implemented critic has Q=-h*C. Holding the state/action fixed,',
        'dQ/dtheta=-h*dC/dtheta. This algebra amplifies sensitivity with horizon;',
        'it does not itself establish a defective objective.',
        'On the primary probe, norm(grad mean Q)/norm(grad mean Q/h) is about',
        '35.3,25.3,15.4,5.4 in successive ten-slot bins, consistent with the',
        'remaining-horizon factor. This is not a normalized-loss experiment.',
        'Fit | early-bin Huber grad norm | terminal-anchor grad norm | early-anchor cosine']
    for fit in range(5):
        g=next(g for g in d['gradients'] if g['arm']=='coupled_sweep' and g['fit']==fit and g['slots'][0]==0)
        lines.append(f"{fit} | {g['gradient_norm']:.4f} | {g['anchor_gradient_norm']:.4f} | {g['uniform_anchor_cosine']:.4f}")
    lines += ['Early-bin original-unit gradients exceed norm10 in all primary fits.',
              'Early-bin gradients oppose the anchor in all five fits. Across four',
              'bins per fit, 12/20 primary uniform-anchor cosines are negative.',
              'Some final-checkpoint bins are below the clipping threshold even',
              'though historical mixed replay updates were all clipped. These are',
              'different batches, states and times, so there is no contradiction.',
              'Clipping limits the combined vector, not each objective separately.',
              'These probes show gradient imbalance/conflict; they do not prove',
              'that changing clipping, loss weights or training duration will help.',
              '', '2. MATCHED DEADLINE OUTCOMES',
              'Fit | primary not on time / 57,600 | heuristic not on time | transmission alone > deadline | queue alone > deadline']
    for t in d['task_timelines']:
        c=t['counts']
        lines.append(f"{t['fit']} | {c['primary_not_on_time']} | {c['heuristic_not_on_time']} | {c.get('transmission_delay_s_alone_exceeds_deadline',0)} | {c.get('queue_delay_s_alone_exceeds_deadline',0)}")
    lines += ['Component counts can overlap and are not an exhaustive partition.',
              'Fit4: 2,792 of 3,079 not-on-time tasks go to UAV1; 2,631 have',
              'transmission duration alone above the deadline. Six are unfinished.',
              'Fit3: 1,177 of 1,499 not-on-time tasks go to RSU2; queue time alone',
              'exceeds the deadline for 206 tasks across all destinations.',
              'Compute duration alone exceeds no deadline among the missed tasks.',
              'This identifies realized transmission/queue bottlenecks, not a',
              'counterfactual proof that changing a single destination fixes a task.',
              'The matched heuristic has different histories, joint assignments and',
              'UAV movements. It is a workload-matched policy comparison.',
              '', '3. CONCRETE RECORDED TIMELINE',
              'Fit4, episode3, task340: deadline .257661s; assigned to UAV1.',
              'Uplink wait 1.234749s + transmission 2.649466s + compute .042017s',
              '= total 3.926232s (zero compute queue). In the matched heuristic',
              'trajectory the same task goes to RSU1 and completes in .223247s.',
              'This is a post-hoc worst-lateness example, not a typical-task estimate.',
              '', '4. WHY THE CURRENT COST MODEL MAY MISS THESE DECISIONS',
              'The learned joint potential explicitly aggregates assigned compute',
              'work and observed remote compute queues. It does not explicitly',
              'calculate assigned-set UAV interference, upload service or task deadlines.',
              'The neural network receives relevant observations and could learn them;',
              'absence of an explicit term does not prove those effects are ignored.',
              'The simulator computes UAV interference from simultaneous active uploads.',
              'The heuristic explicitly estimates radio/link/backlog and shared compute',
              'effects. Its near-perfect deadline result makes radio/queue-aware',
              'assignment a stronger concrete design lead than merely increasing CPU.',
              'The heuristic is not free: preserved standalone engineering timing is',
              '.5378ms median, .68924ms p95. That was not a same-session learned-',
              'controller speed comparison and cannot establish a speedup ratio.',
              '', '5. RETURN GAP IS NOT JUST A MISS-COUNT ISSUE',
              'Across all five fits, primary delay accrual and UAV energy components',
              'are also larger than the heuristic. Their weighted sums plus miss',
              'penalties exactly reproduce the saved return gap. Therefore fixing',
              'only the calibration metric or relabeling unfinished tasks is insufficient.',
              '', '6. AVOID REPEATING THE FAILED CORRECTION',
              'HORIZON_LOSS_DEVELOPMENT_RESULTS_REVIEW_2026-11-14.txt already reports',
              'failed normalized-horizon-loss policy/calibration results. The present',
              'sensitivity probe does not override that experiment. Do not automatically',
              'normalize the loss again, raise clipping, extend training or relax screens.',
              '', 'NEXT BOUNDED GATE',
              'Write ONE prospective radio/queue-aware compact DDQN design brief.',
              'Specify how an efficient action decoder accounts for upload/interference',
              'and deadline slack while retaining DDQN learning and bounded inference.',
              'State the difference from the existing observed-load/radio heuristic,',
              'the current compute-work coupling and plain small DDQN explicitly.',
              'Do not assume adding physics or a safety mask is novel; check literature.',
              'Include a mechanism-removal ablation and the strong observed-load/radio',
              'heuristic. Treat calibration as a separate measured requirement.',
              'Before implementation/training, justify computational cost and choose',
              'fresh development seeds through an explicit prospective amendment.',
              'No architecture, new run, final evaluation or thesis claim is authorized',
              'by this diagnostic report. Preserve all historical evidence.',
              '', 'EVIDENCE', str(OUT/'DIAGNOSTIC_PLAN.json'), str(OUT/'diagnostics.json'),
              str(RUN/'analysis.json'), str(RUN/'optimizer_distributions.json'),
              str(ROOT/'HORIZON_LOSS_DEVELOPMENT_RESULTS_REVIEW_2026-11-14.txt')]
    report=ROOT/'ASSIGNMENT_PRICE_OFFLINE_GRADIENT_TIMELINE_DIAGNOSIS_2026-11-19.txt'
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    checks=dict(passed=True,gradient_batches=80,matched_primary_tasks=288000,
                heuristic_unique_tasks=57600,primary_predictions_reproduced=800,
                terminal_target_identities=20,terminal_boundary_checks=20,
                frozen_definition_hashes_verified=True,input_hashes_verified=len(d['input_sha256']),
                saved_miss_counts_and_return_differences_reproduced=True,
                new_optimizer_updates=0,new_simulator_steps=0,final_seeds_used=False,
                report=str(report))
    (OUT/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks))


if __name__=='__main__': main()
