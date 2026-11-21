"""Record a pre-training decoder amendment. Preserve the version-1 drafts."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/controller_redesign_2026-11-05'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

brief_path=ROOT/'LOW_COMPLEXITY_CONTROLLER_REDESIGN_DESIGN_BRIEF_2026-11-05.txt'
protocol_path=ROOT/'LOW_COMPLEXITY_CONTROLLER_REDESIGN_COMPARISON_PROTOCOL_2026-11-05.txt'
brief=brief_path.read_text(encoding='utf-8')
protocol=protocol_path.read_text(encoding='utf-8')

note='''CURRENT DESIGN: VERSION 2, 2026-11-05
This is the recommended design for the next engineering review. Version 1 is
retained as a superseded design draft; no version has been trained or launched.
Amendment reason: a symmetric synthetic algebra fixture makes the original two
parallel price passes oscillate between equally unbalanced assignments. Their
cost is 0.5 versus exhaustive optimum 0.25. Version 2 adds four deterministic
single-move repair rounds and reaches 0.25 in that fixture. This is mechanics
evidence only; it proves neither real-world performance nor global optimality.
No old research results, seeds, rewards or model definitions were altered.
The 4,052-parameter network and reserved research seeds/budgets are unchanged.

'''
brief=note+brief.replace('CONTROLLER DESIGN BRIEF - VERSION 1','CONTROLLER DESIGN BRIEF - VERSION 2').replace('All DIMENSIONS FIXED FOR VERSION 1','All DIMENSIONS FIXED FOR VERSION 2').replace('ALL DIMENSIONS FIXED FOR VERSION 1','ALL DIMENSIONS FIXED FOR VERSION 2')
old='''Score a^(0),a^(1),a^(2) with the FULL C above and retain the minimum;
ties choose earliest candidate. Keep the selected independent movements.
The output costs no more under this model than a^(0), because a^(0) is retained.
This does NOT guarantee improvement in real return or deadline performance,
nor exact global minimization. Synchronous passes can oscillate or preserve an
unbalanced solution, especially with symmetric costs. Record candidate agreement,
load splits and reduced-fixture exhaustive optimality gaps. No beam search,
full environment transition, hidden queue simulation or adaptive retry occurs.
Keep all five destinations legal; do not invent a deadline-feasibility oracle.'''
new='''Score a^(0),a^(1),a^(2) with the FULL C above and retain the minimum;
ties choose earliest candidate. Keep the selected independent movements.
Then perform exactly four single-move repair rounds. In each round, for EVERY
vehicle i and destination d, recompute the leave-one-out marginal price at the
currently retained assignment and calculate the EXACT single-move cost change:
  Delta_i,d = u_i,d-u_i,a_i + p_i,d-p_i,a_i.
Choose the lowest Delta over all N*5 options; ties choose lowest vehicle index,
then lowest destination index. If Delta < -1e-6, form that one changed assignment;
otherwise use the unchanged assignment. Compute its full C and accept the move
only if C decreases by more than 1e-6. Otherwise retain the old assignment.
Always execute all four rounds; no variable retries or extra network calls.
This requires three initial plus four repair full-cost calculations, seven
in total. Report both predicted Delta and actual full-score change for audits.
The output costs no more under this model than a^(0), because it is retained
and every accepted repair reduces full cost. The symmetric oscillation example
is fixed, but other local minima, multi-task swaps and movement-placement
interactions remain limitations. No real-return/deadline improvement or exact
global minimization is guaranteed. Record assignment/load changes, repair
acceptance rates and reduced-fixture exhaustive optimality gaps. No beam search,
environment look-ahead, hidden queue simulation or adaptive retry occurs.
Keep all five destinations legal; do not invent a deadline-feasibility oracle.'''
assert old in brief
brief=brief.replace(old,new)
brief=brief.replace('three-candidate approximate decoder','bounded approximate decoder with four single-move repairs')
brief=brief.replace('Decoder work is O(K*N*D) with K=2,D=5 plus three full score calculations;',
                    'Decoder work is O((K+R)*N*D), K=2 price passes,R=4 repairs,D=5,\nplus seven full score calculations;')
brief=brief.replace('module, price passes, full candidate scoring','module, price passes, four repair rounds, full scoring')
brief=brief.replace('Do not silently change two passes or choose the best timing run.',
                    'Do not silently change two passes/four repairs or choose the best timing run.')
brief=brief.replace('and three-candidate approximate decoder','and bounded approximate decoder')

protocol=note+protocol.replace('PROSPECTIVE CONTROLLER COMPARISON PROTOCOL - VERSION 1','PROSPECTIVE CONTROLLER COMPARISON PROTOCOL - VERSION 2')
protocol=protocol.replace('controller, two passes.','controller, two price passes plus four repairs.')
protocol=protocol.replace('same dimensions, pooling, cost, two passes and schedule.',
                          'same dimensions, pooling, cost, two price passes/four repairs and schedule.')
protocol=protocol.replace('same shared network/beta/decoder budget as A, but',
                          'same shared network/beta/two-pass/four-repair budget as A, but')
protocol=protocol.replace('before applying the decreasing transform. Tests the cost parameterization.',
                          'before applying the decreasing transform. Repairs use exact single-move\n  changes to that RAW inner cost, not sp-transformed unary costs. Tests the\n  cost parameterization.')
protocol=protocol.replace('Optional untrained forced-no-correction A decoder is a diagnostic only, clearly',
                          'Optional untrained A with its four repairs disabled is a diagnostic only, clearly')
protocol=protocol.replace('heads, load passes, scoring and','heads, load passes, all four repair rounds, seven full scores and')
protocol=protocol.replace('If design changes after this declaration, save v2 with reasons before training;\nkeep v1 and its hash.',
                          'If design changes after version 2, save v3 with reasons before training;\nkeep versions 1 and 2 and their hashes.')

paths=[]
for label,text in [('DESIGN_BRIEF',brief),('COMPARISON_PROTOCOL',protocol)]:
    p=ROOT/f'LOW_COMPLEXITY_CONTROLLER_REDESIGN_{label}_v2_2026-11-05.txt'
    if p.exists():raise FileExistsError(p)
    p.write_text(text,encoding='utf-8');assert p.read_text(encoding='utf-8')==text
    paths.append(p)
spec=json.loads((OUT/'prospective_comparison_v1.json').read_text())
spec['id']='shared-congestion-cost-ddqn-design-v2'
spec['supersedes']='shared-congestion-cost-ddqn-design-v1'
spec['amendment_reason']='symmetric_synthetic_price_oscillation_counterexample_no_research_data_used'
spec['decoder'].update(repair_rounds=4,full_scored_candidates=7,repair_accept_epsilon=1e-6,
                       single_move_cost_change='exact_for_model_but_global_decoder_still_approximate')
spec['fixture']['algebra_seed_already_used_synthetic_only']=99002
p=OUT/'prospective_comparison_v2.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(spec,indent=2),encoding='utf-8');paths.append(p)
record=json.loads((OUT/'design_stage_verification.json').read_text())
record['status']='current_v2_design_files_written_and_read_back_no_learner_implemented'
record['current_design']='v2'
record['v1_draft_retained']=True
record['outputs_sha256'].update({str(p.relative_to(ROOT)):sha(p) for p in paths})
diagnosis=ROOT/'LOW_COMPLEXITY_CONTROLLER_REDESIGN_DIAGNOSIS_2026-11-05.txt'
record['outputs_sha256'][str(diagnosis.relative_to(ROOT))]=sha(diagnosis)
record['synthetic_algebra_checks']=json.loads((OUT/'synthetic_equation_checks.json').read_text())
for rel,digest in record['protected_file_sha256'].items():assert sha(ROOT/rel)==digest
for rel,digest in record['outputs_sha256'].items():assert sha(ROOT/rel)==digest
(OUT/'design_stage_verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print('Current v2 files:')
for p in paths:print(p.name,sha(p))
print('PASS: version 1 preserved; v2 read back; original frozen research files unchanged; no training.')
