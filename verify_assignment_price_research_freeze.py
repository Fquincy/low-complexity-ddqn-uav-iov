"""Independent prospective design checks; no model/simulator execution."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation/assignment_price_research_freeze_2026-11-15_01'
SPEC=ROOT/'ASSIGNMENT_PRICE_RESEARCH_PROTOCOL_2026-11-15.json'
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(p):
    parent=read(ROOT/p['parent_protocol']);assert sha(ROOT/p['parent_protocol'])==p['parent_sha256']
    assert p['primary']=='coupled_sweep' and p['status']=='DESIGN_FREEZE_NOT_EXECUTABLE_RELEASE'
    ids=[d['id'] for d in p['learned_arms']];assert len(ids)==len(set(ids))==15
    originals=[d for d in parent['learned_arms'] if not d.get('normalized_horizon_loss')]
    assert p['learned_arms'][4:]==originals
    assert p['comparisons']['required_reference_arms']==[d['id'] for d in originals]
    assert set(p['comparisons']['primary_report_against_each'])==(set(ids)|set(p['heuristics']))-{p['primary']}
    assert p['comparisons']['reference_each_fit']==parent['comparisons']['reference_each_fit']
    assert p['value_stability']==parent['value_stability']
    assert p['action_cost']['primary_each_fit']==parent['action_cost']['primary_each_fit']
    assert p['final']==parent['final'] and p['simulator']==parent['simulator']
    for k,v in parent['training'].items():
        if k!='loss_definitions':assert p['training'][k]==v,k
    t=p['training'];d=p['development'];fits=len(d['model_seeds']);episodes=t['episodes_per_fit'];h=t['slots_per_episode']
    assert fits==5 and len(d['train_blocks'])==5 and all(x['count']==episodes for x in d['train_blocks'])
    assert d['training_slots']==len(ids)*fits*episodes*h==600000
    assert d['optimizer_updates']==len(ids)*fits*(episodes*h-t['warmup']+1)==523275
    assert d['target_copies']==len(ids)*fits*((episodes*h-t['warmup']+1)//t['target_interval_updates'])==2550
    assert d['evaluation_slots']==(len(ids)*fits+len(p['heuristics']))*d['evaluation_seeds']['count']*h==151680
    assert d['task_records']==(d['training_slots']+d['evaluation_slots'])*30
    assert d['final_checkpoints']==len(ids)*fits==75
    assert p['action_cost']['raw_timed_calls']==len(ids)*fits*5*128==48000
    assert p['scope']==dict(security_trust=False,cloud=False,charging=False,word_thesis_changed=False)
    decision=p['decision'];assert not decision['research_authorization'] and not decision['runner_bound'] and not decision['novelty_established']
    assert decision['every_fit'] and decision['all_scientific_screens_required'] and decision['primary_value_failure_is_failure']
    for m in p['comparisons']['mechanisms']:
        assert m['left']==p['primary'] and m['right'] in ids
        assert [m['each_fit_return_difference_min'],m['each_fit_deadline_fraction_difference_min'],m['equal_fit_mean_return_difference_min'],m['equal_fit_mean_deadline_fraction_difference_min']]==[0.,0.,2.,.01]
    assert {m['id'] for m in p['comparisons']['mechanisms']}=={'sweep','coupling_and_selection','learned_prices'}
    for name,digest in p['definition_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in p['frozen_legacy_definition_sha256'].items():assert sha(ROOT/name)==digest
    seeds=read(OUT/'seed_declaration_audit.json');assert seeds['passed'] and seeds['reserved_no_execution']
    for name,digest in seeds['checked_documents'].items():assert sha(ROOT/name)==digest
    return True

class Checks(unittest.TestCase):
    def test_complete_freeze(self):self.assertTrue(verify(read(SPEC)))
    def test_reject_budget_threshold_and_launch_drift(self):
        original=read(SPEC)
        for path,value in [(['development','training_slots'],1),(['action_cost','primary_each_fit','median_ratio_max'],2.),
                           (['decision','research_authorization'],True),(['final','evaluation_seeds','start'],1)]:
            p=copy.deepcopy(original);parent=p
            for key in path[:-1]:parent=parent[key]
            parent[path[-1]]=value
            with self.assertRaises(AssertionError):verify(p)

if __name__=='__main__':
    for name,digest in read(OUT/'manifest.json').items():assert sha(OUT/name)==digest
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    report=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        protocol_sha256=sha(SPEC),simulator_steps=0,optimizer_updates=0,research_launched=False)
    path=OUT/'independent_design_verification.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    raise SystemExit(not result.wasSuccessful())
