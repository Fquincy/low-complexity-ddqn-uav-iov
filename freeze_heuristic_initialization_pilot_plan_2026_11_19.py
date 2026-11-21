"""Reserve a pilot plan and check arithmetic/seeds. Never executes a model."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = ROOT/'HEURISTIC_INITIALIZED_DDQN_PAIRED_PILOT_PROTOCOL_2026-11-19.json'
EVIDENCE = ROOT/'validation/heuristic_initialization_pilot_plan_2026-11-19_01'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ranges(value, path=''):
    """Read explicit seed scalars/lists and start/count seed-block reservations."""
    result=[]
    if isinstance(value,dict):
        if ('seed' in path.lower() or 'block' in path.lower()) and isinstance(value.get('start'),int) and isinstance(value.get('count'),int):
            result.append((value['start'],value['start']+value['count']-1,path))
            return result
        for k,v in value.items():result.extend(ranges(v,path+'/'+k))
    elif isinstance(value,list):
        for i,v in enumerate(value):result.extend(ranges(v,path+'/'+str(i)))
    elif type(value) is int and 'seed' in path.lower():
        result.append((value,value,path))
    return result


def main():
    assert not SPEC.exists() and not EVIDENCE.exists(),'Do not overwrite a frozen plan'
    history=[]; parsed=[]; errors=[]
    for p in sorted(ROOT.glob('*.json')):
        try:
            data=json.loads(p.read_text(encoding='utf-8-sig'))
            history.extend((lo,hi,str(p.name)+loc) for lo,hi,loc in ranges(data))
            parsed.append({'file':p.name,'sha256':sha(p)})
        except (ValueError,UnicodeError) as e:errors.append({'file':p.name,'error':str(e)})
    assert not errors,errors
    seeds={
        'model_seeds':[918190001,918190011,918190021],
        'demonstration_seed_blocks':[{'start':918200000+1000*i,'count':10} for i in range(3)],
        'online_seed_blocks':[{'start':918210000+1000*i,'count':50} for i in range(3)],
        'evaluation_seed_block':{'start':918220000,'count':16},
        'bc_sampler_seeds':[918230001+10*i for i in range(3)],
        'epsilon_branch_seeds':[918240001+10*i for i in range(3)],
        'replay_sampler_seeds':[918250001+10*i for i in range(3)],
        'job_order_seed':918260001,'timing_order_seed':918260011}
    proposed=ranges(seeds)
    collisions=[]
    for i,(lo,hi,label) in enumerate(proposed):
        for a,b,old in history+proposed[:i]:
            if max(lo,a)<=min(hi,b):collisions.append({'new':label,'existing':old,'intersection':[max(lo,a),min(hi,b)]})
    assert not collisions,collisions
    names=['heuristic_initialized_ddqn_baseline.py','controller_v3_amortized.py',
           'assignment_price_load_radio_heuristic.py','assignment_price_fast_runtime.py',
           'Source/chapter3_iov_env.py','Source/chapter3_iov_env_v2.py',
           'HEURISTIC_INITIALIZED_DDQN_PAIRED_PILOT_PLAN_2026-11-19.txt']
    protocol={
        'protocol_id':'heuristic-initialization-incremental-BC-pilot-v1',
        'date':'2026-11-19','status':'PLAN_FREEZE_NOT_EXECUTABLE_RELEASE',
        'claim_scope':'Descriptive incremental BC development screen; not novelty or thesis confirmation',
        'plan':'HEURISTIC_INITIALIZED_DDQN_PAIRED_PILOT_PLAN_2026-11-19.txt',
        'arms':['teacher_bc_then_ddqn','common_demo_no_bc'],
        'reference':'observed_load_radio','fits':3,'slots_per_episode':40,
        'seeds':seeds,'paired_initial_weights':True,'shared_demonstration_replay':True,
        'student':{'architecture':[533,24,161],'parameters':16841,'parameter_bytes':67364,
                   'buffer_bytes':20,'gamma':1.,'horizon_scaled':False,'terminal_anchor':False},
        'demonstrations':{'episodes_per_fit':10,'transitions_per_fit':400,
                          'teacher_execution_adapter_bound':False,'teacher_motion':'stationary 0,0',
                          'extra_teacher_empty_battery_restriction':True},
        'bc':{'steps_per_fit':200,'batch_size':64,'temperature':1.,'learning_rate':.001,
              'sampling':'without replacement within batch; independent batches',
              'optimizer_reset_before_ddqn':True},
        'online':{'episodes_per_fit':50,'slots_per_fit':2000,'warmup_slots':128,
                  'first_update_zero_based_slot':128,'updates_per_fit':1872,
                  'batch_size':64,'replay_capacity':4096,'learning_rate':.0001,
                  'replay':'uniform with replacement','target_interval_updates':200,
                  'scheduled_target_copies_per_fit':9,'initial_target_copy':True,
                  'loss':'scalar smooth_l1 beta1, detached DDQN target',
                  'epsilon':'1 if t<128 else 1-.95*min(t/2000,1); t zero-based before action'},
        'optimizer_common':{'type':'Adam','betas':[.9,.999],'eps':1e-8,'weight_decay':0.,'clipnorm':10.},
        'evaluation':{'episodes_per_policy':16,'teacher_evaluated_once':True,
                       'student_evaluation_slots':3840,'teacher_evaluation_slots':640,
                       'before_evaluation':'freeze all six final checkpoint hashes'},
        'cost':{'observation_count':128,'warmups':32,'rounds':3,'calls_per_round':128,
                'teacher_in_student_deployment':False,'acceptance_retiming':False,
                'mode':'complete eager action selection, descriptive'},
        'totals':{'student_training_slots':12000,'teacher_demonstration_slots':1200,
                  'evaluation_slots':4480,'all_simulated_slots':17680,
                  'bc_updates':600,'ddqn_updates':11232,'all_updates':11832,
                  'final_student_checkpoints':6},
        'decision':{'promising':'both policy differences >=0 in every fit and both equal-fit means >0',
                    'otherwise':'HOLD expansion, retain mixed/negative evidence, no automatic retuning',
                    'value_calibration':'reported diagnostic; no calibrated-value or historical-gate pass claim',
                    'statistical_confirmatory_claim':False},
        'retention':{'final_checkpoints_only':True,'best_fit_selection':False,'retry':False},
        'definition_sha256':{n:sha(ROOT/n) for n in names},
        'simulator_defaults':json.loads((ROOT/'HORIZON_LOSS_RESEARCH_PROTOCOL_2026-11-14.json').read_text())['simulator']['defaults'],
        'research_launch_authorized':False,'pilot_launched':False,'final_seeds_consumed':False,
        'word_thesis_changed':False,'runner_implemented':False,'auditor_implemented':False,
        'next_gate':'Implement bound pilot runner, teacher adapter and independent auditor; reduced isolated orchestration fixture; separate launch review'}
    assert 3*2*(2000-128)==11232
    assert 3*10*40+3*2*50*40+(3*2+1)*16*40==17680
    assert 3*200+11232==11832
    EVIDENCE.mkdir(parents=True)
    SPEC.write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    audit={'passed':True,'protocol_sha256':sha(SPEC),'historical_root_json_documents_checked':len(parsed),
           'scope':'All root JSON explicit seed fields and start/count blocks; not a full run-log inventory. Refresh before launch.',
           'checked_documents':parsed,'parse_errors':errors,'seed_collisions':collisions,
           'proposed_intervals':proposed,'arithmetic_checked':True,'new_simulator_steps':0,'new_optimizer_updates':0}
    (EVIDENCE/'plan_verification.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    p=ROOT/'LOW_COMPLEXITY_PROJECT_STATUS.json';s=json.loads(p.read_text(encoding='utf-8-sig'))
    s['next']=protocol['next_gate']
    s['heuristic_initialization_pilot_plan_2026_11_19']={
        'protocol':SPEC.name,'protocol_sha256':sha(SPEC),'plan':protocol['plan'],
        'status':protocol['status'],'seed_audit_documents':len(parsed),'seed_collisions':0,
        'planned_simulated_slots':17680,'launch_authorized':False,'launched':False,'word_thesis_changed':False}
    p.write_text(json.dumps(s,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'plan_frozen':True,'seed_documents_checked':len(parsed),'collisions':0,
                      'planned_total_slots':17680,'research_launched':False,'new_optimizer_updates':0}))


if __name__=='__main__':main()
