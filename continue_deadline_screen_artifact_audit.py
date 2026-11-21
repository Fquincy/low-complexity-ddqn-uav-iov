"""Audit-only completion of existing development artifacts. Never retrains."""
import json,traceback
from pathlib import Path
import run_deadline_screen_research_v2 as r
import audit_deadline_screen_research_v3 as auditor
import deadline_screen_research_storage_v2 as storage

ROOT=Path(__file__).resolve().parent
SAVED=ROOT/'development_runs/deadline_screen_development_2026-11-14_01'
OUT=ROOT/'validation/deadline_screen_audit_correction_2026-11-14_01'

def status(state,detail):
    r.write(OUT/'audit_progress.json',dict(state=state,detail=detail,new_training_slots=0,new_evaluation_slots=0,retiming=False))
    (OUT/'LIVE_STATUS.txt').write_text('DEADLINE SCREEN: AUDIT-ONLY CONTINUATION\nState: '+state+'\n'+detail+
        '\nTraining complete: 440000/440000 slots,55 checkpoints.\nEvaluation complete:111360 slots.\nNo retraining, re-evaluation or retiming.\n',encoding='utf-8')

def main():
    try:
        r.require(r.read(OUT/'correction_tests.json')['passed'],'correction tests not passed')
        amendment=r.read(OUT/'amendment.json')
        for record in amendment['files'].values():r.require(r.sha(record['path'])==record['sha256'],'amended source or original manifest changed')
        r.require(not (OUT/'development_saved_artifact_audit.json').exists(),'audit output already exists')
        status('RUNNING','Verifying saved deployment/storage artifacts, then every saved training/evaluation record.')
        deployment=storage.audit(SAVED/'run',SAVED/'deployment_storage')
        r.write(OUT/'deployment_storage_audit.json',deployment)
        result=auditor.audit(SAVED/'run',True,OUT/'development_saved_artifact_audit.json')
        scientific=r.read(SAVED/'run/analysis.json')
        r.write(OUT/'workflow_verification.json',dict(artifact_audit_passed=True,deployment_storage_audit=deployment,
            source_run=str(SAVED),scientific_screens=scientific['computed_screens'],all_scientific_screens_passed=scientific['all_scientific_screens_passed'],
            trained_checkpoints_preserved=True,new_training_slots=0,new_evaluation_slots=0,retiming=False))
        status('COMPLETED','Full saved-artifact and storage audits passed. Scientific results require the completion review.')
        print(json.dumps({k:v for k,v in result.items() if k!='optimizer_distributions'}),flush=True)
        return 0
    except Exception:
        status('STOPPED','Audit failed; inspect stderr.log. Preserve artifacts and do not automatically restart.')
        traceback.print_exc();return 1

if __name__=='__main__':raise SystemExit(main())
