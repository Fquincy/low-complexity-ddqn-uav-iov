"""Saved-data equivalence and scale checks. No simulator/optimizer/timing calls."""
import ast,copy,gzip,json,tracemalloc,unittest
from pathlib import Path
import run_deadline_screen_research as old
import run_deadline_screen_research_v2 as new
import audit_deadline_screen_research_v2 as audit
import deadline_screen_research_storage_v2 as storage

ROOT=Path(__file__).resolve().parent
SAVED=ROOT/'validation/deadline_screen_research_engineering_2026-11-14_01/fixture'
OUT=ROOT/'validation/deadline_screen_launch_review_2026-11-14_01'
MEASUREMENTS={}

def row(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)

class Checks(unittest.TestCase):
    def test_01_training_and_runtime_procedures_unchanged(self):
        def functions(file):
            tree=ast.parse(file.read_text())
            return {x.name:ast.dump(x,include_attributes=False) for x in tree.body if isinstance(x,ast.FunctionDef)}
        a=functions(ROOT/'run_deadline_screen_research.py');b=functions(ROOT/'run_deadline_screen_research_v2.py')
        self.assertEqual(set(a),set(b))
        for name in a:
            if name!='compact':self.assertEqual(a[name],b[name],name)
        self.assertFalse(new.DEVELOPMENT_RELEASED)
        for mode,token in [('development',None),('development',new.CONFIRM),('final',None)]:
            path=OUT/('forbidden_'+mode+str(bool(token)))
            with self.assertRaises(RuntimeError):new.run(path,mode,token)
            self.assertFalse(path.exists())

    def test_02_exact_analysis_with_compact_episode_statistics(self):
        p=new.protocol();pl=new.plan(p,'integration-fixture');ev={d['id']:[[] for _ in pl['model_seeds']] for d in p['learned_arms']};h={x['id']:[] for x in p['heuristics']}
        for arm in ev:
            for fit in range(2):
                for seed in pl['evaluation_seeds']:
                    raw=row(SAVED/arm/f'fit_{fit:02d}'/f'evaluation_{seed}.json.gz')
                    compact=new.compact(raw);self.assertEqual(compact,audit.compact_verified(raw));ev[arm][fit].append(compact)
        for name in h:
            h[name]=[new.compact(row(SAVED/f'heuristic_{name}_{seed}.json.gz')) for seed in pl['evaluation_seeds']]
        calculated=new.analysis.summarize(p,ev,h,old.read(SAVED/'costs.json'),True)
        self.assertEqual(calculated,old.read(SAVED/'analysis.json'))
        MEASUREMENTS['all_comparisons_values_bootstraps_decisions_exact']=True

    def test_03_full_count_summary_memory(self):
        raw=row(next(SAVED.glob('screened_price24/fit_00/evaluation_*.json.gz')))
        tracemalloc.start();start=tracemalloc.get_traced_memory()[0]
        rows=[new.compact(copy.deepcopy(raw)) for _ in range(10)]
        raw_sample_current=tracemalloc.get_traced_memory()[0]-start
        del rows;tracemalloc.stop()
        compact=new.compact(raw)
        tracemalloc.start();start=tracemalloc.get_traced_memory()[0]
        rows=[copy.deepcopy(compact) for _ in range(2784)]
        current,peak=tracemalloc.get_traced_memory();retained=current-start;peak-=start
        tracemalloc.stop()
        self.assertEqual(len(rows),2784);self.assertLess(retained,128*1024*1024)
        self.assertNotIn('tasks',rows[0]);self.assertNotIn('slots',rows[0])
        MEASUREMENTS.update(full_evaluation_rows=2784,retained_summary_bytes=retained,peak_summary_allocation_bytes=peak)

    def test_04_full_saved_fixture_audit_without_refitting(self):
        calculated=audit.audit(SAVED,False)
        self.assertEqual(calculated,old.read(SAVED/'saved_artifact_audit.json'))
        old.write(OUT/'v2_saved_fixture_audit.json',calculated)
        exported=ROOT/'validation/deadline_screen_research_engineering_2026-11-14_01/deployment_storage'
        result=storage.audit(SAVED,exported)
        self.assertTrue(result['passed']);old.write(OUT/'v2_storage_audit.json',result)

    def test_05_streamed_log_totals(self):
        path=OUT/'large_synthetic_log.jsonl'
        # Match 6,977 update rows, including research-size 128x32 action arrays.
        item=dict(optimizer_ns=100,anchor_indices=list(range(32)),next_actions=[[1]*32 for _ in range(128)])
        text=json.dumps(item)+'\n'
        with path.open('w') as stream:
            for _ in range(6977):stream.write(text)
        tracemalloc.start();totals=storage.log_totals(path);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        self.assertEqual(totals,(697700,6977*32));self.assertLess(peak,8*1024*1024)
        MEASUREMENTS.update(stream_test_rows=6977,stream_test_log_bytes=path.stat().st_size,stream_peak_allocation_bytes=peak)
        # Disposable synthetic text only; retain its size/hash, not 90MB of generated repetition.
        MEASUREMENTS['stream_test_log_sha256']=old.sha(path);path.unlink()

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    old.write(OUT/'scale_tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),
        measurements=MEASUREMENTS,new_simulator_steps=0,new_optimizer_updates=0,timing_repeated=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
