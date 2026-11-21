"""Unit checks for the development gate without a simulator or learner run."""
import argparse,json,unittest
from pathlib import Path
import run_emc_tradeoff as r

class ReportingTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.spec=__import__('json').loads(r.SPEC.read_text())
 def test_bootstrap_uses_paired_episode_differences(self):
  left=[dict(return_sum=x,deadline_success=x/100) for x in range(48)]
  right=[dict(return_sum=x-3,deadline_success=x/100-.02) for x in range(48)]
  a=r.paired_bootstrap(left,right,2000,20261113);b=r.paired_bootstrap(left,right,2000,20261113)
  self.assertEqual(a,b);self.assertEqual(a['return_sum']['mean_difference'],3.)
  self.assertAlmostEqual(a['deadline_fraction']['mean_difference'],.02)
  self.assertEqual(a['episodes'],48);self.assertEqual(a['draws'],2000)
 def fixture(self):
  fresh={};diagnostics={}
  for rep in range(3):
   for arm in r.ARMS:
    fresh[f'{arm}_{rep}']=[]
    diagnostics[f'{arm}_{rep}']=dict(fresh_terminal_abs_error_p95=1.,fixed_probe_positive_greedy_fraction=0.,
     median_action_ms=.1 if arm==r.ARMS[0] else .2,online_parameter_bytes=422016 if arm==r.ARMS[0] else 277444,p95_action_ms=.5)
   for i in range(48):
    base=100.+i
    fresh[f'{r.ARMS[0]}_{rep}'].append(dict(return_sum=base,deadline_success=.70))
    fresh[f'{r.ARMS[1]}_{rep}'].append(dict(return_sum=base-3,deadline_success=.72))
    fresh[f'{r.ARMS[2]}_{rep}'].append(dict(return_sum=base,deadline_success=.74))
  heuristics={name:[dict(return_sum=100.+i,deadline_success=.70) for i in range(48)] for name in ('nearest_rsu','uav_selective')}
  return fresh,heuristics,diagnostics
 def test_all_replicates_and_all_frozen_screens_are_required(self):
  fresh,heuristics,diagnostics=self.fixture()
  result=r.analyze_development(self.spec,{},fresh,heuristics,diagnostics)
  self.assertEqual(result['state'],'all_three_emc_initializations_pass')
  self.assertEqual(len(result['replicates']),3)
  for row in fresh[f'{r.ARMS[2]}_2']:row['deadline_success']=.70
  result=r.analyze_development(self.spec,{},fresh,heuristics,diagnostics)
  self.assertEqual(result['state'],'one_or_more_predeclared_screens_failed')
  self.assertFalse(result['replicates']['2']['screens']['versus_additive64'])
  fresh,heuristics,diagnostics=self.fixture();diagnostics[f'{r.ARMS[2]}_1']['median_action_ms']=1.
  result=r.analyze_development(self.spec,{},fresh,heuristics,diagnostics)
  self.assertFalse(result['replicates']['1']['screens']['median_latency_ratio'])

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 stream=__import__('io').StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReportingTests))
 args.output.mkdir(parents=True,exist_ok=False)
 (args.output/'tests.txt').write_text(stream.getvalue(),encoding='utf-8')
 (args.output/'verification.json').write_text(json.dumps(dict(state='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,simulator_steps=0,optimizer_updates=0,pilot_launched=False),indent=2),encoding='utf-8')
 print(stream.getvalue())
 if not result.wasSuccessful():raise SystemExit(1)
