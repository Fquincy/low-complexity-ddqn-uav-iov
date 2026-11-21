"""Full minibatch backward fixture and independent readiness evidence checks."""
import copy
import gc
import json
import math
from pathlib import Path
import statistics
import unittest
from unittest.mock import patch
import numpy as np
import torch
import run_assignment_price_research_bound as w

OUT=w.r.ROOT/'validation/assignment_price_launch_review_2026-11-15_01'
BIND=w.r.ROOT/'validation/assignment_price_research_binding_2026-11-15_01'

class Checks(unittest.TestCase):
    def test_full_batch128_anchor32_backward_no_optimizer_steps(self):
        w.r.configure();full=w.plan('development')['cfg'];fixture=BIND/'fixture_evidence';pl=w.plan('fixture')
        cfg=copy.deepcopy(full);cfg.update(warmup=8,replay_capacity=80)
        reports=[]
        for arm in w.base.ARMS:
            # Engineering weights/task traces only; reserved research seeds never used.
            a=w.base.build(arm,dict(model_seed=pl['model_seeds'][0],cfg=cfg));folder=fixture/arm/'fit_00'
            saved=torch.load(folder/'final.pt',weights_only=True,map_location='cpu')
            a['online'].load_state_dict(saved['online']);a['target'].load_state_dict(saved['target'])
            from audit_assignment_price_fixture_v2 import load
            for ep in range(2):
                row=load(folder/f'train_{ep:03d}.json.gz');arrays=np.load(folder/f'train_{ep:03d}.npz')
                for i,slot in enumerate(row['slots']):a['replay'].add(arrays['states'][i],np.array(slot['action']),slot['reward'],arrays['next_states'][i],slot['done'])
            a['slots']=80;before=w.r.models.original.existing.base.old.digest(a['online'].state_dict())
            with patch.object(a['opt'],'step',return_value=None) as step:
                row=w.base.update(a)
                step.assert_called_once()
            self.assertEqual(len(row['replay_indices']),128);self.assertEqual(len(row['anchor_indices']),32 if (arm in w.base.adp.ARMS or a['online'].anchored) else 0)
            self.assertTrue(np.isfinite(row['loss']) and np.isfinite(row['grad_norm_before_clip']))
            self.assertTrue(all(p.grad is None for p in a['target'].parameters()))
            self.assertEqual(before,w.r.models.original.existing.base.old.digest(a['online'].state_dict()))
            self.assertFalse(a['opt'].state)
            reports.append(dict(arm=arm,batch_size=128,anchor_batch=len(row['anchor_indices']),loss_finite=True,
                grad_finite=True,target_gradients_absent=True,parameters_unchanged=True,actual_optimizer_steps=0,
                scope='Existing engineering weights/replay; actual optimizer.step replaced by no-op for backward-path fixture'))
            del a,saved,row;gc.collect()
        w.r.write(OUT/'full_minibatch_checks.json',dict(passed=True,arms=reports,new_simulator_steps=0,actual_optimizer_steps=0,
            no_research_seed_use=True,scope='Batch128/anchor32 compute and backward feasibility only, not research-length memory/time stability'))

    def test_independent_heuristic_timing_arithmetic_and_projection(self):
        plan=w.r.read(OUT/'PRE_HEURISTIC_TIMING_PLAN.json');rows=w.r.read(OUT/'heuristic_timings.json');saved=w.r.read(OUT/'heuristic_cost_summary.json')
        self.assertEqual([(x['round'],x['index']) for x in rows],[(r,i) for r in range(5) for i in range(128)])
        self.assertTrue(all(isinstance(x['ns'],int) and x['ns']>0 for x in rows))
        samples=sorted(x['ns'] for x in rows);location=(len(samples)-1)*.95;lo=math.floor(location);hi=math.ceil(location)
        self.assertAlmostEqual(statistics.median(samples)/1e6,saved['median_ms'],places=12)
        self.assertAlmostEqual((samples[lo]+(samples[hi]-samples[lo])*(location-lo))/1e6,saved['p95_ms'],places=12)
        self.assertEqual(w.r.sha(w.r.ROOT/plan['input']),plan['input_sha256'])
        self.assertEqual(w.r.sha(w.r.ROOT/'assignment_price_load_radio_heuristic.py'),plan['source_sha256'])
        resource=w.r.read(OUT/'resource_projection.json')
        fields=['projected_optimizer_log_bytes_scaled_by_batch','projected_training_trace_bytes','projected_learned_evaluation_trace_bytes',
                'projected_heuristic_trace_bytes','raw_populated_replay_file_bound_bytes','projected_checkpoint_bytes']
        self.assertAlmostEqual(sum(resource[k] for k in fields),resource['projected_total_bytes'],places=4)
        self.assertEqual(resource['replay_allocated_bytes_per_training_agent'],50000*(2*533*4+32+8))
        self.assertEqual(resource['disk_headroom_2x_pass'],resource['free_disk_bytes']>=2*resource['projected_total_bytes'])
        self.assertFalse(w.DEVELOPMENT_RELEASED)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    path=OUT/'readiness_tests.json';assert not path.exists()
    w.r.write(path,dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        new_simulator_steps=0,actual_optimizer_steps=0,research_launched=False))
    raise SystemExit(not result.wasSuccessful())
