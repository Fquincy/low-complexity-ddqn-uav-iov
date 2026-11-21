"""Focused checks of strict equivalence and audit rejection; no new benchmark."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import audit_compiled_inference_gate as a
import compiled_controller_inference as c
import run_controller_redesign_engineering as r

class Checks(unittest.TestCase):
    def test_actions_must_match_even_when_values_match(self):
        with self.assertRaises(AssertionError):a.assert_equivalent([[0,1]],[-1.],[[1,0]],[-1.])

    def test_nonfinite_values_cannot_pass_equivalence(self):
        with self.assertRaises(AssertionError):a.assert_equivalent([[0]],[float('nan')],[[0]],[float('nan')])

    def test_value_tolerance_is_enforced(self):
        a.assert_equivalent([[0]],[-1.],[[0]],[-1.000001])
        with self.assertRaises(AssertionError):a.assert_equivalent([[0]],[-1.],[[0]],[-1.01])

    def test_wrong_shape_rejected_before_native_runtime(self):
        class Bomb:
            def run(self,*args):raise AssertionError('runtime should not be called')
        for shape in [(532,),(2,533),(1,533)]:
            with self.assertRaises(ValueError):c.compiled_values(Bomb(),np.zeros(shape,np.float32))

    def test_manifest_rejects_altered_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);p=folder/'graph.onnx';p.write_bytes(b'synthetic graph bytes')
            (folder/'run_manifest.json').write_text(json.dumps({'files':{'graph.onnx':r.old.sha(p)}}))
            a.verify_manifest(folder);p.write_bytes(b'altered bytes')
            with self.assertRaises(AssertionError):a.verify_manifest(folder)

    def test_absolute_latency_pass_does_not_override_relative_failure(self):
        metrics={'shared_congestion_cost':dict(parameter_bytes=16208,median_ms=.39,p95_ms=.54),
            'additive64_nonpositive':dict(parameter_bytes=194944,median_ms=.035,p95_ms=.046),
            'additive128_unbounded':dict(parameter_bytes=422016,median_ms=.038,p95_ms=.042)}
        screens=r.cost_screens(metrics,r.spec())
        self.assertEqual(screens,dict(online_storage=True,complete_action_median=False,complete_action_p95=True))

if __name__=='__main__':unittest.main(verbosity=2)
