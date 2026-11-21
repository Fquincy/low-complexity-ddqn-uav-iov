"""Expanded backend/heuristic tests before the full isolated workflow."""
from itertools import product
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
import run_assignment_price_full_fixture as w

BASE=w.r.ROOT/'validation/assignment_price_full_engineering_2026-11-15_01'

class Units(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        w.r.configure();cls.folder=Path(tempfile.mkdtemp(prefix='adp_all_backend_checks_'))
        cls.states=np.load(w.r.ROOT/w.protocol()['action_cost']['common_inputs'])['observations']

    def test_all_eleven_control_exports(self):
        for arm in w.ARMS[4:]:
            model=w.build(arm,w.plan())['online'].eval();runtime=w.references.Runtime(model,self.folder/f'{arm}.onnx')
            for state in self.states:
                with torch.inference_mode():a,q=model.greedy(torch.from_numpy(state)[None])
                b,v=runtime.values(state)
                np.testing.assert_array_equal(a.numpy(),b);np.testing.assert_allclose(q.numpy(),v,rtol=1e-5,atol=1e-6)

    def test_heuristic_indices_interference_and_cost_oracle(self):
        h=w.load_radio;cfg=h.constants();x=self.states[0].copy();parts=h.prepare(x,cfg)
        bits,cycles,deadline,local,remote,backlog,gain,battery=parts
        np.testing.assert_allclose(bits,x[:240].reshape(30,8)[:,5]*cfg[1],rtol=1e-6)
        np.testing.assert_allclose(local,np.expm1(x[240:270].astype(float))*.3)
        np.testing.assert_allclose(backlog,np.expm1(x[274:514].astype(float).reshape(30,8)[:,:4])*cfg[1])
        for i in range(30):
            for j in range(2):
                xy=x[:240].reshape(30,8)[i,:2].astype(float)*cfg[0]
                uv=x[526:532].reshape(2,3)[j,:2].astype(float)*cfg[0]
                distance=np.sqrt(np.dot(xy-uv,xy-uv)+cfg[11]**2)
                los=1/(1+cfg[12]*np.exp(-cfg[13]*(np.degrees(np.arcsin(cfg[11]/distance))-cfg[12])))
                loss=20*np.log10(4*np.pi*cfg[10]/299792458.)+20*np.log10(distance)+los*cfg[14]+(1-los)*cfg[15]
                np.testing.assert_allclose(gain[i,j],10**(-loss/10),rtol=1e-12)
        n=3;g=gain[:n];aa=np.array([1,0,0]);bb=np.array([1,1,0])
        self.assertLess(h.rates(bb,g,cfg)[0,0],h.rates(aa,g,cfg)[0,0])
        small=[v[:n] if v.shape[0]==30 else v.copy() for v in parts[:7]]
        b,c,d,l,q,r,gg=small;order=np.argsort(-c,kind='stable')
        for assignment in product(range(5),repeat=n):
            a=np.array(assignment);rate=h.rates(a,gg,cfg);finish=[];energy=0
            for i in range(n):
                dest=a[i]
                if dest==0:value=l[i]+c[i]/cfg[2]
                else:
                    previous=[k for k in order[:list(order).index(i)+1] if a[k]==dest]
                    value=sum(r[i,k]/rate[i,k] for k in range(4))+b[i]/rate[i,dest-1]+q[dest]+sum(c[k]/cfg[2+dest] for k in previous)
                    if dest<=2:energy+=cfg[20]*c[i]*cfg[2+dest]**2
                finish.append(value)
            expected=cfg[17]*np.mean(np.array(finish)/.3)+cfg[18]*np.mean(np.array(finish)>d+1e-10)+cfg[19]*energy/cfg[21]
            actual=h.objective(a,*small,order,cfg)
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
        a=np.array([1,1,0]);base=h.objective(a,*small,order,cfg)
        grown=[v.copy() for v in small];grown[4][1]+=1
        self.assertGreater(h.objective(a,*grown,order,cfg),base)
        grown=[v.copy() for v in small];grown[5][0,0]+=1e8
        self.assertGreater(h.objective(a,*grown,order,cfg),base)

    def test_heuristic_ties_masks_battery_and_fixed_bound(self):
        h=w.load_radio;cfg=h.constants();x=self.states[0].copy();x[528]=0;x[531]=0
        parts=h.prepare(x,cfg);mask,_=w.adp.allowed(x[None])[0,:30],None
        a,count=h.decode(*parts,mask.copy(),cfg)
        self.assertFalse(np.isin(a,[1,2]).any());self.assertTrue(mask[np.arange(30),a].all())
        self.assertEqual(count,2*(int(mask.sum())-60))
        runtime=h.Runtime();b,_=runtime.values(x);c,_=runtime.values(x);np.testing.assert_array_equal(b,c)
        n=3;zeros=np.zeros(n);mask=np.ones((n,5),bool)
        # All CPU/radio work zero gives exact equal objective: choose local.
        a,count=h.decode(zeros,zeros,zeros+1,zeros,np.zeros(5),np.zeros((n,4)),np.ones((n,2))*.001,np.ones(2),mask,cfg)
        np.testing.assert_array_equal(a,np.zeros(n,int));self.assertEqual(count,2*n*5)
        bad=x.copy();bad[-1]=1
        with self.assertRaises(ValueError):runtime.values(bad)

    def test_protocol_and_execution_guards(self):
        self.assertEqual(len(w.protocol()['learned_arms']),15)
        for mode,confirmation in [('development',None),('development',w.CONFIRM),('final',None)]:
            output=self.folder/f'blocked_{mode}_{confirmation}'
            with self.assertRaises(RuntimeError):w.run(output,mode,confirmation)
            self.assertFalse(output.exists())

if __name__=='__main__':
    BASE.mkdir(exist_ok=False)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Units))
    w.r.write(BASE/'unit_tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),
        control_export_state_checks=1408,heuristic_exhaustive_assignments=125,new_simulator_steps=0,new_optimizer_updates=0))
    raise SystemExit(not result.wasSuccessful())
