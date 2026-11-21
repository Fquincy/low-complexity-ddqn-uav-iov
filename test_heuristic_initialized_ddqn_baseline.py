"""Isolated correctness fixture for the teacher-initialized compact baseline."""
import copy
import ast
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import heuristic_initialized_ddqn_baseline as h

UPDATES = 0
LEARNING = {}
TEACHER = {}
torch.set_num_threads(1)


def teacher_reference():
    """Execute original frozen teacher equations without importing old runners.

    This is a Python arithmetic reference, not its compiled deployment runtime.
    """
    root = Path(__file__).resolve().parent
    sys.path.insert(0,str(root/'Source'))
    from chapter3_iov_env_v2 import Chapter3Config
    ns = {'np': np, 'native': lambda f: f,
          'r': SimpleNamespace(Chapter3Config=Chapter3Config)}
    tree = ast.parse((root/'assignment_price_fast_runtime.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='validate_mask')
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<original mask>', 'exec'), ns)
    tree = ast.parse((root/'assignment_price_load_radio_heuristic.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original teacher reference>', 'exec'), ns)
    return ns['Runtime']()


def observations(n=6):
    rng = np.random.default_rng(2026111901)
    x = rng.uniform(0.05, 0.95, (n, 533)).astype(np.float32)
    task = x[:, :240].reshape(n, 30, 8)
    task[:, :, 6] = rng.uniform(0.60, 1.0, (n, 30))
    task[:, :, 7] = rng.uniform(5 / 6, 1.0, (n, 30))
    x[:, -1] = 0.5
    return x


class BaselineChecks(unittest.TestCase):
    def test_student_shape_storage_and_spec(self):
        model = h.TeacherInitializedCompactDDQN()
        self.assertEqual(h.parameter_bytes(model), 67364)
        self.assertFalse(h.spec_record(model)["teacher_called_by_deploy"])
        self.assertEqual(model.spec.teacher_training_only, True)

    def test_teacher_labels_and_student_deployment_are_separate(self):
        states = observations()

        class FakeTeacher:
            def __init__(self): self.calls = 0
            def values(self, state):
                self.calls += 1
                actions = np.ones((1,32),np.int64); actions[:,30:]=0
                return actions, np.zeros((1,), np.int64)

        teacher = FakeTeacher()
        labels = h.demonstration_actions(teacher, states)
        self.assertEqual(teacher.calls, len(states))
        model = h.TeacherInitializedCompactDDQN().eval()
        action = h.deploy_actions(model, states[0])
        self.assertEqual(action.shape, (32,))
        self.assertEqual(teacher.calls, len(states))
        self.assertTrue(np.all((action >= 0) & (action < 5)))
        self.assertTrue(torch.isfinite(h.imitation_loss(model, torch.from_numpy(states), labels)))

    def test_imitation_updates_student_and_not_teacher(self):
        global UPDATES
        torch.manual_seed(2026111902)
        states = torch.from_numpy(observations())
        labels = torch.ones((len(states), 32), dtype=torch.long); labels[:,30:]=0
        model = h.TeacherInitializedCompactDDQN()
        before = [p.detach().clone() for p in model.parameters()]
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        initial = float(h.imitation_loss(model, states, labels).detach())
        initial_accuracy=float((model.greedy(states)[0][:,:30]==labels[:,:30]).float().mean())
        for _ in range(40):
            opt.zero_grad(); loss = h.imitation_loss(model, states, labels)
            loss.backward(); opt.step(); UPDATES += 1
        final = float(h.imitation_loss(model, states, labels).detach())
        final_accuracy=float((model.greedy(states)[0][:,:30]==labels[:,:30]).float().mean())
        LEARNING.update(initial_loss=initial, final_loss=final, initial_accuracy=initial_accuracy,
                        final_accuracy=final_accuracy, synthetic_updates=40)
        self.assertLess(final, initial)
        self.assertGreater(final_accuracy,initial_accuracy)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def test_detached_ddqn_target_and_terminal_bypass(self):
        global UPDATES
        states = torch.from_numpy(observations(4))
        next_states = states.clone()
        next_states[:, -1] = 0.25
        next_states[[1,3], -1] = 1.
        actions = torch.zeros((4, 32), dtype=torch.long)
        rewards = torch.tensor([-1., -2., -3., -4.])
        dones = torch.tensor([False, True, False, True])
        model = h.TeacherInitializedCompactDDQN()
        target = copy.deepcopy(model).eval()
        for p in target.parameters(): p.requires_grad_(False)
        loss = h.ddqn_loss(model, target, states, actions, rewards, next_states, dones)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        self.assertTrue(all(p.grad is None for p in target.parameters()))
        before=[p.detach().clone() for p in model.parameters()]
        opt=torch.optim.Adam(model.parameters(),lr=1e-3); opt.step(); UPDATES+=1
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.parameters())))

    def test_target_numeric_selection_and_terminal_bypass(self):
        model=h.TeacherInitializedCompactDDQN(); target=copy.deepcopy(model)
        s=torch.from_numpy(observations(2)); ns=s.clone(); ns[1,-1]=1.
        actions=torch.zeros((2,32),dtype=torch.long)
        chosen=torch.full((1,32),2,dtype=torch.long)
        with patch.object(model,'forward',return_value=torch.tensor([-4.,-2.],requires_grad=True)), patch.object(model,'greedy',return_value=(chosen,None,None)) as select, patch.object(target,'forward',return_value=torch.tensor([-3.])) as evaluate:
            loss=h.ddqn_loss(model,target,s,actions,torch.tensor([-1.,-2.]),ns,torch.tensor([False,True]))
            self.assertEqual(float(loss.detach()),0.)
            self.assertEqual(len(select.call_args.args[0]),1)
            self.assertTrue(torch.equal(evaluate.call_args.args[1],chosen))
        ns[:,-1]=1.
        with patch.object(model,'greedy',side_effect=AssertionError('terminal decode')):
            h.ddqn_loss(model,target,s,actions,-torch.ones(2),ns,torch.ones(2,dtype=torch.bool))

    def test_original_screen_and_real_teacher_reference(self):
        x=observations(2)
        scope={'np':np,'torch':torch}
        source=ast.parse(Path('deadline_screen_prototype.py').read_text())
        nodes=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ('intervals','allowed')]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original screen>','exec'),scope)
        np.testing.assert_array_equal(h.allowed(x),scope['allowed'](x))
        teacher=teacher_reference(); begin=time.perf_counter_ns()
        labels=h.demonstration_actions(teacher,x)
        TEACHER.update(reference='original source functions, Python arithmetic; not compiled deployment timing',
                       label_states=2,wall_ns=time.perf_counter_ns()-begin, simulator_steps=0,
                       stationary_motion=True,teacher_only_empty_battery_restriction=True)
        self.assertTrue(torch.all(labels[:,30:]==0))
        self.assertTrue(np.take_along_axis(h.allowed(x),labels.numpy()[:,:,None],2).all())
        self.assertTrue(torch.isfinite(h.imitation_loss(h.TeacherInitializedCompactDDQN(),torch.from_numpy(x),labels)))

    def test_checkpoint_and_deployment_teacher_absence(self):
        model=h.TeacherInitializedCompactDDQN().eval(); x=observations(1)[0]
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'student.pt'; torch.save(model.state_dict(),p)
            other=h.TeacherInitializedCompactDDQN().eval(); other.load_state_dict(torch.load(p,weights_only=True))
            np.testing.assert_array_equal(h.deploy_actions(model,x),h.deploy_actions(other,x))
        self.assertFalse(any('teacher' in name for name in model.state_dict()))
        with patch.object(h,'demonstration_actions',side_effect=AssertionError('training only')):
            h.deploy_actions(model,x)

    def test_invalid_inputs(self):
        model = h.TeacherInitializedCompactDDQN()
        with self.assertRaises(ValueError): h.deploy_actions(model, observations(1)[0].astype(np.float64))
        with self.assertRaises(ValueError): h.imitation_loss(model, torch.from_numpy(observations()), torch.zeros((6, 31), dtype=torch.long))
        with self.assertRaises(ValueError): h.imitation_loss(model, torch.from_numpy(observations()), torch.full((6,32),5,dtype=torch.long))
        x=observations(1); x[0,-1]=1.
        with self.assertRaises(ValueError): h.deploy_actions(model,x[0])


if __name__ == '__main__':
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp()) / 'verification.json'
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BaselineChecks))
    record = {"passed": result.wasSuccessful(), "tests": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "parameter_bytes": 67364, "new_simulator_steps": 0,
              "new_optimizer_updates": UPDATES, "synthetic_learning_check":LEARNING,
              "teacher_label_cost":TEACHER,
              "research_launch_authorized": False}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
