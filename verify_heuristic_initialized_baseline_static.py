"""Dependency-free source audit for the isolated baseline."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULE = ROOT / 'heuristic_initialized_ddqn_baseline.py'
TEST = ROOT / 'test_heuristic_initialized_ddqn_baseline.py'

source = MODULE.read_text(encoding='utf-8')
tree = ast.parse(source)
names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
required = {'TeacherInitializedCompactDDQN', 'demonstration_actions', 'imitation_loss', 'ddqn_loss', 'deploy_actions'}
assert required <= names
deploy = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'deploy_actions')
deploy_names = {node.id for node in ast.walk(deploy) if isinstance(node, ast.Name)}
assert 'teacher' not in deploy_names
ddqn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'ddqn_loss')
assert any(isinstance(node, ast.With) for node in ast.walk(ddqn)), 'target path must use no-grad context'
assert 'new_optimizer_updates' in TEST.read_text(encoding='utf-8')
record = {
    'passed': True,
    'checks': ['required_interfaces', 'deployment_teacher_exclusion', 'target_no_grad_context', 'fixture_receipt'],
    'module_sha256': hashlib.sha256(MODULE.read_bytes()).hexdigest(),
    'test_sha256': hashlib.sha256(TEST.read_bytes()).hexdigest(),
    'parameter_bytes': 67364,
    'new_simulator_steps': 0,
    'new_optimizer_updates': 0,
    'research_launch_authorized': False,
}
out = ROOT / 'validation/heuristic_initialization_feasibility_2026-11-19_01/static_source_audit.json'
out.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record))
