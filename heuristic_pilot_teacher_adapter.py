"""Bound frozen teacher arithmetic without importing historical research runners.

The two original source files are verified against the frozen pilot protocol.
Only named function/class definitions execute, with native decorators replaced
by identity. This is the Python reference backend, not the old compiled path.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def build_teacher(root, config_class):
    root = Path(root)
    namespace = {'np': np, 'native': lambda f: f,
                 'r': SimpleNamespace(Chapter3Config=config_class)}
    for filename, names in [
        ('assignment_price_fast_runtime.py', {'validate_mask'}),
        ('assignment_price_load_radio_heuristic.py',
         {'constants', 'prepare', 'rates', 'objective', 'decode', 'Runtime'}),
    ]:
        tree = ast.parse((root / filename).read_text(encoding='utf-8'))
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        if {n.name for n in nodes} != names:
            raise ValueError('teacher definition inventory changed')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), namespace)
    return namespace['Runtime']()
