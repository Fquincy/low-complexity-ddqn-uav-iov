"""Version the fixed checker for the separately declared original-core route."""
from pathlib import Path
R=Path(__file__).resolve().parents[1];dest=R/'check_anchor_torch_core_runtime.py';assert not dest.exists()
s=(R/'check_anchor_numerical_correction.py').read_text()
s=s.replace('import run_anchor_development_v4 as r','import run_anchor_development_v4 as r\nimport anchor_original_torch_core_runtime as original_core')
s=s.replace('ANCHOR_RUNTIME_NUMERICAL_CORRECTION_PROTOCOL_2026-11-11.json','ANCHOR_TORCH_CORE_RUNTIME_AMENDMENT_2026-11-11.json')
s=s.replace("SOURCES=['check_anchor_numerical_correction.py',PROTOCOL.name,*r.SOURCES]", "SOURCES=['check_anchor_torch_core_runtime.py','anchor_original_torch_core_runtime.py',PROTOCOL.name,*r.SOURCES]")
a=s.index('def candidate_session(');b=s.index('\ndef invoke(',a)
s=s[:a]+'''def candidate_session(path):
    return original_core.session(path)

'''+s[b:]
dest.write_text(s,encoding='utf-8');print(dest.name)
