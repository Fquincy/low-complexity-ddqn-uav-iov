"""Prepare explicit development-only terminal-anchor launch wrappers."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for old,new in [('run_anchor_research_route_v3','run_anchor_development_v4'),('audit_anchor_research_route_v3','audit_anchor_development_v4')]:
    dest=ROOT/(new+'.py');assert not dest.exists()
    text=(ROOT/(old+'.py')).read_text(encoding='utf-8')
    text=text.replace('run_anchor_research_route_v3','run_anchor_development_v4').replace('audit_anchor_research_route_v3','audit_anchor_development_v4')
    if old.startswith('run_'):
        text=text.replace("'terminal_anchor_prototype.py',", "'launch_anchor_development.py','supervise_anchor_development.py','terminal_anchor_prototype.py',")
    dest.write_text(text,encoding='utf-8')
text=(ROOT/'supervise_v3_development.py').read_text(encoding='utf-8')
text=text.replace('launch_v3_development','launch_anchor_development').replace('440000','680000').replace('440,000','680,000').replace('=55','=85').replace('/ 55','/ 85').replace('VERSION 3 DEVELOPMENT RUN','TERMINAL-ANCHOR SEVENTEEN-ARM DEVELOPMENT RUN')
path=ROOT/'supervise_anchor_development.py';assert not path.exists();path.write_text(text,encoding='utf-8')
print('Versioned source-only launch wrappers prepared.')
