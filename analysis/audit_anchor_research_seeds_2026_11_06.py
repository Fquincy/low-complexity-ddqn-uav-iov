"""Reuse bounded metadata/header scanner with a separate prospective reservation."""
from pathlib import Path

source=Path(__file__).with_name('audit_v3_seed_consumption_2026_11_06.py')
code=source.read_text(encoding='utf-8').replace(
    'LOW_COMPLEXITY_V3_RESEARCH_PROTOCOL_2026-11-05.json',
    'TERMINAL_ANCHOR_RESEARCH_PROTOCOL_2026-11-06.json').replace(
    'v3_seed_consumption_review_2026-11-06.json',
    'anchor_research_seed_reservation_2026-11-06.json')
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
