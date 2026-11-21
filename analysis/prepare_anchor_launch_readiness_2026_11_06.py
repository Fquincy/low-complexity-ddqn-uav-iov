"""Create provenance-complete successor without changing training or inference."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
mapping={'run_anchor_research_route_v2':'run_anchor_research_route_v3','audit_anchor_research_route_v2':'audit_anchor_research_route_v3','test_anchor_research_route_v2':'test_anchor_research_route_v3'}
for old,new in mapping.items():
    dest=ROOT/(new+'.py');assert not dest.exists()
    text=(ROOT/(old+'.py')).read_text(encoding='utf-8')
    for before,after in mapping.items():text=text.replace(before,after)
    if old.startswith('run_'):
        text=text.replace('"""Eleven-arm training orchestration; research modes stay disabled pending decision."""','"""Seventeen-arm orchestration with transitive source freeze; research disabled."""')
        text=text.replace("'terminal_anchor_prototype.py',", "'terminal_anchor_prototype.py','run_v3_research_gate_v2.py',")
        text=text.replace('    verify_prior_engineering()\n    out.mkdir', '    verify_prior_engineering()\n    verify_factory_provenance()\n    out.mkdir')
        text=text.replace('def verify_prior_engineering():', '''def verify_factory_provenance():
    name='run_v3_research_gate_v2.py'
    prior=read(ROOT/'validation/terminal_anchor_engineering_completion_2026-11-06.json')
    if base.old.sha(ROOT/name)!=prior['files'][name]:
        raise ValueError('transitive model factory changed:'+name)
    if name not in SOURCES:raise ValueError('transitive model factory missing from source freeze')
    return prior['files'][name]


def verify_prior_engineering():''')
    elif old.startswith('audit_'):
        text=text.replace('"""Independent saved-artifact audit for the eleven-arm synthetic fixture."""','"""Independent seventeen-arm audit including transitive factory provenance."""')
        text=text.replace("    for name,digest in plan['source_sha256'].items():", "    runner.verify_factory_provenance()\n    assert set(plan['source_sha256'])==set(runner.SOURCES)\n    for name,digest in plan['source_sha256'].items():")
    else:text=text.replace('anchor_research_route_v2_tests_2026-11-06.json','anchor_research_route_v3_tests_2026-11-06.json')
    dest.write_text(text,encoding='utf-8')
print('Created v3 runner, auditor and tests; predecessor bytes preserved.')
