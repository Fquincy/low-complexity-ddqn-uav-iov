"""Create a new route-integrated runner/auditor/test set; preserve predecessors."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
amendment=json.loads((ROOT/'ANCHOR_RUNTIME_ROUTE_AMENDMENT_2026-11-06.json').read_text())
assert sha(ROOT/'run_anchor_research_preparation.py')==amendment['old_runtime_sha256']
assert sha(ROOT/'anchor_runtime_single_validation.py')==amendment['new_adapter_sha256']
mapping={'run_anchor_research_preparation':'run_anchor_research_route_v2',
 'audit_anchor_research_preparation':'audit_anchor_research_route_v2',
 'test_anchor_research_preparation':'test_anchor_research_route_v2'}
def rename(text):
    for old,new in mapping.items():text=text.replace(old,new)
    return text
s=rename((ROOT/'run_anchor_research_preparation.py').read_text())
s=s.replace('import terminal_anchor_prototype as anchor',
    'import terminal_anchor_prototype as anchor\nimport anchor_runtime_single_validation as runtime_adapter')
s=s.replace("'terminal_anchor_prototype.py',PROTOCOL.name", 
    "'terminal_anchor_prototype.py','anchor_runtime_single_validation.py','ANCHOR_RUNTIME_ROUTE_AMENDMENT_2026-11-06.json',PROTOCOL.name")
start=s.index('def runtime_values(');end=s.index('\ndef main():',start)
s=s[:start]+'''def runtime_values(runtime,arm,state):
    if arm in CORRECTED:return runtime_adapter.values(runtime,arm,state)
    return fused.values(runtime,arm,state)

'''+s[end:]
s=s.replace("    p=read(PROTOCOL);assert",'''    amendment=read(ROOT/'ANCHOR_RUNTIME_ROUTE_AMENDMENT_2026-11-06.json')
    assert base.old.sha(PROTOCOL)==amendment['base_protocol_sha256']
    assert base.old.sha(ROOT/'anchor_runtime_single_validation.py')==amendment['new_adapter_sha256']
    p=read(PROTOCOL);assert''')
s=s.replace("        arm_order_seed=p['paired_randomization']", "        runtime_route_amendment_sha256=base.old.sha(ROOT/'ANCHOR_RUNTIME_ROUTE_AMENDMENT_2026-11-06.json'),\n        arm_order_seed=p['paired_randomization']")
(ROOT/'run_anchor_research_route_v2.py').write_text(s,encoding='utf-8')
a=rename((ROOT/'audit_anchor_research_preparation.py').read_text())
a=a.replace("    assert plan['protocol_sha256']==base.old.sha(runner.PROTOCOL)",
    "    assert plan['protocol_sha256']==base.old.sha(runner.PROTOCOL)\n    assert plan['runtime_route_amendment_sha256']==base.old.sha(runner.ROOT/'ANCHOR_RUNTIME_ROUTE_AMENDMENT_2026-11-06.json')")
(ROOT/'audit_anchor_research_route_v2.py').write_text(a,encoding='utf-8')
t=rename((ROOT/'test_anchor_research_preparation.py').read_text())
t=t.replace("anchor_research_preparation_tests_2026-11-06.json", "anchor_research_route_v2_tests_2026-11-06.json")
t=t.replace('    def test_protocol_budget_and_prospective_scope(self):', '''    def test_declared_route_is_installed_and_terminal_rejected(self):
        self.assertFalse(r.RESEARCH_ENABLED)
        x=np.zeros(533,np.float32);x[-1]=1.
        with self.assertRaises(ValueError):r.runtime_values(None,'price_both',x)
        with patch.object(r.runtime_adapter,'values',return_value='route_marker') as call:
            self.assertEqual(r.runtime_values(None,'price_both',x),'route_marker')
            call.assert_called_once()
    def test_protocol_budget_and_prospective_scope(self):''')
(ROOT/'test_anchor_research_route_v2.py').write_text(t,encoding='utf-8')
print('Created versioned route integration; original source and protocol bytes preserved.')
