"""Refresh actual local run metadata/episode headers, excluding design declarations."""
import concurrent.futures,gzip,hashlib,json,re,subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from freeze_deadline_screen_research_protocol_2026_11_12 import historical_seed_values,reserve_values

SPEC=ROOT/'HORIZON_LOSS_RESEARCH_PROTOCOL_2026-11-14.json'
OUTPUT=ROOT/'validation/horizon_loss_launch_review_2026-11-14_01/seed_refresh.json'

def main(output=OUTPUT):
    output=Path(output)
    if output.exists():raise SystemExit('Refusing to overwrite seed refresh')
    p=json.loads(SPEC.read_text());groups=reserve_values(p)
    development=set().union(*(values for key,values in groups.items() if key.startswith('development') or key=='research_orders'))
    final=set(historical_seed_values(p['final']))
    for seed in p['final']['model_seeds']:
        final.update(seed+offset for offset in p['rng']['offsets'].values())
    roots=[ROOT/'validation',ROOT/'development_runs',ROOT.parent/'October 31/Working/Project/validation',ROOT.parent/'October 31/Working/Project/development_runs']
    files=[];errors=[]
    for base in roots:
        if not base.exists():errors.append(dict(path=str(base),error='missing root'));continue
        result=subprocess.run(['rg','--files','--hidden','-g','!**/__pycache__/**',str(base)],capture_output=True,text=True,check=True)
        for line in result.stdout.splitlines():
            path=Path(line);rel=path.relative_to(base)
            if any(t in ('source_snapshot','supporting_source_snapshot','release_source_snapshot','source_freeze','frozen_source','sources','continuation_source_snapshot','exports') for t in rel.parts):continue
            name=path.name.lower()
            if name.endswith('.json.gz') or path.suffix=='.json' and any(token in name for token in ('runtime','counters','training_summary','config','run_metadata')):
                files.append(path)
    pattern=re.compile(r'"(?:seed|model_seed|training_seed|train_seed|evaluation_seed|task_seed)"\s*:\s*(\d+)\b')
    def inspect(path):
        try:
            if path.name.endswith('.json.gz'):
                with gzip.open(path,'rt',encoding='utf-8-sig') as f:prefix=f.read(16384)
                values={int(x) for x in pattern.findall(prefix)};kind='compressed_header'
                # Only actual episode headers count as evidence of final seed use.
                final_hits=sorted(values&final)
            else:
                if path.stat().st_size>32*1024*1024:return dict(path=str(path),error='metadata exceeds bounded size')
                values=set(historical_seed_values(json.loads(path.read_text(encoding='utf-8-sig'))));kind='actual_runtime_metadata';final_hits=[]
            return dict(path=str(path),kind=kind,development_hits=sorted(values&development),final_episode_hits=final_hits,
                        size=path.stat().st_size,extracted_seeds_sha256=hashlib.sha256(json.dumps(sorted(values)).encode()).hexdigest())
        except Exception as exc:return dict(path=str(path),error=str(exc))
    entries=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i,entry in enumerate(pool.map(inspect,files),1):
            entries.append(entry)
            if i%10000==0:print('Seed headers/metadata checked:',i,flush=True)
    errors.extend(x for x in entries if 'error' in x)
    collisions=[x for x in entries if x.get('development_hits')];final_hits=[x for x in entries if x.get('final_episode_hits')]
    passed=not errors and not collisions and not final_hits and not development&final
    report=dict(passed=passed,protocol_sha256=hashlib.sha256(SPEC.read_bytes()).hexdigest(),roots=[str(x) for x in roots],
        files_checked=len(entries),compressed_headers=sum(x.get('kind')=='compressed_header' for x in entries),
        actual_runtime_metadata=sum(x.get('kind')=='actual_runtime_metadata' for x in entries),development_collisions=collisions,
        final_episode_seed_hits=final_hits,errors=errors,development_final_sets_disjoint=not bool(development&final),entries=entries,
        limitation='Actual local metadata and first16384 decoded episode-header characters only. Excludes prospective declarations/source snapshots, tensor binaries, external/unlogged runs. No simulator or optimizer execution.',
        research_launched=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='entries'},indent=2))
    return 0 if passed else 1

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT)
    raise SystemExit(main(parser.parse_args().output))
