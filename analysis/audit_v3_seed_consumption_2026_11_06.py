"""Bounded saved-run seed check. Reads metadata and compressed episode headers."""
import gzip
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'LOW_COMPLEXITY_V3_RESEARCH_PROTOCOL_2026-11-05.json'

def main():
    spec=json.loads(PROTOCOL.read_text())
    reserved={}
    for phase in ['development','final']:
        p=spec[phase];seeds=set(p['model_seeds'])
        for b in p['train_blocks']:seeds.update(range(b['start'],b['start']+b['count']))
        e=p['evaluation_seeds'];seeds.update(range(e['start'],e['start']+e['count']))
        seeds.update(p.get('diagnostic_validation_seeds',[]));reserved[phase]=seeds
    roots=[ROOT/'validation',ROOT/'development_runs',ROOT.parent/'October 31/Working/Project/validation',
           ROOT.parent/'October 31/Working/Project/development_runs']
    collisions=[];errors=[];counts={'compressed_headers':0,'metadata_files':0};excluded_large=[];checked=[]
    pattern=re.compile(r'"(?:seed|model_seed|training_seed|train_seed|evaluation_seed|task_seed)"\s*:\s*(\d+)\b')
    def inspect(value,location):
        if isinstance(value,dict):
            for key,item in value.items():
                if 'seed' in key.lower():
                    if isinstance(item,int):yield item
                    elif isinstance(item,list):
                        yield from (x for x in item if isinstance(x,int))
                    elif isinstance(item,dict) and 'start' in item and 'count' in item:
                        yield from range(item['start'],item['start']+item['count'])
                if key=='train_blocks' and isinstance(item,list):
                    for b in item:
                        if isinstance(b,dict) and 'start' in b and 'count' in b:yield from range(b['start'],b['start']+b['count'])
                yield from inspect(item,location)
        elif isinstance(value,list):
            for item in value:yield from inspect(item,location)
    for root in roots:
        if not root.exists():continue
        for path in root.rglob('*'):
            if not path.is_file() or any(x in path.parts for x in ['source_snapshot','exports']):continue
            values=set()
            try:
                if path.name.endswith('.json.gz'):
                    with gzip.open(path,'rt',encoding='utf-8') as stream:prefix=stream.read(16384)
                    values={int(x) for x in pattern.findall(prefix)}
                    counts['compressed_headers']+=1
                elif path.suffix=='.json' and any(token in path.name.lower() for token in ['runtime','counters','training_summary','config','run_metadata']):
                    if path.stat().st_size>16*1024*1024:
                        excluded_large.append(str(path));continue
                    values=set(inspect(json.loads(path.read_text(encoding='utf-8-sig')),path))
                    counts['metadata_files']+=1
                else:continue
                checked.append(str(path))
                for phase,seeds in reserved.items():
                    hit=sorted(values & seeds)
                    if hit:collisions.append(dict(path=str(path),phase=phase,seeds=hit))
            except Exception as exc:errors.append(dict(path=str(path),error=str(exc)))
    report=dict(state='no_reserved_seed_hits_in_checked_metadata' if not collisions and not errors else 'requires_review',
        roots=[str(x) for x in roots if x.exists()],counts=counts,collisions=collisions,errors=errors,
        excluded_large_metadata=excluded_large,
        files_checked=len(checked),checked_path_list_sha256=hashlib.sha256('\n'.join(sorted(checked)).encode()).hexdigest(),
        protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        limitations=['Checks selected actual runtime metadata, including schedules, and first 16384 decoded characters of compressed episode files.',
            'Does not inspect tensor checkpoints, binary or compressed files other than JSON episode headers, unlogged execution, or external folders.',
            'A scheduled seed is conservatively counted as a hit even if its run failed before consuming it.',
            'This is a bounded local check and cannot prove absence of all historical seed use.'],
        research_training_launched=False,new_simulator_slots=0,optimizer_updates=0)
    output=ROOT/'validation/v3_seed_consumption_review_2026-11-06.json'
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
