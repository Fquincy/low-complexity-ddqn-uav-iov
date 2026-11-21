"""Bounded public literature retrieval; no controller, simulator or training."""
import concurrent.futures
import hashlib
import html
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation/value_gradient_isolation_brief_2026-11-20_01'
PAPERS=[('double_dqn','1509.06461'),('dueling','1511.06581'),('branching','1711.08946'),
        ('dqfd','1704.03732'),('gradient_surgery','2001.06782'),('decoupled_representation','2009.08319')]
QUERIES=[('crossref_gradient','dueling deep Q learning stop gradient value advantage shared representation'),
         ('crossref_offloading','DQfD DECO demonstration computation offloading UAV vehicular edge computing'),
         ('crossref_detach','deep Q learning detached value head gradient shared encoder')]


def clean(s):return re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]+>',' ',s))).strip()


def fetch(item):
    name,url,kind=item
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'Research-literature-check/1.0'})
        with urllib.request.urlopen(request,timeout=30) as response:
            raw=response.read();final_url=response.url
        suffix='.json' if kind=='crossref' else '.html'
        with (OUT/(name+suffix)).open('xb') as f:f.write(raw)
        row=dict(name=name,url=url,final_url=final_url,status='retrieved',file=name+suffix,
            sha256=hashlib.sha256(raw).hexdigest(),kind=kind)
        if kind=='abstract':
            text=raw.decode('utf-8','replace')
            title=re.search(r'<title>(.*?)</title>',text,re.S)
            abstract=re.search(r'<blockquote[^>]*class="abstract[^\"]*"[^>]*>(.*?)</blockquote>',text,re.S)
            row.update(title=clean(title[1]) if title else None,abstract=clean(abstract[1]) if abstract else None)
        elif kind=='crossref':
            data=json.loads(raw)
            row['works']=[dict(title=v.get('title',[]),doi=v.get('DOI'),url=v.get('URL'),
                published=v.get('published'),abstract=clean(v.get('abstract',''))) for v in data['message']['items']]
        else:
            text=raw.decode('utf-8','replace')
            # Keep the full response; snippets are pointers, not full-text clearance.
            cleaned=clean(text)
            row['text_characters']=len(cleaned)
            row['snippets']={term:[cleaned[max(0,m.start()-220):m.end()+330] for m in list(re.finditer(re.escape(term),cleaned,re.I))[:6]]
                for term in ['gradient','detach','encoder','stop-gradient']}
        return row
    except Exception as error:
        return dict(name=name,url=url,status='retrieval_failed',kind=kind,error=str(error))


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    jobs=[(name,'https://arxiv.org/abs/'+identifier,'abstract') for name,identifier in PAPERS]
    jobs += [(name,'https://api.crossref.org/works?'+urllib.parse.urlencode(dict(query=query,rows=5)),'crossref') for name,query in QUERIES]
    jobs += [('decoupled_representation_fulltext','https://ar5iv.labs.arxiv.org/html/2009.08319','html_fulltext'),
             ('dueling_fulltext','https://ar5iv.labs.arxiv.org/html/1511.06581','html_fulltext')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(fetch,jobs))
    record=dict(scope='Targeted primary-paper and metadata overlap check; not exhaustive novelty clearance',
        results=results,collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_model_training=False,new_simulator_steps=0,new_teacher_calls=0)
    with (OUT/'literature_retrieval.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
