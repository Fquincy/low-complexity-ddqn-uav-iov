"""Retrieve public papers for read-only full-method review; no research execution."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import fitz

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation/rq24_engineering_2026-11-19_01/literature'
PAPERS=[('dcg','https://arxiv.org/pdf/1910.00091'),
        ('branching','https://arxiv.org/pdf/1711.08946'),
        ('noma_vehicular','https://arxiv.org/pdf/2209.12749'),
        ('graph_ddqn_journal','https://link.springer.com/article/10.1007/s11227-024-06599-4'),
        ('graph_ddqn_preprint','https://www.researchsquare.com/article/rs-4917636/v1')]


def fetch(item):
    name,url=item
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 ThesisMethodReview'}),timeout=25) as r:
            body=r.read()
        is_pdf=body.startswith(b'%PDF')
        path=OUT/(name+('.pdf' if is_pdf else '.html'))
        path.write_bytes(body)
        result=dict(name=name,url=url,status='retrieved',path=str(path),sha256=hashlib.sha256(body).hexdigest())
        if is_pdf:
            doc=fitz.open(path)
            texts=[p.get_text(sort=True) for p in doc]
            (OUT/(name+'.txt')).write_text('\n'.join(f'PAGE {i+1}\n{text}' for i,text in enumerate(texts)),encoding='utf-8')
            result['pages']=len(doc)
            result['method_pages']=[i+1 for i,t in enumerate(texts) if any(s in t.lower() for s in ('message passing','coordination graph','interference','algorithm 1'))]
            if name=='dcg':
                for i in (2,3):
                    doc[i].get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(OUT/f'dcg_page_{i+1}.png')
        return result
    except Exception as e:
        return dict(name=name,url=url,status='unavailable',error=str(e))


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: results=list(pool.map(fetch,PAPERS))
    (OUT/'retrieval.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    for r in results:print(json.dumps(r),flush=True)
