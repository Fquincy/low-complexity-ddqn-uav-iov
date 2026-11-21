"""Bounded public metadata search; records abstracts, not full-text clearance."""
import concurrent.futures
import json
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'validation/radio_queue_design_literature_2026-11-19_01'
QUERIES=[
 ('arxiv-foundations','https://export.arxiv.org/api/query?id_list=1509.06461,1711.08946,1910.00091'),
 ('arxiv-uav','https://export.arxiv.org/api/query?'+urllib.parse.urlencode(dict(search_query='all:UAV AND all:offloading AND all:reinforcement',max_results=6,sortBy='relevance'))),
 ('arxiv-vehicular','https://export.arxiv.org/api/query?'+urllib.parse.urlencode(dict(search_query='all:vehicular AND all:offloading AND all:interference',max_results=6,sortBy='relevance'))),
 ('crossref-radio','https://api.crossref.org/works?'+urllib.parse.urlencode(dict(query='UAV vehicular edge offloading deadline interference double DQN',rows=6))),
 ('crossref-graph','https://api.crossref.org/works?'+urllib.parse.urlencode(dict(query='vehicular offloading graph double DQN coordination',rows=5))),
]


def fetch(item):
    name,url=item
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'ThesisLiteratureReview/1.0'})
        with urllib.request.urlopen(request,timeout=25) as response:
            body=response.read().decode('utf-8')
        records=[]
        if name.startswith('arxiv'):
            root=ET.fromstring(body);ns={'a':'http://www.w3.org/2005/Atom'}
            for entry in root.findall('a:entry',ns):
                records.append(dict(title=' '.join(entry.findtext('a:title','',ns).split()),
                    url=entry.findtext('a:id','',ns),abstract=' '.join(entry.findtext('a:summary','',ns).split()),
                    published=entry.findtext('a:published','',ns),authors=[a.findtext('a:name','',ns) for a in entry.findall('a:author',ns)]))
        else:
            data=json.loads(body)
            for r in data['message']['items']:
                records.append(dict(title=r.get('title'),doi=r.get('DOI'),url=r.get('URL'),abstract=r.get('abstract'),type=r.get('type')))
        return dict(name=name,url=url,status='retrieved',records=records,raw_response=body)
    except Exception as e:
        return dict(name=name,url=url,status='unavailable',error=str(e))


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(fetch,QUERIES))
    evidence=dict(date='2026-11-19',scope='abstract/metadata reconnaissance; full-text overlap unresolved',queries=results)
    (OUT/'search_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    for r in results:
        print(json.dumps({k:v for k,v in r.items() if k!='raw_response'}),flush=True)


if __name__=='__main__':main()
