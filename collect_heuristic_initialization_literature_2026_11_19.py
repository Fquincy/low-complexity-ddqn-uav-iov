"""Bounded literature retrieval only; no simulator, torch or model execution."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'validation/heuristic_initialization_feasibility_2026-11-19_01'
QUERIES = [
    ('dqfd', 'https://arxiv.org/html/1704.03732'),
    ('policy_distillation', 'https://arxiv.org/html/1511.06295'),
    ('dqfd_abstract', 'https://arxiv.org/abs/1704.03732'),
    ('distillation_abstract', 'https://arxiv.org/abs/1511.06295'),
    ('offloading_demonstrations', 'https://api.crossref.org/works?' + urllib.parse.urlencode(
        {'query': 'vehicular UAV edge offloading deep Q learning demonstrations imitation heuristic', 'rows': 8})),
    ('offloading_distillation', 'https://api.crossref.org/works?' + urllib.parse.urlencode(
        {'query': 'vehicular UAV edge offloading policy distillation imitation learning', 'rows': 8})),
]


def fetch(item):
    name, url = item
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'ThesisLiteratureReview/1.0'})
        with urllib.request.urlopen(request, timeout=25) as response:
            body = response.read()
            content_type = response.headers.get('Content-Type', '')
            final_url = response.geturl()
        suffix = '.json' if name.startswith('offloading_') else '.html'
        path = OUT / (name + suffix)
        path.write_bytes(body)
        result = dict(name=name, url=url, final_url=final_url, status='retrieved',
                      path=path.relative_to(ROOT).as_posix(), content_type=content_type,
                      sha256=hashlib.sha256(body).hexdigest(), bytes=len(body))
        if suffix == '.json':
            result['records'] = [dict(title=r.get('title'), doi=r.get('DOI'),
                                      type=r.get('type'), abstract=r.get('abstract'))
                                 for r in json.loads(body)['message']['items']]
        return result
    except Exception as exc:
        return dict(name=name, url=url, status='unavailable', error=str(exc))


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(fetch, QUERIES))
    evidence = dict(date='2026-11-19', scope='Bounded targeted retrieval; metadata is not method verification',
                    queries=results, new_simulator_steps=0, optimizer_updates=0)
    (OUT / 'retrieval.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    for result in results:
        print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
