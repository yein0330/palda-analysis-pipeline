"""Prepare new text-only Validation diagnostics, excluding prior visible judge queues.

No model outputs/reference labels are exported. This is not an independent Test.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir','prior-audit','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    seen=set()
    for path in a.prior_audit.glob('**/judge_queue.jsonl'):
        seen.update(json.loads(s)['itemId'] for s in path.open() if s.strip())
    validation=a.data_dir/'validation.jsonl'
    rows=[json.loads(s) for s in validation.open() if s.strip()]
    taxonomy={c['level2Id']:c for c in json.loads((a.data_dir/'theme_taxonomy.json').read_text())['categories']}
    chosen=[];products=set()
    for task in ('review','qna'):
        candidates=sorted((r for r in rows if r['sourceType']==task and r['itemId'] not in seen),
                          key=lambda r:hashlib.sha256(('blind-laaj-v2:'+r['itemId']).encode()).hexdigest())
        count=0
        for r in candidates:
            if r['productGroupId'] in products:continue
            if re.search(r'\d{2,3}[- ]\d{3,4}[- ]\d{4}|\d+동\s*\d+호|[로길]\s*\d',r['text']):continue
            chosen.append({k:r[k] for k in ('itemId','sourceType','productGroupId','category','text')} | {
                'allowedThemes':[{k:t[k] for k in ('canonicalThemeId','reviewLabel','qnaLabel','annotationReady')}
                                 for t in taxonomy[r['category']['level2Id']]['allowedThemes']],
                'trainingEligible':False})
            products.add(r['productGroupId']);count+=1
            if count==12:break
        if count!=12:raise ValueError('Not enough unexposed product groups')
    a.output_dir.mkdir(parents=True)
    (a.output_dir/'judge_queue.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen))
    (a.output_dir/'manifest.json').write_text(json.dumps({'validationSha256':hashlib.sha256(validation.read_bytes()).hexdigest(),
        'records':len(chosen),'products':len(products),'priorQueueIdsExcluded':len(seen),
        'selection':'12 per task, one per product, deterministic hash order; simple contact/address pattern exclusion',
        'warning':'Assistant has seen prior diagnostics; this queue hides row-specific labels/predictions until judgments are frozen. Not a blind independent model comparison or Test.'},indent=2)+'\n')
    print(json.dumps({'output':str(a.output_dir),'records':len(chosen),'products':len(products)}))


if __name__=='__main__':main()
