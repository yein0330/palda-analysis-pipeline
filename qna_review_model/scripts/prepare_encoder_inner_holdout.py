"""Create a product-held-out development diagnostic using original Train ONLY.

Outer Validation/Test are never opened. This is not a replacement evaluation set.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import shutil


def build(data_dir, output_dir, fraction=.1):
    if output_dir.exists():raise FileExistsError(output_dir)
    if not 0<fraction<1:raise ValueError('Fraction must be in (0,1)')
    source=data_dir/'train.jsonl'
    rows=[json.loads(s) for s in source.open() if s.strip()]
    products=defaultdict(list)
    for row in rows:products[row['productGroupId']].append(row)
    strata=defaultdict(list)
    for pid,group in products.items():
        cat1={r['category']['level1Id'] for r in group}
        if len(cat1)!=1:raise ValueError('Product group has inconsistent Cat1')
        if sum(r['annotation']['grade']=='gold' for r in group)>=3:
            strata[next(iter(cat1))].append(pid)
    heldout=set();selection={}
    for cat1,pids in sorted(strata.items()):
        ordered=sorted(pids,key=lambda pid:hashlib.sha256(('inner-gold-holdout-v1:'+pid).encode()).hexdigest())
        chosen=ordered[:max(1,round(len(ordered)*fraction))]
        heldout.update(chosen);selection[cat1]=chosen
    validation=[r for r in rows if r['productGroupId'] in heldout and r['annotation']['grade']=='gold']
    validation_texts={(r['sourceType'],''.join(r['text'].split())) for r in validation}
    train=[r for r in rows if r['productGroupId'] not in heldout
           and (r['sourceType'],''.join(r['text'].split())) not in validation_texts]
    if {r['sourceType'] for r in train}!={'review','qna'} or {r['sourceType'] for r in validation}!={'review','qna'}:
        raise ValueError('Both tasks must occur in inner Train and Validation')
    train_ids={r['itemId'] for r in train};val_ids={r['itemId'] for r in validation}
    if train_ids & val_ids:raise ValueError('Duplicate IDs across inner splits')
    excluded=[r['itemId'] for r in rows if r['itemId'] not in train_ids|val_ids]
    output_dir.mkdir(parents=True)
    for name in ('label_map.json','theme_taxonomy.json'):shutil.copyfile(data_dir/name,output_dir/name)
    for name,values in (('train.jsonl',train),('validation.jsonl',validation)):
        (output_dir/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))
    report={'schemaVersion':'palda-inner-train-holdout-v1','parentTrain':str(source),
            'parentTrainSha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'selectionByCat1':selection,'heldoutProducts':len(heldout),'excludedTrainIds':excluded,
            'records':{'train':len(train),'validation':len(validation)},
            'tasks':{'train':dict(Counter(r['sourceType'] for r in train)),
                     'validation':dict(Counter(r['sourceType'] for r in validation))},
            'warning':'Original Train only; eligible groups have >=3 Gold rows. Whole selected products excluded from fitting, including Silver. Cross-split exact text duplicates purged. Original annotation provenance retained; not independent Test or representative of all products.'}
    (output_dir/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return {k:v for k,v in report.items() if k not in ('selectionByCat1','excludedTrainIds')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.data_dir,a.output_dir),ensure_ascii=False,indent=2))
