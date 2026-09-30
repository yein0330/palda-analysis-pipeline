"""Paired product-cluster bootstrap of Validation deltas; development uncertainty only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def run_stats(run, rows_by_id, products):
    config=json.loads((run/'best/training_config.json').read_text())
    rows=[json.loads(x) for x in (run/'best-validation-predictions.jsonl').open() if x.strip()]
    if len(rows)!=len(rows_by_id) or {r['itemId'] for r in rows}!=set(rows_by_id):raise ValueError('ID mismatch')
    counts=np.zeros((len(products),8),dtype=float)
    seen=[]
    for r in rows:
        idx=products[rows_by_id[r['itemId']]['productGroupId']]
        if r['sourceType']=='review':
            truth=set(r['reviewTruth']);pred={l for l,s in r['reviewCandidates'] if s>=config['reviewThreshold']}
            counts[idx,:3]+=[len(truth&pred),len(pred-truth),len(truth-pred)]
            seen.append((r['itemId'],sorted(truth)))
        else:
            truth=set(r['secondaryTruth']);pred={l for l,s in r['secondaryCandidates'] if s>=config['secondaryThreshold']}
            counts[idx,3:6]+=[len(truth&pred),len(pred-truth),len(truth-pred)]
            counts[idx,6:]+=[r['primaryTruth']==r['primaryPrediction'],1]
            seen.append((r['itemId'],r['primaryTruth'],sorted(truth)))
    return counts,sorted(seen)


def scores(counts):
    return np.stack([2*counts[...,0]/np.maximum(1,2*counts[...,0]+counts[...,1]+counts[...,2]),
                     counts[...,6]/np.maximum(1,counts[...,7]),
                     2*counts[...,3]/np.maximum(1,2*counts[...,3]+counts[...,4]+counts[...,5])],axis=-1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('baseline','candidate','validation','output'):p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    data=[json.loads(x) for x in a.validation.open() if x.strip()];lookup={r['itemId']:r for r in data}
    products={pid:i for i,pid in enumerate(sorted({r['productGroupId'] for r in data}))}
    base,bt=run_stats(a.baseline,lookup,products);candidate,ct=run_stats(a.candidate,lookup,products)
    if bt!=ct:raise ValueError('Targets changed across runs')
    draws=np.random.default_rng(20260929).integers(0,len(products),size=(2000,len(products)))
    delta=scores(candidate[draws].sum(axis=1))-scores(base[draws].sum(axis=1))
    point=scores(candidate.sum(axis=0))-scores(base.sum(axis=0))
    result={'baseline':str(a.baseline),'candidate':str(a.candidate),'products':len(products),'replicates':2000,
            'validationSha256':hashlib.sha256(a.validation.read_bytes()).hexdigest(),
            'warning':'Paired product bootstrap on development Validation only. Does not capture training-seed uncertainty or repeated model-selection bias.',
            'delta':{name:{'point':float(point[i]),'percentile95':[float(x) for x in np.quantile(delta[:,i],[.025,.975])]}
                     for i,name in enumerate(('reviewMicroF1','qnaPrimaryAccuracy','qnaSecondaryMicroF1'))}}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
