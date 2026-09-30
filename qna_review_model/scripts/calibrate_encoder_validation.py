"""Product-grouped out-of-fold threshold diagnostic on Validation, never Test.

This is development evidence: the checkpoint was itself selected on Validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def metrics(y, pred):
    tp = (y & pred).sum(axis=0); fp = (~y & pred).sum(axis=0); fn = (y & ~pred).sum(axis=0)
    den = 2*tp+fp+fn
    active = den > 0
    return {"microF1": float(2*tp.sum()/max(1,den.sum())),
            "macroF1Observed": float(np.divide(2*tp,den,out=np.zeros_like(den,dtype=float),where=active)[active].mean()) if active.any() else 0,
            "precision": float(tp.sum()/max(1,(tp+fp).sum())), "recall": float(tp.sum()/max(1,(tp+fn).sum()))}


def fit_thresholds(y, probabilities, minimum_positive=10):
    grid = np.arange(0.05,0.951,0.05)
    global_scores = [metrics(y, probabilities>=t)['microF1'] for t in grid]
    global_t = float(grid[max(range(len(grid)),key=lambda i:(global_scores[i],grid[i]))])
    thresholds = np.full(y.shape[1],global_t)
    for j in range(y.shape[1]):
        if y[:,j].sum() < minimum_positive or (~y[:,j]).sum() < minimum_positive:
            continue
        scores = [metrics(y[:,j:j+1],probabilities[:,j:j+1]>=t)['microF1'] for t in grid]
        thresholds[j] = grid[max(range(len(grid)),key=lambda i:(scores[i],grid[i]))]
    return global_t,thresholds


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    validation=[json.loads(x) for x in args.validation.open() if x.strip()]
    lookup={r['itemId']:r for r in validation}
    predictions=[json.loads(x) for x in (args.run/'best-validation-predictions.jsonl').open() if x.strip()]
    if len(lookup)!=len(validation) or len({r['itemId'] for r in predictions})!=len(predictions) or set(lookup)!={r['itemId'] for r in predictions}:
        raise ValueError('Validation ID mismatch')
    products=sorted({r['productGroupId'] for r in validation},key=lambda s:hashlib.sha256(('calibration-v1:'+s).encode()).hexdigest())
    product_fold={pid:i%5 for i,pid in enumerate(products)}
    report={'run':str(args.run),'folds':5,'products':len(products),'minimumPositivePerLabel':10,
            'warning':'Grouped out-of-fold Validation threshold diagnostic; checkpoint selection already used Validation. Not independent Test quality.',
            'validationSha256':hashlib.sha256(args.validation.read_bytes()).hexdigest(),'tasks':{}}
    for task,field,truth in [('review','reviewCandidates','reviewTruth'),('qnaSecondary','secondaryCandidates','secondaryTruth')]:
        rows=[r for r in predictions if field in r]
        labels=sorted({l for r in rows for l,_ in r[field]}|{l for r in rows for l in r[truth]})
        index={l:i for i,l in enumerate(labels)}
        y=np.zeros((len(rows),len(labels)),bool); prob=np.full(y.shape,-1.,dtype=float)
        folds=np.array([product_fold[lookup[r['itemId']]['productGroupId']] for r in rows])
        for i,r in enumerate(rows):
            for l in r[truth]:y[i,index[l]]=True
            for l,s in r[field]:prob[i,index[l]]=s
        global_pred=np.zeros_like(y);label_pred=np.zeros_like(y);fitted=[]
        for fold in range(5):
            train=folds!=fold;holdout=folds==fold
            if not train.any() or not holdout.any():raise ValueError('Empty task fold')
            g,thresholds=fit_thresholds(y[train],prob[train])
            global_pred[holdout]=prob[holdout]>=g
            label_pred[holdout]=prob[holdout]>=thresholds
            fitted.append({'fold':fold,'calibrationRecords':int(train.sum()),'heldoutRecords':int(holdout.sum()),
                           'global':g,'perLabel':dict(zip(labels,thresholds.tolist()))})
        report['tasks'][task]={'records':len(rows),'oofGlobal':metrics(y,global_pred),'oofPerLabel':metrics(y,label_pred),'fitted':fitted}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({t:{k:v for k,v in r.items() if k!='fitted'} for t,r in report['tasks'].items()},indent=2))


if __name__=='__main__':main()
