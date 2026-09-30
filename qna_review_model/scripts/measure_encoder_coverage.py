"""Measure Q&A primary coverage/error at predeclared confidence cutoffs."""
import argparse
import json
import math
from pathlib import Path


def wilson(successes, n):
    if not n:return None
    z=1.959963984540054;rate=successes/n;den=1+z*z/n
    center=(rate+z*z/(2*n))/den
    radius=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
    return [center-radius,center+radius]


def measure(rows, agreement_rows=None):
    qna=[r for r in rows if r['sourceType']=='qna']
    if not qna or any('primaryProbability' not in r for r in qna):
        raise ValueError('Require Q&A primary probabilities')
    if agreement_rows is not None and len({r['itemId'] for r in qna}) != len(qna):
        raise ValueError('Duplicate Q&A IDs')
    agreement = None
    if agreement_rows is not None:
        others = [r for r in agreement_rows if r['sourceType']=='qna']
        agreement = {r['itemId']:r for r in others}
        if len(agreement)!=len(others) or set(agreement)!={r['itemId'] for r in qna}:
            raise ValueError('Agreement model Q&A IDs mismatch')
        if any(agreement[r['itemId']]['primaryTruth']!=r['primaryTruth'] for r in qna):
            raise ValueError('Agreement model Q&A targets mismatch')
    scores=[]
    for cutoff in (0.0,0.5,0.7,0.8,0.9,0.95):
        accepted=[r for r in qna if r['primaryProbability']>=cutoff and
                  (agreement is None or agreement[r['itemId']]['primaryPrediction']==r['primaryPrediction'])]
        n=len(accepted);correct=sum(r['primaryPrediction']==r['primaryTruth'] for r in accepted)
        scores.append({'cutoff':cutoff,'accepted':n,'total':len(qna),'coverage':n/len(qna),
                       'accuracy':correct/n if n else None,'errorRate':1-correct/n if n else None,
                       'accuracyWilson95':wilson(correct,n)})
    return {'scope':'Q&A primary theme only, not whole report quality',
            'agreementRequired':agreement is not None,
            'warning':'Development Validation. Model softmax is not calibrated certainty; intervals assume independent records although products cluster.',
            'cutoffs':scores}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--agreement-predictions',type=Path,help='Optional second model; thresholds stay fixed')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    agreement=[json.loads(x) for x in a.agreement_predictions.open() if x.strip()] if a.agreement_predictions else None
    report=measure([json.loads(x) for x in a.predictions.open() if x.strip()],agreement)
    report['predictions']=str(a.predictions)
    report['agreementPredictions']=str(a.agreement_predictions) if a.agreement_predictions else None
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
