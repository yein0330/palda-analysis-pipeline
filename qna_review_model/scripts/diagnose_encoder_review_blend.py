"""Fixed 50:50 neural/lexical Review blend; Q&A predictions stay neural.

Development-only complementarity diagnostic, not automatic deployment. Mixing is
predeclared, not fit to Validation; threshold is grid-selected unless explicitly frozen.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from scripts.compare_encoder_validation import compare, load_run
from scripts.train_encoder_transformer import choose_threshold, evaluate_predictions


def blend(neural, lexical, output, review_threshold=None):
    if output.exists():raise FileExistsError(output)
    check=compare([neural,lexical])
    config,nrows=load_run(neural);_,lrows=load_run(lexical)
    predictions=deepcopy(list(nrows.values()))
    for row in predictions:
        if row['sourceType']!='review':continue
        source=dict(lrows[row['itemId']]['reviewCandidates'])
        if set(source)!={label for label,_ in row['reviewCandidates']}:
            raise ValueError('Different allowed Review candidate labels')
        row['reviewCandidates']=[[label,.5*score+.5*source[label]] for label,score in row['reviewCandidates']]
    if review_threshold is None:
        threshold,_=choose_threshold(predictions,'reviewCandidates','reviewTruth')
    else:
        if not 0<=review_threshold<=1:raise ValueError('Invalid frozen threshold')
        threshold=review_threshold
    config=deepcopy(config)
    config.update(model='fixed-half-neural-half-lexical-review-diagnostic',reviewThreshold=threshold,
                  reviewThresholdPolicy='validation_grid' if review_threshold is None else 'predeclared_fixed',
                  parents={'neural':str(neural),'lexical':str(lexical)},weightsSaved=False)
    metrics=evaluate_predictions(predictions,threshold,config['secondaryThreshold'])
    (output/'best').mkdir(parents=True)
    (output/'best/training_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    (output/'best-validation-predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions))
    report={'bestValidation':metrics,'bestEpoch':config['epoch'],'testSkipped':True,'weightsSaved':False,
            'reviewThresholdPolicy':config['reviewThresholdPolicy'],
            'validationLabelFingerprint':check['validationLabelFingerprint'],
            'warning':__doc__+' Mixed probabilities are not calibrated confidence.'}
    (output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('neural','lexical','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--review-threshold',type=float,help='Optional predeclared threshold; do not tune on this split')
    a=p.parse_args();print(json.dumps(blend(a.neural,a.lexical,a.output,a.review_threshold),ensure_ascii=False,indent=2))
