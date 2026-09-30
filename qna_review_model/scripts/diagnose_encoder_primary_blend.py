"""Fixed equal-probability Q&A primary blend, using frozen Train-fitted artifacts.

Validation diagnostic only. Review/secondary tasks are not modified or retuned.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score

from scripts.train_encoder_cpu_baseline import model_text
from scripts.train_encoder_transformer import read_jsonl


def run(data_dir, predictions_path, artifact_path, output):
    if output.exists():raise FileExistsError(output)
    artifact=joblib.load(artifact_path)
    if artifact.get('schemaVersion')!='palda-lexical-primary-v1':raise ValueError('Wrong primary artifact')
    for name,digest in artifact['sourceHashes'].items():
        if hashlib.sha256((data_dir/name).read_bytes()).hexdigest()!=digest:raise ValueError('Source hash mismatch')
    rows=[r for r in read_jsonl(data_dir/'validation.jsonl') if r['sourceType']=='qna']
    preds={r['itemId']:r for r in read_jsonl(predictions_path) if r['sourceType']=='qna'}
    if len(preds)!=len(rows) or set(preds)!={r['itemId'] for r in rows}:raise ValueError('Prediction ID mismatch')
    matrix=artifact['vectorizer'].transform([model_text(r) for r in rows])
    probs=artifact['primaryModel'].predict_proba(matrix)
    classes=list(artifact['primaryModel'].classes_)
    outcomes=[]
    for i,row in enumerate(rows):
        pred=preds[row['itemId']]
        truth=row['labels']['qnaPrimaryTheme']['themeId']
        if truth!=pred['primaryTruth']:raise ValueError('Prediction truth mismatch')
        neural=dict(pred['primaryCandidates']);labels=sorted(neural)
        lexical={label:float(probs[i,classes.index(label)]) if label in classes else 0 for label in labels}
        total=sum(lexical.values())
        if total<=0:raise ValueError('No lexical mass on allowed labels')
        lexical={l:v/total for l,v in lexical.items()}
        mixed={l:.5*neural[l]+.5*lexical[l] for l in labels}
        outcomes.append({'itemId':row['itemId'],'truth':truth,'neural':pred['primaryPrediction'],
                         'lexical':max(labels,key=lexical.get),'blend':max(labels,key=mixed.get),
                         'blendScore':max(mixed.values())})
    truth=[r['truth'] for r in outcomes]
    metrics={key:{'accuracy':sum(r[key]==r['truth'] for r in outcomes)/len(outcomes),
                   'macroF1Observed':float(f1_score(truth,[r[key] for r in outcomes],average='macro',zero_division=0))}
             for key in ('neural','lexical','blend')}
    result={'metrics':metrics,'records':len(rows),
        'neuralErrorsCorrected':sum(r['neural']!=r['truth'] and r['blend']==r['truth'] for r in outcomes),
        'neuralCorrectBroken':sum(r['neural']==r['truth'] and r['blend']!=r['truth'] for r in outcomes),
        'outcomes':outcomes,'warning':__doc__+' Scores are uncalibrated; no deployment guarantee.',
        'validationSha256':hashlib.sha256((data_dir/'validation.jsonl').read_bytes()).hexdigest()}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return {k:v for k,v in result.items() if k!='outcomes'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir','predictions','artifact','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.data_dir,a.predictions,a.artifact,a.output),ensure_ascii=False,indent=2))
