"""Reproduce a fixed Review blend from local weights, without cached predictions or Test.

Load only trusted, locally generated joblib/torch artifacts. Does not change current.
The threshold must be supplied from the frozen diagnostic, never retuned here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from scripts.benchmark_encoder_lexical import predict_review_artifact
from scripts.train_encoder_transformer import (
    PaldaCollator,PaldaDataset,PaldaEncoder,collect_predictions,evaluate_predictions,read_jsonl,runtime_label_map,
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir','checkpoint','lexical-artifact','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--review-threshold',type=float,required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    if not 0<=a.review_threshold<=1:raise ValueError('Invalid threshold')
    config=json.loads((a.checkpoint/'training_config.json').read_text())
    mapping=runtime_label_map(json.loads((a.data_dir/'label_map.json').read_text()),'canonical')
    lexical=joblib.load(a.lexical_artifact)
    if lexical.get('schemaVersion')!='palda-lexical-review-v1':raise ValueError('Wrong artifact schema')
    artifact_mapping=lexical['mapping'] | {'specialIndices':sorted(lexical['mapping']['specialIndices'])}
    if mapping!=artifact_mapping or mapping['indexToCanonicalTheme']!=config['indexToCanonicalTheme']:
        raise ValueError('Artifact taxonomy mismatch')
    for name in ('train.jsonl','validation.jsonl','label_map.json'):
        if hashlib.sha256((a.data_dir/name).read_bytes()).hexdigest()!=lexical['sourceHashes'][name]:
            raise ValueError('Source hash mismatch: '+name)
    rows=read_jsonl(a.data_dir/'validation.jsonl')
    review=[r for r in rows if r['sourceType']=='review']
    lexical_predictions={r['itemId']:dict(r['reviewCandidates']) for r in predict_review_artifact(lexical,review)}
    tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    loader=DataLoader(PaldaDataset(rows),batch_size=32,collate_fn=PaldaCollator(tokenizer,mapping,config['args']['max_length']))
    device=torch.device(a.device)
    model=PaldaEncoder(config['model'],config['themeCount'],config['args'].get('pooling','cls'),
                       config['args'].get('review_pooling','shared'),config['args'].get('qna_pooling','shared')).to(device)
    model.load_state_dict(torch.load(a.checkpoint/'model_state.pt',map_location=device,weights_only=True))
    predictions=collect_predictions(model,loader,device,mapping)
    for row in predictions:
        if row['sourceType']!='review':continue
        scores=lexical_predictions[row['itemId']]
        if set(scores)!={label for label,_ in row['reviewCandidates']}:raise ValueError('Candidate mismatch')
        row['reviewCandidates']=[[label,.5*score+.5*scores[label]] for label,score in row['reviewCandidates']]
    report={'metrics':evaluate_predictions(predictions,a.review_threshold,config['secondaryThreshold']),
            'reviewThreshold':a.review_threshold,'secondaryThreshold':config['secondaryThreshold'],
            'checkpoint':str(a.checkpoint),'lexicalArtifact':str(a.lexical_artifact),
            'lexicalArtifactSha256':hashlib.sha256(a.lexical_artifact.read_bytes()).hexdigest(),
            'testSkipped':True,'weightsSaved':False,'warning':'Frozen-threshold Validation reproduction only; no current model replacement.'}
    a.output_dir.mkdir(parents=True)
    (a.output_dir/'validation-predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions))
    (a.output_dir/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
