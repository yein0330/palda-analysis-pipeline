"""Frozen synthetic probes for future intention versus experienced usability.

These assistant-authored probes are diagnostics, not natural-distribution accuracy,
not Gold and never training examples. No Validation or Test inputs are read.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from scripts.train_encoder_transformer import (
    PaldaCollator, PaldaDataset, PaldaEncoder, collect_predictions, read_jsonl, runtime_label_map,
)


PROBES = (
    ('unused', '아직 개봉하지 않았어요. 사용해 본 뒤 후기를 남길게요.', []),
    ('future', '다음 주부터 사용할 예정입니다. 아직 써보지는 않았어요.', []),
    ('no_basis', '아직 사용 전이라 품질과 사용 편의는 평가할 수 없어요.', []),
    ('experience', '사용 방법이 간단해서 혼자서도 쉽게 쓰고 있어요.', ['USABILITY::positive']),
)


def build_probes(train, mapping):
    categories = {r['category']['level2Id']:r['category'] for r in train if r['sourceType']=='review'}
    existing = {''.join(r['text'].split()) for r in train}
    result = []
    for cat1 in ('디지털/가전','가구/인테리어','생활/건강'):
        choices = sorted((c for c in categories.values() if c['level1Name']==cat1),key=lambda c:c['level2Id'])
        if not choices:
            raise ValueError('Probe category unavailable: '+cat1)
        category = choices[0]
        themes = {mapping['indexToCanonicalTheme'][i] for i in mapping['categoryMasks'][category['level2Id']]}
        if 'USABILITY' not in themes:
            raise ValueError('Probe category lacks USABILITY')
        for kind,text,expected in PROBES:
            if ''.join(text.split()) in existing:
                raise ValueError('Synthetic probe text already occurs in Train')
            result.append({'itemId':'probe-'+category['level2Id']+'-'+kind,'sourceType':'review','text':text,
                'category':category,'usable':True,'annotation':{'sampleWeight':0.0,'grade':'automated_judge_proposal'},
                'labels':{'reviewLabels':[],'qnaPrimaryTheme':None,'qnaSecondaryThemes':[]},
                'probeKind':kind,'expectedLabels':expected,'trainingEligible':False})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    mapping=runtime_label_map(json.loads((a.data_dir/'label_map.json').read_text()),'canonical')
    probes=build_probes(read_jsonl(a.data_dir/'train.jsonl'),mapping)
    results=[];device=torch.device(a.device)
    for checkpoint in a.checkpoints:
        config=json.loads((checkpoint/'training_config.json').read_text())
        if config['indexToCanonicalTheme']!=mapping['indexToCanonicalTheme']:raise ValueError('Head mismatch')
        tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
        loader=DataLoader(PaldaDataset(probes),batch_size=12,collate_fn=PaldaCollator(tokenizer,mapping,config['args']['max_length']))
        model=PaldaEncoder(config['model'],config['themeCount'],config['args'].get('pooling','cls'),
                           config['args'].get('review_pooling','shared'),config['args'].get('qna_pooling','shared')).to(device)
        model.load_state_dict(torch.load(checkpoint/'model_state.pt',map_location=device,weights_only=True))
        outcomes=[]
        for probe,pred in zip(probes,collect_predictions(model,loader,device,mapping),strict=True):
            labels=sorted(l for l,s in pred['reviewCandidates'] if s>=config['reviewThreshold'])
            outcomes.append({'itemId':probe['itemId'],'kind':probe['probeKind'],'expected':probe['expectedLabels'],
                             'predicted':labels,'passed':labels==probe['expectedLabels']})
        results.append({'checkpoint':str(checkpoint),'passed':sum(x['passed'] for x in outcomes),
                        'probes':len(outcomes),'outcomes':outcomes})
        del model
    a.output_dir.mkdir(parents=True)
    (a.output_dir/'probes.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in probes))
    report={'warning':__doc__,'judge':'Codex assistant; probe expectations fixed before inference',
            'scriptSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':results}
    (a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='outcomes'} for r in results],indent=2))


if __name__=='__main__':main()
