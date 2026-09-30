"""Apply exact, source-hashed judge proposals to an isolated Train copy only."""
import argparse
import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path


def build(data_dir, proposals_path, output_dir):
    if output_dir.exists():raise FileExistsError(output_dir)
    train_path=data_dir/'train.jsonl'
    proposals=json.loads(proposals_path.read_text())
    if proposals['sourceSha256']!=hashlib.sha256(train_path.read_bytes()).hexdigest():
        raise ValueError('Train source hash mismatch')
    if proposals.get('trainingPolicy')!='isolated_ablation_only':
        raise ValueError('Proposals not marked for isolated ablation')
    rows=[json.loads(s) for s in train_path.open() if s.strip()]
    index={r['itemId']:r for r in rows}
    if len(index)!=len(rows):raise ValueError('Duplicate Train IDs')
    changes=[];seen=set()
    for decision in proposals['decisions']:
        item=decision['itemId']
        if item in seen:raise ValueError('Duplicate proposal')
        seen.add(item)
        if item not in index:raise ValueError('Proposal is not an existing Train ID')
        row=index[item]
        if row['sourceType']!='review' or row['text']!=decision['text']:
            raise ValueError('Proposal text/task mismatch')
        if not decision.get('reason'):raise ValueError('Missing judge rationale')
        changes.append({'itemId':item,'beforeLabels':deepcopy(row['labels']),
                        'beforeAnnotation':deepcopy(row['annotation']),'judgeDecision':decision})
        row['labels']['reviewLabels']=[]
        row['annotation'].update(grade='silver',source='codex-abstention-proposal-experiment-v1',
                                 reviewerStatus='pending',validationStatus='needs_review')
        # Keep original sampleWeight: this isolates target corrections, not reweighting.
    output_dir.mkdir(parents=True)
    for name in ('validation.jsonl','label_map.json','theme_taxonomy.json'):
        shutil.copyfile(data_dir/name,output_dir/name)
    for name,values in (('train.jsonl',rows),('changes.jsonl',changes)):
        (output_dir/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))
    manifest={'parent':str(data_dir),'trainRows':len(rows),'changedRows':len(changes),
              'sourceSha256':proposals['sourceSha256'],'proposalsSha256':hashlib.sha256(proposals_path.read_bytes()).hexdigest(),
              'testPolicy':'Not read or copied; train with --skip-test',
              'warning':'Automated semantic proposals; not new Gold. Original annotations retained in changes.jsonl. Empty targets are experimental abstentions, not a schema-wide policy change.'}
    (output_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data-dir','proposals','output-dir'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.data_dir,a.proposals,a.output_dir),ensure_ascii=False,indent=2))
