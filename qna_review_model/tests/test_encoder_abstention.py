import hashlib
import json

import pytest

from scripts.build_encoder_abstention_ablation import build


def test_abstention_overlay_is_isolated_and_provenance_is_retained(tmp_path):
    data=tmp_path/'input';data.mkdir()
    original={'itemId':'t1','sourceType':'review','text':'아직 사용 전입니다',
              'annotation':{'grade':'gold','reviewerStatus':'approved','sampleWeight':1},
              'labels':{'reviewLabels':[{'themeId':'QUALITY','sentiment':'positive'}]}}
    (data/'train.jsonl').write_text(json.dumps(original)+'\n')
    for name in ('validation.jsonl','label_map.json','theme_taxonomy.json'):(data/name).write_text('{}\n')
    proposals={'sourceSha256':hashlib.sha256((data/'train.jsonl').read_bytes()).hexdigest(),
        'trainingPolicy':'isolated_ablation_only','decisions':[{'itemId':'t1','text':original['text'],'reason':'미사용 진술'}]}
    p=tmp_path/'proposals.json';p.write_text(json.dumps(proposals))
    out=tmp_path/'out';build(data,p,out)
    row=json.loads((out/'train.jsonl').read_text())
    assert row['labels']['reviewLabels']==[]
    assert row['annotation']['grade']=='silver'
    assert row['annotation']['reviewerStatus']=='pending'
    assert row['annotation']['sampleWeight']==1
    assert json.loads((data/'train.jsonl').read_text())==original
    assert json.loads((out/'changes.jsonl').read_text())['beforeAnnotation']==original['annotation']
    assert (out/'validation.jsonl').read_bytes()==(data/'validation.jsonl').read_bytes()
    assert not (out/'test.jsonl').exists()
    proposals['decisions'][0]['itemId']='not-in-train';p.write_text(json.dumps(proposals))
    with pytest.raises(ValueError,match='existing Train ID'):build(data,p,tmp_path/'bad-id')
    proposals['sourceSha256']='stale';p.write_text(json.dumps(proposals))
    with pytest.raises(ValueError,match='hash mismatch'):build(data,p,tmp_path/'stale')
