import json

from scripts.prepare_encoder_inner_holdout import build


def test_inner_holdout_needs_only_train_and_keeps_products_disjoint(tmp_path):
    data=tmp_path/'input';data.mkdir();rows=[]
    for i in range(10):
        for j in range(4):
            rows.append({'itemId':f'{i}-{j}','productGroupId':str(i),'sourceType':'qna' if j==0 else 'review',
                         'text':f'unique {i} {j}','category':{'level1Id':'cat'},
                         'annotation':{'grade':'silver' if j==3 else 'gold'}})
    (data/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for name in ('label_map.json','theme_taxonomy.json'):(data/name).write_text('{}')
    output=tmp_path/'output';report=build(data,output)
    train=[json.loads(s) for s in (output/'train.jsonl').open()]
    val=[json.loads(s) for s in (output/'validation.jsonl').open()]
    assert len(train)==36 and len(val)==3
    assert not {r['productGroupId'] for r in train}&{r['productGroupId'] for r in val}
    assert all(r['annotation']['grade']=='gold' for r in val)
    assert report['heldoutProducts']==1
    assert not (output/'test.jsonl').exists()
