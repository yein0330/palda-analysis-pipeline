import numpy as np
import json
import joblib
import pytest
from scipy.sparse import csr_matrix

from scripts.benchmark_encoder_lexical import binary_scores, predict_review_artifact, run


def test_binary_reference_handles_absent_and_constant_targets():
    x=csr_matrix(np.array([[1.,0.],[0.,1.]],dtype=np.float32))
    for y in (np.array([0,0]),np.array([1,1])):
        assert np.array_equal(binary_scores(x,y,np.ones(2),x,1),y)
    assert np.array_equal(binary_scores(x[:0],np.array([]),np.array([]),x,1),np.zeros(2))


@pytest.mark.parametrize('feature_mode',['char','char-word'])
def test_lexical_run_needs_no_test_file_and_writes_comparable_predictions(tmp_path,feature_mode):
    data=tmp_path/'data';data.mkdir()
    themes=['QUALITY','PRICE']
    mapping={'categoryThemeToIndex':{'quality':0,'price':1},'indexToCategoryTheme':['quality','price'],
             'categoryThemeMetadata':{k:{'canonicalThemeId':v} for k,v in zip(['quality','price'],themes)},
             'categoryMasks':{'cat':[0,1]}}
    (data/'label_map.json').write_text(json.dumps(mapping))
    rows=[]
    for i in range(8):
        theme=themes[i%2];key=theme.lower();task='review' if i<4 else 'qna'
        label={'themeId':theme,'categoryThemeId':key}
        rows.append({'itemId':str(i),'sourceType':task,'text':'품질 좋아요' if theme=='QUALITY' else '가격 저렴해요',
                     'category':{'level1Name':'상위','level2Name':'하위','level2Id':'cat'},'usable':True,
                     'annotation':{'sampleWeight':1.0},'labels':{'reviewLabels':[{**label,'sentiment':'positive'}] if task=='review' else [],
                        'qnaPrimaryTheme':label if task=='qna' else None,'qnaSecondaryThemes':[]}})
    for split in ('train','validation'):
        (data/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=run(data,tmp_path/'out',save_review_artifact=True,feature_mode=feature_mode)
    assert result['testSkipped'] and result['weightsSaved']
    assert (tmp_path/'out/best-validation-predictions.jsonl').exists()
    artifact=joblib.load(tmp_path/'out/review-model.joblib')
    replay=predict_review_artifact(artifact,[r for r in rows if r['sourceType']=='review'])
    saved={r['itemId']:r for r in map(json.loads,(tmp_path/'out/best-validation-predictions.jsonl').open())}
    for row in replay:assert row['reviewCandidates']==saved[row['itemId']]['reviewCandidates']


def test_feature_union_is_normalized_and_does_not_learn_from_validation():
    from scripts.benchmark_encoder_lexical import make_vectorizer
    vectorizer=make_vectorizer('char-word')
    texts=['배송 빠르고 좋아요','배송 빠르고 좋아요','품질 만족합니다','품질 만족합니다']
    x=vectorizer.fit_transform(texts)
    names=vectorizer.named_steps['features'].get_feature_names_out().copy()
    assert np.allclose(np.asarray(x.multiply(x).sum(axis=1)).ravel(),1)
    assert any(n.startswith('word__') for n in names)
    vectorizer.transform(['NEWHELDOUTTEXT'])
    assert np.array_equal(names,vectorizer.named_steps['features'].get_feature_names_out())
