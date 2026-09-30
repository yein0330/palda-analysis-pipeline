import json

from scripts.diagnose_encoder_review_blend import blend


def test_blend_preserves_qna_and_averages_review_only(tmp_path):
    qna={'itemId':'q','sourceType':'qna','primaryTruth':'QUALITY','primaryPrediction':'QUALITY',
         'secondaryTruth':[],'secondaryCandidates':[['QUALITY',.1]],'usableTruth':True,'usableProbability':.9}
    paths=[]
    for i,score in enumerate((.8,.4)):
        path=tmp_path/str(i);(path/'best').mkdir(parents=True);paths.append(path)
        (path/'best/training_config.json').write_text(json.dumps({'epoch':1,'indexToCanonicalTheme':['QUALITY'],
             'reviewThreshold':.5,'secondaryThreshold':.5}))
        review={'itemId':'r','sourceType':'review','reviewTruth':['QUALITY::positive'],
                'reviewCandidates':[['QUALITY::positive',score]]}
        other={**qna,'usableProbability':.2} if i else qna
        (path/'best-validation-predictions.jsonl').write_text(json.dumps(review)+'\n'+json.dumps(other)+'\n')
    blend(*paths,tmp_path/'out')
    rows=[json.loads(s) for s in (tmp_path/'out/best-validation-predictions.jsonl').open()]
    assert rows[1]==qna
    assert abs(rows[0]['reviewCandidates'][0][1]-.6)<1e-7
    frozen=blend(*paths,tmp_path/'frozen',review_threshold=.7)
    assert frozen['bestValidation']['reviewThemeSentiment']['microF1']==0
    config=json.loads((tmp_path/'frozen/best/training_config.json').read_text())
    assert config['reviewThreshold']==.7 and config['reviewThresholdPolicy']=='predeclared_fixed'
