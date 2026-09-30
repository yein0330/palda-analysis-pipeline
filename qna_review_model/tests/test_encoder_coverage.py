from scripts.measure_encoder_coverage import measure
import pytest


def test_empty_acceptance_is_not_perfect_accuracy():
    report=measure([{'sourceType':'qna','primaryProbability':0.6,'primaryPrediction':'A','primaryTruth':'B'}])
    empty=next(x for x in report['cutoffs'] if x['cutoff']==0.9)
    assert empty['coverage']==0
    assert empty['accuracy'] is None
    assert empty['errorRate'] is None
    assert report['cutoffs'][0]['errorRate']==1


def test_agreement_gate_preserves_denominator_and_checks_identity():
    rows=[dict(itemId='a',sourceType='qna',primaryProbability=.9,primaryPrediction='A',primaryTruth='A'),
          dict(itemId='b',sourceType='qna',primaryProbability=.9,primaryPrediction='A',primaryTruth='B')]
    other=[rows[0],rows[1]|{'primaryPrediction':'B'}]
    report=measure(rows,other)
    assert report['agreementRequired']
    assert report['cutoffs'][0]['coverage']==.5
    assert report['cutoffs'][0]['accuracy']==1
    with pytest.raises(ValueError,match='IDs mismatch'):
        measure(rows,other[:1])
    with pytest.raises(ValueError,match='targets mismatch'):
        measure(rows,[other[0]|{'primaryTruth':'Z'},other[1]])
