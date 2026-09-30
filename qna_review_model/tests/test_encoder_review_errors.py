from scripts.diagnose_encoder_review_errors import decompose


def test_error_partition_and_conditional_denominator():
    rows = [dict(sourceType='review', reviewTruth=['A::positive', 'B::negative'],
                 reviewCandidates=[('A::negative', .9), ('C::positive', .9), ('B::negative', .1)]),
            dict(sourceType='review', reviewTruth=['A::positive', 'A::negative'],
                 reviewCandidates=[('A::positive', .9), ('A::negative', .9)])]
    result = decompose(rows, .5)
    assert (result['themeTP'], result['themeFP'], result['themeFN']) == (2, 1, 1)
    assert result['themeOnlyMicroF1'] == 2 / 3
    assert result['sentimentSetAccuracyOnMatchedThemes'] == .5
    assert result['sentimentFPOnMatchedThemes'] == 1
    assert result['sentimentFNOnMatchedThemes'] == 1
    assert result['labelFPFromAbsentThemes'] == 1
    assert result['labelFNFromMissedThemes'] == 1
    assert decompose([], .5)['sentimentSetAccuracyOnMatchedThemes'] is None
