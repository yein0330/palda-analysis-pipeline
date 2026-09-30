from scripts.score_encoder_judge_constraints import evaluate_constraint


def test_optional_labels_are_not_required_and_extras_are_counted():
    d={'required':['PRICE::positive'],'allowed':['PRICE::positive','QUALITY::neutral']}
    assert evaluate_constraint(d,{'PRICE::positive'})['passed']
    assert evaluate_constraint(d,{'PRICE::positive','QUALITY::neutral'})['passed']
    assert evaluate_constraint(d,{'PRICE::positive','QUALITY::positive'})['unsupported']==['QUALITY::positive']
    assert evaluate_constraint({'required':[],'allowed':[]},set())['passed']
