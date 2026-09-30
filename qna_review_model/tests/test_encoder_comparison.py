import json

import pytest

from scripts.compare_encoder_validation import compare, f1_sets


def test_fixed_macro_penalizes_missed_classes():
    result = f1_sets([{"a"}], [{"a", "b"}], ["a", "b"])
    assert result["microF1"] == pytest.approx(2/3)
    assert result["fixedLabelMacroF1"] == 0.5


def test_comparison_rejects_changed_validation_truth(tmp_path):
    runs=[]
    for i,truth in enumerate((["QUALITY::positive"], ["QUALITY::negative"])):
        p=tmp_path/str(i);(p/'best').mkdir(parents=True)
        (p/'best/training_config.json').write_text(json.dumps({"indexToCanonicalTheme":["QUALITY"]}))
        (p/'best-validation-predictions.jsonl').write_text(json.dumps({
            "itemId":"v1","sourceType":"review","reviewTruth":truth})+'\n')
        runs.append(p)
    with pytest.raises(ValueError,match='Different Validation IDs or labels'):
        compare(runs)


def test_comparison_accepts_reordered_multilabel_targets(tmp_path):
    runs=[]
    for i,truth in enumerate((["QUALITY::positive","QUALITY::negative"], ["QUALITY::negative","QUALITY::positive"])):
        p=tmp_path/str(i);(p/'best').mkdir(parents=True)
        (p/'best/training_config.json').write_text(json.dumps({"indexToCanonicalTheme":["QUALITY"],
            "epoch":1,"reviewThreshold":.5,"secondaryThreshold":.5}))
        (p/'best-validation-predictions.jsonl').write_text(json.dumps({
            "itemId":"v1","sourceType":"review","reviewTruth":truth,'reviewCandidates':[]})+'\n')
        runs.append(p)
    assert compare(runs)['records']==1
