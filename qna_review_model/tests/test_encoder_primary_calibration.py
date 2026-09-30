import numpy as np
import pytest
import json

from scripts.calibrate_encoder_primary import fit_temperature, quality, scale


def test_temperature_preserves_rank_and_handles_zeros():
    p = np.array([.8, .15, .05, 0.])
    for t in (.1, 1, 10):
        result = scale(p, t)
        assert result.argmax() == 0
        assert np.isclose(result.sum(), 1)
        assert np.isfinite(result).all()
    assert scale(p, 10).max() < p.max()
    with pytest.raises(ValueError):
        scale(p, 0)
    with pytest.raises(ValueError):
        scale([0, 0], 1)


def test_temperature_fit_reduces_training_nll_not_top1_error():
    probabilities = [np.array([.99, .01])] * 10
    truths = [0] * 6 + [1] * 4
    temperature = fit_temperature(probabilities, truths)
    scaled = [scale(p, temperature) for p in probabilities]
    before, after = quality(probabilities, truths), quality(scaled, truths)
    assert temperature > 1
    assert after['negativeLogLikelihood'] < before['negativeLogLikelihood']
    assert before['accuracy'] == after['accuracy'] == .6
    assert abs(before['ece10EqualWidth'] - .39) < 1e-10


def test_grouped_calibration_cli_uses_no_train_or_test(tmp_path, monkeypatch):
    from scripts.calibrate_encoder_primary import main
    validation, predictions = [], []
    for i in range(10):
        truth = 'A' if i % 2 == 0 else 'B'
        validation.append({'itemId':str(i),'sourceType':'qna','productGroupId':str(i//2),
                           'labels':{'qnaPrimaryTheme':{'themeId':truth}}})
        predictions.append({'itemId':str(i),'sourceType':'qna','primaryTruth':truth,
                            'primaryPrediction':'A','primaryProbability':.9,
                            'primaryCandidates':[['A',.9],['B',.1]]})
    vp, pp, out = tmp_path/'validation.jsonl', tmp_path/'predictions.jsonl', tmp_path/'out'
    vp.write_text(''.join(json.dumps(r)+'\n' for r in validation))
    pp.write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    monkeypatch.setattr('sys.argv',['calibrate','--predictions',str(pp),'--validation',str(vp),'--output-dir',str(out)])
    main()
    report=json.loads((out/'report.json').read_text())
    groups=[p for f in report['fitted'] for p in f['heldoutProducts']]
    assert sorted(groups)==['0','1','2','3','4']
    assert all(f['fitRecords']==8 and f['heldoutRecords']==2 for f in report['fitted'])
    assert report['before']['accuracy']==report['outOfFold']['accuracy']==.5
    assert len((out/'oof-qna-predictions.jsonl').read_text().splitlines())==10
    with pytest.raises(FileExistsError):
        main()
