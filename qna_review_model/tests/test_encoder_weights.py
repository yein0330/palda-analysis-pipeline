import pytest
import torch

from scripts.train_encoder_transformer import training_positive_weights


def test_balanced_weights_count_only_allowed_review_exposure():
    mapping={"indexToCanonicalTheme":["QUALITY","PRICE"],"categoryMasks":{"a":[0],"b":[1]},"targetToIndex":{"quality":0,"price":1}}
    rows=[{"sourceType":"review","category":{"level2Id":"a"},"annotation":{"sampleWeight":1.0},
           "labels":{"reviewLabels":[{"categoryThemeId":"quality","sentiment":"positive"}] if i==0 else []}}
          for i in range(10)]
    rows.append({"sourceType":"qna"})
    weights=training_positive_weights(rows,mapping)
    assert weights.shape==(2,3)
    assert weights[0,0]==3 # sqrt(9 negatives / 1 positive)
    assert torch.equal(weights[1],torch.ones(3)) # No category exposure or positive targets.
    rows[0]['labels']['reviewLabels'][0]['categoryThemeId']='price'
    with pytest.raises(ValueError,match='masked out'):
        training_positive_weights(rows,mapping)
