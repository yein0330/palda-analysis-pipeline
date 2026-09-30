import json
import pytest

from scripts.build_encoder_ablation import build


def fixture_data(tmp_path):
    data = tmp_path / "input"
    data.mkdir()
    rows = [
        {"itemId": "train1", "sourceType": "review", "text": "싼 맛에 쓰기 좋네요",
         "category": {"level2Id": "cat"}, "labels": {"reviewLabels": []},
         "annotation": {"grade": "gold", "sampleWeight": 1.0}},
        {"itemId": "train2", "sourceType": "review", "text": "편해요",
         "category": {"level2Id": "cat"}, "labels": {"reviewLabels": []},
         "annotation": {"grade": "silver", "sampleWeight": 1.0}},
    ]
    (data / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (data / "validation.jsonl").write_text('{"sentinel": "validation unchanged"}\n')
    (data / "label_map.json").write_text(json.dumps({"categoryThemeToIndex": {"price": 0}, "categoryMasks": {"cat": [0]}}))
    (data / "theme_taxonomy.json").write_text("{}")
    # Intentionally no Test file: the pipeline must not need it.
    return data, rows


def test_restore_only_original_allowed_train_price_and_preserve_validation(tmp_path):
    data, rows = fixture_data(tmp_path)
    label = {"themeId": "PRICE", "categoryThemeId": "price", "sentiment": "positive"}
    rows[0]["labels"]["reviewLabels"] = [label]
    rows.append({**rows[0], "itemId": "not-in-train"})
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    output = tmp_path / "output"
    report = build(data, output, "price-inflection", source)
    result = [json.loads(x) for x in (output / "train.jsonl").read_text().splitlines()]
    assert report["changes"] == 1
    assert len(result) == 2
    assert result[0]["labels"]["reviewLabels"] == [label]
    assert (output / "validation.jsonl").read_bytes() == (data / "validation.jsonl").read_bytes()
    assert not (output / "test.jsonl").exists()
    assert json.loads((data / "train.jsonl").read_text().splitlines()[0])["labels"]["reviewLabels"] == []


def test_weight_ablation_changes_only_silver_weight(tmp_path):
    data, original = fixture_data(tmp_path)
    output = tmp_path / "output"
    build(data, output, "silver-weight-03")
    result = [json.loads(x) for x in (output / "train.jsonl").read_text().splitlines()]
    assert result[0] == original[0]
    assert result[1]["annotation"]["sampleWeight"] == 0.3
    result[1]["annotation"]["sampleWeight"] = 1.0
    assert result == original


def test_gold_preserve_requires_existing_approval_and_ignores_silver(tmp_path):
    data, rows = fixture_data(tmp_path)
    label = {"themeId": "PRICE", "categoryThemeId": "price", "sentiment": "positive"}
    for row in rows:
        row['labels']['reviewLabels'] = [label]
        row['annotation']['reviewerStatus'] = 'approved'
    source = tmp_path / 'source.jsonl'
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    out = tmp_path / 'out'
    result = build(data, out, 'gold-preserve', source)
    restored = [json.loads(s) for s in (out/'train.jsonl').read_text().splitlines()]
    assert result['changes'] == 1
    assert restored[0]['labels']['reviewLabels'] == [label]
    assert restored[1]['labels']['reviewLabels'] == []
    assert (out/'validation.jsonl').read_bytes() == (data/'validation.jsonl').read_bytes()
    rows[0]['annotation']['reviewerStatus'] = 'rejected'
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert build(data, tmp_path/'rejected', 'gold-preserve', source)['changes'] == 0


def test_gold_preserve_refuses_stale_text(tmp_path):
    data, rows = fixture_data(tmp_path)
    rows[0]['annotation']['reviewerStatus'] = 'corrected'
    rows[0]['text'] = 'different source text'
    source = tmp_path/'source.jsonl'
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='content mismatch'):
        build(data,tmp_path/'out','gold-preserve',source)


def test_review_only_preservation_leaves_qna_exactly_unchanged(tmp_path):
    data, rows = fixture_data(tmp_path)
    qna = {'itemId':'qna1','sourceType':'qna','text':'얼마인가요?',
           'category':{'level2Id':'cat'},'annotation':{'grade':'gold','sampleWeight':1.0},
           'labels':{'reviewLabels':[],'qnaPrimaryTheme':{'themeId':'PRICE','categoryThemeId':'price'},
                     'qnaSecondaryThemes':[]}}
    rows.append(qna)
    (data/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    source_rows = json.loads(json.dumps(rows))
    source_rows[0]['annotation']['reviewerStatus']='approved'
    source_rows[0]['labels']['reviewLabels']=[{'themeId':'PRICE','categoryThemeId':'price','sentiment':'positive'}]
    source_rows[2]['annotation']['reviewerStatus']='approved'
    source_rows[2]['labels']['qnaSecondaryThemes']=[{'themeId':'PRICE','categoryThemeId':'price'}]
    source=tmp_path/'source.jsonl'
    source.write_text(''.join(json.dumps(r)+'\n' for r in source_rows))
    out=tmp_path/'review-only'
    assert build(data,out,'gold-review-preserve',source)['changes']==1
    result=[json.loads(s) for s in (out/'train.jsonl').read_text().splitlines()]
    assert result[2]==qna
    assert result[1]==rows[1]
    assert not (out/'test.jsonl').exists()
