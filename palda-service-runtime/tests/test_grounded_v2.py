from copy import deepcopy
import pytest
from palda_runtime.grounded_v2 import (
    accepts_support, catalog_for, generic_or_expectation, intent_report,
    sanitize_classification, validate_point, complete_checked, generate_report, source_spans, resolve_classification_spans, verify_records,
)

CAT = [{"themeId": t, "reviewName": t, "qnaName": t} for t in ("PERFORMANCE", "USABILITY", "OTHER", "UNUSABLE")]


def test_verification_never_promotes_evidence_score():
    original = {"id": "r1", "usable": True, "usableConfidence": .7,
                "labels": [{"themeId": "USABILITY", "sentiment": "negative", "confidence": .32}]}
    snapshot = deepcopy(original)
    value = {"usable": True, "labels": [{"themeId": "USABILITY", "sentiment": "negative", "mode": "observed", "quote": "빼기 불편해요"}]}
    p = sanitize_classification("review", "쌀 빼기 불편해요", original, value, CAT)
    assert p["labels"][0]["confidence"] == .32 and p["usableConfidence"] == .7
    value["labels"][0]["themeId"] = "PERFORMANCE"
    assert sanitize_classification("review", "쌀 빼기 불편해요", original, value, CAT)["labels"][0]["confidence"] == 0
    assert original == snapshot


@pytest.mark.parametrize("text", ["좋아요 만족합니다", "벌레 않생기길 바랍니다", "만족스럽습니다~~~"])
def test_no_concrete_claim_from_generic_or_hope(text):
    assert generic_or_expectation(text)
    value = {"usable": True, "labels": [{"themeId": "PERFORMANCE", "sentiment": "positive", "mode": "observed", "quote": text}]}
    assert sanitize_classification("review", text, {"id": "r1"}, value, CAT)["labels"] == []


def test_expectation_and_conditional_not_asserted_as_observed():
    for mode in ("expectation", "conditional"):
        v = {"usable": True, "labels": [{"themeId": "USABILITY", "sentiment": "positive", "mode": mode, "quote": "부드러워서 좋을 것 같아요"}]}
        assert sanitize_classification("review", "부드러워서 좋을 것 같아요", {"id": "r1"}, v, CAT)["labels"] == []


def test_theme_overlay_does_not_mutate_original_or_all_categories():
    assert "DESIGN" in {t["themeId"] for t in catalog_for("출산/육아 > 신생아의류", CAT)}
    assert catalog_for("가전", CAT) == CAT
    assert "DESIGN" not in {t["themeId"] for t in CAT}


def test_qna_intents_are_separate_and_counted_against_analyzed_questions():
    v = {"usable": True, "primaryThemeId": "OTHER", "quote": "재입고 알림", "intents": [{"id": "availability", "quote": "재입고 알림"}]}
    p = sanitize_classification("qna", "재입고 알림 부탁해요", {"id": "q1", "primaryTheme": {"themeId": "OTHER", "confidence": .4}}, v, CAT)
    assert p["primaryTheme"]["themeId"] == "OTHER"
    report = intent_report({"qnaSample": {"items": [{"id": "q1"}, {"id": "q2"}]}}, [p, {"id": "q2", "usable": False}])
    assert report["inputCount"] == 2 and report["analyzedCount"] == 1
    assert report["themes"][0]["id"] == "availability"


def test_valid_intent_is_not_discarded_when_product_theme_is_unusable():
    v = {"usable":True,'primaryThemeId':'UNUSABLE','quote':'재입고 알림','intents':[{'id':'availability','quote':'재입고 알림'}]}
    p = sanitize_classification('qna','재입고 알림 부탁해요',{'id':'q1'},v,CAT)
    assert p['usable'] is False
    report = intent_report({'qnaSample':{'items':[{'id':'q1'}]}},[p])
    assert report['analyzedCount']==1 and report['themes'][0]['id']=='availability'


@pytest.mark.parametrize("quote", ["좋아요", "", None])
def test_hallucinated_support_quote_rejected(quote):
    with pytest.raises(ValueError):
        sanitize_classification("review", "사용이 불편해요", {"id": "r1"}, {"usable": True, "labels": [
            {"themeId": "USABILITY", "sentiment": "positive", "mode": "observed", "quote": quote}]}, CAT)


def test_selling_requires_actual_citation_not_theme_id():
    sources = {"E1": {"text": "부드러워요", "kind": "review", "sentiments": ["positive"]},
               "E2": {"text": "세탁 가능한가요", "kind": "qna", "sentiments": []}}
    point = {"type": "core", "text": "부드러운 촉감", "citations": [{"id": "E1", "quote": "부드러워요"}]}
    assert validate_point(point, sources) == point["citations"]
    for bad in [{**point, "citations": []}, {**point, "citations": [{"id": "E2", "quote": "세탁 가능한가요"}]},
                {**point, "citations": [{"id": "QUALITY", "quote": "부드러워요"}]}]:
        with pytest.raises(ValueError): validate_point(bad, sources)


def test_support_requires_explicit_boolean_and_explanation():
    assert accepts_support({"supported": True, "reason": "원문에 명시"})
    for v in [{}, {"supported": "true", "reason": "a"}, {"supported": True}, {"supported": False, "reason": "반대"}]:
        assert not accepts_support(v)


def test_context_overflow_is_visible_and_not_truncated():
    class Tokenizer:
        def apply_chat_template(self, *a, **k): return "long"
        def __call__(self, text): return {"input_ids": list(range(10))}
    class Decoder:
        SYSTEM = "system"
        tokenizer = Tokenizer()
        max_input_length = 5
        def _complete_many(self, *a, **k): raise AssertionError("Must not call model")
    answers, audit = complete_checked(Decoder(), ["prompt"])
    assert answers == [{}] and audit[0]["status"] == "context_limit"


def run_fake_generation(selling, selling_check=None):
    class Tokenizer:
        def apply_chat_template(self, *a, **k): return "prompt"
        def __call__(self, text): return {"input_ids": [1, 2]}
    class Decoder:
        SYSTEM = "system"
        tokenizer = Tokenizer()
        max_input_length = 100
        def __init__(self):
            self.answers = iter([
                [{"summary": "부드럽다는 리뷰입니다.", "quote": "부드러워요"}],
                [{"supported": True, "reason": "직접 명시"}],
                [selling],
                [selling_check or {"supported": True, "reason": "직접 명시"}],
            ])
        def _complete_many(self, *a, **k): return next(self.answers)
    request = {"analysisId": "a1", "taxonomyVersion": "v1", "product": {"productId": "p1", "categoryPath": "아기"},
               "reviewSample": {"items": [{"id": "r1", "text": "부드러워요"}]}, "qnaSample": {"items": []}}
    predictions = [{"id": "r1", "usable": True, "labels": [{"themeId": "MATERIAL", "sentiment": "positive", "confidence": .5}]}]
    versions = dict(encoderModel="e",decoderModel="d",taxonomy="t",calculation="experimental")
    return generate_report(Decoder(), request, predictions, versions)


def test_failed_selling_generation_is_visible_without_generic_fallback():
    result, diagnostic, _ = run_fake_generation({})
    assert result["sellingPoints"] == []
    assert diagnostic["sellingGeneration"]["status"] == "failed"
    assert diagnostic["sellingGeneration"]["fallbackUsed"] is False
    assert result["reviewAnalysis"]["positiveThemes"]


def test_direct_selling_citations_and_rejected_semantics():
    selling = {"selling_points": [{"type": "core", "text": "부드러운 촉감을 후기 기반 장점으로 안내할 수 있습니다.",
                                   "citations": [{"id": "E1", "spanId": "S1"}]}]}
    result, diagnostic, _ = run_fake_generation(selling)
    assert result["sellingPoints"][0]["evidenceIds"] == ["r1"]
    assert diagnostic["sellingGeneration"]["points"][0]["directCitations"] == [{"id": "r1", "quote": "부드러워요"}]
    result, diagnostic, _ = run_fake_generation(selling, {"supported": False, "reason": "지원 부족"})
    assert result["sellingPoints"] == []
    assert diagnostic["sellingGeneration"]["status"] == "rejected"


def test_span_ids_preserve_original_typos_and_whitespace():
    text = "작업은  쉬워요.\n풀 뭍은거는 없어지네요"
    value = {"usable":True, "labels":[{"spanId":"S2"}]}
    resolved = resolve_classification_spans("review", text, value)
    assert resolved['labels'][0]['quote'] == "풀 뭍은거는 없어지네요"
    assert value['labels'][0] == {"spanId":"S2"}
    with pytest.raises(ValueError):
        resolve_classification_spans('review',text,{'labels':[{'spanId':'S100'}]})


def test_format_retry_is_bounded_and_audited_without_score_promotion():
    class Tokenizer:
        def apply_chat_template(self,*a,**k): return 'prompt'
        def __call__(self,text): return {'input_ids':[1]}
    class Encoder:
        def allowed_theme_catalog(self,path): return CAT
    class Decoder:
        SYSTEM='system'
        tokenizer=Tokenizer()
        max_input_length=100
        def __init__(self): self.calls=0
        def _complete_many(self,prompts,**kw):
            self.calls+=1
            if self.calls==1:return [{}]
            return [{'usable':True,'labels':[{'themeId':' USABILITY','sentiment':'negative',
                     'mode':'observed','spanId':'S1'}]}]
    d=Decoder()
    request={'product':{'categoryPath':'가전'},'reviewSample':{'items':[{'id':'r1','text':'사용이 불편해요'}]},'qnaSample':{'items':[]}}
    old=[{'id':'r1','usable':True,'usableConfidence':.6,'labels':[{'themeId':'USABILITY','sentiment':'negative','confidence':.3}]}]
    result,audit=verify_records(d,Encoder(),request,old)
    assert d.calls==2 and audit['retried']==1 and audit['failed']==0
    assert audit['records'][0]['raw']=={} and audit['records'][0]['retry']
    assert result[0]['labels'][0]['confidence']==.3
