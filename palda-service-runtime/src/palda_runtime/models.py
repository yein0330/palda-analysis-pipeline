from __future__ import annotations

import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .aggregation import _fallback_selling_points


SENTIMENTS = ("positive", "negative", "neutral")
SPECIAL_THEMES = {"OTHER", "UNUSABLE"}


def _strongest_eligible_sentiment(
    scores: list[float], thresholds: list[float]
) -> tuple[int, float] | None:
    eligible = [(index, score) for index, score in enumerate(scores) if score >= thresholds[index]]
    return max(eligible, key=lambda value: value[1]) if eligible else None


class ServiceEncoderModel:
    """Inference adapter for the four-head PALDA service encoder artifact."""

    def __init__(
        self,
        artifact_dir: str | Path,
        taxonomy_path: str | Path,
        *,
        device: str = "auto",
        batch_size: int = 64,
        max_length: int = 256,
    ) -> None:
        try:
            import torch
            from torch import nn
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install model dependencies with: pip install -e '.[models]'") from exc

        artifact = Path(artifact_dir)
        self.torch = torch
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(artifact / "tokenizer", local_files_only=True)
        label_map = _read_json(artifact / "label_map.json")
        self.themes = list(label_map["themes"])
        self.theme_to_id = {theme: index for index, theme in enumerate(self.themes)}
        self.thresholds = _read_json(artifact / "thresholds.json")
        taxonomy = _read_json(taxonomy_path)
        self.category_themes = dict(taxonomy.get("category_themes") or {})
        self.theme_names = {
            str(item["id"]): {
                "review": str(item.get("review_label") or item["id"]),
                "qna": str(item.get("qna_label") or item["id"]),
            }
            for item in taxonomy.get("themes", [])
        }

        class Model(nn.Module):
            def __init__(inner_self: Any) -> None:
                super().__init__()
                inner_self.backbone = AutoModel.from_pretrained(artifact / "backbone", local_files_only=True)
                hidden = int(inner_self.backbone.config.hidden_size)
                dropout = float(getattr(inner_self.backbone.config, "hidden_dropout_prob", 0.1))
                inner_self.dropout = nn.Dropout(dropout)
                inner_self.review_head = nn.Linear(hidden, len(self.themes) * len(SENTIMENTS))
                inner_self.primary_head = nn.Linear(hidden, len(self.themes))
                inner_self.secondary_head = nn.Linear(hidden, len(self.themes))
                inner_self.usable_head = nn.Linear(hidden, 1)
                heads = torch.load(artifact / "heads.pt", map_location="cpu", weights_only=True)
                for name in ("review_head", "primary_head", "secondary_head", "usable_head"):
                    getattr(inner_self, name).load_state_dict(heads[name])

            def forward(inner_self: Any, **encoded: Any) -> dict[str, Any]:
                pooled = inner_self.dropout(inner_self.backbone(**encoded).last_hidden_state[:, 0])
                return {
                    "review": inner_self.review_head(pooled).view(-1, len(self.themes), len(SENTIMENTS)),
                    "primary": inner_self.primary_head(pooled),
                    "secondary": inner_self.secondary_head(pooled),
                    "usable": inner_self.usable_head(pooled).squeeze(-1),
                }

        self.model = Model().to(self.device).eval()

    async def classify(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        return self._classify_sync(request)

    def allowed_theme_catalog(self, category_path: str) -> list[dict[str, str]]:
        """Return the Cat2-scoped taxonomy presented to the decoder verifier."""
        category = _cat2_key(category_path)
        allowed = set(self.category_themes.get(category) or self.themes)
        allowed.update(theme for theme in SPECIAL_THEMES if theme in self.theme_to_id)
        return [{
            "themeId": theme,
            "reviewName": self.theme_names.get(theme, {}).get("review", theme),
            "qnaName": self.theme_names.get(theme, {}).get("qna", theme),
        } for theme in self.themes if theme in allowed]

    def _classify_sync(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        records = [
            *(dict(item, record_type="review") for item in request["reviewSample"]["items"]),
            *(dict(item, record_type="qna") for item in request["qnaSample"]["items"]),
        ]
        category = _cat2_key(request["product"]["categoryPath"])
        allowed = set(self.category_themes.get(category) or self.themes)
        allowed.add("OTHER")
        candidate = self.torch.tensor(
            [theme in allowed for theme in self.themes], dtype=self.torch.bool, device=self.device
        )
        predictions: list[dict[str, Any]] = []
        for offset in range(0, len(records), self.batch_size):
            batch = records[offset : offset + self.batch_size]
            texts = [
                f"[TASK={row['record_type'].upper()}] [CAT1={_cat1(category)}] "
                f"[CAT2={_cat2(category)}] {row['text']}"
                for row in batch
            ]
            encoded = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
            ).to(self.device)
            with self.torch.inference_mode():
                outputs = self.model(**encoded)
            usable = self.torch.sigmoid(outputs["usable"])
            review_scores = self.torch.sigmoid(outputs["review"])
            primary_scores = self.torch.softmax(outputs["primary"].masked_fill(~candidate, -1e4), dim=-1)
            secondary_scores = self.torch.sigmoid(outputs["secondary"])
            for index, row in enumerate(batch):
                is_usable = float(usable[index]) >= float(self.thresholds.get("usable", 0.5))
                result: dict[str, Any] = {
                    "id": str(row["id"]), "usable": is_usable,
                    "usableConfidence": float(usable[index]),
                }
                if row["record_type"] == "review":
                    labels = []
                    review_thresholds = self.thresholds.get("review", 0.5)
                    for theme_index, theme in enumerate(self.themes):
                        if theme not in allowed or theme in SPECIAL_THEMES:
                            continue
                        scores = [
                            float(review_scores[index, theme_index, sentiment_index])
                            for sentiment_index in range(len(SENTIMENTS))
                        ]
                        thresholds = [
                            (
                                float(review_thresholds[theme_index][sentiment_index])
                                if isinstance(review_thresholds, list) else float(review_thresholds)
                            )
                            for sentiment_index in range(len(SENTIMENTS))
                        ]
                        strongest = _strongest_eligible_sentiment(scores, thresholds)
                        # A theme cannot be positive and negative for the same
                        # source item. Keep the strongest eligible sentiment.
                        if strongest:
                            sentiment_index, score = strongest
                            labels.append({
                                "themeId": theme,
                                "theme": self.theme_names.get(theme, {}).get("review", theme),
                                "sentiment": SENTIMENTS[sentiment_index], "confidence": score,
                            })
                    result["labels"] = sorted(labels, key=lambda value: -value["confidence"])
                else:
                    primary_index = int(primary_scores[index].argmax())
                    result["primaryTheme"] = {
                        "themeId": self.themes[primary_index],
                        "theme": self.theme_names.get(self.themes[primary_index], {}).get(
                            "qna", self.themes[primary_index]
                        ),
                        "confidence": float(primary_scores[index, primary_index]),
                    }
                    secondary_thresholds = self.thresholds.get("secondary", 0.5)
                    secondary = []
                    for theme_index, theme in enumerate(self.themes):
                        score = float(secondary_scores[index, theme_index])
                        threshold = (
                            float(secondary_thresholds[theme_index])
                            if isinstance(secondary_thresholds, list) else float(secondary_thresholds)
                        )
                        if (theme in allowed and theme not in SPECIAL_THEMES
                                and theme_index != primary_index and score >= threshold):
                            secondary.append({
                                "themeId": theme,
                                "theme": self.theme_names.get(theme, {}).get("qna", theme),
                                "confidence": score,
                            })
                    result["secondaryThemes"] = sorted(secondary, key=lambda value: -value["confidence"])
                predictions.append(result)
        return predictions


class GroundedDecoderModel:
    """Runs the PALDA adapter as group summaries followed by selling points."""

    SYSTEM = (
        "당신은 PALDA의 한국어 커머스 분석 모델입니다. 입력 근거만 사용하고, "
        "인용문은 원문에서 그대로 가져오며, 지정된 JSON 형식으로만 답하세요."
    )

    def __init__(
        self,
        adapter_dir: str | Path,
        *,
        device_map: str = "auto",
        load_in_4bit: bool = True,
        max_input_length: int = 2048,
        max_new_tokens: int = 512,
        batch_size: int = 4,
    ) -> None:
        try:
            import torch
            from peft import AutoPeftModelForCausalLM
            from transformers import AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Install model dependencies with: pip install -e '.[models]'") from exc
        kwargs: dict[str, Any] = {"device_map": device_map}
        if torch.cuda.is_available():
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            kwargs["dtype"] = compute_dtype
            if load_in_4bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                )
        self.torch = torch
        self.model = AutoPeftModelForCausalLM.from_pretrained(str(adapter_dir), **kwargs).eval()
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    async def generate(self, report_input: dict[str, Any]) -> dict[str, Any]:
        return self._generate_sync(report_input)

    async def verify_predictions(
        self, request: dict[str, Any], predictions: list[dict[str, Any]],
        allowed_themes: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Re-read source text and filter/correct encoder candidates before aggregation."""
        sources = {
            str(item["id"]): (record_type, str(item["text"]))
            for record_type, sample_key in (("review", "reviewSample"), ("qna", "qnaSample"))
            for item in request[sample_key]["items"]
        }
        jobs = []
        for prediction in predictions:
            source = sources.get(str(prediction.get("id")))
            if source is None:
                jobs.append(None)
                continue
            record_type, source_text = source
            jobs.append(_verification_prompt(
                record_type=record_type, category_path=str(request["product"]["categoryPath"]),
                source_text=source_text, candidate=prediction, allowed_themes=allowed_themes,
            ))
        completions = iter(self._complete_many(
            [prompt for prompt in jobs if prompt is not None],
            max_new_tokens=256, disable_adapter=True,
        ))
        verified = []
        for prediction, prompt in zip(predictions, jobs):
            if prompt is None:
                verified.append(prediction)
                continue
            record_type = sources[str(prediction["id"])][0]
            verified.append(_sanitize_verification(
                record_type, next(completions), prediction, allowed_themes
            ))
        return verified

    def _generate_sync(self, report_input: dict[str, Any]) -> dict[str, Any]:
        summaries = []
        positive, negative, qna = [], [], []
        jobs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for group in report_input.get("reviewGroups", []):
            if group.get("sentiment") not in {"positive", "negative"}:
                continue
            jobs.append(("review", group, _summary_task(report_input, group, "review_summary")))
        for group in report_input.get("qnaGroups", []):
            jobs.append(("qna", group, _summary_task(report_input, group, "qna_summary")))
        completions = self._complete_many([_summary_prompt(task) for _, _, task in jobs])
        for (kind, group, task), completion in zip(jobs, completions):
            summary = str(completion.get("summary") or "").strip()
            fallback_used = not summary
            if fallback_used:
                summary = _fallback_summary(group, kind)
            if kind == "review":
                item = {
                "themeId": group["themeId"], "displayTheme": group["displayTheme"],
                "mentionCount": group["mentionCount"], "sentiment": group["sentiment"],
                "quoteEvidenceId": _exact_quote_id(completion.get("quote"), group["evidence"]),
                "summary": summary, "fallbackUsed": fallback_used,
                }
                (positive if group["sentiment"] == "positive" else negative).append(item)
            else:
                qna.append({
                    "themeId": group["themeId"], "displayTheme": group["displayTheme"],
                    "primaryCount": group["primaryCount"], "ratio": group["ratio"],
                    "questionEvidenceId": _exact_quote_id(completion.get("quote"), group["evidence"]),
                    "summary": summary, "fallbackUsed": fallback_used,
                })
            summaries.append({
                "task_type": task["task_type"], "theme_id": group["themeId"],
                "sentiment": task.get("sentiment"), "percentage": task.get("percentage"),
                "quote": completion.get("quote", ""), "summary": completion.get("summary", ""),
            })
        selling_task = {
            "task_id": report_input["analysisId"], "task_type": "selling_point",
            "analysis_id": report_input["analysisId"],
            "category_key": report_input["category"]["path"], "summary_items": summaries,
        }
        selling = self._complete(_selling_prompt(selling_task)) if summaries else {"selling_points": []}
        selling_points = selling.get("selling_points")
        if not isinstance(selling_points, list) or not selling_points:
            selling_points = _fallback_selling_points(report_input)
        return {
            "reviewAnalysis": {"positiveThemes": positive, "negativeThemes": negative},
            "qnaAnalysis": {"themes": qna},
            "sellingPoints": selling_points,
        }

    def _complete(self, user_prompt: str) -> dict[str, Any]:
        return self._complete_many([user_prompt])[0]

    def _complete_many(
        self, user_prompts: list[str], *, max_new_tokens: int | None = None,
        disable_adapter: bool = False,
    ) -> list[dict[str, Any]]:
        results = []
        for offset in range(0, len(user_prompts), self.batch_size):
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": value}],
                    tokenize=False, add_generation_prompt=True,
                )
                for value in user_prompts[offset : offset + self.batch_size]
            ]
            encoded = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_input_length,
            ).to(self.model.device)
            adapter_context = (
                self.model.disable_adapter()
                if disable_adapter and hasattr(self.model, "disable_adapter") else nullcontext()
            )
            with adapter_context:
                with self.torch.inference_mode():
                    generated = self.model.generate(
                        **encoded, do_sample=False,
                        max_new_tokens=max_new_tokens or self.max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
            width = encoded["input_ids"].shape[1]
            results.extend(
                _parse_json_or_empty(value)
                for value in self.tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
            )
        return results


def _verification_prompt(
    *, record_type: str, category_path: str, source_text: str,
    candidate: dict[str, Any], allowed_themes: list[dict[str, str]],
) -> str:
    catalog = [{
        "themeId": item["themeId"],
        "name": item["reviewName" if record_type == "review" else "qnaName"],
    } for item in allowed_themes]
    common = (
        "원문 한 건을 다시 읽고 encoder 후보를 검증하세요. 후보는 참고일 뿐이며 틀리면 삭제하거나 교정하세요. "
        "원문에 명시된 의미만 분류하고, allowed_themes의 themeId만 사용하세요. JSON 객체만 출력하세요."
        f"\nCATEGORY={category_path}\nTYPE={record_type}\nTEXT={source_text}"
        f"\nENCODER_CANDIDATE={json.dumps(candidate, ensure_ascii=False)}"
        f"\nALLOWED_THEMES={json.dumps(catalog, ensure_ascii=False)}\n"
    )
    if record_type == "review":
        return common + (
            "한 테마에 감정 하나만 지정하고 labels는 명시적으로 평가된 테마만 최대 3개입니다. "
            "sentiment는 positive, negative, neutral 중 하나입니다. 사용할 수 없는 문장이면 usable=false입니다.\n"
            'OUTPUT={"usable":true,"labels":[{"themeId":"...","sentiment":"positive"}]}'
        )
    return common + (
        "질문의 핵심 의도 하나를 primaryThemeId로 선택하세요. secondaryThemeIds는 실제로 함께 물은 의도만 최대 2개입니다. "
        "잡음이면 usable=false와 primaryThemeId=UNUSABLE을 출력하세요.\n"
        'OUTPUT={"usable":true,"primaryThemeId":"...","secondaryThemeIds":["..."]}'
    )


def _sanitize_verification(
    record_type: str, value: dict[str, Any], original: dict[str, Any],
    allowed_themes: list[dict[str, str]],
) -> dict[str, Any]:
    """Convert verifier JSON to encoder schema; malformed output safely falls back."""
    allowed = {str(item["themeId"]) for item in allowed_themes}
    name_key = "reviewName" if record_type == "review" else "qnaName"
    names = {str(item["themeId"]): str(item[name_key]) for item in allowed_themes}
    if not isinstance(value, dict) or not any(
        key in value for key in ("usable", "labels", "primaryThemeId", "primary_theme_id")
    ):
        return original
    usable = value.get("usable")
    if not isinstance(usable, bool):
        return original
    result = {"id": str(original["id"]), "usable": usable, "usableConfidence": 1.0}
    if record_type == "review":
        labels_value = value.get("labels")
        if not isinstance(labels_value, list):
            return original
        labels, seen = [], set()
        for item in labels_value:
            if not isinstance(item, dict):
                continue
            theme = str(item.get("themeId") or item.get("theme_id") or "")
            sentiment = str(item.get("sentiment") or "").lower()
            if (theme not in allowed or theme in SPECIAL_THEMES or theme in seen
                    or sentiment not in SENTIMENTS):
                continue
            seen.add(theme)
            labels.append({"themeId": theme, "theme": names.get(theme, theme),
                           "sentiment": sentiment, "confidence": 1.0})
            if len(labels) == 3:
                break
        result["labels"] = labels if usable else []
        return result
    primary = str(value.get("primaryThemeId") or value.get("primary_theme_id") or "")
    if not usable:
        primary = "UNUSABLE" if "UNUSABLE" in allowed else primary
    if primary not in allowed:
        return original
    result["primaryTheme"] = {"themeId": primary, "theme": names.get(primary, primary),
                              "confidence": 1.0}
    secondary_value = value.get("secondaryThemeIds", value.get("secondary_theme_ids", []))
    if not isinstance(secondary_value, list):
        secondary_value = []
    secondary, seen = [], {primary}
    for raw_theme in secondary_value:
        theme = str(raw_theme)
        if theme not in allowed or theme in SPECIAL_THEMES or theme in seen:
            continue
        seen.add(theme)
        secondary.append({"themeId": theme, "theme": names.get(theme, theme), "confidence": 1.0})
        if len(secondary) == 2:
            break
    result["secondaryThemes"] = secondary if usable else []
    return result

def _summary_task(report: dict[str, Any], group: dict[str, Any], task_type: str) -> dict[str, Any]:
    count = int(group.get("mentionCount", group.get("primaryCount", 0)))
    total = int(report["sample"]["reviewUsableSize" if task_type == "review_summary" else "qnaUsableSize"])
    return {
        "task_id": report["analysisId"], "task_type": task_type,
        "analysis_id": report["analysisId"], "category_key": report["category"]["path"],
        "theme_id": group["themeId"], "theme_name": group["displayTheme"],
        "sentiment": group.get("sentiment"), "group_count": count, "total_count": total,
        "percentage": round(count / max(total, 1) * 100, 1),
        "source_items": [
            {"sample_id": item["id"], "text": item["text"]} for item in group.get("evidence", [])
        ],
    }


def _summary_prompt(task: dict[str, Any]) -> str:
    sentiment = (
        f"sentiment는 반드시 {task['sentiment']}입니다."
        if task["task_type"] == "review_summary" else "sentiment는 null입니다."
    )
    return (
        "한국어 쇼핑 리뷰/Q&A 묶음을 PALDA 화면용으로 요약하세요. quote는 입력 text의 정확한 부분 문자열이어야 합니다. "
        "summary는 반복 패턴을 한두 문장으로 설명하고 입력에 없는 사실을 만들지 마세요. " + sentiment +
        f"\nINPUT={json.dumps(task, ensure_ascii=False)}\n"
        f'OUTPUT={{"theme_id":"{task["theme_id"]}","sentiment":{json.dumps(task.get("sentiment"))},"quote":"...","summary":"..."}}'
    )


def _selling_prompt(task: dict[str, Any]) -> str:
    return (
        "다음 테마별 분석만 사용해 셀러가 활용할 수 있는 핵심 소구점 1~3개를 만드세요. "
        "과장하거나 새로운 사실을 추가하지 마세요.\n"
        f"INPUT={json.dumps(task, ensure_ascii=False)}\n"
        'OUTPUT={"selling_points":[{"title":"...","message":"...","evidence_theme_ids":["..."]}]}'
    )


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("decoder returned no JSON object")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("decoder output must be a JSON object")
    return value


def _parse_json_or_empty(text: str) -> dict[str, Any]:
    try:
        return _parse_json(text)
    except (ValueError, json.JSONDecodeError):
        return {}


def _fallback_summary(group: dict[str, Any], kind: str) -> str:
    theme = str(group.get("displayTheme") or group.get("themeId") or "해당 테마")
    if kind == "qna":
        return f"{theme}가 확인됩니다." if theme.endswith("문의") else f"{theme}에 대한 문의가 확인됩니다."
    sentiment = "긍정적인" if group.get("sentiment") == "positive" else "부정적인"
    return f"{theme}에 대한 {sentiment} 평가가 확인됩니다."


def _exact_quote_id(quote: Any, evidence: list[dict[str, Any]]) -> str | None:
    value = str(quote or "")
    exact = next(
        (str(item["id"]) for item in evidence if value and value in str(item["text"])), None
    )
    # Never display the decoder's corrected or hallucinated wording. Falling
    # back to the backend-ranked evidence keeps the theme visible and the quote
    # exactly grounded even when the model normalizes a source typo.
    return exact or (str(evidence[0]["id"]) if evidence else None)


def _cat2_key(path: str) -> str:
    return " > ".join(part.strip() for part in path.split(">")[:2])


def _cat1(path: str) -> str:
    return path.split(">", 1)[0].strip()


def _cat2(path: str) -> str:
    parts = path.split(">", 1)
    return parts[1].strip() if len(parts) > 1 else ""


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
