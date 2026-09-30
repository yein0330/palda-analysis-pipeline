from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from palda_decoder.audit import audit_release
from palda_decoder.io import iter_jsonl, write_json, write_jsonl
from palda_decoder.llm import _decode_chat_body, validate_completion
from palda_decoder.pipeline import run_pipeline
from palda_decoder.prepare import prepare_insight_tasks, prepare_summary_tasks
from palda_decoder.release import create_training_release
from palda_decoder.rewards import (
    concise_reward, format_reward, grounding_reward, label_consistency_reward,
)
from palda_decoder.select import select_candidates


class DecoderPipelineTest(unittest.TestCase):
    def test_chat_response_supports_json_and_sse(self) -> None:
        standard = json.dumps({"choices": [{"message": {"content": "{\"ok\":true}"}}]})
        stream = "\n\n".join([
            'data: {"choices":[{"delta":{"content":"{\\"ok\\":"}}]}',
            'data: {"choices":[{"delta":{"content":"true}"}}]}',
            "data: [DONE]",
        ])
        self.assertEqual(_decode_chat_body(standard), '{"ok":true}')
        self.assertEqual(_decode_chat_body(stream, "text/event-stream"), '{"ok":true}')

    def test_prepare_separates_review_sentiment_and_qna_percentage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                self._row("r1", "review", "연결이 빨라요", "CONNECT", "positive"),
                self._row("r2", "review", "연결이 자꾸 끊겨요", "CONNECT", "negative"),
                self._row("q1", "qna", "언제 배송되나요?", "DELIVERY", None),
                self._row("q2", "qna", "주말 전에 오나요?", "DELIVERY", None),
                self._row("q3", "qna", "XS 이어팁이 있나요?", "SIZE", None),
            ]
            write_jsonl(root / "input.jsonl", rows)
            report = prepare_summary_tasks([root / "input.jsonl"], root / "tasks.jsonl")
            tasks = list(iter_jsonl(root / "tasks.jsonl"))
            review = [task for task in tasks if task["task_type"] == "review_summary"]
            delivery = next(task for task in tasks if task["theme_id"] == "DELIVERY")
            self.assertEqual({task["sentiment"] for task in review}, {"positive", "negative"})
            self.assertEqual(delivery["percentage"], 66.7)
            self.assertEqual(report["tasks"], 4)

    def test_completion_requires_grounded_quote(self) -> None:
        task = {
            "task_type": "review_summary",
            "theme_id": "CONNECT",
            "sentiment": "positive",
            "source_items": [{"text": "연결이 빨라요"}],
        }
        valid = {"theme_id": "CONNECT", "sentiment": "positive", "quote": "연결이 빨라요", "summary": "초기 연결이 빠르다는 평가가 반복됩니다."}
        invalid = {**valid, "quote": "배터리가 오래가요"}
        self.assertEqual(validate_completion(task, valid), [])
        self.assertIn("quote_not_grounded", validate_completion(task, invalid))

    def test_positive_selling_point_with_complaint_content_is_rejected(self) -> None:
        task = {"task_type": "selling_point", "summary_items": [{"theme_id": "USABILITY"}, {"theme_id": "DIMENSIONS"}]}
        mislabeled = {"selling_points": [{"title": "숨은 강점", "message": "신발 모양에 따라 들어가지 않는 문제가 발생합니다.", "evidence_theme_ids": ["DIMENSIONS"]}]}
        barrier = {"selling_points": [{"title": "구매 장벽", "message": "신발 모양에 따라 들어가지 않는 문제가 발생합니다.", "evidence_theme_ids": ["DIMENSIONS"]}]}
        strength = {"selling_points": [{"title": "핵심 소구점", "message": "슬림한 디자인이라 좁은 현관에도 둘 수 있습니다.", "evidence_theme_ids": ["USABILITY"]}]}
        self.assertIn("selling_point_type_content_mismatch", validate_completion(task, mislabeled))
        self.assertEqual(validate_completion(task, barrier), [])
        self.assertEqual(validate_completion(task, strength), [])

    def test_selection_uses_independent_judges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = {"task_id": "t1", "task_type": "review_summary", "analysis_id": "p1", "category_key": "이어폰", "theme_id": "CONNECT", "sentiment": "positive"}
            write_jsonl(root / "tasks.jsonl", [task])
            write_jsonl(
                root / "candidates.jsonl",
                [
                    {"task_id": "t1", "model_id": "a", "valid": True, "completion": {"summary": "A", "quote": "Q"}},
                    {"task_id": "t1", "model_id": "b", "valid": True, "completion": {"summary": "B", "quote": "Q"}},
                ],
            )
            write_jsonl(
                root / "judgments.jsonl",
                [
                    {"task_id": "t1", "candidate_model_id": "a", "judge_model_id": "b", "valid": True, "scores": {"grounding": 0.8, "coverage": 0.8, "specificity": 0.8, "style": 0.8, "hallucination": False}},
                    {"task_id": "t1", "candidate_model_id": "b", "judge_model_id": "a", "valid": True, "scores": {"grounding": 0.6, "coverage": 0.6, "specificity": 0.6, "style": 0.6, "hallucination": False}},
                ],
            )
            report = select_candidates(root / "tasks.jsonl", root / "candidates.jsonl", root / "selected.jsonl", judgments_path=root / "judgments.jsonl")
            selected = next(iter_jsonl(root / "selected.jsonl"))
            self.assertEqual(report["accepted"], 1)
            self.assertEqual(selected["generator_model_id"], "a")

    def test_hallucination_flag_disqualifies_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = {"task_id": "t1", "task_type": "review_summary", "analysis_id": "p1", "category_key": "이어폰", "theme_id": "CONNECT", "sentiment": "positive"}
            write_jsonl(root / "tasks.jsonl", [task])
            write_jsonl(
                root / "candidates.jsonl",
                [
                    {"task_id": "t1", "model_id": "a", "valid": True, "completion": {"summary": "A", "quote": "Q"}},
                    {"task_id": "t1", "model_id": "b", "valid": True, "completion": {"summary": "B", "quote": "Q"}},
                ],
            )
            # "a" scores higher on the judge that did not flag it. Dropping the
            # flagged vote would leave it with a 0.95 average and win the task.
            write_jsonl(
                root / "judgments.jsonl",
                [
                    {"task_id": "t1", "candidate_model_id": "a", "judge_model_id": "b", "valid": True, "scores": {"grounding": 0.95, "coverage": 0.95, "specificity": 0.95, "style": 0.95, "hallucination": False}},
                    {"task_id": "t1", "candidate_model_id": "a", "judge_model_id": "c", "valid": True, "scores": {"grounding": 0.1, "coverage": 0.1, "specificity": 0.1, "style": 0.1, "hallucination": True}},
                    {"task_id": "t1", "candidate_model_id": "b", "judge_model_id": "a", "valid": True, "scores": {"grounding": 0.7, "coverage": 0.7, "specificity": 0.7, "style": 0.7, "hallucination": False}},
                    {"task_id": "t1", "candidate_model_id": "b", "judge_model_id": "c", "valid": True, "scores": {"grounding": 0.7, "coverage": 0.7, "specificity": 0.7, "style": 0.7, "hallucination": False}},
                ],
            )
            select_candidates(root / "tasks.jsonl", root / "candidates.jsonl", root / "selected.jsonl", judgments_path=root / "judgments.jsonl")
            selected = next(iter_jsonl(root / "selected.jsonl"))
            self.assertEqual(selected["generator_model_id"], "b")
            self.assertEqual(selected["status"], "accepted")
            self.assertEqual(selected["judge_count"], 2)

    def test_hallucinated_sole_candidate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = {"task_id": "t1", "task_type": "review_summary", "analysis_id": "p1", "category_key": "이어폰", "theme_id": "CONNECT", "sentiment": "positive"}
            write_jsonl(root / "tasks.jsonl", [task])
            write_jsonl(
                root / "candidates.jsonl",
                [{"task_id": "t1", "model_id": "a", "valid": True, "completion": {"summary": "A", "quote": "Q"}}],
            )
            write_jsonl(
                root / "judgments.jsonl",
                [
                    {"task_id": "t1", "candidate_model_id": "a", "judge_model_id": "b", "valid": True, "scores": {"grounding": 0.9, "coverage": 0.9, "specificity": 0.9, "style": 0.9, "hallucination": False}},
                    {"task_id": "t1", "candidate_model_id": "a", "judge_model_id": "c", "valid": True, "scores": {"grounding": 0.9, "coverage": 0.9, "specificity": 0.9, "style": 0.9, "hallucination": True}},
                ],
            )
            report = select_candidates(root / "tasks.jsonl", root / "candidates.jsonl", root / "selected.jsonl", judgments_path=root / "judgments.jsonl")
            selected = next(iter_jsonl(root / "selected.jsonl"))
            self.assertEqual(report["accepted"], 0)
            self.assertEqual(selected["quality_score"], 0.0)
            self.assertIn("hallucination_flagged", selected["reasons"])

    def test_insights_release_and_rewards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = {
                "task_id": "t1",
                "status": "accepted",
                "quality_score": 0.9,
                "task": {"task_id": "t1", "task_type": "review_summary", "analysis_id": "p1", "category_key": "이어폰", "theme_id": "CONNECT", "sentiment": "positive"},
                "completion": {"theme_id": "CONNECT", "sentiment": "positive", "quote": "연결이 빨라요", "summary": "초기 연결이 빠르다는 평가가 반복됩니다."},
            }
            write_jsonl(root / "selected.jsonl", [selected])
            prepare_insight_tasks(root / "selected.jsonl", root / "insights.jsonl")
            insight = next(iter_jsonl(root / "insights.jsonl"))
            self.assertEqual(insight["summary_items"][0]["theme_id"], "CONNECT")
            manifest = create_training_release([root / "selected.jsonl"], root / "release")
            self.assertEqual(manifest["examples"], 1)
            completion = json.dumps(selected["completion"], ensure_ascii=False)
            self.assertEqual(format_reward([completion], ["review_summary"]), [1.0])
            self.assertEqual(concise_reward([completion], ["review_summary"]), [1.0])
            self.assertEqual(grounding_reward([completion], ["review_summary"], [["연결이 빨라요"]], [[]]), [1.0])
            self.assertEqual(label_consistency_reward(
                [completion], [completion], ["review_summary"]
            ), [1.0])

    def test_release_audit_checks_grounding_and_split_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            example = {
                "example_id": "e1",
                "analysis_id": "p1",
                "task_type": "review_summary",
                "source_texts": ["연결이 빨라요"],
                "allowed_theme_ids": ["CONNECT"],
                "reference_completion": json.dumps({
                    "theme_id": "CONNECT", "sentiment": "positive",
                    "quote": "연결이 빨라요", "summary": "연결이 빠르다는 평가입니다.",
                }, ensure_ascii=False),
            }
            write_jsonl(root / "train.jsonl", [example])
            write_jsonl(root / "validation.jsonl", [{**example, "example_id": "e2", "analysis_id": "p2"}])
            write_jsonl(root / "test.jsonl", [{**example, "example_id": "e3", "analysis_id": "p3"}])
            report = audit_release(root)
            self.assertTrue(report["hard_gate_passed"])
            self.assertEqual(report["quality"]["validation"]["grounding_pass_rate"], 1.0)

    def test_end_to_end_data_pipeline_with_mock_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "input.jsonl", [self._row("r1", "review", "연결이 빨라서 편해요", "CONNECT", "positive")])
            write_json(
                root / "models.json",
                {"models": [
                    {"id": "a", "model": "a", "base_url": "http://unused"},
                    {"id": "b", "model": "b", "base_url": "http://unused"},
                    {"id": "c", "model": "c", "base_url": "http://unused"},
                ]},
            )
            write_json(
                root / "config.json",
                {
                    "input_jsonl_paths": ["input.jsonl"],
                    "models_path": "models.json",
                    "work_dir": "work",
                    "release_dir": "release",
                    "artifact_dir": "artifact",
                    "train": {"sft_enabled": False},
                },
            )

            def fake_chat(_endpoint: object, prompt: str, _retries: int, **_: object) -> str:
                if "평가하세요" in prompt:
                    return json.dumps({"grounding": 0.9, "coverage": 0.9, "specificity": 0.9, "style": 0.9, "hallucination": False})
                if "PALDA 셀러용 소구점" in prompt:
                    return json.dumps({"selling_points": [{"title": "빠른 연결", "message": "꺼내자마자 빠르게 연결되는 편의성을 강조합니다.", "evidence_theme_ids": ["CONNECT"]}]}, ensure_ascii=False)
                return json.dumps({"theme_id": "CONNECT", "sentiment": "positive", "quote": "연결이 빨라서 편해요", "summary": "초기 연결이 빠르고 사용이 편하다는 평가가 반복됩니다."}, ensure_ascii=False)

            with patch("palda_decoder.llm._chat", side_effect=fake_chat):
                summary = run_pipeline(root / "config.json", skip_training=True)
            self.assertEqual(summary["select_summaries"]["accepted"], 1)
            self.assertEqual(summary["select_insights"]["accepted"], 1)
            self.assertEqual(summary["release"]["examples"], 2)
            payload = next(iter_jsonl(root / "work/service_payloads.jsonl"))
            self.assertEqual(payload["reviews"]["positive"][0]["theme_id"], "CONNECT")
            self.assertEqual(payload["selling_points"][0]["title"], "빠른 연결")

    @staticmethod
    def _row(sample_id: str, record_type: str, text: str, theme: str, sentiment: str | None) -> dict[str, object]:
        return {
            "sample_id": sample_id,
            "analysis_id": "product-1",
            "product_id": "product-1",
            "record_type": record_type,
            "text": text,
            "category_key": "가전 > 이어폰",
            "agreement_score": 0.9,
            "labels": [{"theme_id": theme, "sentiment": sentiment, "evidence": text}],
        }


if __name__ == "__main__":
    unittest.main()
