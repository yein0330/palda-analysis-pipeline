from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from palda_encoder.consensus import build_consensus
from palda_encoder.benchmark import build_human_benchmark
from palda_encoder.cli import run_pipeline
from palda_encoder.generation_plan import plan_tail_generation
from palda_encoder.io import iter_jsonl, write_json, write_jsonl
from palda_encoder.iteration import run_iterations
from palda_encoder.llm import (
    ModelEndpoint,
    _decode_chat_body,
    _extract_json,
    _generate_one,
    _parse_labels,
    label_samples,
)
from palda_encoder.normalize import normalize_annotator_csvs
from palda_encoder.release import create_isolated_release, create_release
from palda_encoder.strategy_plan import build_strategy_plans


class PipelineTest(unittest.TestCase):
    def test_qna_labels_are_canonicalized_and_deduplicated(self) -> None:
        labels = _parse_labels(
            '{"labels":['
            '{"theme_id":"DELIVERY","sentiment":"positive","role":"primary"},'
            '{"theme_id":"DELIVERY","sentiment":null}]}'
            , record_type="qna"
        )
        self.assertEqual(len(labels), 1)
        self.assertIsNone(labels[0].sentiment)
        self.assertEqual(labels[0].role, "primary")

    def test_review_allows_opposite_sentiments_for_same_theme(self) -> None:
        labels = _parse_labels(
            '{"labels":['
            '{"theme_id":"PRICE","sentiment":"positive"},'
            '{"theme_id":"PRICE","sentiment":"negative"}]}',
            record_type="review",
        )
        self.assertEqual([(label.theme_id, label.sentiment) for label in labels], [
            ("PRICE", "positive"), ("PRICE", "negative")
        ])

    def test_human_benchmark_keeps_products_and_duplicates_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for index in range(40):
                rows.append({
                    "itemId": f"r{index}",
                    "productGroupId": f"p{index}",
                    "sourceType": "review",
                    "category": {
                        "level1Id": "1", "level2Id": "2",
                        "level1Name": "가전", "level2Name": "이어폰",
                        "path": "가전 > 이어폰 > 무선",
                    },
                    "text": "같은 문장" if index in (0, 1) else f"문장 {index}",
                    "usable": True,
                    "labels": {"reviewLabels": [{"themeId": "QUALITY", "sentiment": "positive"}]},
                    "annotation": {"grade": "gold", "validationStatus": "valid", "source": "human"},
                })
            write_jsonl(root / "source.jsonl", rows)
            write_json(root / "taxonomy.json", {
                "categories": [{
                    "level1Id": "1", "level2Id": "2", "path": "가전 > 이어폰",
                    "allowedThemes": [{"canonicalThemeId": "QUALITY", "annotationReady": True}],
                }]
            })
            manifest = build_human_benchmark(
                root / "source.jsonl", root / "taxonomy.json", root / "benchmark", seed=9
            )
            locations = {}
            texts = {}
            for split in ("simulation", "validation", "test"):
                for row in iter_jsonl(root / f"benchmark/{split}.jsonl"):
                    locations[row["product_id"]] = split
                    self.assertNotIn(row["text"], texts)
                    texts[row["text"]] = split
            self.assertEqual(locations["p0"], locations.get("p1", locations["p0"]))
            self.assertEqual(manifest["leakage"], {
                "product_simulation_validation": 0,
                "product_simulation_test": 0,
                "product_validation_test": 0,
                "text_simulation_validation": 0,
                "text_simulation_test": 0,
                "text_validation_test": 0,
            })

            write_jsonl(root / "train-samples.jsonl", [
                {
                    "sample_id": "blocked", "product_id": "p0", "record_type": "review",
                    "text": "새 학습 문장", "category_path": "가전 > 이어폰",
                    "category_key": "가전 > 이어폰", "usable": True,
                },
                {
                    "sample_id": "safe", "product_id": "p100", "record_type": "review",
                    "text": "안전한 학습 문장", "category_path": "가전 > 이어폰",
                    "category_key": "가전 > 이어폰", "usable": True,
                },
            ])
            write_jsonl(root / "consensus.jsonl", [
                {"sample_id": value, "status": "accepted", "labels": [
                    {"theme_id": "QUALITY", "sentiment": "positive", "evidence": None}
                ]}
                for value in ("blocked", "safe")
            ])
            release = create_isolated_release(
                root / "train-samples.jsonl", root / "consensus.jsonl",
                root / "benchmark", root / "release", taxonomy_path=root / "taxonomy.json",
            )
            self.assertEqual(release["counts"]["train"], 1)
            self.assertEqual(release["excluded_train"]["benchmark_product"], 1)
            self.assertEqual(release["simulation_benchmark"]["included_in_release"], False)

    def test_chat_response_supports_json_and_sse(self) -> None:
        standard = json.dumps({"choices": [{"message": {"content": "{\"ok\":true}"}}]})
        stream = "\n\n".join([
            'data: {"choices":[{"delta":{"content":"{\\"ok\\":"}}]}',
            'data: {"choices":[{"delta":{"content":"true}"}}]}',
            "data: [DONE]",
        ])
        self.assertEqual(_decode_chat_body(standard), '{"ok":true}')
        self.assertEqual(_decode_chat_body(stream, "text/event-stream"), '{"ok":true}')

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        fields = [
            "reviewer_id",
            "annotation_id",
            "product_id",
            "source_split",
            "category_path",
            "record_type",
            "text",
            "usable",
            "review_labels_json",
            "qna_primary_theme_id",
            "qna_secondary_theme_ids_json",
            "model_confidence",
            "validation_status",
            "reviewer_status",
            "reviewer_notes",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_normalize_legacy_review_and_qna(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "review.csv"
            self._write_csv(
                csv_path,
                [
                    {
                        "reviewer_id": "a1",
                        "annotation_id": "r1",
                        "product_id": "p1",
                        "source_split": "train",
                        "category_path": "가전 > 이어폰",
                        "record_type": "review",
                        "text": "연결이 빨라요",
                        "usable": "true",
                        "review_labels_json": json.dumps(
                            [{"themeId": "CONNECT", "sentiment": "positive"}]
                        ),
                    },
                    {
                        "reviewer_id": "a1",
                        "annotation_id": "q1",
                        "product_id": "p2",
                        "source_split": "test",
                        "category_path": "가전 > 이어폰",
                        "record_type": "qna",
                        "text": "언제 오나요?",
                        "usable": "true",
                        "review_labels_json": "[]",
                        "qna_primary_theme_id": "DELIVERY",
                        "qna_secondary_theme_ids_json": "[]",
                    },
                ],
            )
            report = normalize_annotator_csvs([csv_path], root / "out")
            self.assertEqual(report["samples"], 2)
            taxonomy = json.loads((root / "out/taxonomy.derived.json").read_text())
            self.assertEqual({item["id"] for item in taxonomy["themes"]}, {"CONNECT", "DELIVERY"})

    def test_two_of_three_consensus_and_product_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = [
                {
                    "sample_id": f"s{i}",
                    "product_id": "same-product" if i < 2 else f"p{i}",
                    "record_type": "review",
                    "text": f"연결이 빨라요 {i}",
                    "category_path": "가전 > 이어폰",
                    "source_kind": "real",
                    "usable": True,
                    "metadata": {},
                }
                for i in range(10)
            ]
            write_jsonl(root / "samples.jsonl", samples)
            write_json(
                root / "taxonomy.json",
                {
                    "version": "v1",
                    "themes": [{"id": "CONNECT"}, {"id": "DESIGN"}],
                    "category_themes": {"가전 > 이어폰": ["CONNECT", "DESIGN"]},
                },
            )
            runs = []
            for sample in samples:
                for model, theme in (("kanana", "CONNECT"), ("llama", "CONNECT"), ("qwen", "DESIGN")):
                    runs.append(
                        {
                            "sample_id": sample["sample_id"],
                            "model_id": model,
                            "valid": True,
                            "labels": [{"theme_id": theme, "sentiment": "positive", "evidence": None}],
                        }
                    )
            write_jsonl(root / "runs.jsonl", runs)
            report = build_consensus(
                root / "samples.jsonl",
                [root / "runs.jsonl"],
                root / "taxonomy.json",
                root / "consensus.jsonl",
                root / "hard.jsonl",
            )
            self.assertEqual(report["accepted"], 10)
            first = next(iter_jsonl(root / "consensus.jsonl"))
            self.assertEqual(first["labels"][0]["theme_id"], "CONNECT")
            manifest = create_release(
                root / "samples.jsonl",
                root / "consensus.jsonl",
                root / "release",
                taxonomy_path=root / "taxonomy.json",
                seed=7,
            )
            split_by_sample = {}
            for split in ("train", "validation", "test"):
                for row in iter_jsonl(root / f"release/{split}.jsonl"):
                    split_by_sample[row["sample_id"]] = split
                    self.assertEqual(row["candidate_theme_ids"], ["CONNECT", "DESIGN"])
            self.assertEqual(split_by_sample["s0"], split_by_sample["s1"])
            self.assertEqual(sum(manifest["counts"].values()), 10)

    def test_qna_consensus_assigns_one_primary_and_secondary_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "samples.jsonl", [{
                "sample_id": "q1", "product_id": "p1", "record_type": "qna",
                "text": "언제 오고 반품도 되나요?", "category_path": "가전 > 이어폰",
                "category_key": "가전 > 이어폰", "usable": True,
            }])
            write_json(root / "taxonomy.json", {
                "version": "v1", "themes": [{"id": "DELIVERY"}, {"id": "RETURN_EXCHANGE"}],
                "category_themes": {"가전 > 이어폰": ["DELIVERY", "RETURN_EXCHANGE"]},
            })
            runs = []
            for model, primary in (("kanana", "DELIVERY"), ("llama", "DELIVERY"), ("qwen", "RETURN_EXCHANGE")):
                secondary = "RETURN_EXCHANGE" if primary == "DELIVERY" else "DELIVERY"
                runs.append({
                    "sample_id": "q1", "model_id": model, "valid": True,
                    "labels": [
                        {"theme_id": primary, "sentiment": None, "role": "primary"},
                        {"theme_id": secondary, "sentiment": None, "role": "secondary"},
                    ],
                })
            write_jsonl(root / "runs.jsonl", runs)
            build_consensus(
                root / "samples.jsonl", [root / "runs.jsonl"], root / "taxonomy.json",
                root / "consensus.jsonl", root / "hard.jsonl",
            )
            labels = next(iter_jsonl(root / "consensus.jsonl"))["labels"]
            self.assertEqual(sum(label["role"] == "primary" for label in labels), 1)
            self.assertEqual(next(label["theme_id"] for label in labels if label["role"] == "primary"), "DELIVERY")

    def test_synthetic_generator_claim_is_not_a_consensus_vote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "samples.jsonl", [{
                "sample_id": "syn1", "product_id": "syn1", "record_type": "review",
                "text": "허리를 잘 잡아줘요", "category_path": "패션 > 언더웨어",
                "category_key": "패션 > 언더웨어", "source_kind": "synthetic",
                "usable": True, "metadata": {"generator_model_id": "qwen"},
            }])
            write_json(root / "taxonomy.json", {
                "version": "v1", "themes": [{"id": "FIT"}, {"id": "QUALITY"}],
                "category_themes": {"패션 > 언더웨어": ["FIT", "QUALITY"]},
            })
            write_jsonl(root / "runs.jsonl", [
                {"sample_id": "syn1", "model_id": "qwen", "source": "generator_claim",
                 "valid": True, "labels": [{"theme_id": "FIT", "sentiment": "positive"}]},
                {"sample_id": "syn1", "model_id": "kanana", "source": "llm_labeler",
                 "valid": True, "labels": [{"theme_id": "FIT", "sentiment": "positive"}]},
                {"sample_id": "syn1", "model_id": "llama", "source": "llm_labeler",
                 "valid": True, "labels": [{"theme_id": "QUALITY", "sentiment": "positive"}]},
            ])
            report = build_consensus(
                root / "samples.jsonl", [root / "runs.jsonl"], root / "taxonomy.json",
                root / "consensus.jsonl", root / "hard.jsonl", min_models=2,
            )
            self.assertEqual(report["accepted"], 0)
            self.assertEqual(report["review_needed"], 1)

    def test_synthetic_samples_are_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = [
                {
                    "sample_id": "real-1",
                    "product_id": "real-1",
                    "record_type": "review",
                    "text": "실제 리뷰",
                    "category_key": "가전 > 이어폰",
                    "source_kind": "real",
                    "usable": True,
                },
                {
                    "sample_id": "syn-1",
                    "product_id": "syn-1",
                    "record_type": "review",
                    "text": "합성 리뷰",
                    "category_key": "가전 > 이어폰",
                    "source_kind": "synthetic",
                    "usable": True,
                },
            ]
            write_jsonl(root / "samples.jsonl", samples)
            write_jsonl(
                root / "consensus.jsonl",
                [
                    {
                        "sample_id": row["sample_id"],
                        "status": "accepted",
                        "labels": [{"theme_id": "CONNECT", "sentiment": "positive"}],
                    }
                    for row in samples
                ],
            )
            manifest = create_release(
                root / "samples.jsonl", root / "consensus.jsonl", root / "release"
            )
            train_ids = {row["sample_id"] for row in iter_jsonl(root / "release/train.jsonl")}
            validation_ids = {
                row["sample_id"] for row in iter_jsonl(root / "release/validation.jsonl")
            }
            test_ids = {row["sample_id"] for row in iter_jsonl(root / "release/test.jsonl")}
            self.assertIn("syn-1", train_ids)
            self.assertNotIn("syn-1", validation_ids | test_ids)
            self.assertEqual(manifest["source_kinds"], {"real": 1, "synthetic": 1})

    def test_extract_json_from_fence(self) -> None:
        value = _extract_json('```json\n{"labels": []}\n```')
        self.assertEqual(value, {"labels": []})

    def test_generation_retries_are_not_multiplied_by_chat_retries(self) -> None:
        endpoint = ModelEndpoint(id="qwen", model="q", url="http://unused")
        target = {
            "record_type": "review",
            "category_path": "가전 > 이어폰",
            "theme_id": "QUALITY",
            "sentiment": "positive",
        }
        with patch(
            "palda_encoder.llm._chat",
            side_effect=['{"text":""}', '{"text":"좋아요"}'],
        ) as chat:
            self.assertEqual(_generate_one(target, endpoint, 1, 7, {"themes": []}), "좋아요")
        self.assertEqual(chat.call_count, 2)
        self.assertTrue(all(call.kwargs["retries"] == 0 for call in chat.call_args_list))

    def test_strategy_plans_are_bounded_and_source_transform_is_anchored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "release").mkdir()
            write_json(root / "release/taxonomy.json", {
                "themes": [{"id": "FIT"}, {"id": "COMFORT"}],
                "category_themes": {"패션 > 속옷": ["FIT", "COMFORT"]},
            })
            write_jsonl(root / "release/train.jsonl", [
                {"sample_id": f"s{i}", "record_type": "review",
                 "text": f"허리를 잘 잡아줘요 {i}", "category_key": "패션 > 속옷",
                 "category_path": "패션 > 속옷 > 코르셋",
                 "labels": [{"theme_id": "FIT", "sentiment": "positive"}]}
                for i in range(4)
            ])
            report = build_strategy_plans(
                root / "release", root / "plans", max_per_strategy=3, seed=9
            )
            self.assertEqual(set(report["strategies"]), {
                "plain", "attributed", "contrastive", "source_transform"
            })
            for strategy in report["strategies"]:
                self.assertEqual(report["strategies"][strategy]["samples"], 3)
            transformed = list(iter_jsonl(root / "plans/source_transform.jsonl"))
            self.assertTrue(all(row.get("source_text") for row in transformed))
            contrasted = list(iter_jsonl(root / "plans/contrastive.jsonl"))
            self.assertTrue(all(row["forbidden_theme_ids"] == ["COMFORT"] for row in contrasted))

    def test_tail_generation_plan_is_budget_capped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(
                root / "samples.jsonl",
                [
                    {
                        "sample_id": "r1",
                        "record_type": "review",
                        "category_key": "가전 > 이어폰",
                        "category_path": "가전 > 이어폰 > 무선",
                    }
                ],
            )
            write_jsonl(
                root / "consensus.jsonl",
                [
                    {
                        "sample_id": "r1",
                        "status": "accepted",
                        "labels": [{"theme_id": "CONNECT", "sentiment": "positive"}],
                    }
                ],
            )
            report = plan_tail_generation(
                root / "samples.jsonl",
                root / "consensus.jsonl",
                root / "plan.jsonl",
                target_per_combination=10,
                max_per_combination=5,
                max_total=7,
            )
            rows = list(iter_jsonl(root / "plan.jsonl"))
            self.assertEqual(report["planned_samples"], 7)
            self.assertEqual({row["sentiment"] for row in rows}, {"positive", "negative"})

    def test_multi_model_labeling_skips_generator_self_vote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(
                root / "samples.jsonl",
                [
                    {
                        "sample_id": "syn1",
                        "product_id": "syn1",
                        "record_type": "review",
                        "text": "연결이 빨라요",
                        "category_path": "가전 > 이어폰 > 무선",
                        "category_key": "가전 > 이어폰",
                        "source_kind": "synthetic",
                        "usable": True,
                        "metadata": {"generator_model_id": "qwen"},
                    }
                ],
            )
            write_json(
                root / "taxonomy.json",
                {
                    "version": "v1",
                    "themes": [{"id": "CONNECT"}],
                    "category_themes": {"가전 > 이어폰": ["CONNECT"]},
                },
            )
            write_json(
                root / "models.json",
                {
                    "models": [
                        {"id": "kanana", "model": "k", "base_url": "http://unused"},
                        {"id": "llama", "model": "l", "base_url": "http://unused"},
                        {"id": "qwen", "model": "q", "base_url": "http://unused"},
                    ]
                },
            )
            response = json.dumps(
                {
                    "labels": [
                        {
                            "theme_id": "CONNECT",
                            "sentiment": "positive",
                            "evidence": "연결이 빨라요",
                        }
                    ]
                },
                ensure_ascii=False,
            )
            with patch("palda_encoder.llm._chat", return_value=response):
                report = label_samples(
                    root / "samples.jsonl",
                    root / "taxonomy.json",
                    root / "models.json",
                    root / "runs.jsonl",
                    max_workers=2,
                )
            self.assertEqual(report["valid"], 2)
            self.assertEqual(
                {row["model_id"] for row in iter_jsonl(root / "runs.jsonl")},
                {"kanana", "llama"},
            )

    def test_one_command_data_pipeline_with_mock_labelers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_csv(
                root / "input.csv",
                [
                    {
                        "reviewer_id": "old",
                        "annotation_id": f"r{i}",
                        "product_id": f"p{i}",
                        "source_split": "train",
                        "category_path": "가전 > 이어폰 > 무선",
                        "record_type": "review",
                        "text": f"연결이 빨라요 {i}",
                        "usable": "true",
                        "review_labels_json": json.dumps(
                            [{"themeId": "CONNECT", "sentiment": "positive"}]
                        ),
                    }
                    for i in range(20)
                ],
            )
            write_json(
                root / "models.json",
                {
                    "models": [
                        {"id": "kanana", "model": "k", "base_url": "http://unused"},
                        {"id": "llama", "model": "l", "base_url": "http://unused"},
                    ]
                },
            )
            write_json(
                root / "pipeline.json",
                {
                    "input_csv_globs": ["input.csv"],
                    "work_dir": "work",
                    "release_dir": "release",
                    "artifact_dir": "artifacts",
                    "models_path": "models.json",
                    "min_models": 2,
                    "train": {"enabled": False},
                },
            )
            response = json.dumps(
                {
                    "labels": [
                        {"theme_id": "CONNECT", "sentiment": "positive", "evidence": None}
                    ]
                }
            )
            with patch("palda_encoder.llm._chat", return_value=response):
                summary = run_pipeline(root / "pipeline.json", skip_train=True)
            self.assertEqual(summary["consensus"]["accepted"], 20)
            self.assertEqual(sum(summary["release"]["counts"].values()), 20)

    def test_iteration_promotes_baseline_and_stops_after_patience(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv = root / "input.csv"
            input_csv.write_text("placeholder", encoding="utf-8")
            write_json(
                root / "pipeline.json",
                {
                    "input_csv_globs": ["input.csv"],
                    "work_dir": "unused-work",
                    "release_dir": "unused-release",
                    "artifact_dir": "unused-artifact",
                    "train": {"enabled": True},
                    "auto_generation": {"enabled": False, "target_per_combination": 20},
                    "iteration": {
                        "root_dir": "iterations",
                        "max_rounds": 5,
                        "patience": 2,
                        "min_improvement": 0.002,
                    },
                },
            )
            scores = [0.7, 0.701, 0.7005]
            generated_flags = []

            def fake_runner(config_path: Path, **_: object) -> dict[str, object]:
                config = json.loads(Path(config_path).read_text(encoding="utf-8"))
                generated_flags.append(config["auto_generation"]["enabled"])
                self.assertEqual(config["split_seed"], 42)
                self.assertEqual(config["seed"], 42)
                self.assertEqual(config["generation_seed"], 42 + len(generated_flags) - 1)
                self.assertEqual(config["train"]["seed"], 42 + len(generated_flags) - 1)
                score = scores[len(generated_flags) - 1]
                return {
                    "release": {
                        "counts": {"train": 80, "validation": 10, "test": 10},
                        "source_kinds": {"real": 90, "synthetic": 10},
                    },
                    "train": {
                        "validation": {
                            "theme_macro_f1": score,
                            "tail_theme_macro_f1": 0.6,
                            "sentiment_accuracy_on_gold_themes": 0.8,
                            "samples_per_second": 100.0,
                        },
                        "test": {
                            "theme_macro_f1": score - 0.01,
                            "tail_theme_macro_f1": 0.6,
                            "sentiment_accuracy_on_gold_themes": 0.8,
                            "samples_per_second": 100.0,
                        },
                        "evaluation_label_tiers": {
                            "validation": {"gold": 10},
                            "test": {"gold": 10},
                        },
                    },
                }

            state = run_iterations(root / "pipeline.json", pipeline_runner=fake_runner)
            self.assertEqual(state["stop_reason"], "no_meaningful_improvement")
            self.assertEqual(len(state["rounds"]), 3)
            self.assertEqual(state["champion"]["round"], 0)
            self.assertEqual(generated_flags, [False, True, True])
            resumed = run_iterations(
                root / "pipeline.json", resume=True, pipeline_runner=fake_runner
            )
            self.assertEqual(resumed["champion"]["round"], 0)


if __name__ == "__main__":
    unittest.main()
