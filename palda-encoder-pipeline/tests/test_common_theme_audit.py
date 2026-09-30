"""Structural audit checks; these are not semantic annotation accuracy tests."""
import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_common_themes.py"
SPEC = importlib.util.spec_from_file_location("common_theme_audit", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class CommonThemeAuditTests(unittest.TestCase):
    def test_verbatim_preserves_negation_and_condition(self):
        text = "예전에는 몰랐어요. 겨울에는 따뜻하지 않아요."
        result = AUDIT.span(text, "겨울에는 따뜻하지 않아요.")
        self.assertEqual(text[result["start"]:result["end"]], result["text"])

    def test_rewritten_quote_rejected(self):
        with self.assertRaises(ValueError):
            AUDIT.span("따뜻하지 않아요", "따뜻해요")

    def test_ambiguous_occurrence_rejected(self):
        with self.assertRaises(ValueError):
            AUDIT.span("좋아요. 좋아요.", "좋아요.")

    def test_codepoint_offsets_not_utf16(self):
        result = AUDIT.span("🙂 좋아요", "좋아요")
        self.assertEqual((result["start"], result["end"]), (2, 5))

    def test_deterministic_selection_and_filter(self):
        rows = [
            {"sample_id": str(i), "category_key": "식품 > 농산물", "record_type": kind, "text": text}
            for i, (kind, text) in enumerate([
                ("review", "맛있어요"), ("review", "별로예요"),
                ("review", "주소 확인"), ("review", "길" * 651),
                ("qna", "언제 오나요?"), ("qna", "몇 개예요?"),
            ])
        ]
        forward = AUDIT.selected_rows(rows, "seed")
        backward = AUDIT.selected_rows(list(reversed(rows)), "seed")
        self.assertEqual(forward, backward)
        self.assertEqual({r["sample_id"] for r in forward}, {"0", "1", "4", "5"})


if __name__ == "__main__":
    unittest.main()
