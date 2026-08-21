"""Unit test cho fusion score-gating, TRAKE monotonic và QA refinement."""

import os
from pathlib import Path
import sys
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from fusion import rrf_fuse  # noqa: E402
from search_types import SearchResult  # noqa: E402
from trake_engine import enforce_increasing_frames  # noqa: E402
from qa_engine import (  # noqa: E402
    is_text_question,
    refine_vqa_question,
)


def make_result(vector_index, frame_index, **scores):
    return SearchResult(
        vector_index=vector_index,
        frame_index=frame_index,
        **scores,
    )


class RrfScoreGatingTests(unittest.TestCase):
    def setUp(self):
        # Tránh env của máy chạy test can thiệp kết quả.
        self._old_gate = os.environ.pop("AIC_RRF_SCORE_GATE", None)
        self._old_overlap = os.environ.pop("AIC_TEXT_MIN_OVERLAP", None)

    def tearDown(self):
        if self._old_gate is not None:
            os.environ["AIC_RRF_SCORE_GATE"] = self._old_gate
        if self._old_overlap is not None:
            os.environ["AIC_TEXT_MIN_OVERLAP"] = self._old_overlap

    def test_weak_ocr_match_does_not_contribute(self):
        visual = [make_result(0, 5, clip_score=0.30)]
        ocr = [make_result(1, 32, ocr_score=0.14)]

        fused = rrf_fuse({"visual": visual, "ocr": ocr})
        by_index = {row.vector_index: row for row in fused}

        self.assertEqual(fused[0].vector_index, 0)
        self.assertNotIn(1, by_index)

    def test_strong_ocr_match_boosts_above_visual_only(self):
        visual = [
            make_result(0, 5, clip_score=0.30),
            make_result(1, 20, clip_score=0.19),
        ]
        ocr = [make_result(1, 20, ocr_score=0.71)]

        fused = rrf_fuse({"visual": visual, "ocr": ocr})

        self.assertEqual(fused[0].vector_index, 1)

    def test_gating_disabled_keeps_legacy_behavior(self):
        visual = [make_result(0, 5, clip_score=0.30)]
        ocr = [make_result(1, 32, ocr_score=0.14)]

        fused = rrf_fuse({"visual": visual, "ocr": ocr}, score_gating=False)
        by_index = {row.vector_index: row for row in fused}

        self.assertIn(1, by_index)
        self.assertGreater(by_index[1].final_score, 0.0)

    def test_visual_contribution_unchanged_by_gating(self):
        visual = [make_result(0, 5, clip_score=0.30)]

        gated = rrf_fuse({"visual": visual}, score_gating=True)
        legacy = rrf_fuse({"visual": visual}, score_gating=False)

        self.assertAlmostEqual(
            gated[0].final_score, legacy[0].final_score
        )


class TrakeMonotonicTests(unittest.TestCase):
    def setUp(self):
        self.video_rows = [
            {"frame_index": frame} for frame in (10, 20, 30, 40)
        ]

    def test_duplicate_frames_are_pushed_forward(self):
        fixed = enforce_increasing_frames(
            [10, 10, 10, 10], self.video_rows
        )
        self.assertEqual(fixed, [10, 20, 30, 40])

    def test_already_increasing_chain_is_untouched(self):
        fixed = enforce_increasing_frames(
            [10, 20, 30, 40], self.video_rows
        )
        self.assertEqual(fixed, [10, 20, 30, 40])

    def test_no_successor_keeps_frame(self):
        fixed = enforce_increasing_frames([40, 40], self.video_rows)
        self.assertEqual(fixed, [40, 40])

    def test_empty_chain(self):
        self.assertEqual(enforce_increasing_frames([], self.video_rows), [])


class QaQuestionRefinementTests(unittest.TestCase):
    def test_description_before_comma_is_removed(self):
        question = (
            "cảnh có biển cảnh báo sạt lở nguy hiểm, "
            "biển cảnh báo có màu gì?"
        )
        self.assertEqual(
            refine_vqa_question(question), "biển cảnh báo có màu gì?"
        )

    def test_multi_comma_keeps_question_body(self):
        question = (
            "trong cảnh đó, có bao nhiêu người, đang khiêng thùng?"
        )
        self.assertEqual(
            refine_vqa_question(question),
            "có bao nhiêu người, đang khiêng thùng?",
        )

    def test_no_marker_keeps_original(self):
        question = "màu sắc của chiếc xe"
        self.assertEqual(refine_vqa_question(question), question)

    def test_text_question_detection(self):
        self.assertTrue(is_text_question("biển báo ghi gì?"))
        self.assertTrue(is_text_question("What does the sign say?"))
        self.assertFalse(is_text_question("có bao nhiêu con trâu?"))


if __name__ == "__main__":
    unittest.main()
