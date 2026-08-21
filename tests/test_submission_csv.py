import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from search_types import save_submission_csv


class SubmissionCsvTests(unittest.TestCase):
    def test_kis_is_utf8_lf_without_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query-1-kis.csv"
            summary = save_submission_csv(
                path,
                "kis",
                [
                    {"video_id": "L00_V000", "frame_id": 1234},
                    {"video_id": "L00_V055", "frame_id": 5555},
                ],
            )

            self.assertEqual(summary["row_count"], 2)
            self.assertEqual(
                path.read_bytes(),
                b"L00_V000,1234\nL00_V055,5555\n",
            )

    def test_qa_escapes_csv_and_truncates_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query-2-qa.csv"
            long_answer = "đ" * 101
            summary = save_submission_csv(
                path,
                "qa",
                [
                    {
                        "video_id": "L03_V005",
                        "frame_id": 2800,
                        "answer": "Màu đỏ,\nrất đẹp",
                    },
                    {
                        "video_id": "L04_V012",
                        "frame_id": 4100,
                        "answer": 'Anh ấy nói "Tuyệt vời"',
                    },
                    {
                        "video_id": "L05_V001",
                        "frame_id": 5000,
                        "answer": long_answer,
                    },
                ],
            )

            self.assertEqual(summary["truncated_answers"], 1)
            with path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.reader(file))

            self.assertEqual(rows[0], ["L03_V005", "2800", "Màu đỏ, rất đẹp"])
            self.assertEqual(
                rows[1],
                ["L04_V012", "4100", 'Anh ấy nói "Tuyệt vời"'],
            )
            self.assertEqual(rows[2], ["L05_V001", "5000", "đ" * 100])
            text = path.read_text(encoding="utf-8")
            self.assertIn('"Màu đỏ, rất đẹp"', text)
            self.assertIn('"Anh ấy nói ""Tuyệt vời"""', text)

    def test_trake_requires_event_count_and_strict_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query-3-trake.csv"
            summary = save_submission_csv(
                path,
                "trake",
                [
                    {
                        "video_id": "L10_V001",
                        "frame_ids": [1200, 1850, 2100, 2450],
                    },
                    {
                        "video_id": "L10_V002",
                        "frame_ids": [1200, 1850, 1850, 2450],
                    },
                    {
                        "video_id": "L10_V003",
                        "frame_ids": [1200, 1850, 2100],
                    },
                ],
                event_count=4,
            )

            self.assertEqual(summary["row_count"], 1)
            self.assertEqual(summary["skipped_rows"], 2)
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "L10_V001,1200,1850,2100,2450\n",
            )

    def test_rejects_submission_when_every_row_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query-3-trake.csv"

            with self.assertRaisesRegex(ValueError, "Không có dòng"):
                save_submission_csv(
                    path,
                    "trake",
                    [{"video_id": "L10_V001", "frame_ids": [2, 1]}],
                    event_count=2,
                )

            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
