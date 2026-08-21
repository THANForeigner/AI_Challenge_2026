import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_object_index import build  # noqa: E402
from build_object_label_catalog import build_catalog  # noqa: E402
from object_io import iter_object_documents  # noqa: E402
from object_label_catalog import ObjectLabelCatalog  # noqa: E402
from object_retriever import ObjectRetriever  # noqa: E402


ALIASES_PATH = PROJECT_DIR / "config" / "object_aliases.vi.json"


class ObjectLabelResolverTests(unittest.TestCase):
    def setUp(self):
        self.vocabulary = {
            "bull": "Bull",
            "cattle": "Cattle",
            "car": "Car",
            "cardboard": "Cardboard",
            "motorcycle": "Motorcycle",
            "helmet": "Helmet",
            "hat": "Hat",
            "palm tree": "Palm tree",
            "tree": "Tree",
            "man": "Man",
            "woman": "Woman",
            "person": "Person",
            "skyscraper": "Skyscraper",
            "building": "Building",
        }
        self.catalog = ObjectLabelCatalog.from_vocabulary(
            self.vocabulary, ALIASES_PATH
        )

    def labels(self, query, **kwargs):
        return [
            match.label_key
            for match in self.catalog.resolve(
                query, self.vocabulary, **kwargs
            )
        ]

    def test_vietnamese_alias_and_accent_fold(self):
        self.assertEqual(
            self.labels("đàn trâu ăn cỏ")[:2], ["cattle", "bull"]
        )
        self.assertEqual(
            self.labels("dan trau an co")[:2], ["cattle", "bull"]
        )

    def test_longest_phrase_wins(self):
        self.assertEqual(self.labels("mũ bảo hiểm"), ["helmet"])
        self.assertEqual(self.labels("cây cọ"), ["palm tree"])
        self.assertEqual(
            self.labels("tòa nhà chọc trời"), ["skyscraper"]
        )

    def test_whole_word_matching(self):
        self.assertEqual(self.labels("car"), ["car"])
        self.assertNotIn("cardboard", self.labels("car"))
        self.assertNotIn("woman", self.labels("man"))

    def test_short_accent_fold_does_not_create_false_flag_match(self):
        vocabulary = {**self.vocabulary, "flag": "Flag"}
        catalog = ObjectLabelCatalog.from_vocabulary(
            vocabulary, ALIASES_PATH
        )
        labels = [
            match.label_key
            for match in catalog.resolve("dan trau an co", vocabulary)
        ]
        self.assertNotIn("flag", labels)

    def test_single_word_accent_fold_avoids_common_false_positives(self):
        vocabulary = {
            **self.vocabulary,
            "dog": "Dog",
            "house": "House",
            "chair": "Chair",
        }
        catalog = ObjectLabelCatalog.from_vocabulary(vocabulary, ALIASES_PATH)
        for query, forbidden in (
            ("cho tre em qua", "dog"),
            ("ban dang di bo", "table"),
            ("di nhanh len nha", "house"),
            ("ghe dang chay tren song", "chair"),
        ):
            labels = [
                match.label_key
                for match in catalog.resolve(query, vocabulary)
            ]
            self.assertNotIn(forbidden, labels)

    def test_translation_and_semantic_fallback(self):
        vocabulary = {**self.vocabulary, "rhinoceros": "Rhinoceros"}
        catalog = ObjectLabelCatalog.from_vocabulary(
            vocabulary, ALIASES_PATH
        )
        translated = catalog.resolve(
            "sinh vật lạ",
            vocabulary,
            translated_query="a rhinoceros",
        )
        self.assertEqual(translated[0].label_key, "rhinoceros")

        semantic = catalog.resolve(
            "sinh vật kỳ lạ",
            vocabulary,
            semantic_scores={"cattle": 0.71, "rhinoceros": 0.83},
            semantic_threshold=0.72,
        )
        self.assertEqual([match.label_key for match in semantic], ["rhinoceros"])


class ObjectCatalogBuilderTests(unittest.TestCase):
    def test_skip_invalid_catches_deferred_generator_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            object_root = root / "objects" / "V1"
            object_root.mkdir(parents=True)
            (object_root / "001.json").write_text(
                json.dumps(
                    {
                        "detection_class_entities": ["Car"],
                        "detection_scores": [0.9],
                    }
                ),
                encoding="utf-8",
            )
            (object_root / "002.json").write_text(
                json.dumps(
                    {
                        "detection_class_entities": ["Person", "Car"],
                        "detection_scores": [0.9],
                    }
                ),
                encoding="utf-8",
            )
            (object_root / "003.json").write_text(
                json.dumps(
                    {
                        "detection_class_entities": ["Bull"],
                        "detection_scores": [1.5],
                    }
                ),
                encoding="utf-8",
            )
            (object_root / "004.json").write_text(
                "{JSON bị hỏng",
                encoding="utf-8",
            )

            payload = build_catalog(
                inputs=[root / "objects"],
                output_path=root / "catalog.json",
                aliases_path=ALIASES_PATH,
                strict=False,
            )

            self.assertEqual(payload["stats"]["documents"], 2)
            self.assertEqual(payload["stats"]["invalid_documents"], 2)
            self.assertEqual(payload["stats"]["invalid_detections"], 1)
            self.assertEqual(payload["stats"]["unique_labels"], 1)


class ObjectIndexIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.mapping_path = self.root / "clip_row_mapping.jsonl"
        self.object_root = self.root / "objects"
        self.index_path = self.root / "object_index.sqlite3"
        self.rows = [
            {
                "vector_index": 0,
                "video_id": "V1",
                "keyframe_name": "001.jpg",
                "frame_index": 10,
                "timestamp_ms": 1000,
            },
            {
                "vector_index": 1,
                "video_id": "V1",
                "keyframe_name": "002.jpg",
                "frame_index": 20,
                "timestamp_ms": 2000,
            },
            {
                "vector_index": 2,
                "video_id": "V1",
                "keyframe_name": "003.jpg",
                "frame_index": 30,
                "timestamp_ms": 3000,
            },
            {
                "vector_index": 3,
                "video_id": "V2",
                "keyframe_name": "001.jpg",
                "frame_index": 40,
                "timestamp_ms": 4000,
            },
        ]
        self._write_mapping(self.rows)
        self._write_object("V1", "001", ["Person", "Person"], [0.9, 0.6])
        self._write_object("V1", "002", ["Bull", "Cattle"], [0.9, 0.92])
        self._write_object("V1", "003", ["Car"], [0.88])
        self._write_object("V2", "001", ["Traffic light"], [0.91])

        self.metadata = build(
            inputs=[self.object_root],
            output_path=self.index_path,
            mapping_path=self.mapping_path,
            require_complete=True,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _write_mapping(self, rows):
        self.mapping_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _write_object(self, video_id, stem, labels, scores):
        directory = self.object_root / video_id
        directory.mkdir(parents=True, exist_ok=True)
        boxes = [[0.1, 0.1, 0.5, 0.5] for _ in labels]
        (directory / f"{stem}.json").write_text(
            json.dumps(
                {
                    "detection_class_entities": labels,
                    "detection_scores": scores,
                    "detection_boxes": boxes,
                }
            ),
            encoding="utf-8",
        )

    def retriever(self):
        return ObjectRetriever(
            sqlite_index_path=self.index_path,
            legacy_index_path=self.root / "missing.json",
            mapping_path=self.mapping_path,
            catalog_path=self.root / "missing_catalog.json",
            aliases_path=ALIASES_PATH,
        )

    def test_builder_writes_fingerprint_and_idf(self):
        self.assertEqual(self.metadata["schema"], "aic_object_sqlite_v1")
        self.assertEqual(self.metadata["complete"], "1")
        connection = sqlite3.connect(self.index_path)
        try:
            person_df, person_idf = connection.execute(
                "SELECT document_frequency, idf FROM classes "
                "WHERE label_key='person'"
            ).fetchone()
            bull_df, bull_idf = connection.execute(
                "SELECT document_frequency, idf FROM classes "
                "WHERE label_key='bull'"
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual(person_df, 1)
        self.assertEqual(bull_df, 1)
        self.assertAlmostEqual(person_idf, math.log(5 / 2) + 1.0)
        self.assertAlmostEqual(bull_idf, math.log(5 / 2) + 1.0)

    def test_json_locator_mismatch_is_rejected(self):
        path = self.object_root / "V1" / "001.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["keyframe_name"] = "002.jpg"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mâu thuẫn"):
            build(
                inputs=[self.object_root],
                output_path=self.root / "bad.sqlite3",
                mapping_path=self.mapping_path,
            )

    def test_composite_keyframe_id_matches_filename(self):
        path = self.object_root / "V1" / "001.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw.update({"video_id": "V1", "keyframe_id": "V1_K001"})
        path.write_text(json.dumps(raw), encoding="utf-8")
        document = next(iter_object_documents([path]))
        self.assertEqual(document.logical_id, "V1/001")

    def test_partial_index_is_disabled_by_default(self):
        partial_root = self.root / "partial" / "V1"
        partial_root.mkdir(parents=True)
        (partial_root / "001.json").write_text(
            (self.object_root / "V1" / "001.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        partial_index = self.root / "partial.sqlite3"
        build(
            inputs=[self.root / "partial"],
            output_path=partial_index,
            mapping_path=self.mapping_path,
        )
        retriever = ObjectRetriever(
            sqlite_index_path=partial_index,
            legacy_index_path=self.root / "missing.json",
            mapping_path=self.mapping_path,
            catalog_path=self.root / "missing_catalog.json",
            aliases_path=ALIASES_PATH,
        )
        self.assertFalse(retriever.retrieval_ready())
        self.assertIsNone(retriever.search("person"))
        retriever.close()

    def test_sqlite_retriever_can_be_used_from_another_thread(self):
        retriever = self.retriever()
        _ = retriever.vocabulary
        outcome = []

        def search_in_thread():
            try:
                outcome.append(retriever.search("car", top_k=1)[0].frame_index)
            except Exception as error:
                outcome.append(error)

        worker = threading.Thread(target=search_in_thread)
        worker.start()
        worker.join()
        self.assertEqual(outcome, [30])
        retriever.close()

    def test_vietnamese_object_retrieval_and_no_double_count(self):
        retriever = self.retriever()
        results = retriever.search("đàn trâu ăn cỏ", top_k=3)
        self.assertEqual(results[0].video_id, "V1")
        self.assertEqual(results[0].frame_index, 20)
        # Bull/Cattle là hai target thay thế của một concept "trâu".
        self.assertEqual(len(results[0].object_match_details), 1)
        self.assertLessEqual(results[0].object_score, 1.0)

        car = retriever.search("một chiếc ô tô", top_k=1)[0]
        self.assertEqual(car.frame_index, 30)
        self.assertFalse(Path(car.keyframe_path).is_absolute())
        retriever.close()

    def test_translation_alternatives_are_not_double_counted(self):
        retriever = self.retriever()
        results = retriever.search(
            "sinh vật lạ",
            translated_query="buffalo",
            top_k=1,
        )
        self.assertEqual(results[0].frame_index, 20)
        self.assertEqual(len(results[0].object_match_details), 1)
        self.assertLessEqual(results[0].object_score, 1.0)
        retriever.close()

    def test_mapping_change_is_rejected(self):
        changed = [dict(row) for row in self.rows]
        changed[0]["frame_index"] = 999
        self._write_mapping(changed)
        with self.assertRaisesRegex(RuntimeError, "không khớp"):
            _ = self.retriever().vocabulary


if __name__ == "__main__":
    unittest.main()
