"""Object retriever có catalog Việt–Anh, IDF và kiểm tra mapping.

Ưu tiên ``artifacts/object_index.sqlite3``. Nếu chưa build lại, reader vẫn
hỗ trợ ``object_index.json`` cũ và cảnh báo vì artifact legacy không có SHA.
"""

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import warnings

from object_label_catalog import (
    ALIASES_PATH,
    CATALOG_PATH,
    EMBEDDINGS_PATH,
    LabelMatch,
    ObjectLabelCatalog,
)
from search_types import (
    ARTIFACTS_DIR,
    MAPPING_PATH,
    load_mapping,
    result_from_row,
)
from stdio_setup import configure_stdio


configure_stdio()


OBJECT_SQLITE_INDEX_PATH = ARTIFACTS_DIR / "object_index.sqlite3"
OBJECT_INDEX_PATH = ARTIFACTS_DIR / "object_index.json"
SQLITE_SCHEMA = "aic_object_sqlite_v1"


# Giữ API cũ cho code ngoài đang import các helper này.
OBJECT_TERM_ALIASES = {
    "buffalo": {"bull", "cattle"},
    "buffaloes": {"bull", "cattle"},
    "calf": {"cattle"},
    "cow": {"cattle"},
    "cows": {"cattle"},
    "ox": {"bull", "cattle"},
    "oxen": {"bull", "cattle"},
}

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "with", "and", "or",
    "is", "are", "was", "were", "to", "from", "by", "for", "about",
    "that", "this", "there", "here", "it", "its", "as", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "into",
    "over", "under", "near", "while", "when", "who", "whom", "which",
    "what", "where", "how", "some", "any", "no", "not", "very",
}


def variants(term):
    term = str(term).casefold().strip()
    result = {term}
    irregular = {
        "children": "child",
        "men": "man",
        "people": "person",
        "women": "woman",
    }
    if term in irregular:
        result.add(irregular[term])
    result.update(OBJECT_TERM_ALIASES.get(term, set()))
    for plural, singular in irregular.items():
        if term == singular:
            result.add(plural)
    if term.endswith("s") and len(term) > 3:
        result.add(term[:-1])
    else:
        result.add(term + "s")
    return result


def term_matches_class(term, class_name):
    term_variants = variants(term)
    class_lower = str(class_name).casefold()
    if class_lower in term_variants:
        return True
    return any(
        word in term_variants
        for word in re.split(r"[\s\-_/]+", class_lower)
    )


def extract_query_objects(query, vocabulary):
    tokens = re.findall(r"[a-zA-Z]+", str(query).casefold())
    matched = []
    seen = set()
    for token in tokens:
        if token in STOPWORDS or token in seen:
            continue
        for class_key, display_name in vocabulary.items():
            if term_matches_class(token, display_name):
                matched.append(class_key)
                seen.add(token)
                break
    return matched


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping_path(path):
    path = Path(path)
    if path.resolve() == Path(MAPPING_PATH).resolve():
        return load_mapping()

    rows = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: int(row["vector_index"]))
    if [int(row["vector_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("vector_index trong mapping phải liên tục từ 0")
    return rows


class ObjectRetriever:
    name = "object"

    def __init__(
        self,
        sqlite_index_path=OBJECT_SQLITE_INDEX_PATH,
        legacy_index_path=OBJECT_INDEX_PATH,
        mapping_path=MAPPING_PATH,
        catalog_path=CATALOG_PATH,
        aliases_path=ALIASES_PATH,
        semantic_matcher=None,
        allow_partial=None,
        postings_per_label=None,
    ):
        self.sqlite_index_path = Path(sqlite_index_path)
        self.legacy_index_path = Path(legacy_index_path)
        self.mapping_path = Path(mapping_path)
        self.catalog_path = Path(catalog_path)
        self.aliases_path = Path(aliases_path)
        self.semantic_matcher = semantic_matcher
        self.allow_partial = (
            os.getenv("AIC_ALLOW_PARTIAL_OBJECT_INDEX", "0") == "1"
            if allow_partial is None
            else bool(allow_partial)
        )
        self.postings_per_label = int(
            postings_per_label
            if postings_per_label is not None
            else os.getenv("AIC_OBJECT_POSTINGS_PER_LABEL", "5000")
        )
        if self.postings_per_label < 0:
            raise ValueError("postings_per_label không được âm")

        self._kind = None
        self._data = None
        self._mapping = None
        self._connection = None
        self._metadata = {}
        self._vocabulary = None
        self._class_stats = None
        self._catalog = None
        self._legacy_aggregates = {}
        self._semantic_error = None
        self._db_lock = threading.RLock()
        self._query_state = threading.local()
        self.last_query_details = {
            "translated_query": None,
            "matches": [],
            "translation_error": None,
            "semantic_error": None,
        }

    @property
    def last_query_details(self):
        details = getattr(self._query_state, "details", None)
        if details is None:
            return {
                "translated_query": None,
                "matches": [],
                "translation_error": None,
                "semantic_error": None,
            }
        return {
            **details,
            "matches": [dict(match) for match in details.get("matches", [])],
        }

    @last_query_details.setter
    def last_query_details(self, details):
        self._query_state.details = dict(details)

    def available(self):
        return self.sqlite_index_path.exists() or self.legacy_index_path.exists()

    def close(self):
        with self._db_lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def index_kind(self):
        self._ensure_loaded()
        return self._kind

    @property
    def metadata(self):
        self._ensure_loaded()
        return dict(self._metadata)

    @property
    def complete(self):
        self._ensure_loaded()
        return self._metadata.get("complete", "1") == "1"

    def retrieval_ready(self):
        """Chỉ full index được dùng mặc định; partial cần cờ thử nghiệm."""

        if not self.available():
            return False
        self._ensure_loaded()
        return self.complete or self.allow_partial

    def _open_sqlite(self):
        uri = f"file:{self.sqlite_index_path.resolve().as_posix()}?mode=ro"
        connection = sqlite3.connect(
            uri,
            uri=True,
            check_same_thread=False,
        )
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        if metadata.get("schema") != SQLITE_SCHEMA:
            connection.close()
            raise RuntimeError(
                f"Object index sai schema: {metadata.get('schema')!r}. "
                "Chạy lại build_object_index.py."
            )
        actual_hash = _sha256(self.mapping_path)
        if metadata.get("mapping_sha256") != actual_hash:
            connection.close()
            raise RuntimeError(
                "Object index không khớp clip_row_mapping.jsonl. "
                "Chạy lại build_object_index.py với đúng bộ Object."
            )
        if int(metadata.get("mapping_rows", -1)) != len(self._mapping):
            connection.close()
            raise RuntimeError("Object index có số dòng mapping không khớp")
        return connection, metadata

    def _load_legacy(self):
        data = json.loads(self.legacy_index_path.read_text(encoding="utf-8-sig"))
        metadata = data.get("metadata") or {}
        expected_hash = metadata.get("mapping_sha256")
        if expected_hash:
            if expected_hash != _sha256(self.mapping_path):
                raise RuntimeError(
                    "Legacy Object index không khớp mapping. Hãy build lại."
                )
        else:
            warnings.warn(
                "Đang dùng object_index.json legacy không có mapping SHA256. "
                "Hãy chạy lại build_object_index.py để tạo SQLite index an toàn.",
                RuntimeWarning,
                stacklevel=2,
            )
        stats = data.get("stats") or {}
        mapped_rows = stats.get("mapped_keyframes")
        if mapped_rows is not None and int(mapped_rows) != len(self._mapping):
            raise RuntimeError(
                "Legacy Object index có số mapping khác hiện tại; "
                "không thể ánh xạ vector_index an toàn. Hãy build lại SQLite."
            )
        mapped_documents = int(mapped_rows or len(self._mapping)) - int(
            stats.get("missing_object_files", 0)
        )
        metadata = {
            **{str(key): str(value) for key, value in metadata.items()},
            "schema": data.get("schema", "legacy_object_json_v1"),
            "mapping_rows": str(mapped_rows or len(self._mapping)),
            "mapped_object_documents": str(max(1, mapped_documents)),
            "complete": "1"
            if int(stats.get("missing_object_files", 0)) == 0
            else "0",
        }
        return data, metadata

    def _ensure_loaded(self):
        if self._kind is not None:
            return
        with self._db_lock:
            if self._kind is None:
                self._load_index()

    def _load_index(self):
        if not (
            self.sqlite_index_path.exists() or self.legacy_index_path.exists()
        ):
            raise FileNotFoundError(
                "Thiếu Object index. Chạy scripts/build_object_index.py."
            )

        self._mapping = _load_mapping_path(self.mapping_path)
        if self.sqlite_index_path.exists():
            self._connection, self._metadata = self._open_sqlite()
            self._kind = "sqlite"
            rows = self._connection.execute(
                "SELECT label_key, display_name, document_frequency, "
                "detection_count, idf FROM classes"
            ).fetchall()
            self._vocabulary = {row[0]: row[1] for row in rows}
            self._class_stats = {
                row[0]: {
                    "document_frequency": int(row[2]),
                    "detection_count": int(row[3]),
                    "idf": float(row[4]),
                }
                for row in rows
            }
        else:
            self._data, self._metadata = self._load_legacy()
            self._kind = "legacy_json"
            self._vocabulary = dict(self._data.get("vocabulary") or {})
            self._class_stats = self._legacy_class_stats()

        if not self._vocabulary:
            raise RuntimeError("Object index không có vocabulary")
        self._catalog = ObjectLabelCatalog.load(
            self.catalog_path,
            vocabulary=self._vocabulary,
            aliases_path=self.aliases_path,
        )
        if self.semantic_matcher is None and Path(EMBEDDINGS_PATH).exists():
            try:
                from object_semantic_matcher import ObjectSemanticMatcher

                self.semantic_matcher = ObjectSemanticMatcher(
                    EMBEDDINGS_PATH, catalog_path=self.catalog_path
                )
            except Exception as error:
                self._semantic_error = str(error)

        if not self.complete:
            warnings.warn(
                "Object index mới bao phủ một phần keyframe. Retrieval Object "
                "bị tắt mặc định để không thiên lệch về batch đã xử lý; chỉ "
                "bật thử nghiệm bằng AIC_ALLOW_PARTIAL_OBJECT_INDEX=1.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _legacy_rows_for_label(self, label_key):
        if label_key in self._legacy_aggregates:
            return self._legacy_aggregates[label_key]

        aggregates = {}
        for posting in self._data.get("inverted_index", {}).get(label_key, []):
            if len(posting) < 2:
                continue
            vector_index = int(posting[0])
            confidence = float(posting[1])
            area = float(posting[2]) if len(posting) >= 3 else 0.0
            count = int(posting[3]) if len(posting) >= 4 else 1
            current = aggregates.setdefault(
                vector_index,
                {"max_confidence": 0.0, "max_area": 0.0, "box_count": 0},
            )
            current["max_confidence"] = max(
                current["max_confidence"], confidence
            )
            current["max_area"] = max(current["max_area"], area)
            current["box_count"] += count

        rows = [
            (
                vector_index,
                values["max_confidence"],
                values["max_area"],
                values["box_count"],
            )
            for vector_index, values in aggregates.items()
        ]
        self._legacy_aggregates[label_key] = rows
        return rows

    def _legacy_class_stats(self):
        document_count = max(
            1, int(self._metadata.get("mapped_object_documents", 1))
        )
        stats = {}
        for label_key in self._vocabulary:
            rows = self._legacy_rows_for_label(label_key)
            document_frequency = len(rows)
            stats[label_key] = {
                "document_frequency": document_frequency,
                "detection_count": sum(row[3] for row in rows),
                "idf": math.log(
                    (document_count + 1) / (document_frequency + 1)
                ) + 1.0,
            }
        return stats

    @property
    def vocabulary(self):
        self._ensure_loaded()
        return dict(self._vocabulary)

    def _semantic_scores(self, query):
        if self.semantic_matcher is None:
            return None
        try:
            if hasattr(self.semantic_matcher, "scores"):
                return self.semantic_matcher.scores(
                    query, set(self._vocabulary)
                )
            return self.semantic_matcher(query, set(self._vocabulary))
        except Exception as error:
            self._semantic_error = str(error)
            return None

    def resolve_query(self, query, objects=None, translated_query=None, top_k=8):
        self._ensure_loaded()
        queries = [
            str(value).strip()
            for value in (objects or [])
            if str(value).strip()
        ]
        if not queries:
            queries = [str(query or "").strip()]

        matches = []
        translations = []
        translation_errors = []
        for position, source_query in enumerate(queries):
            source_translation = (
                translated_query if position == 0 else None
            )
            if source_translation is None:
                try:
                    from query_translator import translate_for_object

                    source_translation = translate_for_object(source_query)
                except Exception as error:  # model có thể chưa gắn trên Kaggle
                    translation_errors.append(str(error))
                    source_translation = None
            if source_translation:
                translations.append(str(source_translation))

            resolved = self._catalog.resolve(
                source_query,
                self._vocabulary,
                translated_query=source_translation,
                top_k=max(16, top_k * 3),
            )
            # Direct alias đáng tin hơn. Bỏ bản dịch của chính label đã nhận
            # trực tiếp, nhưng giữ label mới từ phần query chưa được phủ
            # (vd. người→Person, diều→Kite qua bản dịch toàn câu).
            direct_labels = {
                match.label_key
                for match in resolved
                if match.method != "translation_fallback"
            }
            resolved = [
                match
                for match in resolved
                if match.method != "translation_fallback"
                or match.label_key not in direct_labels
            ]
            # objects=... biểu diễn các concept độc lập; thêm prefix để cùng
            # span ở hai phần tử không bị gộp nhầm khi chấm điểm.
            for match in resolved:
                matches.append(
                    LabelMatch(
                        **{
                            **match.to_dict(),
                            "concept_id": f"input:{position}:{match.concept_id}",
                        }
                    )
                )

        if not matches and queries:
            semantic_scores = self._semantic_scores(queries[0])
            matches = self._catalog.resolve(
                queries[0],
                self._vocabulary,
                top_k=top_k,
                semantic_scores=semantic_scores,
                semantic_threshold=float(
                    os.getenv("AIC_OBJECT_SEMANTIC_THRESHOLD", "0.72")
                ),
            )

        # Dedupe ổn định, giữ match mạnh nhất cho label/concept.
        best = {}
        for match in matches:
            # Giữ nguyên concept_id theo phrase/span. Nếu một bản dịch như
            # ``buffalo`` trỏ đồng thời Bull và Cattle, scorer phải lấy MAX
            # giữa hai label thay thế thay vì coi chúng là hai vật độc lập.
            key = (match.concept_id, match.label_key)
            if key not in best or match.score > best[key].score:
                best[key] = match
        matches = sorted(
            best.values(), key=lambda match: (-match.score, match.label_key)
        )[:top_k]
        used_translation = (
            translated_query
            or ". ".join(dict.fromkeys(translations))
            or None
        )
        self.last_query_details = {
            "translated_query": used_translation,
            "matches": [match.to_dict() for match in matches],
            "translation_error": "; ".join(dict.fromkeys(translation_errors))
            or None,
            "semantic_error": self._semantic_error,
        }
        return matches

    def _posting_rows(self, label_keys):
        if self._kind == "sqlite":
            rows = []
            query = """
                SELECT p.label_key, p.vector_index, p.max_confidence,
                       p.max_area, p.box_count, c.idf, c.display_name
                FROM postings p
                JOIN classes c ON c.label_key = p.label_key
                WHERE p.label_key = ?
                ORDER BY p.max_confidence DESC, p.vector_index ASC
            """
            if self.postings_per_label:
                query += " LIMIT ?"
            # Một connection dùng chung được bảo vệ khi backend xử lý nhiều
            # request; đồng thời giới hạn posting phổ biến như Person để RAM
            # không tăng theo toàn bộ corpus.
            with self._db_lock:
                for label_key in sorted(label_keys):
                    parameters = (
                        (label_key, self.postings_per_label)
                        if self.postings_per_label
                        else (label_key,)
                    )
                    rows.extend(
                        self._connection.execute(query, parameters).fetchall()
                    )
            return rows

        rows = []
        for label_key in label_keys:
            stats = self._class_stats[label_key]
            display_name = self._vocabulary[label_key]
            for vector_index, confidence, area, count in (
                self._legacy_rows_for_label(label_key)
            ):
                rows.append(
                    (
                        label_key,
                        vector_index,
                        confidence,
                        area,
                        count,
                        stats["idf"],
                        display_name,
                    )
                )
        return rows

    def score_map(self, matches):
        """Score theo concept; label thay thế lấy MAX, concept khác mới SUM."""

        self._ensure_loaded()
        if matches and isinstance(matches[0], str):
            # Tương thích API cũ score_map(["person", "car"]).
            matches = [
                LabelMatch(
                    label_key=label_key,
                    display_name=self._vocabulary.get(label_key, label_key),
                    score=1.0,
                    method="explicit",
                    matched_text=label_key,
                    concept_id=f"explicit:{position}",
                )
                for position, label_key in enumerate(matches)
                if label_key in self._vocabulary
            ]
        if not matches:
            return {}

        match_by_label = defaultdict(list)
        for match in matches:
            if match.label_key in self._vocabulary:
                match_by_label[match.label_key].append(match)
        if not match_by_label:
            return {}

        max_idf = max(
            self._class_stats[label_key]["idf"]
            for label_key in match_by_label
        )
        by_vector_concept = defaultdict(dict)
        classes_by_vector = defaultdict(set)
        details_by_vector = defaultdict(dict)

        for (
            label_key,
            vector_index,
            confidence,
            area,
            box_count,
            idf,
            display_name,
        ) in self._posting_rows(set(match_by_label)):
            # Area chỉ là tín hiệu phụ rất nhẹ; file không có bbox nhận 0.95,
            # không bị giả định là phủ toàn ảnh.
            area_quality = 0.95 + 0.05 * math.sqrt(
                max(0.0, min(1.0, float(area)))
            )
            idf_quality = 0.6 + 0.4 * (float(idf) / max_idf)

            for match in match_by_label[label_key]:
                contribution = (
                    float(confidence)
                    * area_quality
                    * idf_quality
                    * float(match.score)
                )
                current = by_vector_concept[vector_index].get(match.concept_id)
                if current is None or contribution > current:
                    by_vector_concept[vector_index][match.concept_id] = contribution
                    details_by_vector[vector_index][match.concept_id] = {
                        "label": display_name,
                        "confidence": float(confidence),
                        "box_count": int(box_count),
                        "idf": float(idf),
                        "match_score": float(match.score),
                        "method": match.method,
                    }
                classes_by_vector[vector_index].add(display_name)

        return {
            vector_index: (
                sum(concepts.values()),
                sorted(classes_by_vector[vector_index]),
                details_by_vector[vector_index],
            )
            for vector_index, concepts in by_vector_concept.items()
        }

    def search(
        self,
        query=None,
        objects=None,
        top_k=100,
        translated_query=None,
    ):
        if top_k <= 0 or not self.available():
            return None
        if not self.retrieval_ready():
            return None

        matches = self.resolve_query(
            query,
            objects=objects,
            translated_query=translated_query,
        )
        if not matches:
            return None

        scores = self.score_map(matches)
        if not scores:
            return None

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1][0], item[0]),
        )[:top_k]
        results = []
        for rank, (vector_index, (score, classes, details)) in enumerate(
            ranked, start=1
        ):
            if vector_index < 0 or vector_index >= len(self._mapping):
                raise RuntimeError(
                    f"Object index chứa vector_index ngoài mapping: {vector_index}"
                )
            row = self._mapping[int(vector_index)]
            result = result_from_row(row, object_score=float(score))
            result.rank = rank
            result.matched_classes = classes
            result.object_match_details = details
            results.append(result)
        return results
