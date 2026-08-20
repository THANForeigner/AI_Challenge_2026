"""
Object retriever (C4/C5): xếp hạng keyframe theo vật thể phát hiện
bởi Faster R-CNN OpenImages V4.

- available(): True khi artifacts/object_index.json tồn tại.
- search(query, top_k): tự đoán vật thể từ query (hoặc nhận danh sách
  qua tham số objects), trả list[SearchResult] theo object_score giảm.

KHÔNG lọc cứng keyframe thiếu object — Faster R-CNN có thể bỏ sót;
object chỉ là điểm cộng, do search_engine quyết định (C5).
"""

import json
import re
from pathlib import Path

from search_types import (
    ARTIFACTS_DIR,
    load_mapping,
    result_from_row,
)
from stdio_setup import configure_stdio


configure_stdio()


OBJECT_INDEX_PATH = ARTIFACTS_DIR / "object_index.json"

# Từ đồng nghĩa/nhãn gần nghĩa thường xuất hiện sau khi dịch Việt→Anh nhưng
# OpenImages dùng tên class khác. Ví dụ "trâu" → buffalo trong khi detector
# của bộ dữ liệu trả Bull/Cattle.
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
    "black", "white", "red", "blue", "green", "yellow",
}


def variants(term):
    """Biến thể số ít/số nhiều đơn giản của một từ."""

    term = term.lower().strip()
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
    """Khớp theo từng từ để 'car' không ăn 'cardboard'."""

    term_variants = variants(term)
    class_lower = class_name.lower()

    if class_lower in term_variants:
        return True

    words = re.split(r"[\s\-_/]+", class_lower)

    return any(word in term_variants for word in words)


def extract_query_objects(query, vocabulary):
    """Tìm từ trong query khớp với class có trong object index."""

    tokens = re.findall(r"[a-zA-Z]+", query.lower())

    seen = set()
    matched = []

    for token in tokens:
        if token in STOPWORDS or token in seen:
            continue

        for class_key in vocabulary:
            if term_matches_class(token, vocabulary[class_key]):
                matched.append(token)
                seen.add(token)
                break

    return matched


class ObjectRetriever:
    name = "object"

    def __init__(self):
        self._data = None
        self._mapping = None
        self._score_cache = {}

    def available(self):
        return OBJECT_INDEX_PATH.exists()

    def _ensure_loaded(self):
        if self._data is not None:
            return

        if not self.available():
            raise FileNotFoundError(
                f"Thiếu object index: {OBJECT_INDEX_PATH}. "
                "Chạy scripts/build_object_index.py trước."
            )

        self._data = json.loads(
            OBJECT_INDEX_PATH.read_text(encoding="utf-8")
        )
        self._mapping = load_mapping()

    @property
    def vocabulary(self):
        self._ensure_loaded()
        return self._data["vocabulary"]

    def matched_classes(self, objects):
        """Các class OpenImages khớp với danh sách vật thể."""

        vocabulary = self.vocabulary
        matched = set()

        for term in objects:
            for class_key, display_name in vocabulary.items():
                if term_matches_class(term, display_name):
                    matched.add(class_key)

        return matched

    def score_map(self, objects):
        """{vector_index: (object_score, [tên class khớp])}."""

        self._ensure_loaded()

        cache_key = tuple(sorted(objects))

        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        # Một frame có thể chứa nhiều box cùng class. Cộng toàn bộ box khiến
        # các cảnh đông vật thể được điểm rất lớn (ví dụ 10 box Tree), dù mức
        # tin cậy semantic không cao hơn. Lấy max cho mỗi class rồi mới cộng
        # giữa các class truy vấn khác nhau.
        scores_by_class = {}
        classes_by_vector = {}

        for class_key in self.matched_classes(objects):
            display_name = self._data["vocabulary"][class_key]

            for vector_index, score, _area in (
                self._data["inverted_index"][class_key]
            ):
                class_scores = scores_by_class.setdefault(vector_index, {})
                class_scores[class_key] = max(
                    class_scores.get(class_key, 0.0),
                    score,
                )

                classes_by_vector.setdefault(vector_index, set()).add(
                    display_name
                )

        result = {
            vector_index: (
                sum(class_scores.values()),
                sorted(classes_by_vector[vector_index]),
            )
            for vector_index, class_scores in scores_by_class.items()
        }

        self._score_cache[cache_key] = result
        return result

    def search(self, query=None, objects=None, top_k=100):
        """
        query: tự đoán vật thể từ câu.
        objects: danh sách vật thể chỉ định (ưu tiên hơn query).
        Trả None nếu không có vật thể nào khớp.
        """

        if objects:
            terms = [term.lower().strip() for term in objects]
        elif query:
            terms = extract_query_objects(query, self.vocabulary)
        else:
            return None

        if not terms:
            return None

        scores = self.score_map(terms)

        if not scores:
            return None

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1][0],
            reverse=True,
        )[:top_k]

        results = []

        for rank, (vector_index, (score, classes)) in enumerate(
            ranked, start=1
        ):
            row = self._mapping[int(vector_index)]
            result = result_from_row(row, object_score=float(score))
            result.rank = rank
            # Giữ tên class khớp để hiển thị (ngoài schema SearchResult)
            result.matched_classes = classes
            results.append(result)

        return results
