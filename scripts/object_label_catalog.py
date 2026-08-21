"""Catalog nhãn Object và resolver truy vấn Việt/Anh.

Không dịch hay thay đổi nhãn gốc của detector. Catalog chỉ bổ sung alias và
trả về các nhãn canonical thật sự có trong Object index hiện tại.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from search_types import ARTIFACTS_DIR, PROJECT_DIR


CATALOG_PATH = ARTIFACTS_DIR / "object_label_catalog.json"
ALIASES_PATH = PROJECT_DIR / "config" / "object_aliases.vi.json"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "object_label_embeddings.npz"


@lru_cache(maxsize=8192)
def normalize_text(text):
    """NFKC + casefold + chuẩn hóa dấu câu/khoảng trắng."""

    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    normalized = re.sub(r"[_/\\-]+", " ", normalized)
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


@lru_cache(maxsize=8192)
def fold_accents(text):
    """Bỏ dấu tiếng Việt; xử lý riêng ``đ`` vì NFKD không tách ký tự này."""

    normalized = normalize_text(text).replace("đ", "d")
    decomposed = unicodedata.normalize("NFKD", normalized)
    return "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LabelMatch:
    label_key: str
    display_name: str
    score: float
    method: str
    matched_text: str
    concept_id: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class _AliasEntry:
    concept_id: str
    language: str
    text: str
    normalized: str
    folded: str
    weight: float
    source: str
    targets: tuple


@dataclass(frozen=True)
class _Hit:
    entry: _AliasEntry
    span: tuple
    method: str
    method_weight: float

    @property
    def length(self):
        return self.span[1] - self.span[0]

    @property
    def domain(self):
        return "translated" if self.method == "translation_fallback" else "original"


def _phrase_spans(text, phrase):
    if not text or not phrase:
        return []
    pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.UNICODE)
    return [match.span() for match in pattern.finditer(text)]


def load_alias_config(path=ALIASES_PATH):
    path = Path(path)
    if not path.exists():
        return []

    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if raw.get("schema") != "aic_object_aliases_vi_v1":
        raise ValueError(f"Alias Object sai schema: {path}")
    concepts = raw.get("concepts")
    if not isinstance(concepts, list):
        raise ValueError(f"Alias Object thiếu danh sách concepts: {path}")
    return concepts


def canonical_concepts(labels):
    concepts = []
    for label_key, record in sorted(labels.items()):
        display_name = record.get("display_name", label_key)
        concepts.append(
            {
                "id": f"label:{label_key}",
                "source": "canonical",
                "aliases": {"en": [display_name, label_key], "vi": []},
                "targets": [{"label": label_key, "weight": 1.0}],
            }
        )
    return concepts


class ObjectLabelCatalog:
    """Resolve cụm từ truy vấn thành label canonical của detector."""

    def __init__(self, labels, concepts):
        self.labels = {
            str(key).casefold(): dict(record)
            for key, record in labels.items()
        }
        self._entries = self._compile_entries(concepts)

    @classmethod
    def from_vocabulary(cls, vocabulary, aliases_path=ALIASES_PATH):
        labels = {
            str(key).casefold(): {
                "key": str(key).casefold(),
                "display_name": str(display_name),
                "aliases": [],
            }
            for key, display_name in vocabulary.items()
        }
        concepts = canonical_concepts(labels) + load_alias_config(aliases_path)
        return cls(labels, concepts)

    @classmethod
    def load(
        cls,
        path=CATALOG_PATH,
        vocabulary=None,
        aliases_path=ALIASES_PATH,
    ):
        path = Path(path)
        if not path.exists():
            return cls.from_vocabulary(vocabulary or {}, aliases_path)

        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if raw.get("schema") != "aic_object_label_catalog_v1":
            raise ValueError(f"Object label catalog sai schema: {path}")

        labels = {
            str(record["key"]).casefold(): dict(record)
            for record in raw.get("labels", [])
        }
        for key, display_name in (vocabulary or {}).items():
            labels.setdefault(
                str(key).casefold(),
                {
                    "key": str(key).casefold(),
                    "display_name": str(display_name),
                    "aliases": [],
                },
            )

        # Catalog có canonical/auto-translation; file config được merge lần
        # cuối để chỉnh alias thủ công mà không phải build lại catalog.
        concepts_by_id = {
            concept["id"]: concept
            for concept in canonical_concepts(labels) + raw.get("concepts", [])
            if isinstance(concept, dict) and concept.get("id")
        }
        for concept in load_alias_config(aliases_path):
            concepts_by_id[concept["id"]] = concept

        return cls(labels, list(concepts_by_id.values()))

    def _compile_entries(self, concepts):
        entries = []
        for concept in concepts:
            concept_id = str(concept.get("id", "")).strip()
            aliases = concept.get("aliases") or {}
            targets = []
            for target in concept.get("targets") or []:
                label_key = str(target.get("label", "")).casefold().strip()
                if not label_key:
                    continue
                targets.append(
                    (label_key, float(target.get("weight", 1.0)))
                )
            if not concept_id or not targets:
                continue

            source = str(concept.get("source", "curated"))
            for language in ("vi", "en"):
                raw_aliases = aliases.get(language) or []
                if isinstance(raw_aliases, str):
                    raw_aliases = [raw_aliases]
                for raw_alias in raw_aliases:
                    if isinstance(raw_alias, dict):
                        text = str(raw_alias.get("text", "")).strip()
                        weight = float(raw_alias.get("weight", 1.0))
                        alias_source = str(raw_alias.get("source", source))
                    else:
                        text = str(raw_alias).strip()
                        weight = 1.0
                        alias_source = source
                    normalized = normalize_text(text)
                    if not normalized:
                        continue
                    entries.append(
                        _AliasEntry(
                            concept_id=concept_id,
                            language=language,
                            text=text,
                            normalized=normalized,
                            folded=fold_accents(normalized),
                            weight=weight,
                            source=alias_source,
                            targets=tuple(targets),
                        )
                    )

        # Duyệt cụm dài trước giúp deterministic và hỗ trợ longest phrase.
        entries.sort(
            key=lambda entry: (
                -len(entry.normalized.split()),
                -len(entry.normalized),
                entry.concept_id,
                entry.text,
            )
        )
        return entries

    def _find_hits(self, query, translated_query=None):
        normalized = normalize_text(query)
        folded = fold_accents(query)
        translated = normalize_text(translated_query or "")
        hits = []

        for entry in self._entries:
            for span in _phrase_spans(normalized, entry.normalized):
                method = "vi_exact" if entry.language == "vi" else "en_exact"
                hits.append(_Hit(entry, span, method, 1.0))

            # Accent-fold chỉ bổ sung khi exact chưa thể khớp alias đó.
            if entry.language == "vi" and entry.folded != entry.normalized:
                # Mọi alias đơn từ đều có thể va chạm sau khi bỏ dấu:
                # chó→cho, bàn→ban, nhà→nha, ghế→ghe. Chỉ nhận alias đơn từ
                # khi toàn query chính là từ đó. Trong câu dài, dùng alias có
                # ngữ cảnh như "con chó", "cái bàn", "ngôi nhà" hoặc model
                # dịch/semantic fallback.
                if (
                    len(entry.folded.split()) == 1
                    and folded != entry.folded
                ):
                    continue
                for span in _phrase_spans(folded, entry.folded):
                    if any(
                        hit.entry == entry and hit.span == span
                        for hit in hits
                    ):
                        continue
                    hits.append(_Hit(entry, span, "vi_accent_fold", 0.96))

            if translated and entry.language == "en":
                for span in _phrase_spans(translated, entry.normalized):
                    hits.append(
                        _Hit(entry, span, "translation_fallback", 0.92)
                    )

        return hits

    @staticmethod
    def _keep_longest_hits(hits):
        """Bỏ alias ngắn nằm trong alias dài, nhưng giữ các target thay thế."""

        ordered = sorted(
            hits,
            key=lambda hit: (
                -hit.length,
                -(hit.entry.weight * hit.method_weight),
                hit.entry.concept_id,
            ),
        )
        accepted = []
        accepted_spans = []
        for hit in ordered:
            contained = any(
                start <= hit.span[0]
                and hit.span[1] <= end
                and hit.span != (start, end)
                and domain == hit.domain
                for domain, start, end in accepted_spans
            )
            if contained:
                continue
            accepted.append(hit)
            domain_span = (hit.domain, *hit.span)
            if domain_span not in accepted_spans:
                accepted_spans.append(domain_span)
        return accepted

    def resolve(
        self,
        query,
        available_labels,
        translated_query=None,
        top_k=8,
        semantic_scores=None,
        semantic_threshold=0.72,
    ):
        """Trả các label match; luôn giao với vocabulary của index thật."""

        if top_k <= 0 or not normalize_text(query):
            return []

        available = {str(label).casefold() for label in available_labels}
        hits = self._keep_longest_hits(
            self._find_hits(query, translated_query=translated_query)
        )
        best = {}

        for hit in hits:
            # Các label cùng khớp đúng một phrase dùng chung concept_id để
            # scorer lấy MAX, tránh cộng Bull + Cattle hai lần cho "trâu".
            phrase_key = fold_accents(hit.entry.text)
            concept_id = (
                f"phrase:{hit.domain}:{hit.span[0]}:{hit.span[1]}:{phrase_key}"
            )
            for label_key, target_weight in hit.entry.targets:
                if label_key not in available:
                    continue
                score = hit.entry.weight * hit.method_weight * target_weight
                match = LabelMatch(
                    label_key=label_key,
                    display_name=self.labels.get(label_key, {}).get(
                        "display_name", label_key
                    ),
                    score=float(score),
                    method=hit.method,
                    matched_text=hit.entry.text,
                    concept_id=concept_id,
                )
                key = (concept_id, label_key)
                if key not in best or match.score > best[key].score:
                    best[key] = match

        # Semantic chỉ là fallback khi lexical/translation không tìm được
        # concept nào; tránh semantic yếu làm nhiễu một query đã rõ nhãn.
        if not best and semantic_scores:
            for label_key, score in semantic_scores.items():
                label_key = str(label_key).casefold()
                score = float(score)
                if label_key not in available or score < semantic_threshold:
                    continue
                best[("semantic:query", label_key)] = LabelMatch(
                    label_key=label_key,
                    display_name=self.labels.get(label_key, {}).get(
                        "display_name", label_key
                    ),
                    score=score,
                    method="semantic",
                    matched_text=str(query),
                    concept_id="semantic:query",
                )

        matches = sorted(
            best.values(),
            key=lambda match: (
                -match.score,
                -len(normalize_text(match.matched_text)),
                match.label_key,
            ),
        )
        return matches[:top_k]
