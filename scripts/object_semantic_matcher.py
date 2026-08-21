"""Semantic matching tùy chọn giữa query Việt và catalog nhãn Object.

Khuyến nghị ``intfloat/multilingual-e5-small``. Model chỉ được tải khi thật
sự cần semantic fallback; alias/translation exact không chịu chi phí này.
"""

import json
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np

from object_label_catalog import CATALOG_PATH, EMBEDDINGS_PATH
from stdio_setup import configure_stdio


configure_stdio()


DEFAULT_MODEL = "intfloat/multilingual-e5-small"


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(model_name):
    import torch
    from transformers import AutoModel, AutoTokenizer

    allow_download = os.getenv("AIC_ALLOW_MODEL_DOWNLOAD", "0") == "1"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=True
        )
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
    except OSError:
        if not allow_download:
            raise RuntimeError(
                f"Không tìm thấy semantic model offline: {model_name}. "
                "Hãy gắn model vào Kaggle hoặc đặt "
                "AIC_ALLOW_MODEL_DOWNLOAD=1 khi có Internet."
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tokenizer, model.to(device).eval(), device


def encode_texts(texts, tokenizer, model, device, batch_size=32):
    import torch

    vectors = []
    for start in range(0, len(texts), max(1, int(batch_size))):
        batch = texts[start:start + max(1, int(batch_size))]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(
                min=1e-9
            )
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.cpu().numpy().astype(np.float32))
    return np.ascontiguousarray(np.concatenate(vectors, axis=0))


def _label_descriptions(catalog):
    aliases_by_label = {}
    for concept in catalog.get("concepts", []):
        vi_aliases = concept.get("aliases", {}).get("vi", [])
        vi_aliases = [
            str(alias.get("text", "") if isinstance(alias, dict) else alias)
            for alias in vi_aliases
        ]
        for target in concept.get("targets", []):
            label_key = str(target.get("label", "")).casefold()
            aliases_by_label.setdefault(label_key, set()).update(
                alias for alias in vi_aliases if alias
            )

    keys = []
    descriptions = []
    for record in catalog.get("labels", []):
        label_key = str(record["key"]).casefold()
        display_name = str(record["display_name"])
        aliases = sorted(aliases_by_label.get(label_key, set()))
        description = display_name
        if aliases:
            description += ". Vietnamese aliases: " + ", ".join(aliases)
        keys.append(label_key)
        descriptions.append("passage: " + description)
    return keys, descriptions


def build_embeddings(
    catalog_path=CATALOG_PATH,
    output_path=EMBEDDINGS_PATH,
    model_name=DEFAULT_MODEL,
    batch_size=32,
):
    catalog_path = Path(catalog_path)
    output_path = Path(output_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
    if catalog.get("schema") != "aic_object_label_catalog_v1":
        raise ValueError(f"Catalog sai schema: {catalog_path}")

    label_keys, descriptions = _label_descriptions(catalog)
    print(f"Đang tải Object semantic model: {model_name}")
    tokenizer, model, device = _load_model(model_name)
    print(f"Đang encode {len(label_keys):,} nhãn trên {device}...")
    embeddings = encode_texts(
        descriptions, tokenizer, model, device, batch_size=batch_size
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix="object_embeddings_", suffix=".npz", dir=output_path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            schema=np.asarray(["aic_object_embeddings_v1"]),
            model_name=np.asarray([str(model_name)]),
            catalog_sha256=np.asarray([_file_sha256(catalog_path)]),
            label_keys=np.asarray(label_keys),
            embeddings=embeddings,
        )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Đã tạo embedding {embeddings.shape}: {output_path}")
    return embeddings.shape


class ObjectSemanticMatcher:
    def __init__(
        self,
        embeddings_path=EMBEDDINGS_PATH,
        model_name=None,
        catalog_path=CATALOG_PATH,
    ):
        with np.load(Path(embeddings_path), allow_pickle=False) as artifact:
            schema = str(artifact["schema"][0])
            if schema != "aic_object_embeddings_v1":
                raise ValueError(f"Object embeddings sai schema: {schema}")
            self.label_keys = [str(value) for value in artifact["label_keys"]]
            self.embeddings = np.ascontiguousarray(
                artifact["embeddings"].astype(np.float32)
            )
            artifact_model_name = str(artifact["model_name"][0])
            expected_catalog_hash = str(artifact["catalog_sha256"][0])
        if Path(catalog_path).exists() and expected_catalog_hash != _file_sha256(
            catalog_path
        ):
            raise RuntimeError(
                "Object label embeddings không khớp catalog. "
                "Chạy lại build_object_label_embeddings.py."
            )
        if self.embeddings.shape[0] != len(self.label_keys):
            raise ValueError("Số label và embedding row không khớp")
        requested_model = model_name or os.getenv("AIC_OBJECT_SEMANTIC_MODEL")
        if requested_model and str(requested_model) != artifact_model_name:
            raise RuntimeError(
                "Semantic model runtime khác model đã build Object embedding: "
                f"{requested_model!r} != {artifact_model_name!r}"
            )
        self.model_name = artifact_model_name
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure_model(self):
        if self._model is None:
            self._tokenizer, self._model, self._device = _load_model(
                self.model_name
            )

    def scores(self, query, available_labels):
        self._ensure_model()
        query_vector = encode_texts(
            ["query: " + str(query)],
            self._tokenizer,
            self._model,
            self._device,
            batch_size=1,
        )[0]
        similarities = self.embeddings @ query_vector
        available = {str(label).casefold() for label in available_labels}
        return {
            label_key: float(score)
            for label_key, score in zip(self.label_keys, similarities)
            if label_key in available
        }
