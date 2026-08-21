"""Build SQLite ASR segment index cho tập dữ liệu lớn."""

import argparse
import json
import os
import sqlite3
import tempfile
from pathlib import Path

from asr_retriever import segment_bounds_ms, tokenize
from search_types import ARTIFACTS_DIR, PROJECT_DIR, load_mapping, mapping_sha256
from stdio_setup import configure_stdio


configure_stdio()

DEFAULT_INPUT = PROJECT_DIR / "data" / "asr"
DEFAULT_OUTPUT = ARTIFACTS_DIR / "asr_index.sqlite3"


def iter_payloads(path):
    paths = sorted(path.rglob("*.json")) + sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    for source in paths:
        if source.suffix.lower() == ".jsonl":
            with source.open("r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        yield json.loads(line), source
        else:
            raw = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("videos"), list):
                for item in raw["videos"]:
                    yield item, source
            elif isinstance(raw, dict) and "segments" not in raw:
                # {"video_id": {"segments": [...]}} hoặc
                # {"video_id": [{start/end/text}, ...]}
                for video_id, value in raw.items():
                    if isinstance(value, dict) and isinstance(
                        value.get("segments"), list
                    ):
                        item = dict(value)
                        item.setdefault("video_id", str(video_id))
                        yield item, source
                    elif isinstance(value, list):
                        yield {"video_id": str(video_id), "segments": value}, source
            elif isinstance(raw, list) and raw and isinstance(raw[0], dict) and "segments" in raw[0]:
                for item in raw:
                    yield item, source
            else:
                yield raw, source


def iter_segments(payload, source, input_root):
    if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
        video_id = payload.get("video_id")
        if video_id is None and input_root.is_dir() and source.parent == input_root:
            video_id = source.stem
        for segment in payload["segments"]:
            yield str(video_id or ""), segment
        return

    if isinstance(payload, dict):
        video_id = payload.get("video_id")
        if video_id is not None and any(
            key in payload for key in ("start", "start_ms")
        ):
            yield str(video_id), payload
            return

    if isinstance(payload, list):
        fallback_video = source.stem if input_root.is_dir() else ""
        for segment in payload:
            if isinstance(segment, dict):
                yield str(segment.get("video_id", fallback_video)), segment


def build(input_path, output_path):
    mapping = load_mapping()
    mapping_videos = {str(row["video_id"]) for row in mapping}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix="asr_index_", suffix=".sqlite3", dir=output_path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    indexed = empty = unknown_video = 0

    try:
        connection = sqlite3.connect(temporary)
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE segments (
                id INTEGER PRIMARY KEY,
                video_id TEXT NOT NULL,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                text TEXT NOT NULL
            );
            CREATE TABLE segment_tokens (
                token TEXT NOT NULL,
                segment_id INTEGER NOT NULL,
                PRIMARY KEY (token, segment_id)
            ) WITHOUT ROWID;
            """
        )

        for payload, source in iter_payloads(input_path):
            for video_id, segment in iter_segments(payload, source, input_path):
                if video_id not in mapping_videos:
                    unknown_video += 1
                    continue
                text = str(segment.get("text", "")).strip()
                tokens = tokenize(text)
                if not tokens:
                    empty += 1
                    continue
                start_ms, end_ms = segment_bounds_ms(segment)
                cursor = connection.execute(
                    "INSERT INTO segments(video_id,start_ms,end_ms,text) VALUES (?,?,?,?)",
                    (video_id, start_ms, end_ms, text),
                )
                segment_id = int(cursor.lastrowid)
                connection.executemany(
                    "INSERT OR IGNORE INTO segment_tokens(token,segment_id) VALUES (?,?)",
                    ((token, segment_id) for token in tokens),
                )
                indexed += 1
                if indexed % 10000 == 0:
                    connection.commit()
                    print(f"Đã index {indexed:,} ASR segment...")

        connection.executescript(
            """
            CREATE INDEX segment_tokens_token_idx ON segment_tokens(token);
            CREATE INDEX segments_video_time_idx ON segments(video_id,start_ms,end_ms);
            """
        )
        metadata = {
            "schema": "aic_asr_sqlite_v1",
            "mapping_sha256": mapping_sha256(),
            "mapping_rows": str(len(mapping)),
            "indexed_segments": str(indexed),
        }
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)", metadata.items()
        )
        connection.commit()
        connection.close()

        if indexed == 0:
            raise RuntimeError(
                "Không index được ASR segment nào. Kiểm tra schema và video_id."
            )
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print("Tạo ASR index thành công:", output_path)
    print(f"  Segments: {indexed:,}; rỗng: {empty:,}; sai video_id: {unknown_video:,}")


def main():
    parser = argparse.ArgumentParser(description="Build persistent ASR SQLite index")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Không tìm thấy ASR input: {args.input}")
    build(args.input, args.output)


if __name__ == "__main__":
    main()
