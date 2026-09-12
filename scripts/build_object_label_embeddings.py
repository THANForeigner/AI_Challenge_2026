"""CLI tạo semantic embeddings cho Object label catalog."""

import argparse
from pathlib import Path

from object_label_catalog import CATALOG_PATH, EMBEDDINGS_PATH
from object_semantic_matcher import DEFAULT_MODEL, build_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Encode catalog nhãn Object bằng multilingual E5"
    )
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--output", type=Path, default=EMBEDDINGS_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    build_embeddings(
        catalog_path=args.catalog,
        output_path=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

