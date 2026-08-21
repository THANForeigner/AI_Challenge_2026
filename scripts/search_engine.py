"""
Search engine tổng (C3 + C4 + router).

Luồng KIS (search_kis):
    CLIP lấy pool ứng viên
    → fusion RRF với object/ocr/asr (modality thiếu thì bỏ qua)
    → xếp theo final_score
    → khử các frame quá gần nhau trong cùng video
    → trả tối đa top_k kết quả <video_id>, <frame_id>

Cách chạy:
    python search_engine.py "a person riding a motorcycle"
    python search_engine.py "..." --type kis|qa|trake --top_k 100
"""

import argparse
import json
from pathlib import Path

from fusion import DEFAULT_WEIGHTS, rrf_fuse
from stdio_setup import configure_stdio
from search_types import (
    ARTIFACTS_DIR,
    load_mapping,
    print_submission_results,
    save_json,
    save_submission_csv,
    serialize_results,
    submission_from_results,
    video_keyframes,
)
from visual_retriever import VisualRetriever
from object_retriever import ObjectRetriever
from ocr_retriever import OcrRetriever
from asr_retriever import AsrRetriever


configure_stdio()


ENGINE_RESULTS_PATH = ARTIFACTS_DIR / "engine_results.json"
SUPPORTED_MODALITIES = ("visual", "object", "ocr", "asr")


def write_cli_submission(
    query_type, answers, output_path=None, event_count=None
):
    path = (
        Path(output_path)
        if output_path
        else ARTIFACTS_DIR / f"query-{query_type}.csv"
    )
    summary = save_submission_csv(
        path,
        query_type,
        answers,
        event_count=event_count,
    )
    print(f"Đã lưu CSV submission: {path} ({summary['row_count']} dòng)")

    if summary["truncated_answers"]:
        print(
            "Cảnh báo: đã tự cắt "
            f"{summary['truncated_answers']} Answer còn tối đa 100 ký tự."
        )

    if summary["skipped_rows"]:
        print(
            "Cảnh báo: đã bỏ qua "
            f"{summary['skipped_rows']} dòng submission không hợp lệ."
        )

    return path


def auto_min_gap(mapping):
    """
    Khoảng cách tối thiểu giữa 2 frame được chọn trong cùng video:
    trung vị khoảng cách giữa các keyframe liên tiếp (auto).
    """

    frames_by_video = {}

    for row in mapping:
        frames_by_video.setdefault(row["video_id"], []).append(
            int(row["frame_index"])
        )

    diffs = []

    for frames in frames_by_video.values():
        frames.sort()

        for previous, current in zip(frames, frames[1:]):
            if current > previous:
                diffs.append(current - previous)

    if not diffs:
        return 1

    diffs.sort()
    return max(1, diffs[len(diffs) // 2])


def suppress_nearby_frames(results, min_gap):
    """
    C3: hạn chế nhiều frame gần nhau trong danh sách nộp bài.
    Chọn tham lam theo thứ hạng; bỏ qua ứng viên có frame_index quá
    gần một frame đã chọn trong cùng video.
    """

    accepted_frames = {}
    kept = []

    for result in results:
        frames = accepted_frames.setdefault(result.video_id, [])

        if any(
            abs(result.frame_index - frame) <= min_gap
            for frame in frames
        ):
            continue

        frames.append(result.frame_index)
        kept.append(result)

    return kept


class SearchEngine:
    def __init__(self, weights=None, verbose=True):
        self.visual = VisualRetriever()
        self.object = ObjectRetriever()
        self.ocr = OcrRetriever()
        self.asr = AsrRetriever()

        self.weights = (
            dict(DEFAULT_WEIGHTS) if weights is None else dict(weights)
        )
        self.mapping = self.visual.mapping
        self._videos = video_keyframes(self.mapping)
        self._min_gap = auto_min_gap(self.mapping)

        if verbose:
            print("SearchEngine sẵn sàng:")
            print(f"  - visual : OK ({self.visual.ntotal} vectors)")
            print(
                "  - object : "
                + (
                    "OK"
                    if self.object.available()
                    else "thiếu object_index.sqlite3/object_index.json — bỏ qua"
                )
            )
            print(
                "  - ocr    : "
                + (
                    "OK"
                    if self.ocr.available()
                    else "thiếu ocr_index.sqlite3 hoặc data/ocr — bỏ qua"
                )
            )
            print(
                "  - asr    : "
                + (
                    "OK"
                    if self.asr.available()
                    else "thiếu asr_index.sqlite3 hoặc data/asr — bỏ qua"
                )
            )

    # ---------- C3: KIS Top-100 ----------

    def search_kis(
        self,
        query,
        top_k=100,
        pool_k=300,
        objects=None,
        min_gap_frames=None,
        modalities=None,
        visual_query=None,
        save=True,
    ):
        if top_k <= 0:
            raise ValueError("top_k phải lớn hơn 0")

        if pool_k <= 0:
            raise ValueError("pool_k phải lớn hơn 0")

        pool_k = max(pool_k, top_k)
        selected_modalities = list(
            SUPPORTED_MODALITIES if modalities is None else modalities
        )
        unknown = sorted(set(selected_modalities) - set(SUPPORTED_MODALITIES))

        if unknown:
            raise ValueError(
                "Modality không hỗ trợ: " + ", ".join(unknown)
            )

        if not selected_modalities:
            raise ValueError("Phải chọn ít nhất một modality")

        ranked = {}

        if "visual" in selected_modalities:
            if visual_query is None:
                from query_translator import translate_for_visual

                try:
                    visual_query = translate_for_visual(query)
                except Exception as error:
                    print(
                        "Cảnh báo: không dịch được query cho Visual: "
                        f"{error}. Visual sẽ dùng query gốc."
                    )
                    visual_query = query

                if visual_query != query:
                    print(f"Visual query (dịch tự động): {visual_query}")

            ranked["visual"] = self.visual.search(
                visual_query or query, top_k=pool_k
            )

        object_query = None
        object_label_matches = []

        if (
            "object" in selected_modalities
            and self.object.available()
            and self.object.retrieval_ready()
        ):
            ranked["object"] = self.object.search(
                query, objects=objects, top_k=pool_k
            )
            object_details = self.object.last_query_details
            object_query = object_details.get("translated_query") or query
            object_label_matches = object_details.get("matches", [])

            if object_details.get("translation_error"):
                print(
                    "Cảnh báo: Object không dịch được query; đã dùng "
                    "catalog Việt–Anh/semantic nếu có: "
                    + object_details["translation_error"]
                )
            if object_details.get("semantic_error"):
                print(
                    "Cảnh báo: Object semantic fallback không sẵn sàng: "
                    + object_details["semantic_error"]
                )
            if object_label_matches:
                labels = ", ".join(
                    f"{match['display_name']} "
                    f"({match['method']}, {match['score']:.2f})"
                    for match in object_label_matches
                )
                print(f"Object labels: {labels}")

        if "ocr" in selected_modalities and self.ocr.available():
            ranked["ocr"] = self.ocr.search(query, top_k=pool_k)

        if "asr" in selected_modalities and self.asr.available():
            ranked["asr"] = self.asr.search(query, top_k=pool_k)

        modalities_used = [
            name for name, rows in ranked.items() if rows
        ]

        fused = rrf_fuse(ranked, weights=self.weights)

        min_gap = (
            self._min_gap
            if min_gap_frames is None
            else max(0, int(min_gap_frames))
        )
        deduplicated = suppress_nearby_frames(fused, min_gap)

        # Khử frame gần nhau có thể làm danh sách ngắn hơn top_k, gây mất
        # recall (điểm BTC tính R@k trên số câu trả lời nộp). Lấp lại bằng
        # các ứng viên bị loại, xếp sau nhóm đa dạng để không đổi top đầu.
        if len(deduplicated) < top_k:
            kept_indices = {
                result.vector_index for result in deduplicated
            }

            for result in fused:
                if len(deduplicated) >= top_k:
                    break

                if result.vector_index in kept_indices:
                    continue

                deduplicated.append(result)
                kept_indices.add(result.vector_index)

        results = deduplicated[:top_k]

        for rank, result in enumerate(results, start=1):
            result.rank = rank

        # C9: gọi frame_refinement của thành viên A nếu có
        results, refined = self.apply_frame_refinement(results)

        if save:
            save_json(
                ENGINE_RESULTS_PATH,
                {
                    "query": query,
                    "visual_query": visual_query or query,
                    "object_query": object_query,
                    "object_label_matches": object_label_matches,
                    "type": "kis",
                    "modalities_used": modalities_used,
                    "modalities_requested": selected_modalities,
                    "weights": self.weights,
                    "min_gap_frames": min_gap,
                    "refined": refined,
                    "top_k": len(results),
                    "submission": submission_from_results(results),
                    "results": serialize_results(results),
                },
            )

        return results

    # ---------- C9: hook gọi frame_refinement của A ----------

    def apply_frame_refinement(self, results):
        """
        Nếu thành viên A cung cấp scripts/frame_refinement.py với hàm
        refine(engine, video_id, frame_index) -> frame_index mới,
        engine tự gọi cho kết quả hạng 1. Chưa có thì bỏ qua.
        """

        if not results:
            return results, False

        try:
            import frame_refinement
        except ImportError:
            return results, False

        top = results[0]

        try:
            refined_frame = frame_refinement.refine(
                self, top.video_id, top.frame_index
            )
        except Exception as error:
            print(f"Cảnh báo frame_refinement lỗi: {error}")
            return results, False

        if refined_frame is not None and refined_frame != top.frame_index:
            top.frame_index = int(refined_frame)
            return results, True

        return results, False

    # ---------- Router: KIS / Q&A / TRAKE ----------

    def answer(self, query, top_k=100):
        import query_router

        routed = query_router.route(query)
        query_type = routed["type"]

        print(f"Router: query thuộc dạng {query_type.upper()}")

        if query_type == "qa":
            import qa_engine

            return qa_engine.answer_question(
                event_description=routed["event_description"],
                question=routed["question"],
                engine=self,
            )

        if query_type == "trake":
            import trake_engine

            return trake_engine.search_trake(
                events=routed["events"],
                engine=self,
                answer_k=top_k,
            )

        results = self.search_kis(query, top_k=top_k)
        return {
            "type": "kis",
            "query": query,
            "submission": submission_from_results(results),
            "results": results,
        }


def main():
    parser = argparse.ArgumentParser(
        description="AIC 2026 search engine (KIS / Q&A / TRAKE)"
    )
    parser.add_argument("query", nargs="+", help="Câu truy vấn")
    parser.add_argument(
        "--type",
        default="auto",
        choices=["auto", "kis", "qa", "trake"],
        help="Ép loại truy vấn (mặc định auto: router tự đoán)",
    )
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Đường dẫn CSV submission; mặc định là "
            "artifacts/query-<type>.csv"
        ),
    )
    parser.add_argument(
        "--modalities",
        default=None,
        help=(
            "Các nhánh cần dùng, cách nhau bằng dấu phẩy: "
            "visual,object,ocr,asr"
        ),
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Đối chứng giống notebook baseline: chỉ Visual CLIP, "
            "không fusion và không khử frame gần nhau"
        ),
    )
    parser.add_argument(
        "--objects",
        default=None,
        help="Vật thể cho object retriever, phân tách bằng dấu phẩy",
    )
    args = parser.parse_args()
    query = " ".join(args.query).strip()

    objects = None

    if args.objects:
        objects = [
            term.strip()
            for term in args.objects.split(",")
            if term.strip()
        ]

    if args.baseline and args.modalities:
        parser.error("Không dùng đồng thời --baseline và --modalities")

    if args.baseline:
        modalities = ["visual"]
        min_gap_frames = 0
    elif args.modalities:
        modalities = [
            name.strip().lower()
            for name in args.modalities.split(",")
            if name.strip()
        ]
        min_gap_frames = None
    else:
        modalities = None
        min_gap_frames = None

    engine = SearchEngine()

    if args.type == "auto":
        routed_type = None

        import query_router

        routed_type = query_router.route(query)["type"]
    else:
        routed_type = args.type

    print()

    if routed_type == "qa":
        if args.type == "auto":
            outcome = engine.answer(query, top_k=args.top_k)
        else:
            import qa_engine
            import query_router

            event_description, question = query_router.split_qa(query)
            outcome = qa_engine.answer_question(
                event_description=event_description,
                question=question or query,
                engine=engine,
                top_k=min(args.top_k, 5),
            )

        print("Video:", outcome["video_id"])
        print("Frame:", outcome["frame_id"])
        print("Answer:", outcome["answer"] or "(VQA không tạo được đáp án)")
        write_cli_submission(
            "qa",
            outcome.get("answers", []),
            output_path=args.output,
        )

        return

    if routed_type == "trake":
        if args.type == "auto":
            outcome = engine.answer(query, top_k=args.top_k)
        else:
            import query_router
            import trake_engine

            outcome = trake_engine.search_trake(
                events=query_router.decompose_events(query),
                engine=engine,
                answer_k=args.top_k,
            )

        if outcome.get("video_id") is None:
            print(
                "Không tìm được chuỗi:",
                outcome.get("error", "không rõ lý do"),
            )
            return

        print("Video:", outcome["video_id"])
        print("Frame IDs:", outcome["frame_ids"])
        print(
            "Chuỗi đầy đủ:",
            "có" if outcome["complete_chain"] else "không (đã nội suy)",
        )

        for event, frame_id in zip(
            outcome["events"], outcome["frame_ids"]
        ):
            print(f"  - {event} → frame {frame_id}")

        write_cli_submission(
            "trake",
            outcome.get("answers", []),
            output_path=args.output,
            event_count=len(outcome["events"]),
        )

        return

    results = engine.search_kis(
        query,
        top_k=args.top_k,
        objects=objects,
        modalities=modalities,
        min_gap_frames=min_gap_frames,
    )

    print_submission_results(
        results,
        header_lines=[f"QUERY: {query}", f"KIS Top-{args.top_k}"],
    )
    write_cli_submission(
        "kis",
        submission_from_results(results),
        output_path=args.output,
    )

    print()
    print("Đã lưu kết quả tại:", ENGINE_RESULTS_PATH)


if __name__ == "__main__":
    main()
