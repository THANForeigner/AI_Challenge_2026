# AIC 2026 — Pipeline Thành viên C (CLIP, FAISS, Fusion, KIS, Q&A, TRAKE)

Pipeline tìm kiếm video cho vòng Sơ tuyển. Đầu ra mỗi câu trả lời
theo format BTC:

- KIS: `<video_id>, <frame_id>`
- Q&A: `<video_id>, <frame_id>, <answer>`
- TRAKE: `<video_id>, <frame_id_1>, ..., <frame_id_n>`

`frame_id` là **frame index thật trong video** (không phải số thứ tự
keyframe). Mỗi truy vấn nộp tối đa 100 câu trả lời; Final Score =
trung bình R@1/5/20/50/100 → **xếp hạng đúng quan trọng hơn tìm ra**.

> Các phần đang là mã giả / chờ dữ liệu: xem **PLACEHOLDERS.md**.

## Cấu trúc thư mục

```
aic_practice/
├── data/
│   ├── keyframes/<video_id>/0001.jpg ...   # thành viên khác cắt, đặt tên = video_id
│   ├── metadata/<video_id>.json            # frame index mỗi keyframe (bắt buộc)
│   ├── objects/<video_id>/0001.json        # Faster R-CNN OpenImages V4 (format TF Hub)
│   ├── ocr/<video_id>/0001.json            # chữ nhận dạng theo keyframe
│   ├── asr/<video_id>.json                 # transcript theo segment thời gian
│   └── videos/<video_id>.mp4               # tham khảo
├── features/clip/<video_id>.npy            # C1: mỗi video một file features
├── artifacts/
│   ├── clip_row_mapping.jsonl              # vector_index ↔ video/frame/keyframe
│   ├── clip.index                          # FAISS index chung
│   ├── object_index.json                   # inverted index vật thể
│   ├── ocr_index.sqlite3                   # token OCR → vector_index
│   ├── asr_index.sqlite3                   # token ASR → segment thời gian
│   └── engine_results.json                 # kết quả search mới nhất
├── evals/                                  # GT + dự đoán mẫu để test chấm điểm
└── scripts/                                # toàn bộ module (bảng bên dưới)
```

## Quy trình tổng thể (đã xây dựng cùng nhau)

### Giai đoạn 1 — BUILD (chạy mỗi khi nhận dữ liệu mới)

```
keyframes + metadata JSON          →  build_mapping.py      → clip_row_mapping.jsonl
ảnh keyframe                       →  encode_clip_features.py → features/clip/<video>.npy
các .npy + mapping                 →  build_faiss_index.py  → clip.index
                                      (bất biến C1: index.ntotal = tổng keyframe = số dòng mapping)
objects JSON (Faster R-CNN)        →  build_object_index.py → object_index.json
OCR JSON + mapping                 →  build_ocr_index.py    → ocr_index.sqlite3
ASR JSON + mapping                 →  build_asr_index.py    → asr_index.sqlite3
```

```bash
python scripts/build_mapping.py
python scripts/encode_clip_features.py
python scripts/build_faiss_index.py
python scripts/build_object_index.py
python scripts/build_ocr_index.py
python scripts/build_asr_index.py
python scripts/validate_baseline_setup.py
python scripts/test_faiss_image_query.py   # sanity: hạng 1 phải là chính ảnh query, score ~1
```

`build_object_index.py` mặc định làm đúng notebook baseline: chỉ index
detection có `score > 0.4`. Có thể đổi ngưỡng để thí nghiệm bằng
`--min_score`, nhưng nên giữ `0.4` khi đối chứng baseline.

Khi dùng Object để xếp hạng, nhiều box cùng một class trong một ảnh chỉ lấy
confidence lớn nhất; hệ thống không còn cộng lặp tất cả box và đẩy sai các
ảnh có nhiều detection cùng loại lên đầu.

OCR/ASR dùng SQLite để không phải mở lại hàng trăm nghìn JSON ở mỗi lần
khởi động. Hai index lưu SHA-256 của `clip_row_mapping.jsonl`; nếu mapping
thay đổi, retriever sẽ yêu cầu build lại thay vì âm thầm trả sai keyframe.
Retriever vẫn fallback về JSON thô khi chưa có SQLite để tiện test nhỏ.

### Giai đoạn 2 — QUERY (lúc thi)

Một cửa duy nhất: `search_engine.py` tự phân loại và xử lý cả 3 dạng:

```bash
python scripts/search_engine.py "a person riding a motorcycle"
```

Đối chứng đúng nhánh retrieval của notebook baseline (chỉ CLIP Visual,
không trộn Object/OCR/ASR và không khử keyframe gần nhau):

```bash
python scripts/search_engine.py "forest" --type kis --top_k 100 --baseline
```

Chọn thủ công các nhánh khi thử nghiệm ablation:

```bash
python scripts/search_engine.py "rừng cây" --type kis --top_k 100 --modalities visual,object
```

```
query
 │
 ├─ query_router: đây là KIS, Q&A hay TRAKE?
 │
 ├─ KIS:   CLIP lấy pool 300 ứng viên
 │         → fusion RRF với Object/OCR/ASR (modality nào thiếu dữ liệu
 │           thì TỰ BỎ QUA, không làm search thất bại)
 │         → xếp theo final_score
 │         → khử các frame quá gần nhau trong cùng video
 │         → trả top-100 <video_id>, <frame_id>
 │         → (hook C9: gọi frame_refinement của A nếu có module)
 │
 ├─ Q&A:   tách (event_description, question) bằng rule
 │         → retrieval sự kiện → EvidenceBundle (frame hạng 1 ± 1 keyframe)
 │         → ViLT VQA trả lời, kèm confidence và bằng chứng OCR/ASR
 │         → trả <video_id>, <frame_id>, <answer>
 │
 └─ TRAKE: tách chuỗi events bằng rule
           → dịch từng event Việt→Anh rồi lấy top-K ứng viên CLIP
           → beam search nhiều chuỗi frame tăng dần trong cùng video
           → event thiếu ứng viên thì nội suy + bám keyframe gần nhất
           → trả <video_id>, [frame_id_1..n]
```

Công thức fusion (C4): `RRF(d) = Σ_r w_r / (60 + rank_r(d))`, trọng số
mặc định visual=1.0, object=0.7, ocr=0.6, asr=0.6.

Tùy chọn nhanh:

```bash
python scripts/search_engine.py "..." --type kis --top_k 100 --objects person,car
python scripts/search_engine.py "Trong cảnh có biển cảnh báo, biển có màu gì?" --type qa --top_k 5
python scripts/search_engine.py "1. chạy đà 2. giậm nhảy 3. tiếp đất" --type trake --top_k 100
```

Mỗi lần chạy sẽ tạo CSV submission UTF-8, phân cách bằng dấu phẩy, dùng
LF và không có header. Mặc định file nằm tại `artifacts/query-<type>.csv`;
dùng `--output` để đặt tên theo query của BTC:

```bash
python scripts/search_engine.py "..." --type kis --output query-1-kis.csv
python scripts/search_engine.py "..." --type qa --output query-2-qa.csv
python scripts/search_engine.py "..." --type trake --output query-3-trake.csv
```

Answer Q&A được chuẩn hóa về một dòng và tự cắt còn tối đa 100 ký tự.
TRAKE chỉ ghi các chuỗi có đúng số event và frame tăng nghiêm ngặt.

Script cũ vẫn dùng được nếu muốn đi từng bước thủ công:
`search_text_query.py` (chỉ CLIP) rồi `rerank_objects.py` (chỉ object).

### Giai đoạn 3 — EVALUATE

```bash
python scripts/evaluation.py --gt evals/sample_gt.json --pred evals/sample_predictions.json
```

Chấm đúng công thức BTC: KIS/Q&A/TRAKE đều tính R-Score từng câu trả
lời → R@1/5/20/50/100 → final. TRAKE có thêm chẩn đoán top-1:
đúng video không, đúng thứ tự không, tỉ lệ event đúng.

## Bảng module (đầu ra bắt buộc của C)

| File | Nhiệm vụ |
|---|---|
| `build_mapping.py` | quét keyframe + metadata → mapping (bước 1) |
| `encode_clip_features.py` | CLIP ViT-B-32 openai → `features/clip/<video>.npy` |
| `build_faiss_index.py` | gộp features → 1 FAISS IndexFlatIP + bất biến C1 |
| `build_object_index.py` | objects JSON → inverted index class→keyframe |
| `build_ocr_index.py` | OCR JSON/JSONL → SQLite token index |
| `build_asr_index.py` | ASR segment JSON/JSONL → SQLite token/time index |
| `search_types.py` | `SearchResult` thống nhất (C2) + helpers |
| `visual_retriever.py` | CLIP text → FAISS (clip_score) |
| `object_retriever.py` | vật thể Faster R-CNN (object_score) |
| `ocr_retriever.py` | tìm chữ trong ảnh từ SQLite, fallback JSON |
| `asr_retriever.py` | tìm lời thoại/segment từ SQLite, ánh xạ thời gian về keyframe |
| `fusion.py` | Reciprocal Rank Fusion (C4) |
| `query_router.py` | phân loại KIS/Q&A/TRAKE + tách events (rule-based) |
| `search_engine.py` | KIS top-100 (C3) + dispatch Q&A/TRAKE + hook C9 |
| `qa_engine.py` | Q&A solver: retrieval + ViLT VQA + OCR/ASR evidence |
| `trake_engine.py` | TRAKE K-best sequence search bằng beam search (C8) |
| `evaluation.py` | chấm R@k theo công thức BTC (C10) |

Model: OpenAI CLIP `ViT-B-32` (`ViT-B-32-quickgelu` trong open_clip) —
trùng họ model BTC cung cấp. Features chuẩn hóa L2, index inner
product ⇒ điểm là cosine similarity.

Với KIS tiếng Việt, Object tự dịch Việt→Anh trước khi tra vocabulary
OpenImages; Visual/OCR/ASR giữ query gốc. Riêng Q&A và TRAKE dịch phần mô tả
sự kiện Việt→Anh cho CLIP vì ViT-B/32 hoạt động tốt hơn với tiếng Anh.

## Ghi chú chiến lược

- Object re-ranking KHÔNG lọc cứng keyframe thiếu object (Faster
  R-CNN bỏ sót) — chỉ cộng điểm, theo đúng C5.
- Q&A dùng mặc định `dandelin/vilt-b32-finetuned-vqa`. Trên Kaggle offline,
  gắn model dưới dạng Dataset rồi đặt `AIC_VQA_MODEL` tới thư mục model.
- Kết quả chi tiết được lưu tại `artifacts/qa_results.json` và
  `artifacts/trake_results.json`.
- TRAKE: đoạn khung hình đáp án thường < 10 frame, nên sau khi có
  chuỗi thô cần frame refinement (C9, chờ thành viên A) để chốt frame.
