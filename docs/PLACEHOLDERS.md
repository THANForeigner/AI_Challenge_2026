# PLACEHOLDERS — các phần đang là MÃ GIẢ / chờ dữ liệu

Mỗi mục ghi rõ: đang thiếu gì, lấy từ đâu, định dạng mong đợi, và
code hiện xử lý ra sao khi chưa có. Khi nhận được dữ liệu thật,
làm theo bước "Khi có dữ liệu" rồi chạy lại lệnh tương ứng.

---

## 1. Metadata frame index của 4 video L21 (GIẢ)

- **File**: `data/metadata/L21_V008.json`, `L21_V014.json`,
  `L21_V027.json`, `L21_V029.json` (+ `PLACEHOLDER_NOTE.txt`)
- **Hiện trạng**: frame_index = số thứ tự keyframe (1, 2, 3...) —
  KHÔNG phải frame thật trong video. Chỉ dùng để test pipeline.
- **Cần từ**: thành viên cắt keyframe (file JSON theo format BTC).
- **Khi có dữ liệu**: xóa/ghi đè 4 file (hoặc đặt 1 file gộp tên
  khác tên video vào `data/metadata/`) → chạy lại
  `build_mapping.py` + `build_faiss_index.py`
  (KHÔNG cần chạy lại `encode_clip_features.py` nếu danh sách video
  không đổi — features chỉ phụ thuộc ảnh).

## 2. Objects JSON trong data/objects/ (GIẢ)

- **Hiện trạng**: toàn bộ `data/objects/` là detection random sinh
  để test, không phải kết quả Faster R-CNN thật.
- **Cần từ**: thành viên phụ trách object detection / dữ liệu BTC.
- **Khi có dữ liệu**: XÓA toàn bộ `data/objects/` → copy dữ liệu
  thật vào (giữ layout `data/objects/<video_id>/<keyframe>.json`) →
  chạy `build_object_index.py`.

## 3. OCR retriever — chưa có dữ liệu OCR

- **Chờ**: kết quả OCR từ thành viên khác (team repo có
  `scripts/run_ocr.py`).
- **Định dạng mong đợi**:
  `data/ocr/<video_id>/<keyframe>.json` chứa
  `{"texts": ["dòng chữ 1", ...]}` (hoặc `{"text": "..."}`).
- **Hiện tại**: `ocr_retriever.available() = False` → fusion RRF tự
  bỏ qua modality OCR, search vẫn chạy bình thường (đúng yêu cầu C4).

## 4. ASR retriever — đã tích hợp, timestamp hiện là xấp xỉ

- **Dữ liệu hiện có**: `data/asr/<video_id>.json`, mỗi video một file.
- **Định dạng hỗ trợ**: `start_ms`/`end_ms` tính bằng mili giây hoặc
  `start`/`end` kiểu Whisper tính bằng giây.
- **Hiện tại**: ASR đã tham gia fusion. Khi mapping chưa có
  `timestamp_ms`, retriever nội suy vị trí keyframe theo thứ tự trên toàn
  thời lượng transcript.
- **Còn cần cho dữ liệu thi thật**: timestamp thật của từng keyframe, hoặc
  frame index thật cùng FPS video. Nội suy hiện tại chỉ phù hợp chạy thử.

## 5. VQA model cho Q&A (C6) — ĐÃ TÍCH HỢP

- **Model mặc định**: `dandelin/vilt-b32-finetuned-vqa`.
- **Luồng**: dịch câu hỏi Việt→Anh, retrieval cảnh bằng CLIP, chạy VQA trên
  keyframe chính và frame lân cận, trả answer ngắn cùng confidence/evidence.
- **Kaggle offline**: gắn model dưới dạng Dataset và đặt biến môi trường
  `AIC_VQA_MODEL=/kaggle/input/<dataset>/<model-folder>`.
- **Giới hạn**: ViLT là baseline VQA tiếng Anh; câu hỏi suy luận phức tạp vẫn
  cần model thị giác-ngôn ngữ mạnh hơn để tăng độ chính xác.

## 6. LLM phân tích query (C6/C7) — đang rule-based

- **Vị trí mã giả**: `query_router.llm_parse_query()`.
- **Hiện tại**: router tách Q&A / TRAKE bằng rule (dấu `?`, từ để
  hỏi, mục đánh số 1./2./...). Đủ dùng cho phần lớn truy vấn; khi
  nào rule-based tách sai mới cần LLM (cần API key).

## 7. Frame refinement (C9) — chờ thành viên A

- **Chờ**: module `scripts/frame_refinement.py` của thành viên A,
  interface mong đợi:
  ```python
  def refine(engine, video_id, frame_index) -> int:
      """encode các frame lân cận, trả frame chính xác hơn"""
  ```
- **Hiện tại**: `search_engine.apply_frame_refinement()` tự thử
  import; chưa có module thì bỏ qua lặng lẽ, có module thì tự gọi
  cho kết quả hạng 1 — không cần sửa search_engine.

## 8. Ground truth evaluation (C10) — dữ liệu mẫu

- **File**: `evals/sample_gt.json`, `evals/sample_predictions.json`
  là dữ liệu GIẢ để kiểm tra công thức chấm.
- **Cần từ**: đề + đáp án thật của BTC khi thi.
- **Khi có dữ liệu**: giữ đúng schema trong `evaluation.py`, chạy
  `python scripts/evaluation.py --gt <gt.json> --pred <pred.json>`.

## 9. Ngôn ngữ truy vấn — Object đã có dịch Việt→Anh offline

- Query gốc tiếng Việt được giữ nguyên cho Visual, OCR và ASR.
- Riêng Object tự dịch query sang tiếng Anh bằng
  `Helsinki-NLP/opus-mt-vi-en`, vì vocabulary OpenImages là tiếng Anh.
- Model OpenAI CLIP ViT-B/32 vẫn nhận nguyên query tiếng Việt theo cấu hình
  hiện tại; chưa thay bằng Multilingual CLIP.
