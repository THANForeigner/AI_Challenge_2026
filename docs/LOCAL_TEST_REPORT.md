# Báo cáo kiểm thử local KIS, Q&A và TRAKE

Ngày chạy: 19/08/2026

## Phạm vi

Bộ test dùng video local giả lập có `video_id = testing` và 101 keyframe.
Ground truth được gán theo khoảng frame của từng cảnh sau khi kiểm tra trực
quan. Đây là kiểm thử phát triển để phát hiện lỗi pipeline và so sánh cấu hình,
không phải điểm chính thức trên tập đánh giá AIC 2026.

Dữ liệu local tại thời điểm chạy:

- 101 keyframe.
- Object JSON đã được build thành `object_index.json`.
- 101 OCR JSON.
- 1 ASR JSON của video `testing`.
- FAISS có 101 vector.

Cấu hình của lần chạy báo cáo:

- KIS: `visual,object,ocr,asr`, Top-20.
- Q&A retrieval: `visual`; trả lời bằng ViLT.
- TRAKE: chế độ tự động; Visual cho mọi event và thêm OCR cho event có dấu
  hiệu văn bản/biển báo.
- Query đầu vào bằng tiếng Việt, được dịch nội bộ cho CLIP/Object.

## Kết quả tổng hợp

| Nhóm | Chỉ số | Kết quả |
|---|---:|---:|
| KIS (7 query) | R@1 | 42,9% |
| KIS (7 query) | R@5 | 85,7% |
| KIS (7 query) | R@20 | 100% |
| Q&A (3 query) | Đúng vị trí | 100% |
| Q&A (3 query) | Đúng câu trả lời | 100% |
| Q&A (3 query) | Đúng cả vị trí và đáp án | 100% |
| TRAKE (3 chuỗi, 9 event) | Đúng event | 100% |
| TRAKE (3 chuỗi) | Đúng toàn bộ chuỗi | 100% |

## Chi tiết KIS

| ID | Query | Khoảng đúng | Rank đúng đầu tiên |
|---|---|---:|---:|
| kis_01 | con đường ven sông bị sạt lở | 1–15 | 2 |
| kis_02 | biển cảnh báo sạt lở nguy hiểm | 16–22 | 2 |
| kis_03 | bờ biển và rừng nhìn từ trên cao | 27–34 | 9 |
| kis_04 | vòi nước phun cao lên trời | 47–60 | 2 |
| kis_05 | hai người đàn ông khiêng một chiếc thùng | 64–73 | 1 |
| kis_06 | đàn cá trong ao nuôi | 76–85 | 1 |
| kis_07 | đàn trâu đang ăn cỏ trên cánh đồng | 90–101 | 1 |

Điểm cần cải thiện rõ nhất là query `kis_03`, vì kết quả đúng chỉ xuất hiện ở
rank 9. Các query còn lại đều có kết quả đúng trong Top-2.

## Chi tiết Q&A

| ID | Câu hỏi | Frame | Đáp án | Kết quả |
|---|---|---:|---|---|
| qa_01 | Biển cảnh báo có màu gì? | 18 | vàng | Đúng |
| qa_02 | Có bao nhiêu người đang khiêng thùng? | 68 | 2 | Đúng |
| qa_03 | Có bao nhiêu con trâu? | 101 | 3 | Đúng |

## Chi tiết TRAKE

| ID | Chuỗi sự kiện | Frame dự đoán | Kết quả |
|---|---|---|---|
| trake_01 | sạt lở → biển cảnh báo → bờ biển | `[5, 18, 30]` | 3/3 event |
| trake_02 | vòi nước → khiêng thùng → ao cá | `[58, 68, 80]` | 3/3 event |
| trake_03 | biển cảnh báo → khiêng thùng → đàn trâu | `[18, 68, 101]` | 3/3 event |

## Cách chạy lại

Mở PowerShell và chuyển vào project:

```powershell
cd D:\competition\aic2026\aic_practice
```

Chạy toàn bộ bộ test:

```powershell
..\.venv\Scripts\python.exe scripts\run_local_tests.py --group all --top_k 20
```

Chạy riêng từng nhóm:

```powershell
..\.venv\Scripts\python.exe scripts\run_local_tests.py --group kis --top_k 20
..\.venv\Scripts\python.exe scripts\run_local_tests.py --group qa
..\.venv\Scripts\python.exe scripts\run_local_tests.py --group trake
```

So sánh ablation cho KIS:

```powershell
# Chỉ CLIP Visual
..\.venv\Scripts\python.exe scripts\run_local_tests.py --group kis --modalities visual

# Visual + Object
..\.venv\Scripts\python.exe scripts\run_local_tests.py --group kis --modalities visual,object

# Đầy đủ bốn nhánh
..\.venv\Scripts\python.exe scripts\run_local_tests.py --group kis --modalities visual,object,ocr,asr
```

Test Q&A với nhiều modality retrieval hơn:

```powershell
$env:AIC_QA_MODALITIES="visual,ocr,asr"
..\.venv\Scripts\python.exe scripts\run_local_tests.py --group qa
Remove-Item Env:AIC_QA_MODALITIES
```

## Test trực tiếp từng chức năng

KIS:

```powershell
..\.venv\Scripts\python.exe scripts\search_engine.py "đàn trâu đang ăn cỏ trên cánh đồng" --type kis --top_k 20 --modalities visual,object,ocr,asr
```

Q&A:

```powershell
..\.venv\Scripts\python.exe scripts\search_engine.py "cảnh có biển cảnh báo sạt lở nguy hiểm, biển cảnh báo có màu gì?" --type qa --top_k 5
```

TRAKE:

```powershell
..\.venv\Scripts\python.exe scripts\search_engine.py "1. con đường ven sông bị sạt lở; 2. xuất hiện biển cảnh báo sạt lở nguy hiểm; 3. bờ biển và rừng nhìn từ trên cao" --type trake --top_k 5
```

## File đầu ra

Báo cáo máy đọc của lần chạy gần nhất:

```text
artifacts/local_test_suite_results.json
```

Các chức năng cũng cập nhật file riêng:

```text
artifacts/engine_results.json
artifacts/qa_results.json
artifacts/trake_results.json
```

Lưu ý: mỗi lần chạy trực tiếp có thể ghi đè file kết quả tương ứng. File
`local_test_suite_results.json` gom kết quả của toàn bộ case trong lần chạy
runner gần nhất.
