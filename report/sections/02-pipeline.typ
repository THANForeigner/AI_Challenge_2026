= Giải pháp, Thuật toán và Mô hình AI

Hệ thống tích hợp nhiều mô hình Trí tuệ Nhân tạo hiện đại nhất để bóc tách thông tin từ đa chiều (hình ảnh, âm thanh, văn bản).

== Xử lý Keyframe và Deduplication
- Sử dụng **OpenCV** để trích xuất với tần suất 1 frame/giây. 
- Nhằm tối ưu hóa số lượng ảnh cần tìm kiếm, hệ thống tự động loại bỏ các frame trùng lặp bằng cách tính toán 3D Color Histogram (không gian BGR, 8x8x8 bins) và sử dụng hàm tương quan (correlation). Các frame có độ tương đồng lớn hơn hoặc bằng 0.95 sẽ bị loại bỏ, giúp giảm nhiễu.

== Nhận dạng Giọng nói (ASR)
- **Mô hình chính**: OpenAI **Whisper-medium**. Phân mảnh âm thanh thành các đoạn (segments) có dấu thời gian (start, end).
- **Hậu xử lý (LMM)**: Sử dụng mô hình **Qwen3-VL-8B-Instruct**. Lời thoại từ video tiếng Việt thường bị nhận dạng sai dấu hoặc sai từ do nhiễu âm thanh. Qwen3-VL được tinh chỉnh qua prompt để sửa lỗi chính tả, chuẩn hóa văn bản mà không tự ý thêm bớt nội dung.
- **Index**: SQLite FTS5 (Full-text search), dung lượng 7.62MB, 11.061 segments.

== Nhận dạng Văn bản trong Ảnh (OCR)
- **Mô hình chính**: **EasyOCR** hỗ trợ song ngữ (vi + en). Trích xuất các hộp văn bản (bounding boxes) và sắp xếp theo thứ tự đọc (từ trên xuống, trái sang phải).
- **Hậu xử lý (LMM)**: Sử dụng **Qwen3-VL-2B-Instruct**. Sửa lỗi nhận diện sai ký tự đặc thù của tiếng Việt và ngắt câu hợp lý.
- **Index**: SQLite FTS5, dung lượng 338.87MB.

== Phát hiện Vật thể (Object Detection)
- **Mô hình**: **Faster R-CNN InceptionResNetV2** (TensorFlow Hub, huấn luyện trên OpenImages V4).
- **Tính năng**: Phát hiện hơn 600 lớp vật thể. Hệ thống sử dụng thêm bộ từ điển (alias) để ánh xạ các nhãn tiếng Anh sang tiếng Việt, hỗ trợ truy vấn bằng cả hai ngôn ngữ.
- **Index**: SQLite Database, ghi nhận 905.974 detections thuộc 503 classes.

== Chỉ mục Không gian Visual (CLIP)
- **Mô hình**: **CLIP ViT-B-32-quickgelu** (từ thư viện open_clip_torch). 
- **Chỉ mục**: Chuyển đổi toàn bộ 196.590 keyframe thành vector 512 chiều. Xây dựng chỉ mục **FAISS** (IndexFlatIP) để thực hiện tìm kiếm ảnh bằng văn bản (Text-to-Image) với độ trễ tính bằng mili-giây.

== Chiến lược kết hợp Đa phương thức (Multi-modal Fusion)
Hệ thống cho phép thực hiện song song các loại truy vấn: Semantic (CLIP), Lời thoại (ASR), Chữ viết trên màn hình (OCR) và Vật thể (Object). Sau đó, kết quả được hợp nhất bằng thuật toán Reciprocal Rank Fusion (RRF) có trọng số và áp dụng cơ chế score gating để loại bỏ các nhiễu từ tín hiệu yếu.

