= Tổng quan hệ thống

Hệ thống của đội được phát triển nhằm mục đích giải quyết bài toán truy xuất thông tin đa phương thức từ tập dữ liệu video quy mô lớn trong khuôn khổ Hội thi AI Challenge Thành phố Hồ Chí Minh 2026. 

Với **873 video** (từ L21 đến L30), hệ thống đã trích xuất thành công **196.590 keyframe** và xây dựng một pipeline end-to-end hoàn chỉnh bao gồm 4 giai đoạn chính:

1.  **Trích xuất Keyframe**: Xử lý dữ liệu thô (ZIP), trích xuất hình ảnh, loại bỏ trùng lặp (deduplication) và chuẩn hóa.
2.  **Trích xuất Đặc trưng Đa phương thức**: Thực hiện nhận dạng giọng nói (ASR), nhận dạng văn bản (OCR) và phát hiện vật thể (Object Detection).
3.  **Lập Chỉ mục (Indexing)**: Xây dựng các cấu trúc tìm kiếm tối ưu (FAISS cho vector, SQLite FTS cho văn bản).
4.  **Hệ thống Truy vấn (Search Engine)**: Giao diện web cho phép tìm kiếm và kết hợp đa phương thức (Multi-modal Fusion).

Điểm nổi bật của hệ thống là khả năng xử lý khối lượng dữ liệu khổng lồ trong điều kiện tài nguyên hạn chế (bộ nhớ trong ~35GB) nhờ thiết kế luồng xử lý "Tải -> Giải nén -> Xử lý -> Xóa" hiệu quả, kết hợp cơ chế lưu điểm kiểm tra (checkpoint) để có thể tiếp tục tự động sau khi bị gián đoạn.

== Môi trường và Các Công cụ AI
Để xây dựng và vận hành toàn bộ hệ thống, đội đã sử dụng phối hợp nhiều tài nguyên và công cụ mã nguồn mở:
- **Môi trường tính toán**: Hệ thống notebook của **Kaggle** được sử dụng để tận dụng phần cứng GPU miễn phí nhằm xử lý khối lượng lớn video, trích xuất đặc trưng và xây dựng chỉ mục.
- **Mô hình Trí tuệ Nhân tạo (AI Models)**: Sử dụng mô hình **CLIP (ViT-B-32-quickgelu)** cho đối sánh hình ảnh - văn bản; **Whisper-medium** của OpenAI cho nhận dạng giọng nói (ASR); **EasyOCR** cho nhận dạng chữ trên màn hình; **Faster R-CNN** cho phát hiện vật thể. Đặc biệt, đội ứng dụng mô hình ngôn ngữ lớn đa phương thức **Qwen3-VL (2B và 8B)** để hậu xử lý tự động.
- **Lưu trữ và Cơ sở dữ liệu**: Sử dụng **FAISS** để tìm kiếm vector tốc độ cao và **SQLite FTS5** để tìm kiếm văn bản toàn văn (Full-Text Search).
- **Phát triển phần mềm**: Giao diện người dùng (UI) được xây dựng bằng **React/Vite**, kết nối với server **FastAPI** (Python).
