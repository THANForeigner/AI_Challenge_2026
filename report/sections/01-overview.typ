= Tổng quan hệ thống

Hệ thống của đội được phát triển nhằm mục đích giải quyết bài toán truy xuất thông tin đa phương thức từ tập dữ liệu video quy mô lớn trong khuôn khổ Hội thi AI Challenge Thành phố Hồ Chí Minh 2026. 

Với **873 video** (từ L21 đến L30), hệ thống đã trích xuất thành công **196.590 keyframe** và xây dựng một pipeline end-to-end hoàn chỉnh bao gồm 4 giai đoạn chính:

1.  **Trích xuất Keyframe**: Xử lý dữ liệu thô (ZIP), trích xuất hình ảnh, loại bỏ trùng lặp (deduplication) và chuẩn hóa.
2.  **Trích xuất Đặc trưng Đa phương thức**: Thực hiện nhận dạng giọng nói (ASR), nhận dạng văn bản (OCR) và phát hiện vật thể (Object Detection).
3.  **Lập Chỉ mục (Indexing)**: Xây dựng các cấu trúc tìm kiếm tối ưu (FAISS cho vector, SQLite FTS cho văn bản).
4.  **Hệ thống Truy vấn (Search Engine)**: Giao diện web cho phép tìm kiếm và kết hợp đa phương thức (Multi-modal Fusion).

Điểm nổi bật của hệ thống là khả năng xử lý khối lượng dữ liệu khổng lồ trong điều kiện tài nguyên hạn chế (bộ nhớ trong ~35GB) nhờ thiết kế luồng xử lý "Tải -> Giải nén -> Xử lý -> Xóa" hiệu quả, kết hợp cơ chế lưu điểm kiểm tra (checkpoint) để có thể tiếp tục tự động sau khi bị gián đoạn.
