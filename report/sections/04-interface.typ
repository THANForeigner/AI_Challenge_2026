= Giao diện Hệ thống và Công tác Vận hành

Hệ thống được thiết kế dưới dạng ứng dụng Web với backend **FastAPI** và frontend tối ưu cho thao tác nhanh trong giờ thi.

== Luồng tương tác
1.  **Giao diện Truy vấn (Query Input)**: Ô tìm kiếm hỗ trợ nhập ngôn ngữ tự nhiên. Người dùng có thể bật/tắt các bộ lọc modality (Chỉ tìm CLIP, Chỉ tìm ASR, Chỉ tìm OCR...).
2.  **Lưới Kết quả (Grid View)**: Kết quả trả về trong vài chục mili-giây, hiển thị dưới dạng lưới các thumbnail keyframe. Mỗi thumbnail đính kèm thông tin ideo_id, rame_index, 	imestamp_ms và điểm số tương đồng (score).
3.  **Chi tiết Keyframe (Detail View)**: Khi click vào một frame, hệ thống sẽ pop-up chi tiết, hiển thị bối cảnh thời gian (các frame trước/sau) nhờ file 	emporal_relations.jsonl. Giao diện cũng hiện rõ các hộp kiểm OCR và danh sách Vật thể (Object) trong frame đó.
4.  **Tích hợp Nộp bài (Submission)**: Cung cấp nút bấm để gửi trực tiếp kết quả (KIS, QA, TRaKE) thông qua định dạng chuẩn (CSV-like) hoặc API hệ thống chấm.

== Kinh nghiệm Vận hành
Qua 3 đợt thi, đội nhận thấy việc có nhiều Index chuyên biệt (ASR, OCR, Object, Visual) giúp chia nhỏ công việc hiệu quả cho các thành viên. Một người chuyên xử lý các câu hỏi chữ (tìm OCR), một người chuyên tìm theo âm thanh (ASR), một người tìm theo ngữ nghĩa hình ảnh (CLIP). Việc kết hợp linh hoạt giúp bao quát toàn bộ phổ câu hỏi của ban tổ chức.
