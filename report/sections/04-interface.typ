= Giao diện Hệ thống và Công tác Vận hành

Hệ thống được thiết kế dưới dạng ứng dụng Web hiện đại (xây dựng bằng React/Vite) giao tiếp với backend thông qua REST API, tối ưu hóa cho thao tác tìm kiếm tốc độ cao trong giờ thi.


#figure(
  image("../image/image.png", width: 100%),
  caption: [Giao diện ứng dụng tìm kiếm đa phương thức của đội]
) <ui_image>
== Kiến trúc Giao diện Người dùng (UI)
Giao diện được chia thành hai phần chính: Sidebar (Cột điều hướng bên trái) và Main Area (Vùng hiển thị kết quả chính).

1.  **Sidebar - Bảng Điều khiển Truy vấn**:
    - **Mode Search**: Hỗ trợ chuyển đổi nhanh giữa tìm kiếm văn bản (Text), tìm kiếm bằng hình ảnh (Image/Upload) và tìm kiếm chuỗi sự kiện (Temporal).
    - **Model Selection**: Cho phép người dùng chọn mô hình xử lý (CLIP, BEiT3, BLIP2, MiniGPT-4, LLaVA).
    - **Bộ lọc đa phương thức**: Cung cấp ô nhập liệu (Filter Panel) để lọc riêng theo các từ khóa xuất hiện trong **OCR** (văn bản trên màn hình) hoặc **ASR** (lời thoại).
    - **Tùy chỉnh hệ thống**: Cho phép điều chỉnh số lượng kết quả trả về (Top K) và thay đổi API Base URL linh hoạt.

2.  **Main Area - Lưới Kết quả (Result Grid)**:
    - Hiển thị tổng số kết quả tìm được và thời gian truy vấn thực tế (chỉ vài chục mili-giây).
    - **Grid Density**: Cho phép thay đổi mật độ hiển thị (từ 2 đến 7 cột) để dễ dàng lướt qua hàng trăm keyframe cùng lúc.
    - Mỗi thẻ (Keyframe Card) hiển thị ảnh thumbnail và điểm số tương đồng (Score).

3.  **Detail Modal - Trình xem chi tiết**:
    - Khi click vào một keyframe, hệ thống sẽ mở pop-up phóng to hình ảnh cùng các nút điều hướng (Next/Previous).
    - Hiển thị đầy đủ thông số: `video_id`, `frame_index`, `score`, và `timestamp` (được định dạng chuẩn mm:ss).
    - **Trích xuất thông tin**: Nếu frame có văn bản, hộp thoại sẽ tự động hiển thị box text **OCR** (chữ trên hình) và **ASR** (lời thuyết minh).
    - **Công cụ nộp bài**: Cung cấp nút *"Copy video_id, frame"* sao chép nhanh chuỗi định dạng nộp kết quả chuẩn (ví dụ: `L26_V156, 4925`), và nút *"Find similar"* để tự động chuyển sang chế độ tìm kiếm bằng ảnh tương tự.

== Kinh nghiệm Vận hành
Qua các đợt thi, đội nhận thấy việc tách biệt bộ lọc OCR/ASR ngay trên giao diện giúp các thành viên chia nhỏ công việc rất hiệu quả. Một người có thể tìm kiếm theo ngữ nghĩa hình ảnh thông thường, trong khi một người khác khai thác sâu các khung hình có lời thoại cụ thể. Các nút Copy nhanh (Clipboard) và Find Similar tiết kiệm được rất nhiều thời gian thao tác tay, giúp đội tăng tốc độ chốt đáp án đối với các câu hỏi khó.


