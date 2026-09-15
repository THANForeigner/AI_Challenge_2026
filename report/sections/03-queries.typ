= Phân tích Tình huống Truy vấn Tiêu biểu

Bộ câu hỏi P2 gồm 36 câu: 26 KIS, 8 Q&A và 2 TRAKE. Theo yêu cầu kỹ thuật, bộ đề gồm 15 câu KIS thị giác–đối tượng, 6 câu KIS có diễn tiến thời gian, 5 câu KIS có chữ hoặc sơ đồ, 6 câu Q&A dựa trên OCR/ASR, 1 câu đếm thị giác, 1 câu suy luận logic và 2 câu căn chỉnh chuỗi sự kiện.

Dưới đây là 5 tình huống tiêu biểu được chọn lọc nhằm bao quát các năng lực và giới hạn hiện tại của hệ thống:

== Tình huống 1: Concept thị giác hiếm (KIS - Xe lội nước kiểu ô tô cổ)
- **Mã câu hỏi**: p2-10-kis
- **Đặc điểm & Tiếp cận**: Cần tìm một hình ảnh rất đặc trưng ("xe lội nước").
- **Kết quả & Bài học**: Với concept có hình dạng thị giác đặc trưng như xe lội nước kiểu ô tô cổ, CLIP là nhánh truy xuất phù hợp nhất; Object Index đóng vai trò bổ trợ do tập nhãn chỉ biểu diễn các lớp vật thể tổng quát.

== Tình huống 2: Phối hợp OCR và Visual (KIS - Slide bài giảng tiếng Anh)
- **Mã câu hỏi**: p2-4-kis
- **Đặc điểm & Tiếp cận**: Truy vấn tìm hình ảnh chứa các ví dụ ngữ pháp tiếng Anh. Nếu chỉ dùng CLIP tìm "giáo viên nữ", hệ thống sẽ bị nhiễu do có nhiều frame tương tự. Visual định vị bối cảnh lớp học, OCR đọc các câu ví dụ và công thức trên màn hình, còn ASR bổ trợ bằng nội dung giảng giải được phát âm trong video.
- **Bài học**: Sự kết hợp đa phương thức này được sử dụng để thu hẹp không gian ứng viên. Text index (OCR) là tín hiệu "vân tay" xuất sắc cho các truy vấn chứa văn bản rõ ràng.

== Tình huống 3: Trích xuất số lượng và đơn vị (QA - Lượng nước tương)
- **Mã câu hỏi**: p2-24-qa
- **Đặc điểm & Tiếp cận**: Cần tìm số lượng nước tương được sử dụng. Hệ thống định vị scene, chạy OCR/ASR, và trích xuất ra các con số. Tuy nhiên, các ứng viên trả về bao gồm "200g", "3l", "3muỗng" phân tán ở nhiều video khác nhau.
- **Kết quả & Bài học**: Hệ thống đã nhận dạng được chữ và số, nhưng bộc lộ hạn chế trong việc "gắn số đo với đúng thực thể" (nước tương). Cần cải thiện việc liên kết (grounding) thông tin số lượng với ngữ cảnh thay vì chỉ bốc tách con số nổi bật nhất.

== Tình huống 4: Suy luận logic nhiều bước (QA - Số nhóm X)
- **Mã câu hỏi**: p2-35-qa
- **Đặc điểm & Tiếp cận**: Đáp án không xuất hiện trực tiếp mà phải suy luận từ 4 câu trắc nghiệm trên màn hình (đều chứa chữ số 0). Cơ chế text-mode hiện tại chỉ trích xuất các chữ số thô (ví dụ: 3020, 73, 2, 0) mà chưa giải quyết được bài toán điều kiện logic.
- **Bài học**: Có một khoảng trống lớn giữa việc "truy xuất văn bản thuần túy" và "suy luận logic". Cần một mô-đun LLM/VLM đủ mạnh chạy trên nhiều frame để giải quyết dạng câu hỏi này.

== Tình huống 5: Căn chỉnh chuỗi sự kiện (TRAKE - Bốn mốc làm món tôm)
- **Mã câu hỏi**: p2-21-trake
- **Đặc điểm & Tiếp cận**: Khác với định dạng cũ, TRAKE đợt 3 yêu cầu 4 mốc thời gian diễn ra **trong cùng một video** và phải **tăng nghiêm ngặt** về frame (E1 < E2 < E3 < E4). Engine TRAKE tách 4 event, tìm ứng viên, chuẩn hóa điểm và chạy thuật toán quy hoạch động (K-best DP) để tìm ra chuỗi sự kiện liền mạch nhất.
- **Kết quả**: Hệ thống xếp hạng cao nhất chuỗi ứng viên trong video L26_V156 với các frame 2875, 3825, 4900 và 4925. Chuỗi thỏa điều kiện cùng video và thứ tự frame tăng nghiêm ngặt; độ chính xác nội dung cần được xác nhận bằng kết quả chấm chính thức.
