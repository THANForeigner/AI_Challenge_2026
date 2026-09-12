= Phân tích Tình huống Truy vấn Tiêu biểu

Phân tích kết quả thi từ 3 đợt Vòng Sơ tuyển (21/8, 28/8 và 4/9/2026), đội đã rút ra một số tình huống truy vấn tiêu biểu, thể hiện độ khó của đề thi cũng như sức mạnh của hệ thống.

== Tình huống 1: Hỏi địa danh (QA - Đèo Hải Vân)
- **Đặc điểm câu hỏi (R1/Q17)**: Tìm frame có xuất hiện địa danh cụ thể. Đáp án đúng là "đèo Hải Vân".
- **Cách tiếp cận & Xử lý**: Hệ thống không thể chỉ dựa vào CLIP vì "đèo Hải Vân" khó nhận dạng thuần qua hình ảnh địa hình đối với model zero-shot. Đội sử dụng **ASR Index** và **OCR Index** để tìm các từ khóa "đèo", "Hải Vân". Hệ thống phát hiện lời thuyết minh (Whisper) hoặc biển báo (EasyOCR) và gợi ý các video L22_V020, L21_V006, L22_V005. 
- **Kết quả**: Truy xuất chính xác và điền câu trả lời "đèo Hải Vân".

== Tình huống 2: Ký hiệu đặc biệt (QA - Tỷ lệ 15‰)
- **Đặc điểm câu hỏi (R2/Q23)**: Tìm frame chứa con số tỷ lệ, đáp án "15‰".
- **Cách tiếp cận & Xử lý**: Ký hiệu phần nghìn "‰" rất dễ bị các engine OCR nhận diện nhầm thành "%" hoặc "0/00". Đội thi kết hợp tìm kiếm con số "15" trong OCR và lắng nghe transcript ASR (nơi người nói đọc "mười lăm phần nghìn"). Sự chỉnh lý của **Qwen3-VL-2B** ở bước hậu xử lý OCR đã giúp bảo toàn được dạng thức của ký hiệu này.
- **Kết quả**: Chốt được đáp án "15‰" trong video L25_V087, L30_V085.

== Tình huống 3: Truy vấn TRaKE phân tán
- **Đặc điểm câu hỏi (R2/Q21)**: Tìm 4 keyframe có liên hệ logic với nhau nhưng nằm rải rác trên 26 video khác nhau. Độ khó cực cao do không thể dò cục bộ (local context) trong 1 video.
- **Cách tiếp cận & Xử lý**: Sử dụng tính năng truy vấn bằng hình ảnh mẫu (Image-to-Image) hoặc CLIP text search trừu tượng. Các vector của 196.590 keyframe được FAISS quét toàn cục. Hệ thống backend hỗ trợ "gom nhóm" (grouping) kết quả theo concept, giúp thành viên phát hiện ra mối nối giữa các video L29_V019, L22_V030, L21_V015...
- **Bài học**: Hệ thống tìm kiếm theo vector (FAISS) hoạt động đặc biệt hiệu quả cho dạng đề TRaKE phân tán.

== Tình huống 4: Văn bản Công thức Khoa học (QA)
- **Đặc điểm câu hỏi (R3/Q6)**: Trích xuất một đoạn văn bản dài về bài toán Vật lý (con lắc đơn). 
- **Cách tiếp cận & Xử lý**: Hình ảnh là một slide bài giảng có chứa các ký hiệu toán học (g = 9.8, biên độ góc). OCR truyền thống thường bị vỡ vụn (fragmented) khi gặp dạng này. Nhờ pipeline sử dụng EasyOCR gom dòng theo chiều dọc và Qwen3-VL-2B sửa lỗi context, đoạn text được khôi phục nguyên vẹn: "CÂU 6 Một con lắc đơn có chiều dài 81cm đang dao động điều hòa...".
- **Kết quả**: Truy vấn chính xác text từ video L25_V085, L25_V049. 

== Tình huống 5: Hỏi tên loài sinh vật (QA - Cá chim)
- **Đặc điểm câu hỏi (R2/Q9)**: Nhận diện loài vật. Đáp án "cá chim".
- **Cách tiếp cận & Xử lý**: Tên loài vật cụ thể bằng tiếng Việt rất hiếm có trong tập nhãn của CLIP hay Faster R-CNN (chủ yếu là nhãn chung "fish" hoặc "animal"). Do đó, nếu nhập query "cá chim" vào CLIP sẽ ít có kết quả chính xác. Đội thi linh hoạt chuyển sang **ASR Index** (tìm đoạn âm thanh MC nhắc đến "cá chim") và kết hợp với **Object Index** (bộ lọc class "fish" đã được alias tiếng Việt).
- **Kết quả**: Tìm thấy frame chứa "cá chim" ở L26_V360, L26_V446.
