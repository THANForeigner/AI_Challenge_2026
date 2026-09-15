# Ghi chú xây dựng lại Phần 3 — Phân loại câu hỏi vòng P2

## 1. Mục đích và phạm vi bằng chứng

Tài liệu này dùng để thay nội dung **Phần 3. Phân tích tình huống truy vấn tiêu biểu** trong báo cáo hiện tại. Việc phân loại dựa trên ba nguồn:

- 36 câu hỏi vòng P2: 26 câu KIS, 8 câu Q&A và 2 câu TRAKE.
- Các câu trả lời ứng viên trong `submission.zip`.
- Luồng xử lý thực tế trong `scripts/query_router.py`, `scripts/search_engine.py`, `scripts/qa_engine.py`, `scripts/trake_engine.py`, `scripts/fusion.py` và các retriever.

Lưu ý quan trọng: `submission.zip` chỉ chứa **các dự đoán đã nộp**, không phải ground truth. Vì vậy, tài liệu chỉ dùng độ nhất quán hoặc bất thường của các dự đoán để chẩn đoán hệ thống; không gọi các dự đoán đó là đáp án đúng nếu chưa có kết quả chấm chính thức.

## 2. Những nội dung cần thay trong Phần 3 cũ

1. Năm tình huống hiện tại lấy ví dụ từ R1/R2/R3, không đại diện cho bộ 36 câu P2 vừa nộp. Nên thay bằng phân tích theo toàn bộ phân bố câu hỏi P2.
2. Mục 3.3 cũ mô tả TRAKE là các keyframe “nằm rải rác trên 26 video”. Mô tả này không phù hợp với code và định dạng nộp hiện tại. `trake_engine.py` bắt buộc mọi event của một đáp án thuộc **cùng một `video_id`** và thỏa `frame_1 < frame_2 < ... < frame_n`.
3. Không nên dùng kết quả 100% trong `LOCAL_TEST_REPORT.md` như kết quả thi thật. Báo cáo đó ghi rõ chỉ chạy trên một video giả lập gồm 101 keyframe và ground truth tự gán.
4. Không nên khẳng định một mô hình hay bước hậu xử lý đã quyết định kết quả P2 nếu repository và artifact hiện tại không lưu được bằng chứng theo từng câu. Ví dụ, phần code truy vấn hiện tại có OCR/ASR retriever nhưng không có log chứng minh Qwen đã sửa đúng một ký hiệu cụ thể trong câu thi.
5. Nên tách hai tầng phân loại:
   - **Tầng thể lệ:** KIS, Q&A, TRAKE.
   - **Tầng kỹ thuật:** nguồn bằng chứng và kiểu suy luận cần dùng, như thị giác, vật thể, OCR, ASR, đếm, suy luận logic hoặc căn chỉnh thời gian.

## 3. Phân loại tổng hợp mới

| Nhóm kỹ thuật | Số câu | Mã câu | Nhánh xử lý chính |
|---|---:|---|---|
| KIS thị giác–ngữ nghĩa và quan hệ đối tượng | 15 | 7, 9, 10, 11, 12, 13, 17, 22, 23, 25, 28, 30, 31, 32, 36 | CLIP; Object/OCR/ASR bổ trợ tùy câu |
| KIS có chuỗi hành động hoặc quan hệ thời gian | 6 | 1, 3, 15, 19, 29, 33 | CLIP + Object; cần kiểm tra diễn tiến trong video |
| KIS có chữ, số hoặc sơ đồ trên màn hình | 5 | 2, 4, 16, 20, 26 | OCR + CLIP; ASR có thể bổ trợ |
| Q&A trích xuất đáp án từ OCR/ASR | 6 | 5, 6, 8, 14, 18, 24 | Định vị cảnh bằng CLIP, tìm bằng chứng OCR/ASR, trích entity hoặc số |
| Q&A đếm trực tiếp bằng thị giác | 1 | 27 | CLIP + VQA/đếm đối tượng |
| Q&A cần OCR kết hợp suy luận logic | 1 | 35 | OCR nhiều frame + bộ suy luận; code hiện tại chưa đủ |
| TRAKE căn chỉnh chuỗi sự kiện | 2 | 21, 34 | Truy xuất từng event + K-best DP trong cùng video |
| **Tổng** | **36** |  |  |

Phân bố này cho thấy phần lớn đề vẫn là truy xuất thị giác, nhưng 14/36 câu có yêu cầu rõ về thời gian, chữ/lời thoại, đếm hoặc suy luận đáp án. Do đó, chỉ báo cáo CLIP và FAISS là chưa phản ánh đúng độ khó của vòng P2.

## 4. Ánh xạ chi tiết 36 câu hỏi

Ký hiệu modality: **V** = Visual/CLIP, **O** = Object, **C** = OCR, **A** = ASR, **VQA** = hỏi đáp trên ảnh.

### 4.1. Textual KIS — 26 câu

| ID | Phân loại kỹ thuật | Chính | Bổ trợ | Nhận xét theo code hiện tại |
|---|---|---|---|---|
| p2-1 | Chuỗi thao tác nấu ăn | V | O | Nhiều mốc: đổ trứng → cắt đậu hũ → cho vào nồi → khuấy. `search_kis()` hiện mã hóa cả mô tả thành một query, chưa ràng buộc thứ tự các mốc. |
| p2-2 | Cảnh trường học có bảng số lớp | V + C | O | CLIP định vị đám đông học sinh; OCR trên bảng số lớp là tín hiệu phân biệt tốt. |
| p2-3 | Chuỗi thêm nguyên liệu vào nồi | V | O | Cần nhận màu, hình dạng nguyên liệu và thứ tự; là KIS nhưng có bản chất temporal. |
| p2-4 | Slide bài giảng tiếng Anh | C + V | A | Hai câu ví dụ và công thức ngữ pháp là tín hiệu chữ; chỉ tìm “giáo viên nữ” bằng CLIP sẽ có nhiều nhiễu. |
| p2-7 | Hành vi nguy hiểm khi đi xe | V | O | Quan hệ người–xe–làn đường và tư thế bất cẩn là tín hiệu thị giác chính. |
| p2-9 | Thời trang và đồ thủ công ghép vải | V | O | Cần nối các cảnh trình diễn, trưng bày và đồ thủ công bằng cùng concept màu sắc/họa tiết. |
| p2-10 | Xe lội nước kiểu ô tô cổ | V | O | Concept hiếm nhưng rất đặc trưng về hình ảnh; CLIP quan trọng hơn nhãn Object chung. |
| p2-11 | Thành phẩm trắng nở thành các sợi que | V | O | Cần nhận biến đổi hình dạng/vật liệu; nhãn OpenImages có thể không đủ chi tiết. |
| p2-12 | Hiện tượng bùng lửa màu hồng trong chảo | V | O | Sự kiện động ngắn, dễ bị bỏ lỡ nếu keyframe không nằm đúng thời điểm. |
| p2-13 | Lờ/lợp cá và cần xé tre bên sông | V | A, O | Vật thể văn hóa địa phương hiếm trong OpenImages; ASR có thể giúp nếu lời thuyết minh gọi đúng tên. |
| p2-15 | Chuỗi sơ chế, áo bột và chiên | V | O | Mô tả nhiều bước; một embedding toàn câu có thể thiên về bước nổi bật nhất. |
| p2-16 | Công thức có các lượng 1.5 L, 1/2 và 2 muỗng | C + V | A | Các con số/đơn vị trên video là dấu vân tay tốt cho OCR; cảnh nấu ăn dùng để giới hạn video. |
| p2-17 | Tách lõi và vỏ bằng dụng cụ | V | O | Cần nhận tương tác tay–dụng cụ–nguyên liệu, không chỉ sự có mặt của vật thể. |
| p2-19 | Chuỗi động tác lân vàng trên cọc | V | O | Nhiều hành động nhỏ và phép đếm 4 nhịp; KIS hiện tại không kiểm chứng đủ toàn bộ chuỗi. |
| p2-20 | Slide hợp chất hữu cơ và sơ đồ phân tử | V + C | A | Hình vẽ cấu trúc là tín hiệu visual; OCR/ASR giúp nhận chủ đề bài giảng. |
| p2-22 | Thiết bị bay điện có khung tròn tại Tây Âu | V | A | Hình dáng thiết bị định vị bằng CLIP; quốc gia khó suy ra chỉ từ ảnh nên ASR có thể rất quan trọng. |
| p2-23 | Đua trâu trên ruộng bùn | V | O | Cần cảnh đua, số cặp trâu và biến thể trâu màu sáng; Visual là chính. |
| p2-25 | Nhúng thịt và chan nước dùng vào tô | V | O | Chuỗi món ăn có nhiều cảnh tương tự trong tập; Object chỉ bổ trợ bằng thành phần chung. |
| p2-26 | Chuỗi slide hình học không gian | V + C | A | Ký hiệu vuông góc, nét đứt và khối bốn đỉnh mang tính sơ đồ; OCR thuần có thể không đọc được hình học. |
| p2-28 | Phỏng vấn người phụ nữ ôm bao có logo xanh | V | C | CLIP định vị người và bao đồ; OCR/logo có thể giúp phân biệt tổ chức hoặc địa điểm. |
| p2-29 | Múc canh đúng hai lần | V | — | Đây là đếm hành động theo thời gian, không phải chỉ đếm vật thể trong một frame. |
| p2-30 | Đua xe đạp và người quay bằng hai thiết bị | V | O | Cần quan hệ hai tay cầm điện thoại và gậy quay, là truy vấn compositional. |
| p2-31 | Người nước ngoài gói bánh lá, quạt xanh | V | O | Nhiều thuộc tính đồng thời tạo nên cảnh cần tìm. |
| p2-32 | Hai mẹ con gọi điện rồi người con phỏng vấn | V | A | Chuỗi hai cảnh và một người đi ngang; ASR có thể hỗ trợ nhận đúng đoạn phỏng vấn. |
| p2-33 | Hai tay đua nước rút, phân biệt mũ đen/mũ trắng và kết quả | V | — | Cần theo dõi đối tượng và quan hệ trước–sau; một keyframe riêng lẻ khó chứng minh “đuối sức rồi thua”. |
| p2-36 | Ảnh bắt tay Bác Hồ rồi nghệ nhân vẽ chân dung | V | A | Camera lia nối ảnh gốc với thao tác vẽ; tên dụng cụ/chất liệu không được nêu nên Object khó hỗ trợ trực tiếp. |

### 4.2. Question Answering — 8 câu

| ID | Kiểu đáp án | Bằng chứng chính | Luồng hiện tại | Đánh giá từ file dự đoán |
|---|---|---|---|---|
| p2-5 | Số lượng tỉnh/thành | C + A | `is_text_question()` nhận “theo nội dung/bài giảng”; trích số từ OCR/ASR | Các đáp án ứng viên `6`, `3260`, `3`, `0`, `10` rất phân tán, cho thấy nhiễu số và định vị cảnh chưa ổn định. |
| p2-6 | Tên vùng miền núi | A + C | Entity extraction từ lời giảng hoặc slide | Các dự đoán là những đoạn OCR Vật lý/AI không liên quan; đây là lỗi retrieval hoặc trích nguyên fragment, không phải đáp án đáng tin. |
| p2-8 | Chuỗi chữ neon | C | OCR trực tiếp trên frame và frame lân cận | Bốn dự đoán đầu cùng video/frame lân cận và cùng cụm `tiếng Anh Greatest of All Time`; ứng viên ổn định nhưng vẫn cần chuẩn hóa thành đúng phần chữ cần nộp. |
| p2-14 | Một chữ số chỉ lớp học | A, C | Text-evidence mode; `answer_type()` là number | Năm dòng đều dự đoán `1` trong cùng video; tính nhất quán cao, chưa đồng nghĩa ground truth đã được xác nhận. |
| p2-18 | Tên trường tiểu học | A + C | Entity extraction từ lời kể/lower-third | Năm dòng đều dự đoán `Công Hải` trong cùng video; ứng viên ổn định. |
| p2-24 | Lượng nước tương | C + A | Trích số kèm đơn vị từ bằng chứng công thức | Các ứng viên `200g`, `3l`, `3muỗng` khác video và sai khác đơn vị; cần gắn số với đúng cụm “nước tương”, không chọn số nổi bật nhất. |
| p2-27 | Số màu bột trên đĩa | V + VQA | Không có cue text nên chạy ViLT trên shortlist frame | Tất cả dự đoán là `0`, trong khi mô tả đã nêu ít nhất bột trắng và một màu khác. Đây là dấu hiệu rõ rằng baseline VQA/định vị cảnh thất bại. |
| p2-35 | Số nhóm X sau khi giải bốn câu trắc nghiệm | C + suy luận logic | Text mode chỉ có thể trích số, chưa giải điều kiện “4 đáp án khác nhau và đều chứa chữ số 0” | Dự đoán `3020`, `73`, `0`, `2` phản ánh việc bắt nhầm các chữ số trên slide. Câu này cần OCR nhiều frame rồi suy luận, không thể giải bằng regex trích số hiện tại. |

### 4.3. TRAKE — 2 câu

| ID | Đặc điểm | Luồng hiện tại | Dự đoán đã nộp |
|---|---|---|---|
| p2-21 | Bốn mốc rất gần nhau trong quá trình làm món tôm, gồm bắt đầu thao tác và phần tử thứ tư | Tách E1–E4, truy xuất từng event, chuẩn hóa score, tìm chuỗi tăng nghiêm ngặt trong cùng video bằng K-best DP | Các chuỗi đầu đều thuộc `L26_V156`, có dạng khoảng `2875/3200 → 3825/4300 → 4800/4900 → 4925`. |
| p2-34 | Bốn tư thế/động tác tinh vi của lân đỏ trên cọc, các mốc trải dài trong video | Cùng pipeline TRAKE; compactness chỉ là tie-break nhẹ nên vẫn cho phép chuỗi dài khi relevance tốt | Top đầu chủ yếu thuộc `L22_V007`; một số chuỗi từ `L28_V013`, cho thấy còn cạnh tranh giữa hai video ứng viên. |

## 5. Phần 3 mới — bản viết gọn có thể đưa vào báo cáo

### 3. Phân tích và phân loại truy vấn vòng P2

Bộ đề P2 gồm 36 câu, trong đó có 26 câu Textual KIS, 8 câu Q&A và 2 câu TRAKE. Ngoài cách chia theo thể lệ, đội phân loại lại theo nguồn bằng chứng và kiểu suy luận mà hệ thống cần thực hiện. Kết quả cho thấy 15 câu KIS chủ yếu dựa trên ngữ nghĩa thị giác và quan hệ đối tượng; 6 câu KIS mô tả một chuỗi hành động; 5 câu KIS chứa chữ, số hoặc sơ đồ trên màn hình; 6 câu Q&A cần trích đáp án từ OCR/ASR; 1 câu Q&A là bài toán đếm trực tiếp bằng thị giác; 1 câu Q&A cần OCR kết hợp suy luận logic; và 2 câu TRAKE cần căn chỉnh chuỗi sự kiện theo thời gian.

#### 3.1. Nhóm truy xuất cảnh bằng thị giác và vật thể

Các câu p2-7, p2-9, p2-10, p2-11, p2-12, p2-13, p2-17, p2-22, p2-23, p2-25, p2-28, p2-30, p2-31, p2-32 và p2-36 mô tả cảnh, vật thể hoặc quan hệ giữa người và vật. Hệ thống dùng CLIP để lấy pool ứng viên theo ngữ nghĩa, Object Index để bổ sung các vật thể nhận diện được, rồi hợp nhất thứ hạng bằng Reciprocal Rank Fusion. OCR hoặc ASR chỉ đóng vai trò phụ ở các câu có logo, lời phỏng vấn, địa danh hoặc tên vật thể văn hóa khó nhận biết từ ảnh.

#### 3.2. Nhóm KIS có diễn tiến thời gian và nội dung trên màn hình

Các câu p2-1, p2-3, p2-15, p2-19, p2-29 và p2-33 không chỉ mô tả một cảnh tĩnh mà còn yêu cầu nhận biết thứ tự hoặc số lần thực hiện hành động. Trong khi đó, p2-2, p2-4, p2-16, p2-20 và p2-26 có tín hiệu mạnh từ bảng lớp, công thức, con số hoặc sơ đồ. Pipeline hiện tại vẫn xử lý các câu này bằng `search_kis()`: CLIP, Object, OCR và ASR tạo các ranking độc lập; RRF có score gating giảm ảnh hưởng của các text match yếu; cuối cùng các frame quá gần nhau được khử trùng. Giới hạn là KIS hiện mã hóa toàn bộ mô tả thành một truy vấn, nên chưa kiểm chứng tường minh rằng mọi bước đều xuất hiện đúng thứ tự.

#### 3.3. Nhóm Q&A theo nguồn bằng chứng

Sáu câu p2-5, p2-6, p2-8, p2-14, p2-18 và p2-24 cần đọc thông tin trên slide, phụ đề hoặc lời nói. Với nhóm này, hệ thống dùng Visual để định vị scene, tìm OCR/ASR riêng, giới hạn text match vào các video scene ứng viên, gom frame chính cùng hai frame lân cận và transcript quanh thời điểm đó, rồi trích số hoặc thực thể từ bằng chứng. Câu p2-27 là bài toán đếm thị giác nên được chuyển sang VQA. Câu p2-35 là trường hợp khó hơn: đáp án không xuất hiện trực tiếp dưới dạng một số đơn lẻ mà phải được suy ra từ bốn câu trắc nghiệm, vì vậy cần thêm bước OCR nhiều frame và suy luận logic.

Kết quả nộp cho thấy hệ thống ổn định hơn ở các câu có đáp án lặp lại trên nhiều frame lân cận như p2-8, p2-14 và p2-18. Ngược lại, p2-5, p2-6, p2-24, p2-27 và p2-35 có dự đoán phân tán hoặc không hợp lý, phản ánh ba lỗi chính: định vị nhầm video, chọn nhầm con số trong văn bản và giới hạn của VQA/suy luận hiện tại. Đây là chẩn đoán từ các dự đoán, không phải kết luận đúng–sai khi chưa có ground truth.

#### 3.4. Nhóm TRAKE căn chỉnh sự kiện

Hai câu p2-21 và p2-34 đều gồm bốn event có thứ tự. Router tách các dòng E1–E4, sau đó hệ thống truy xuất top ứng viên đa phương thức cho từng event. `trake_engine.py` chuẩn hóa score theo từng ranking và dùng K-best dynamic programming để tìm các chuỗi đầy đủ thuộc cùng một video, có frame tăng nghiêm ngặt. Điểm chuỗi kết hợp mức liên quan với một thành phần compactness nhỏ để ưu tiên diễn tiến liền mạch khi các kết quả có relevance gần nhau. Nếu không tìm được chuỗi đủ bốn event, mặc định hệ thống không nội suy frame giả.

#### 3.5. Bài học và hướng cải thiện

Phân loại trên cho thấy hệ thống đa phương thức là cần thiết nhưng chưa đủ cho mọi câu. Các hướng ưu tiên là: tách long-query KIS thành các sub-event để kiểm tra độ phủ và thứ tự; tăng khả năng đọc chữ/sơ đồ nhỏ trên slide; gắn số và đơn vị với đúng thực thể được hỏi; thay baseline ViLT bằng VLM mạnh hơn cho đếm và quan hệ; bổ sung mô-đun suy luận nhiều bước cho p2-35; và hoàn thiện frame refinement để chọn đúng ranh giới hành động ở hai câu TRAKE.

## 6. Đối chiếu trực tiếp với code hiện tại

| Thành phần | Hành vi đã có trong code | Ý nghĩa đối với Phần 3 |
|---|---|---|
| `query_router.py` | Rule-based route KIS/Q&A/TRAKE; tách câu hỏi cuối và các dòng E1–En | Có thể mô tả hệ thống tự phân luồng, nhưng không nên gọi là LLM query understanding. |
| `search_engine.py` | KIS lấy pool mặc định 300; chạy Visual/Object/OCR/ASR; RRF; khử frame gần nhau; trả top-K | Là pipeline chung cho 26 câu KIS và bước retrieval nền cho QA/TRAKE. |
| `fusion.py` | Trọng số mặc định Visual 1.0, Object 0.7, OCR 0.6, ASR 0.6; text overlap gate mặc định 0.3 | Có cơ sở để giải thích vì sao Visual là trục chính và text/object là bằng chứng bổ trợ. |
| `qa_engine.py` | Auto dùng Visual để định vị scene; câu text dùng OCR/ASR riêng; gom ±1 keyframe và ASR quanh ±15 giây; trích số/entity hoặc chạy ViLT | Phù hợp với phân loại QA thành text evidence và visual VQA. |
| `trake_engine.py` | Top-300/event; chuẩn hóa điểm; K-best DP; cùng video; frame tăng nghiêm ngặt; không nội suy mặc định | Phải thay hoàn toàn mô tả “TRAKE phân tán qua nhiều video” trong bản cũ. |
| `PLACEHOLDERS.md` | Router vẫn rule-based, ViLT là baseline, frame refinement chưa có; dữ liệu local có phần giả/xấp xỉ | Các giới hạn này cần được trình bày trung thực thay vì suy rộng kết quả local thành kết quả thi. |

## 7. Các câu nên chọn làm tình huống tiêu biểu nếu báo cáo bị giới hạn trang

Nếu chỉ đủ chỗ trình bày 4–5 tình huống, nên chọn các câu sau vì chúng bao phủ đúng các năng lực và giới hạn của hệ thống:

1. **p2-10-kis — xe lội nước:** đại diện cho CLIP ở một concept thị giác hiếm, rõ ràng.
2. **p2-4-kis — slide ngữ pháp:** đại diện cho phối hợp OCR và Visual.
3. **p2-24-qa — lượng nước tương:** đại diện cho định vị scene, OCR/ASR và trích số kèm đơn vị; đồng thời thể hiện lỗi gắn số với thực thể.
4. **p2-35-qa — nhóm X:** đại diện cho khoảng trống giữa text retrieval và suy luận logic nhiều bước.
5. **p2-21-trake — bốn mốc làm món tôm:** đại diện cho tách event và K-best DP với ràng buộc cùng video, đúng thứ tự.

