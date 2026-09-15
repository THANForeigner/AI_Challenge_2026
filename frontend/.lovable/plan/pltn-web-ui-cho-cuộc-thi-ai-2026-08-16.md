# PLTN — Web UI cho cuộc thi AI

Xây một trang tìm kiếm video/keyframe kiểu "PIKA SEARCH", giữ layout gốc (sidebar trái + grid keyframe) nhưng nâng cấp thị giác để trình diễn ấn tượng. Kết quả lấy từ API search có sẵn của bạn.

## Giao diện

- Dark theme cao cấp: nền xanh đen sâu, viền kính mờ, accent neon xanh lá / cyan cho logo "PIKA SEARCH" và nút Search.
- Sidebar trái (cố định, cuộn riêng):
  - Logo neon
  - 5 tab chế độ: Text / Image / OCR / ASR / Temporal
  - Vùng input đổi theo tab: textarea query, khung upload ảnh (kéo-thả + preview), ô OCR/ASR query, editor Temporal nhiều bước (thêm/xoá/kéo thứ tự các sự kiện)
  - Select Model (danh sách model lấy từ API nếu có, mặc định: beit3, clip, blip2)
  - Filter Panel: OCR filter, ASR filter
  - Slider Top K (10–500)
  - Nút Search lớn, có trạng thái loading
- Khu kết quả: grid keyframe responsive (2–7 cột), mỗi ô hiện video ID + frame, badge rank/score, hover hiện nút "xem chi tiết" và "tìm ảnh tương tự".
- Trạng thái: skeleton khi loading, empty state, error banner rõ ràng khi API lỗi.
- Thanh trên khu kết quả: số kết quả, thời gian truy vấn, mật độ grid, nút xoá kết quả.

## Chi tiết keyframe

Modal lớn khi click ảnh:
- Ảnh keyframe cỡ lớn, điều hướng ← → giữa các kết quả
- Metadata: video id, frame idx, timestamp, score, OCR text, ASR text
- Nút: copy `video_id, frame` cho file submit, mở video ở timestamp (nếu API trả link), "tìm ảnh tương tự" (chạy search Image với keyframe này)
- Giỏ Submission: chọn nhiều keyframe, xem danh sách, tải về CSV để nộp thi

## Kết nối API của bạn

- Base URL cấu hình được: đọc từ biến môi trường, kèm ô nhập tạm trong UI (Settings) để đổi nhanh khi demo.
- Gọi qua một server function proxy để tránh lỗi CORS và giữ URL nội bộ không lộ ra client.
- Lớp adapter chuẩn hoá kết quả: chấp nhận nhiều dạng response phổ biến (`results`/`data`/mảng thuần; các key `video_id`, `frame`/`frame_idx`, `score`, `path`/`url`/`image`, `ocr`, `asr`) rồi map về một kiểu chung, nên không cần đổi backend.
- Nếu chưa cấu hình API, UI hiện chế độ demo với dữ liệu mẫu để vẫn trình diễn được.

## Chi tiết kỹ thuật

- Route: `/` là trang search (thay placeholder), kèm `head()` title/description/OG riêng.
- Files chính: `src/routes/index.tsx`, `src/components/search/{Sidebar,ModeTabs,ResultGrid,KeyframeCard,DetailModal,SubmissionCart,TemporalEditor}.tsx`, `src/lib/pltn.functions.ts` (server fn proxy), `src/lib/pltn-normalize.ts` (adapter + types).
- Tokens màu/neon/shadow thêm vào `src/styles.css`; không hardcode màu trong component.
- State query bằng TanStack Query (`useMutation` cho search), giữ lịch sử truy vấn trong session.
- Upload ảnh gửi dạng base64 qua proxy tới API.

## Cần bạn cung cấp sau khi duyệt

- URL API search + dạng payload/response (hoặc mình dùng adapter mặc định rồi tinh chỉnh sau khi bạn gửi 1 response mẫu).
