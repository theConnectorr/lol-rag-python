### 1. Phân tích Cấu trúc Infobox (Metadata EDA)

Infobox chính là bộ khung xương của Đồ thị tri thức. Bạn cần đánh giá chất lượng của nó:

- **Phân tích Độ khuyết thiếu (Missing Value Analysis):**
  - Thống kê xem trong 160+ tướng, bao nhiêu % có trường `Family`, bao nhiêu % có trường `Related character`, `Weapon`, `Species`.
  - _Điểm mù:_ Bạn sẽ phát hiện ra nhiều tướng cũ (như Shaco, Rammus) có Infobox gần như trống rỗng.
  - _Giải pháp:_ Khi viết luồng GraphRAG, phải có cơ chế **Fallback (dự phòng)**. Nếu Graph không tìm thấy `Family` của Shaco, nó phải tự động tìm trong VectorDB thay vì báo lỗi "Không có dữ liệu".
- **Phân bố Thuộc tính (Categorical Distribution):**
  - Vẽ biểu đồ Bar chart xem Vùng đất (Region) nào có nhiều tướng nhất (Ionia vs. Demacia vs. Hư Không). Thống kê các Loài (Species - Human, Yordle, Vastaya, Cyborg).
  - _Mỏ vàng:_ Nếu hệ thống biết Ionia là vùng đất đông đúc nhất, bạn có thể tạo trước các Index (Chỉ mục) đặc biệt trong Neo4j cho node Ionia để tối ưu tốc độ truy vấn.

### 2. Phân tích Mạng lưới Thực thể sơ bộ (Network EDA)

Đây là bước "nhá hàng" trước khi đưa vào Neo4j, tập trung vào trường `Related character`:

- **Phân tích Mật độ (Degree Centrality):**
  - Đếm số lượng mối quan hệ của mỗi tướng. Ai là người có nhiều "kẻ thù/bạn bè" nhất? (Thường là các "Hub" như Swain, Jarvan IV, Azir). Ai là "Kẻ cô độc" (Isolated nodes)?
  - _Mỏ vàng:_ Các tướng "Hub" chính là **điểm bắt đầu (Starting nodes) tuyệt vời nhất** để test thuật toán Graph Traversal (Duyệt đồ thị).
- **Vấn đề Định danh (Entity Resolution Check):**
  - _Điểm mù chí mạng:_ Trong tiểu sử Lux, Garen được gọi là "Garen Crownguard". Nhưng trong tiểu sử Garen, anh ta có thể chỉ tên là "Garen".
  - _Giải pháp:_ Dữ liệu của bạn chắc chắn cần một bảng map (Alias Mapping) từ trường `Alias` trong Infobox để quy chuẩn mọi tên gọi về một Node duy nhất trong Neo4j (VD: _The Lady of Luminosity = Luxanna = Lux_).

### 3. Phân tích Đặc trưng Văn bản (Textual EDA)

Tập trung vào phần `mainContent` (TOC) để tối ưu cho thuật toán Vector và Chunking:

- **Phân phối Độ dài Tiểu sử (Document Length Distribution):**
  - Vẽ biểu đồ Histogram số lượng từ (Word count) của các mục chính (như "Early Life", "Personality").
  - _Điểm mù:_ Tiểu sử của Aphelios hoặc Jhin có thể dài 3000 từ, trong khi Cho'Gath chỉ có 200 từ.
  - _Giải pháp:_ Điều chỉnh tham số `chunkSize` trong LangChain. Đối với các tướng có tiểu sử quá ngắn, có thể không cần cắt (chunk) mà nhét luôn nguyên bài vào 1 Vector để giữ trọn vẹn ngữ cảnh.
- **Đồng nhất Cấu trúc Mục lục (TOC Consistency):**
  - Bao nhiêu % tướng có mục "Early Life"? Bao nhiêu % có mục "Abilities"?
  - _Mỏ vàng:_ Sự đồng nhất này cho phép bạn xây dựng một **Hệ thống lọc Metadata siêu việt**. Ví dụ: Người dùng hỏi _"Kỹ năng của Sylas là gì?"_, Router của bạn không chỉ tìm Target Champion = "Sylas", mà còn ép DB chỉ quét trong Section = "Abilities" hoặc "Magic", tăng độ chính xác lên 100%.
