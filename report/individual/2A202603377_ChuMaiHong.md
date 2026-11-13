# Báo cáo cá nhân — Day 06: Data Ingestion Engineering

> **Còn chờ trước khi nộp:** tạo commit cuối rồi điền commit hash bên dưới; spot-check các nhãn audit AI-assisted và một số trang PDF. Các thông tin cá nhân và tên file đã được điền theo thông tin hiện có.

## Thông tin người nộp

- Họ và tên: Chu Mai Hồng
- MSSV: 2A202603377
- Lớp: L3A
- Repo bài nộp: https://github.com/maihong1208/K5-L3-Day06-ChuMaiHong-2A202603377-DataIngestionEngineering
- Commit chốt nội dung: `c3a90de4bb0823aea53b091884c29c8dd4cff64e`

## Dữ liệu và pipeline

- **Chủ đề/corpus:** OWASP Top 10 for LLM Applications 2025, tài liệu về các rủi ro và biện pháp giảm thiểu trong ứng dụng LLM.
- **Nguồn/quyền:** tài liệu từ OWASP GenAI Security Project, giấy phép CC BY-SA 4.0 theo thông tin trên PDF. Manifest ghi attribution và yêu cầu chia sẻ bản phái sinh theo cùng giấy phép.
- **Trang:** PDF có 45 trang vật lý; dùng 35 trang nội dung, trang 7–13 và 15–42. Loại trang 14 vì là danh sách tham chiếu ngắn. Tổng text chuẩn hóa là 78.077 ký tự; mọi trang được chọn đều trích xuất được.
- **Chuẩn hóa/truy vết:** pypdf 6.21.4; Unicode NFC và chuẩn hóa whitespace; giữ số trang vật lý 1-based. Metadata section được lấy từ heading trên PDF và áp dụng đến heading kế tiếp.
- **Lỗi/giới hạn xử lý:** pypdf báo `incorrect startxref pointer` và lỗi giải nén object stream khi mở PDF. Các trang được trích xuất thành công, nhưng cần spot-check trực quan một số trang trước khi nộp.
- **Embedding:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, Sentence Transformers 5.7.0, CPU, 384 chiều; cùng model dùng cho query và document; không dùng API key, chi phí API $0. Model revision trong summary hiện là `null`, nên cần pin revision nếu muốn tái lập tuyệt đối.
- **Vector store:** Chroma PersistentClient 1.5.9, cosine distance, approximate HNSW; lưu tại `starter/results/chroma/`.
- **Metadata:** `document_id`, `source`, `page`, `section`. `section` được kiểm chứng từ heading trong PDF, hữu ích để lọc theo LLM01–LLM10; `source` và `page` hỗ trợ truy vết/citation.
- **Lệnh chạy:** từ thư mục `starter`, chạy `.\.venv\Scripts\python.exe -m src.run` (sau khi cài `requirements.txt`). Model chạy local; lần đầu cần tải weights.

## Cấu hình và benchmark

- **A:** cửa sổ cố định 500 ký tự, overlap 50; không filter.
- **B:** chunk theo đoạn, tối đa 800 ký tự; không filter.
- **C:** cùng chunk và vectors với B; equality filter theo `section` khi câu hỏi có nêu section.
- B–C chỉ khác filter; A–B thay cách chunk. C tái sử dụng vectors của B.
- Bộ benchmark có 20 câu, chia trước thành 15 dev / 5 test; có 5 câu filter tổng cộng, trong đó 2 câu test. Mỗi câu có expected answer, trang nguồn và evidence quote.
- **Tiêu chí chọn:** chọn trên dev theo Evidence Hit@3 cao nhất, rồi Hit@1; hòa thì ưu tiên A, sau đó B, rồi C. Chỉ sau khi chốt lựa chọn trên dev mới đánh giá test. B và C hòa trên dev nên chọn B theo quy tắc tie-break đã định.

## Kết quả truy xuất

| Strategy | Split | Hit@1 | Evidence Hit@3 | Số chunk | Thời gian ingest (s) | Chi phí API |
|---|---|---:|---:|---:|---:|---:|
| A | Dev | 5/15 (33,3%) | 9/15 (60,0%) | 189 | 4,583 | $0 |
| A | Test | 2/5 (40,0%) | 3/5 (60,0%) | 189 | 4,583 | $0 |
| B | Dev | 5/15 (33,3%) | 10/15 (66,7%) | 124 | 4,497 | $0 |
| B | Test | 2/5 (40,0%) | 4/5 (80,0%) | 124 | 4,497 | $0 |
| C | Dev | 5/15 (33,3%) | 10/15 (66,7%) | 124 | 0,132* | $0 |
| C | Test | 2/5 (40,0%) | 4/5 (80,0%) | 124 | 0,132* | $0 |

`*` C dùng lại embeddings của B nên thời gian 0,132 giây chỉ bao gồm ghi/upsert vào store, không thể so trực tiếp với thời gian ingest đầy đủ của A/B. Thời gian truy xuất trung bình mỗi query ở mức khoảng 0,94–1,20 ms; số đo nhỏ và không đại diện cho tải production.

## Năm khía cạnh đánh giá

### 1. Độ chính xác truy xuất

Dev: A đạt 5/15 Hit@1 và 9/15 Evidence Hit@3; B/C đạt 5/15 và 10/15. Test: cả ba đạt 2/5 Hit@1; A đạt 3/5 Evidence Hit@3, B/C đạt 4/5.

### 2. Chunk coherence

Chấm cùng 10 vị trí nguồn đã chọn trong coherence audit:

| Strategy | Điểm 0 | Điểm 1 | Điểm 2 | Trung bình /2 |
|---|---:|---:|---:|---:|
| A | 0 | 3 | 7 | 1,7 |
| B | 0 | 3 | 7 | 1,7 |
| C | 0 | 3 | 7 | 1,7 |

Các điểm 1 chủ yếu ở chunk bắt đầu giữa câu hoặc thiếu đầu mối của đoạn nguồn (ví dụ các vị trí ở trang vật lý 8, 10 và 17). B và C dùng cùng chunk text nên coherence giống nhau.

### 3. Tính hữu dụng của metadata

Đã kiểm tra 45 bản ghi (5 câu test × top-3 × 3 chiến lược): `document_id`, `source`, `page` và `section` đều đủ, khớp manifest — **45/45**. `source`/`page` cho phép truy vết; `section` hỗ trợ giới hạn phạm vi. Hạn chế: metadata đúng không đảm bảo kết quả của A/B liên quan, vì hai chiến lược này không áp dụng filter.

Trong hai câu test có yêu cầu filter:

| Strategy | q17: hit thuộc LLM08 | q20: hit thuộc LLM10 | Top-3 tuân thủ đầy đủ |
|---|---:|---:|---:|
| A | 2/3 | 2/3 | 0/2 câu |
| B | 2/3 | 1/3 | 0/2 câu |
| C | 3/3 | 3/3 | 2/2 câu; tổng 6/6 hit |

A/B không lọc theo thiết kế, nên các hit ngoài section là giới hạn có chủ ý của baseline, không phải metadata sai.

### 4. Grounding quality

Đây là đánh giá độ đủ của evidence trong top-3, **không phải độ chính xác của câu trả lời do LLM sinh**.

| Strategy | Điểm 0 | Điểm 1 | Điểm 2 | Trung bình /2 |
|---|---:|---:|---:|---:|
| A | 0 | 1 | 4 | 1,8 |
| B | 0 | 0 | 5 | 2,0 |
| C | 0 | 0 | 5 | 2,0 |

Các nhãn và lý do từng câu nằm trong `starter/results/manual_audit_template.json`. Đây là lượt rà soát AI-assisted; tác giả cần kiểm tra và xác nhận trước khi nộp.

### 5. Tác động chiến lược dữ liệu và kết luận

- **A so với B:** B giảm số chunk từ 189 xuống 124 (khoảng 34%). Evidence Hit@3 tăng từ 60% lên 66,7% trên dev và từ 60% lên 80% trên test; Hit@1 không đổi. Grounding thủ công tăng từ 1,8 lên 2,0, còn coherence trung bình bằng nhau. Một ví dụ lỗi của A là q19: các chunk trang 37 chỉ chứa phần cuối của khuyến nghị RAG, thiếu đoạn nêu việc lấy thông tin đã xác minh từ nguồn ngoài đáng tin cậy; B/C có chunk đầy đủ hơn.
- **B so với C:** cùng chunking và vectors nên số chunk và retrieval accuracy bằng nhau. C đảm bảo top-3 của cả q17 và q20 nằm trong section yêu cầu (6/6 hit); B không lọc nên lẫn section khác. C phù hợp khi phạm vi section là đầu vào đáng tin cậy; B phù hợp truy vấn không có filter và là cấu hình được chọn trên dev.
- **Lựa chọn:** B được chọn trên dev theo quy tắc tie-break đã định. Trên test nhỏ này, C bằng B về Evidence Hit@3 và Hit@1; lợi thế của C thể hiện ở tuân thủ phạm vi filter, không phải tỷ lệ hit cao hơn.
- **Ví dụ lỗi filter:** q20 ở B trả về một chunk trang 40 thuộc LLM10 nhưng các chunk khác thuộc LLM04/LLM07. C lọc `LLM10:2025` nên cả top-3 thuộc đúng section. Đây là ví dụ tác động metadata filter, không phải lỗi metadata.
- **Giới hạn:** chỉ một tài liệu và 35 trang, test có 5 câu nên không suy rộng kết quả; thời gian CPU rất nhỏ và có nhiễu; chưa có bước sinh câu trả lời LLM; pypdf có cảnh báo cấu trúc; manual audit hiện do một lượt AI-assisted thực hiện, chưa phải đánh giá độc lập của tác giả. Cần xác nhận nhãn và spot-check PDF trước khi nộp.
- **Cải tiến nếu làm lại:** chốt/pin revision embedding model, xác minh PDF trực quan, ghi giả thuyết trước khi chạy vào experiment log và bổ sung người chấm độc lập hoặc vòng đối soát nhãn.

## Vị trí artifacts

- Summary: `starter/results/run_summary.json`
- Trace top-3: `starter/results/traces.json`
- Metadata/grounding audit: `starter/results/manual_audit_template.json`
- Coherence audit: `starter/results/coherence_audit_template.json`
- Manifest và benchmark: `starter/data/manifest.json`, `starter/benchmark/questions.json`
