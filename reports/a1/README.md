# Khung report Assignment 1

Thư mục này được thiết kế lại từ mẫu HCMUT có sẵn, theo Section 3, 4 và 10–14 của `docs/handbook-ene-v2.pdf` (revision 14/09/2026). Hạn A1 được cập nhật thành **28/10/2026, 23:59 GMT+7** theo thông tin nhóm cung cấp; cần lưu thông báo LMS làm căn cứ.

## Cấu trúc và nơi chỉnh sửa

```text
DL_Placeholder_A1/
├── main.tex                         # Điểm vào, ghép toàn bộ report
├── metadata.tex                     # Tên môn, nhóm, tiêu đề, ngày nộp, URL
├── preamble.tex                     # Package, layout, macro chung
├── hcmut.sty                        # Nhận diện HCMUT, header/footer
├── Sections/
│   ├── 1-Title.tex                  # Trang bìa và thành viên
│   ├── 2-Abstract.tex               # Tóm tắt; viết sau khi xong kết quả
│   ├── 3-Problem-and-Data.tex       # PHẦN 1: Problem and Data Description
│   ├── 4-Methodology.tex            # PHẦN 2: Methodology
│   ├── 5-Implementation-Results.tex # PHẦN 3: Implementation Results
│   ├── 6-References.tex             # Tài liệu tham khảo, dùng thebibliography
│   ├── 7-Reproducibility.tex        # Phụ lục tái lập, liên kết và đóng góp
│   ├── 8-AI-Usage.tex               # Phụ lục khai báo AI đủ 9 trường
│   └── 9-Model-Evidence.tex         # Hình của từng model, nạp từ phụ lục 7
├── Tables/                         # Bảng số liệu tách khỏi văn bản phân tích
├── Images/                         # Logo, EDA và bằng chứng dự đoán
├── Evidence/                       # Snapshot JSON/YAML/AI log làm nguồn
└── Original/                       # Bản gốc main, style, trang bìa (.bak)
```

`main.tex` dùng `\input` để ghép bìa, abstract, ba phần chính, references và hai phụ lục. Mỗi phần chính là một file; mọi mục con nằm trong file đó. `Sections/9-Model-Evidence.tex` chỉ là thành phần của phụ lục, không phải phần chính thứ tư. Giữ nguyên cách viết hoa `Sections`, `Tables`, `Images`, `Evidence` để build được trên Linux/Overleaf.

## Đối chiếu handbook

| Yêu cầu | Nơi điền trong report |
|---|---|
| Problem formulation, motivation, input/output, prediction unit, khó khăn | Phần 1 / Problem formulation |
| Nguồn, version, license, collection/labeling, số mẫu, định dạng, dung lượng, bias | Phần 1 / Dataset description |
| EDA: phân bố lớp, kích thước input, imbalance, ảnh đại diện | Phần 1 / EDA |
| Split, seed, stratification thực tế, split unit, leakage, subset | Phần 1 / Data splitting |
| Preprocessing, augmentation, shape, một batch thật sau biến đổi | Phần 1 / Preprocessing |
| Sơ đồ pipeline, Dataset/DataLoader, train/val/test | Phần 2 / End-to-end pipeline |
| Linear, MLP, CNN tự thiết kế, LSTM hoặc GRU, Transformer | Phần 2 / Mandatory architecture families |
| Input representation, kiến trúc, loss, metric, ưu/nhược, tham số, tự viết/thư viện | Phần 2 / Architectures + Metrics + Training setup |
| Optimizer, LR, batch, epochs, scheduler, regularization, early stopping, checkpoint, seed, hardware, precision, tuning | Phần 2 / Training setup |
| Hypothesis, changed/fixed factors, metric, decision criterion, fairness | Phần 2 / Experimental design |
| Curves, checkpoint, over/underfit, thời gian train, vấn đề triển khai | Phần 3 / Training behavior + phụ lục hình |
| Accuracy, macro-F1, parameters, train/inference time, mean ± std nếu khả thi | Phần 3 / Quantitative comparison |
| Đúng, khó nhưng đúng, sai, GT/prediction và lựa chọn ví dụ | Phần 3 / Qualitative evaluation |
| So sánh, inductive bias, representation, trade-offs, giới hạn kết luận | Phần 3 / Comparison and discussion |
| Confusion matrix, nhóm lỗi, tần suất, ví dụ, nguyên nhân giả định, cải thiện | Phần 3 / Error analysis + phụ lục hình |
| Ablation: khuyến khích ở A1; không được mô tả như bắt buộc | Phần 3 / Optional ablation |
| Limitations, conclusion, future work | Cuối phần 3 |
| Config, split, checkpoint, log/run ID, commit/tag, dependency và hardware | Phụ lục A / Reproducibility |
| AI disclosure đủ 9 trường và người kiểm chứng | Phụ lục B / AI Usage |

## Cách điền nội dung

- Report giữ tiếng Anh như mẫu gốc. Các đoạn màu xanh `\reporttodo{...}` giải thích chính xác cần viết gì. Thay chúng bằng nội dung có bằng chứng; không xóa yêu cầu chỉ để report hết màu xanh.
- `\reportfigure{đường_dẫn}{caption}{label}` chèn hình nếu có; nếu chưa có thì hiện khung báo thiếu. Hiện còn thiếu hình `Images/eda/preprocessed_batch.png`: phải tạo từ một batch DataLoader thật.
- `\codepath{...}` dùng cho đường dẫn có dấu gạch dưới, tránh lỗi LaTeX.
- Dùng `\label`, `\ref`, `\cite` cho tham chiếu. Thêm nguồn đã đọc và kiểm chứng trong `Sections/6-References.tex`; hiện mới có handbook và repo, cần bổ sung nguồn dataset/mô hình/thư viện.
- Bảng định lượng đã lấy từ JSON Fashion-MNIST của 6 model. 5 họ bắt buộc chỉ cần LSTM **hoặc** GRU; nếu giữ cả hai thì giải thích phần mở rộng.
- Chỉ có một seed trong snapshot hiện tại. Không trình bày std = 0 như bằng chứng ổn định. Chưa được kết luận quan hệ nhân quả hoặc một model tốt hơn chỉ nhờ accuracy.
- Kiểm tra fairness và cách đo thời gian trước khi so sánh. Snapshot có revision `-dirty` ở một số run: cần patch tương ứng hoặc tái lập từ revision sạch để hoàn thiện provenance.
- Hình qualitative đang minh họa bằng Transformer; cần so sánh trên cùng ảnh giữa các model và thêm ca khó nhưng đúng. Hình curves và confusion matrix đã có cho cả 6 model.
- Dữ liệu trong `Evidence/` và hình trong `Images/` là bản sao tĩnh. Khi chạy thí nghiệm mới, cập nhật đồng thời evidence, hình và các bảng; không trộn số từ draft 5 epoch với run hiện tại 50 epoch.
- Điền tên người dùng AI, người chịu trách nhiệm kiểm chứng, các bước đã thực sự kiểm chứng; đối chiếu `AI_USAGE.md` rồi cập nhật nhất quán ở report, landing page và trang A1.

## Biên dịch

Từ thư mục chứa `main.tex`, dùng **pdfLaTeX** (không cần BibTeX/Biber):

```sh
pdflatex -interaction=nonstopmode -halt-on-error -jobname=Placeholder_A1_Report main.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=Placeholder_A1_Report main.tex
```

Có thể upload toàn bộ thư mục lên Overleaf, chọn `main.tex` làm Main document, compiler pdfLaTeX; khi tải PDF, đặt tên `Placeholder_A1_Report.pdf`. Không cần upload `Original/` để biên dịch. Report không dùng đường dẫn tuyệt đối sang repo và không cần tải ảnh từ mạng.

Trong phiên chỉnh sửa này chưa tìm thấy công cụ LaTeX khả dụng, nên chưa biên dịch hoặc kiểm tra hình thức PDF. Việc kiểm tra source không thay thế cho bước đó.

## Trước khi nộp

- Hoàn thiện mọi `\reporttodo`, `\pending` và hình thiếu; viết lại abstract/conclusion theo kết quả cuối.
- Kiểm tra bảng số, checkpoint, hardware/timing, provenance và nguồn trích dẫn.
- Build ít nhất hai lần, xử lý undefined reference/citation và overfull box; đọc từng trang PDF để kiểm tra layout và độ đọc được của bảng/hình.
- Kiểm tra tên nhóm đúng đăng ký và deadline cập nhật trên LMS.
- Hoàn thiện slides, YouTube, GitHub Pages và links checkpoint; các sản phẩm này không được thay thế bằng phụ lục report.
- Người đại diện nộp PDF đúng tên; cập nhật disclosure ở đủ các nơi.

Lưu ý: `.tmp/` đang bị `.gitignore` của repo bỏ qua. Khung report này hiện là bản làm việc cục bộ; chưa được commit hoặc chuyển vào `reports/`.
