# Cross-encoder theo ngữ cảnh sở thích

## Câu hỏi

Mô hình hiện tại biết hai paper gần nhau về chủ đề, venue, tác giả và năm, nhưng không
biết *lý do* một paper trong cùng chủ đề đáng đọc hơn paper còn lại. Thí nghiệm này kiểm
tra liệu biểu diễn cặp `(paper cần quyết định, paper từng được đánh giá)` có thêm tín hiệu
để phân biệt đóng góp, phương pháp, kết quả và loại bài hay không.

Không dùng nhãn test, URL, citation count, hay bất kỳ metadata bên ngoài nào. Abstract đã
được thu thập trước đó là phần nội dung công khai của paper.

## Input và reference set (đã khoá)

Với mỗi query trong một lần fit:

1. Xét **chỉ** các paper train thuộc `fit` và khác publication-group với query.
2. Xếp hạng bằng cosine trong `retrieval_margin` đã tồn tại, rồi `id` để phá hoà.
3. Nếu venue của query có ít nhất ba candidate hợp lệ trong một lớp reference, chỉ lấy
   venue đó; ngược lại lấy tất cả venue.
4. Lấy ba nearest paper `Label=1` và ba nearest paper `Label>1`. Nếu một lớp có ít hơn ba
   candidate thì lặp reference gần nhất của chính lớp đó và ghi vào audit; preflight sẽ
   fail nếu không có candidate nào.

Nhãn reference hợp lệ vì chúng thuộc training partition. Nhãn query không đi vào input.
Mỗi pair mang `query title + abstract` và `reference title + abstract`. Mỗi phía bị cắt
độc lập còn 126 subword trước khi ghép; pair có tối đa `126 + 126 + 3 = 255` tokens.
Vì vậy không xảy ra việc abstract của reference bị tokenizer cắt hết do query dài.

## Mô hình và loss

Base là checkpoint local cố định `allenai/specter2_base@344764...`; base transformer frozen.
Chỉ regression adapter local `allenai/specter2_regression@a9d105...` và head được cập nhật.
Đây là adapter tuning, không phải full fine-tuning.

Một batch logic là bốn query, mỗi query sáu cặp. CLS của sáu cặp cộng embedding 8 chiều
của **label reference** đi qua pair head. Mean logits sáu pair tạo logits năm lớp cho query.
Loss cố định:

`CE(query label) + 0.5 BCE(query is read) + 0.25 BCE(query label > reference label)`.

Pairwise BCE bỏ qua hoà nhãn. Loss này buộc mô hình so sánh preference tương đối, trong khi
CE vẫn giữ bài toán năm lớp gốc. Arm đối chứng `query_only` dùng cùng base/adapter/head,
input query và `CE + 0.5 BCE`, nhưng không có reference hay loss pairwise.

## Đánh giá đã khoá

Sau khi preflight thành công:

- 5 outer group folds; mỗi outer dùng 4 inner group folds.
- Mỗi arm train 2 epoch (không chọn epoch bằng outer); AdamW head=1e-3, adapter=5e-5,
  weight decay=.01, clip=1, seed=20261029.
- Mỗi arm chọn threshold từ `0.00..0.90` bước `.05` theo mean inner H.
- Outer refit, rồi báo H/Q/R, TP/FP/FN, label 4/5 bị bỏ và AUC. Không tạo submission trừ
  khi contextual arm vượt incumbent ít nhất .005 H trung bình và thắng >=3/5 folds.

Preflight chỉ kiểm tra thực thi, memory, gradient, group isolation và audit token/reference;
nó không tính điểm outer/inner và không được dùng để lựa chọn cấu hình.

## Tiêu chí diễn giải

Nếu contextual arm không hơn query-only, có hai khả năng được tách rõ: representation pair
không có tín hiệu preference, hoặc adapter/head chưa học được tín hiệu ấy trong dữ liệu nhỏ.
Nếu nó hơn query-only nhưng không hơn incumbent, cross-encoder có tín hiệu mới nhưng chưa đủ
để thay policy selection. Không diễn giải một lần thử thất bại thành "semantic vô ích".
