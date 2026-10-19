# Thử nghiệm SPECTER2 cho reading preference

Pipeline: `title + [SEP] + abstract` → SPECTER2 regression adapter → vector 768 chiều → hồi quy dự đoán `Label` → chọn và xếp thứ tự đọc.

Kết quả đã chạy: [RESULTS.md](RESULTS.md).

Phân tích lỗi và kiểm tra chất lượng input: [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md).

Encoder và adapter pretrained được đóng băng. Ridge/SVR được huấn luyện bằng nhãn thật trong `train.csv`; đây là phần supervised regression đáp ứng phương pháp của môn học. Thử nghiệm này chưa fine-tune Transformer theo nhãn của assignment.

## Chạy lại

Từ thư mục Assignment, dùng Python 3.10 và môi trường riêng:

```bash
python3 -m venv .venv-specter2
source .venv-specter2/bin/activate
python -m pip install -r experiments/specter2/requirements.txt
python experiments/specter2/run_experiment.py all --batch-size 8
python -m unittest discover -s experiments/specter2 -p 'test_*.py' -v
```

Lần thực thi hiện tại dùng `/tmp/assignment-specter2-venv/bin/python`, kế thừa PyTorch CUDA sẵn có. Hai thư viện cần thiết được cài trong môi trường riêng; `bitsandbytes==0.43.1` cũng được cài để thay bản tùy chọn có sẵn không tương thích với PyTorch 2.1.2. Pipeline không dùng quantization/bitsandbytes. Môi trường mới không cài bitsandbytes thì không cần dependency này.

Có thể chạy riêng `embed` hoặc `train` thay cho `all`. Embedding được cache và kiểm tra hash của title/abstract, thứ tự ID và revision model trước khi tái sử dụng. Weights tải từ Hugging Face và lưu trong cache của Hugging Face; văn bản được encode tại máy, không gửi tới dịch vụ inference.

## Dữ liệu và feature

- Đọc metadata và nhãn gốc từ `assignment-data-mining-261.zip`; tự xử lý lớp RTF bọc quanh `Test.csv`.
- Ghép `abstracts_collected.csv` bằng `(split, id)`, kiểm tra khóa duy nhất và đủ 3.090 dòng.
- 80 dòng thiếu abstract dùng title. 35 dòng có `identity_review_note` cũng tạm dùng title; không tự sửa nhãn hay tác giả theo suy đoán.
- Decode HTML entities, bỏ tag HTML, chuẩn hóa khoảng trắng.
- Dùng `allenai/specter2_base` với adapter `allenai/specter2_regression`. Revision cố định nằm trong code và `outputs/embedding_manifest.json`.
- Input tối đa 512 tokens; vector lấy ở token CLS theo hướng dẫn của tác giả. Cache vector float32; inference GPU dùng mixed precision fp16.
- Các biến thể metadata dùng venue one-hot, year chuẩn hóa và TF-IDF tên tác giả. Các bộ biến đổi được fit riêng trên phần training của từng fold.
- Không dùng `id`, DOI, nhãn hay các cột kết quả tra cứu làm feature. ID chỉ dùng ghép dữ liệu, chia nhóm trùng và áp dụng quy tắc hòa điểm của evaluator.

## Đánh giá và chọn mô hình

Giữ một fold trong StratifiedGroupKFold 5 folds làm holdout. Tập development dùng StratifiedGroupKFold 4 folds để chọn hyperparameter và ngưỡng loại bài. Nhóm được tạo bằng normalized title trùng nhau hoặc DOI/Semantic Scholar identifier trùng; URL CEUR bị gắn nhầm hàng loạt không được dùng để nối nhóm. Những bài gần giống nhưng khác cả title lẫn identifier có thể vẫn chưa được nhận diện.

So sánh sáu họ mô hình: metadata + Ridge, TF-IDF + Ridge, TF-IDF + metadata + Ridge, SPECTER2 + Ridge, SPECTER2 + metadata + Ridge, SPECTER2 + SVR. Grid nhỏ, được định nghĩa trước trong code; các trọng số feature block được cố định trước validation. Mọi họ đều dùng cùng split và grid ngưỡng. TF-IDF có word unigrams/bigrams; đây là baseline đơn giản, chưa tối ưu sâu.

Với dự đoán rating `p` và ngưỡng `t`, bài được chọn khi `p > t` và `p > 1`. Khi chọn, `priority = 2.5 * (p - 1)`, nếu không thì 0. Không clip priority để tránh đổi thứ tự/tạo hòa điểm; đề cho phép mọi giá trị hữu hạn không âm.

Metric dùng phần **Competition evaluation chi tiết** trong `overview.md`: gain `(Label - 1) / 4`, discount `1 / log2(rank + 1)`, relevance là gain trung bình có discount, coverage là recall của nhãn 2–5, điểm là harmonic mean `2QR/(Q+R)`. Phần tóm tắt đầu overview có công thức mô tả khác; các con số ở đây chỉ có ý nghĩa với công thức chi tiết này.

Chọn mỗi họ và mô hình chung bằng điểm trung bình trên các inner folds. Sau khi chốt model/hyperparameter/threshold mới chấm holdout một lần. Model SPECTER2 và model chung được refit trên toàn bộ 2.494 nhãn để tạo CSV test cục bộ. Không dùng điểm holdout để thay đổi lựa chọn, không có nhãn test, không submit lên Kaggle tự động.

## Artifacts

- `outputs/holdout_metrics.csv`: so sánh holdout; baseline recommend-all xếp hòa điểm theo ID dạng text đúng quy tắc đề.
- `outputs/development_cv.csv`: cấu hình và ngưỡng tốt nhất cho từng candidate trên development.
- `outputs/development_threshold_curves.csv`: toàn bộ ngưỡng đã thử trên development.
- `outputs/validation_split.csv`: danh sách nhóm, development/holdout và inner fold.
- `outputs/development_oof_predictions.csv`, `outputs/holdout_predictions.csv`: dự đoán để kiểm tra lỗi.
- `outputs/embeddings.npz`, `outputs/embedding_index.csv`, `outputs/embedding_manifest.json`: vector, thứ tự record và provenance.
- `outputs/model_specter2.joblib`, `outputs/model_best_dev.joblib`: pipeline hồi quy đã fit cùng cấu hình chọn bằng development.
- `outputs/submission_specter2.csv`, `outputs/submission_best_dev.csv`: mỗi file đủ 596 ID và hai cột `id,priority`.
- `outputs/results.json`: kết quả và cấu hình đầy đủ.

Holdout là một lần chia với seed cố định, không phải ước lượng độ bất định qua nhiều lần thử. Chất lượng abstract, metadata và phân bố train/test đều ảnh hưởng kết quả. Điểm local không bảo đảm điểm Public/Private leaderboard.

## Nguồn phương pháp

- [SPECTER2 repository của Allen AI](https://github.com/allenai/SPECTER2).
- [Model card regression adapter và cách encode title/abstract](https://huggingface.co/allenai/specter2_regression).
- [SciRepEval paper](https://aclanthology.org/2023.emnlp-main.338/).
