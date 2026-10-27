# Prompt giao coding agent trên remote

Bạn đang tiếp tục assignment Data Mining tại repo Oct19. Mục tiêu của task
này là chạy pilot **Qwen supervised QLoRA 5 nhãn** đã triển khai, không lên
thêm một hướng mới hoặc lặp lại frozen binary prompting.

Đọc theo thứ tự:

1. `experiments/qwen_remote_v1/README.md`
2. `experiments/qwen_remote_v1/PLAN.md`
3. `experiments/qwen_remote_v1/config.json`
4. `overview.md` phần Evaluation, `data.md`, `rules.md`

Ưu tiên entry point qwen_remote_v1 cho task này. Các HANDOFF/AGENT_PROMPT của
llm_preference là lịch sử, có các protocol đã NO_GO hoặc chưa implement nested.

Bối cảnh: một người dùng, 2494 rating1–5, 596 test chưa nhãn. Incumbent full277
public .64349 theo user; pilot baseline mean H .6475806245807401 trên hai split
đã commit. Không so lẫn các tập. Các folds đã được xem nhiều lần, nên kết quả
pilot exploratory. Qwen trước đây chưa có quality fine-tune run; technical
preflight2 updates và frozen inference không chứng minh supervised quality.

`inputs/` đã có đầy đủ dữ liệu pilot và baseline cache dạng JSON, có SHA256.
Không cần các ignored outputs cũ, embeddings hoặc thư viện CatBoost. Không
chạy lại export_inputs.py/scraper để chuẩn bị dữ liệu trên remote.

Việc cần thực hiện ngay:

- Tạo môi trường riêng theo README; kiểm tra NVIDIA driver/GPU/BF16 và disk.
- Chạy CPU verification +evaluation contract checks. Tải đúng Qwen7B revision
  từ config, verify tokenizer. Không đổi model ID hoặc weights sang AWQ.
- Chạy `bash experiments/qwen_remote_v1/run_pilot.sh` trong tmux. Pipeline
  tự smoke, frozen score, train độc lập1 epoch cho mỗi partition, score và evaluate.
- Smoke phải có LoRA updates hữu hạn, base frozen và save/reload parity.
  Khi runtime/source đổi, dùng output namespace mới và smoke lại. Không xóa
  failed artifacts hoặc sửa fingerprint để resume. Không sửa file locked sau fit.
- Monitor training.jsonl khoảng10 updates: loss/norm hữu hạn, updates chạy,
  GPU memory, time/update. Đo training/scoring/loading riêng; đừng dùng ETA
  93–120h cũ, cũng không hứa remote sẽ bằng0.9h/epoch laptop.
- Run.py chưa resume optimizer giữa training; đọc README để xử lý interruption.
  Có thể resume prediction cache hoặc completed adapter cùng fingerprint.
- Không đọc held-out labels để chỉnh epoch, prompt, cutoff, loss, checkpoint.
  Evaluator yêu cầu cả2 partitions đã score xong mới báo quality.

Sau khi xong, report bằng tiếng Việt:

1. GPU/stack, model revision, output path và terminal status từng job.
2. Frozen→trained: 5class logloss, read AUC/Brier; có evidence học sở thích không?
3. Trained→incumbent: H/Q/R, TP/FP/FN/TN, missed4/missed5 của từng fold và tổng.
4. Confusion matrix5×5, recall/precision từng label; thêm/bỏ đúng nhãn nào.
5. Native/selection-only/ranking-only/matched-budget: bottleneck selection,
   ranking hay calibration? Không gọi argmax accuracy là metric cuộc thi.
6. Gate có đạt không và bước tiếp nào được evidence hỗ trợ. Không hứa H>.7.

Chưa train final2494, predict test hoặc upload Kaggle trong task pilot. Nhãn
đầu ra phải giữ đủ1…5; priority0 loại paper, priority dương xếp bằng expected
gain. Không ép các class test bằng nhau. Không thay incumbent/submission cũ.

Author là phase tiếp trong PLAN, chưa có CLI. Nó từng giúp recall nhưng tăng
FP; đừng gọi việc thêm19 smoothed author stats là mới. Nếu mở phase author,
implement đối chứng, cross-fitted lineage và author×topic/support rõ ràng trước;
không dùng in-sample Qwen scores để huấn luyện stacking như thể OOF sạch.

Nếu gặp lỗi kỹ thuật, sửa trước khi scoring và ghi phiên bản mới. Chỉ chạy
GPU jobs cần thiết, không chạy grid rộng. Commit code/docs nếu bạn thay đổi;
giữ checkpoint/logs trong outputs (gitignored). Task kết thúc bằng báo cáo
thực tế và artifacts kiểm chứng được, không chỉ một kế hoạch tiếp theo.
