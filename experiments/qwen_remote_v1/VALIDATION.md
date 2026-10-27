# Kiểm tra trước khi bàn giao — 2026-10-27

Đã thực hiện local, không khởi chạy GPU training cho package này:

- `verify.py`: SHA256 mọi input khớp; 2.494 nhãn khớp ZIP gốc; IDs/splits/
  publication groups và baseline row order hợp lệ; metric toy cases gồm
  empty selection, irrelevant-only và lexical ID tie-break đều pass.
- Tái tính incumbent đúng H .6487764769062809 và .6463847722551991;
  tổng TP513/FP135/FN118/TN235, missed4=8/missed5=3 trên hai prediction sets.
- `verify.py --tokenizer` với đúng checkpoint cached: 2.494 prompts hợp lệ;
  min105/median280/max677 tokens; answer token IDs16–20 riêng biệt; thay
  label và metadata không đổi input tokens của arm nội dung.
- `verify_evaluation.py`: 3 CPU integration checks pass. Prediction fixtures
  giống incumbent cho delta H=0 và không được gate giả; thiếu prediction arm
  hoặc prediction có foreign lineage bị từ chối trước khi xuất summary.
- Python compile và `bash -n run_pilot.sh` pass.
- Copy riêng package sang một thư mục tạm không có ZIP/caches cũ: input
  verification và cả 3 evaluation integration checks vẫn pass.

Chưa xác nhận: cài mới toàn bộ remote stack torch2.6/cu124, CUDA kernels của
GPU remote, GPU smoke của runner mới, train một epoch hoặc quality gain.
Model loader kế thừa dtype/LoRA fixes đã được preflight trước đây; runner mới
có orchestration/AMP riêng nên **bắt buộc smoke lại**, không dùng preflight cũ
để bỏ qua bước đó. Chưa có nested CV, author quality run hoặc submission mới.

Chỉ code/docs/portable inputs được commit; runtime logs, model shards và
adapters nằm ngoài Git. Các kiểm tra GPU/quality còn lại được thực thi theo
README và PLAN trên remote, không được ghi nhận là đã pass ở local.
