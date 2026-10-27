# Remote Qwen: tải checkpoint, kiểm tra, train và đánh giá

Đây là entry point mới cho **supervised QLoRA 5 nhãn**, thay cho các hướng dẫn
frozen inference/binary gate cũ. Đọc [PLAN.md](PLAN.md) để hiểu thiết kế và
[AGENT_PROMPT.md](AGENT_PROMPT.md) để bàn giao nguyên task cho coding agent.

Package này độc lập với `experiments/llm_preference/outputs/` và các cache cũ.
`inputs/` được commit, gồm 2.494 title/abstract đã làm sạch, nhãn gốc, hai split
theo publication group và xác suất incumbent đúng từng split. **Clone là đủ
dữ liệu cho pilot**; không chạy lại scraper, SPECTER2, CatBoost hoặc exporter.
Metadata author gốc được giữ để audit, chưa đưa vào arm nội dung.

## 1. Môi trường remote

Linux, Python 3.11, NVIDIA GPU hỗ trợ BF16. Khuyến nghị >=12 GB VRAM,
32 GB RAM và >=40 GB disk trống; laptop 8 GB từng chạy được technical preflight
nhưng remote vẫn phải vượt smoke test. Một GPU là đủ; script không dùng DDP.

Chạy từ repository root sau khi clone/pull commit chứa package này:

```bash
python3.11 -m venv .venv-qwen
source .venv-qwen/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r experiments/qwen_remote_v1/requirements.txt
python -m pip check
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())'
```

Stack torch 2.6/CUDA 12.4 là recipe remote đề xuất, **chưa chạy GPU trên máy
remote**. Torch wheel phải tương thích driver và kiến trúc GPU. Nếu server có
GPU mới không được wheel này hỗ trợ, agent xử lý môi trường trước, ghi rõ
phiên bản thay thế và chạy smoke; không đổi model/protocol để né lỗi.
Không cài `flash-attn`, không nâng cấp môi trường legacy tại chỗ.
Mỗi run ghi lại package versions và runtime fingerprint.

Recipe torch theo [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).
Recipe NF4/LoRA theo [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization).

## 2. Tải đúng checkpoint

```bash
python experiments/qwen_remote_v1/verify.py
python experiments/qwen_remote_v1/verify_evaluation.py
python experiments/qwen_remote_v1/run.py download
python experiments/qwen_remote_v1/verify.py --tokenizer
```

Model: `Qwen/Qwen2.5-7B-Instruct`, revision
`a09a35458c702b33eeacc393d103063234e8bc28`. Tải weights safetensors gốc,
không tải AWQ/GPTQ thay thế. `download` dùng Hugging Face cache mặc định;
có thể đặt `HF_HOME` theo storage server trước khi chạy. Training chỉ đọc
local cache, không tự tải revision khác khi thiếu file.

## 3. Một lệnh chạy toàn bộ pilot

```bash
bash experiments/qwen_remote_v1/run_pilot.sh
```

Lệnh này tuần tự chạy tokenizer verification → smoke → frozen scoring + train
outer1_inner0 → frozen scoring + train outer1_inner1 → evaluation. Hai adapter
được khởi tạo **độc lập** từ cùng base. Mỗi adapter train đúng một epoch.
Chạy trong tmux nếu SSH có thể ngắt. Không đồng thời chạy job khác trên GPU đó.

Output mặc định: `experiments/qwen_remote_v1/outputs/pilot_v1/`, bị gitignore.
Chọn namespace khác bằng `QWEN_OUTPUT`; chọn Python khác bằng `QWEN_PYTHON`.

Có thể chạy từng bước để monitor:

```bash
python experiments/qwen_remote_v1/run.py smoke
python experiments/qwen_remote_v1/run.py train --partition outer1_inner0
python experiments/qwen_remote_v1/run.py train --partition outer1_inner1
python experiments/qwen_remote_v1/evaluate.py
```

Nếu bitsandbytes không tìm thấy thư viện CUDA ở cách chạy trực tiếp, dùng
`run_pilot.sh` trên output mới; wrapper thêm thư viện từ torch wheel vào
`LD_LIBRARY_PATH`. Giữ toàn bộ stderr để chẩn đoán lỗi.

## 4. Kết quả cần đọc

Trong output root:

- `summary.csv`: mean H/Q/R và tổng TP/FP/FN/TN, missed label 4/5 theo arm/policy.
- `confusion.csv`, `class_metrics.csv`: ma trận 5×5, recall/precision từng nhãn.
- `label_transitions.csv`: mỗi nhãn được nhận thêm hay bỏ đi bao nhiêu so incumbent.
- `cases.csv`: từng paper, năm xác suất, true/predicted label, title/venue/authors.
- `decision.json`: có tín hiệu học so frozen không; có vượt gate so incumbent không.

Trong mỗi partition: adapter safetensors, raw `frozen.jsonl`/`trained.jsonl`,
`training.jsonl`, `training_summary.json`, prompt audit, runtime lock,
save/reload parity và trạng thái. Không diễn giải smoke thành quality result.

## 5. Gián đoạn và chạy lại

Không tự ghi đè run cũ. Nếu đã có smoke hoàn tất, chạy trực tiếp partition
còn thiếu thay vì chạy lại toàn bộ shell script. Nếu inference bị gián đoạn:

```bash
python experiments/qwen_remote_v1/run.py train --partition outer1_inner0 --resume
```

`--resume` chỉ dùng khi thư mục partition đã tồn tại và fingerprint khớp.
Nó tiếp tục prediction JSONL hoặc nạp adapter đã có `TRAIN_COMPLETE.json`.
**Chưa hỗ trợ resume giữa training từ optimizer/RNG state.** Nếu training đã
bắt đầu mà chưa có TRAIN_COMPLETE, giữ artifacts, dùng `--output` mới, chạy
smoke và train lại. Không sửa/xóa training log để ép resume.
Nếu sửa code/config/environment, dùng output mới và chạy smoke lại.

## 6. Thời gian và giới hạn bàn giao

Benchmark laptop 4060 cho Q candidate-only: khoảng 0,87–0,89 giờ tính toán
forward/backward cho một epoch 1.494 mẫu, chưa gồm optimizer, frozen/trained
scoring, reload và tải checkpoint. Đây **không phải ETA remote**. Dùng thời
gian thực ở `training.jsonl` sau 10 updates để ước lượng phần training còn
lại; đo riêng inference và loading. Cả pilot có hai training jobs, không chỉ
một. Không dùng lại ước tính 93–120 giờ đã bị thay thế.

Runner ở đây thực thi pilot nội dung một epoch và smoke. Author ablation,
nested confirmation, calibration theo inner OOF, final refit và submission
được thiết kế trong PLAN.md nhưng **chưa có CLI**. Agent không được gọi hai
fold pilot là validation độc lập hoặc tự tuyên bố đã có model tốt hơn.

## 7. Kiểm tra local trước commit

Xem [VALIDATION.md](VALIDATION.md). CPU/schema/metric/tokenizer checks có thể
chạy local. GPU smoke và train phải chạy trên remote; chưa có quality result
cho package này tại thời điểm bàn giao.
