# Cross-encoder preference v2 — sửa collapse trước CV

## Lý do tạo v2

V1 đã trả cùng một probability cho toàn bộ prediction của nhiều partition. Chẩn đoán
cho thấy v1 đưa CLS thô (norm khoảng 25) vào head vẫn nhân 10, trái với implementation
SPECTER2 cũ đã L2-normalize CLS. ReLU của head chết và gradient về adapter biến mất.
Artifact V1 được giữ nguyên, không tái diễn giải kết quả của nó.

V2 chỉ sửa và kiểm chứng lỗi training trước. Không thay reference rule, label, split,
encoder, token budget hay objective để không gộp nhiều giả thuyết trong một lần sửa.

## Điều giữ nguyên

- SPECTER2 base và regression adapter local, base frozen.
- Title + abstract, tối đa 126 wordpiece cho query và reference; pair tối đa 255.
- Ba nearest Label 1 và ba nearest Label >1 từ pool fit, khác publication group;
  ưu tiên venue nếu đủ ba; retrieval cosine rồi id.
- Query-only: CE5 + .5 BCE(read). Contextual: thêm .25 BCE(query label > reference label),
  bỏ tie. Query label không vào input; label reference chỉ đến từ fit partition.

## Sửa và guardrail

CLS được `F.normalize(..., dim=1)` ngay sau encoder trước khi cả query và pair head
nhân 10. Training dùng `torch.cuda.amp.GradScaler`, unscale trước clip. Mỗi run lưu:

- số batch có gradient adapter khác 0;
- active fraction của ReLU head;
- số probability vectors khác nhau, largest duplicate group, read-score standard deviation;
- đối chiếu logits AMP và FP32 trước update;
- group isolation, token cap, input hash và adapter drift.

Không được vào nested CV nếu probability không thay đổi theo input, hidden ReLU chết,
gradient adapter mất, hoặc checkpoint reload cho probability khác.

## Hai stage

1. **Smoke test:** outer0/inner0, lấy deterministic 512 query train và 128 held-out
   group query. Chỉ để phát hiện failure mode; không chọn threshold/model và không báo H.
   Chạy cả query-only lẫn contextual với hai epoch cố định.
2. **Nested CV:** chỉ khi smoke pass. Tạo protocol mới và chạy 5 outer x 4 inner group
   folds. Trước khi thử token budget hay cấu trúc pair khác, báo tách riêng ảnh hưởng
   normalization để biết semantic strategy có tín hiệu hay chỉ hết lỗi training.

Kết quả outer vẫn exploratory vì labels đã được xem trong nghiên cứu trước.
