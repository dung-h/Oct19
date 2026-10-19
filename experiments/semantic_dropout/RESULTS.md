# Nội dung của các bài đáng đọc đang bị bỏ

Phạm vi: 313 false negatives (FN) của incumbent, từ 2.494 dự đoán outer OOF.
Đọc toàn bộ 38 FN nhãn 4–5 và 14 FN nhãn 2–3 được lấy có chủ đích theo venue
và xác suất thấp/trung vị/sát ngưỡng. Các nhận xét semantic dưới đây dựa trên
52 trường hợp này; không gán phân loại semantic đã đọc thủ công cho cả 313 bài.
Không train thêm model, không thay nhãn hoặc prediction, không sửa submission.

## Những gì đo được trên toàn bộ FN

| Kiểm tra | Kết quả |
|---|---:|
| Tổng FN | 313 |
| Nhãn 2 / 3 / 4 / 5 trong FN | 183 / 92 / 33 / 5 |
| FN có abstract | 309/313 |
| FN nhãn 4–5 có abstract | 37/38 |
| FN ở CAV / LICS | 116 / 142 |
| FN có P(read)<0,45 | 176 |
| FN có 0,50<=P(read)<=0,55 | 77 |
| FN mà title-SPECTER2 cũng bỏ | 233/313 |
| FN nhãn 4–5 mà title-SPECTER2 cũng bỏ | 23/38 |

Thiếu abstract là vấn đề thật ở một số bài, nhưng không giải thích được đa số
FN. 275/313 FN là nhãn 2–3: nên phân biệt những bài chỉ hơi đáng đọc với 38
trường hợp bị mất nhãn cao. 176 FN ở dưới 0,45 cũng cho thấy không phải tất cả
đều chỉ mắc ở sát ngưỡng 0,55.

Với mỗi bài nhãn dương, tìm riêng bài nhãn 1 gần nhất và bài nhãn 2–5 gần nhất
trong chính outer-fit, cùng venue, bằng frozen SPECTER2 retrieval đã có. Loại
toàn publication group của query. Không đưa bài validation vào nguồn tham chiếu.

| Nhóm bài nhãn dương | Có Label 1 gần hơn bài dương gần nhất |
|---|---:|
| Bị bỏ (FN) | 187/313 = 59,7% |
| Được nhận (TP) | 274/1.278 = 21,4% |

Đây là liên hệ giữa neighborhood và lỗi, không chứng minh encoder là nguyên
nhân duy nhất. Similarity là độ gần theo một biểu diễn; không phải xác suất
cùng sở thích. Kết quả không cho phép suy ra lý do thật của người gán nhãn.

## 1. Cùng chủ đề, khác đối tượng và câu hỏi nghiên cứu

**ID2383, Label 5, P(read)=0,4622: Soundness of reset workflow nets.**

Abstract tập trung các biến thể soundness của workflow nets có reset, chứng
minh generalised soundness undecidable và chỉ ra một thuộc tính trung gian có
hành vi decidability khác. Bài Label 1 gần nhất trong cùng-venue fit là ID3071,
*The Complexity of Coverability in ν-Petri Nets* (cosine 0,933): nghiên cứu
coverability, safety và complexity của một loại Petri net có quản lý tên.

Hai bài cùng nets/verification/complexity, nhưng khác **loại net → thuộc tính
được xét → kết quả**. Các cờ như automata/nets, verification, complexity không
biểu diễn đầy đủ quan hệ này. Đây là một trường hợp nhầm sự gần chủ đề với sự
phù hợp sở thích. Không suy ra từ một cặp rằng soundness luôn được thích hơn
coverability.

## 2. Cùng công nghệ, khác kiểu đóng góp

**ID1580, Label 4, P(read)=0,4081: Towards Benchmarking of Solidity Verification
Tools.** Abstract đề xuất benchmark và so sánh SolCMC/Certora theo soundness,
completeness, expressiveness. Label 1 gần nhất là ID1685, *SolCMC: Solidity
Compiler's Model Checker* (cosine 0,965), giới thiệu kiến trúc và cách dùng
một công cụ.

Đối tượng chung là Solidity verification, còn đóng góp một bên là **đánh giá
so sánh nhiều công cụ**, một bên là **giới thiệu một công cụ**. Trong cached
features của ID1580, phân bố top15 retrieval cùng venue có p1=0,666 và Ridge
rating=1,733. Đây là bằng chứng feature lịch sử nghiêng về negative quanh bài
này; chưa phải attribution chứng minh CatBoost dùng riêng chúng để bỏ bài.

**ID366, Label 4: Google vs IBM: A Constraint Solving Challenge on the Job-Shop
Scheduling Problem** cũng là đánh giá so sánh solver/benchmark, thay vì đề xuất
solver mới. Tuy nhiên không được biến nhận xét này thành luật 'benchmark thì
đọc': ID1926, *The Move Prover*, cũng Label 4 dù là giới thiệu công cụ.

## 3. Title ít thông tin; đóng góp cụ thể nằm trong abstract

**ID2456, Label 4, P(read)=0,5013: Folding interpretations.** Title không nói
rõ polyregular functions. Abstract mới cho biết bài xây hệ combinators với
fold và type system để biểu diễn chính xác lớp hàm đó.

Label 1 gần nhất là ID2725, *Pebble Minimization of Polyregular Functions*
(cosine 0,957): mối liên hệ giữa số pebbles, độ tăng kích thước output và biểu
diễn transducer tối thiểu. Chủ đề chung rất gần nhưng hai đóng góp khác nhau.

**ID2521, Label 4, P(read)=0,3547: Curry and Howard Meet Borel** có title ngắn,
liên tưởng. Abstract mới lộ quan hệ giữa counting propositional logic, type
system và probabilistic lambda calculus. **ID2281, Label 4, P(read)=0,2588:
BigraphER** cần thông tin về rewriting, simulation, bigraphs và PRISM để hiểu
khả năng của công cụ, ngoài tên riêng trong title.

Trong incumbent, title được nén SVD32; abstract đi vào cờ lexical và thống kê
láng giềng, không có một khối biểu diễn đầy đủ problem/method/result. Trong
title-adapted SPECTER2, abstract không phải input. Đây là giới hạn biểu diễn
cụ thể, chưa chứng minh chỉ cần thêm abstract embedding là sửa được lỗi:
abstract-adapter pilot trước đó cũng chưa thắng incumbent.

## 4. Đóng góp phân biệt bằng giả thiết, biến thể hoặc kết luận

**ID2758, Label 4, P(read)=0,4154: Topology is relevant...** Abstract trình bày
phản ví dụ và phân biệt local/global satisfaction trong infinite-domain CSP.
Label 1 gần nhất ID3016 (cosine 0,953) nghiên cứu reduction và lifting
tractability. Hai bài có đúng cùng bộ content-category flags trong audit cũ,
dù vấn đề kỹ thuật cụ thể khác nhau.

**ID162, Label 4, P(read)=0,3815: Recursive Aggregates as Intensional
Functions** giải thích semantics của recursive aggregates theo cách diễn giải
functions, gồm liên hệ clingo/dlv. Label 1 gần nhất ID191 nghiên cứu level
ranking constraints và pipeline dịch ASP sang SAT/SMT/MIP. 'Aggregates' hoặc
'logic programming' chung không mô tả được khác biệt semantics/encoding.

Không phải mọi FN đều thiếu positive neighbors. Với ID162, top15 retrieval
cùng venue có p1=0,234 và p4+p5=0,588, nhưng Ridge rating=1,585 và quyết định
cuối vẫn bỏ. **ID519, Label 5, P(read)=0,4472: Non-deterministic Action
Reversibility** cũng gần bài dương ID978, Label 4 (cosine 0,950), hơn negative
gần nhất (0,916). Có tín hiệu phù hợp nhưng các khối feature có thể bất đồng;
không thể quy mọi lỗi cho embedding tìm sai hàng xóm.

## 5. Ứng dụng và phương pháp quan trọng bị trộn thành một chủ đề rộng

**ID1041, Label 4, P(read)=0,3246: Intelligent Clustering as a Means to Improve
K-means Based Horizontal Attacks** kết hợp k-means với side-channel analysis;
đóng góp là cách chọn centroids để tăng khả năng khôi phục key. Label 1 gần
nhất ID1040 tập trung ảnh hưởng của sampling rate trong horizontal DEMA attack.
Ứng dụng chung là side-channel attacks, nhưng thao tác và mục tiêu cải thiện
khác nhau. Cũng có bài dương gần hơn (ID1026, Label 2, cosine 0,962), nên
trường hợp này không phù hợp lời giải thích 'chỉ có negative neighbors'.

Các FN nhãn 4 như ID1942 (MCTS cho phối hợp tránh va chạm), ID1928 (điều
khiển/coordination tại vòng xuyến), ID2051 (RL scheduling cho V2V) cho thấy
không thể định nghĩa toàn bộ sở thích bằng một nhãn logic/verification duy
nhất. Không có căn cứ coi tất cả ứng dụng xe cộ hay tất cả bài mang venue CAV
là nên bỏ. Cũng không được từ ba ví dụ suy ra mọi bài về xe cộ đều đáng đọc.

## 6. Document type và lượng thông tin thật sự có sẵn

**ID1045, Label 5, P(read)=0,2523** là sách/tuyển tập workshop AIME về KR,
health care và explainable systems. Abstract đang có chỉ 27 từ, là mô tả
tuyển tập bị cắt, không trình bày một problem/method/result của một paper.
Label 1 gần nhất ID1184 cũng là tuyển tập KR/health care, cosine 0,954.

Không thể dùng một quy tắc ngữ nghĩa như 'proceedings thì không đọc'. Cũng
không nên giả sử mọi item phải có abstract nghiên cứu tiêu chuẩn. **ID1154,
Label 4, Universal Composability: A Comparison of Different Models** là FN
nhãn cao duy nhất còn thiếu abstract. Những trường hợp này là input-poor
thật sự; khác với đa số FN đã có abstract nhưng representation/learner chưa
phân biệt tốt.

## Kết luận có thể dùng để định hướng

Các trường hợp khó thường đòi hỏi mô tả quan hệ **đối tượng nghiên cứu → câu
hỏi/thuộc tính → phương pháp → kết quả/đóng góp**, thay vì chỉ nhận biết topic.
Có ít nhất ba nhóm cần xử lý khác nhau: thiếu input thật sự; representation
gộp các đóng góp khác nhau; có tín hiệu dương nhưng các feature/model bất đồng.

Đổi threshold không tạo ra các khác biệt nội dung còn thiếu. Một bộ trích
problem/method/result tốt hơn phải kiểm tra được các cặp đối chứng trên; bộ
regex chọn câu cộng embedding/SVD đã thử trước đây chưa chứng minh làm được
điều đó. Không chỉ đổi tên các cờ keyword thành 'semantic features'.

Các nhận xét ở đây không xác định được lý do thật của nhãn cá nhân, không là
bằng chứng label noise, và không hứa một kiến trúc mới sẽ tăng điểm. Những
case đã đọc là diagnostic probes, không phải dữ liệu chưa từng xem để tuyên
bố tổng quát hóa của extractor mới.

## Tái lập và dữ liệu đối chiếu

- `python experiments/semantic_dropout/analyze.py`: không fit model.
- `outputs/false_negatives.csv`: đủ 313 FN với title/abstract/probability.
- `outputs/review_sample.csv`: đúng 52 case được chọn đọc.
- `outputs/neighbors.csv`: 3 nearest Label 1 và 3 nearest Label 2–5, trong
  cùng venue và toàn venue, chỉ từ outer-fit.
- `outputs/positive_neighborhoods.csv`: kiểm tra neighborhood của cả 1.591
  bài dương, gồm TP và FN.
- `outputs/REVIEW_PACKET.md`: source texts và láng giềng cho 52 case.
- `outputs/newly_lost_true_positives.csv`: riêng 15 TP incumbent bị gate
  author_venue bỏ, gồm 2 nhãn 4; chưa gộp vào 52 case review ở trên.
- `outputs/summary.json`, `outputs/counts.csv`, `outputs/manifest.json`: số
  liệu và hashes. Mọi nhãn/prediction nguồn giữ nguyên.
