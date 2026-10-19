#!/usr/bin/env python3
"""Describe saved outer OOF errors; do not train or choose new parameters."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'


def markdown(frame):
    columns = list(frame.columns)
    rows = ['| ' + ' | '.join(columns) + ' |',
            '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for values in frame.itertuples(index=False, name=None):
        rows.append('| ' + ' | '.join(str(v).replace('|', '\\|').replace('\n', ' ') for v in values) + ' |')
    return '\n'.join(rows)


def main():
    d = pd.read_csv(OUT / 'nested_oof_predictions.csv')
    clean = pd.read_csv(OUT / 'abstracts_cleaned.csv')
    clean = clean[clean.split.eq('train')]
    d = d.merge(clean[['id', 'abstract_usable_v2', 'identity_review_note']],
                on='id', validate='one_to_one')
    assert len(d) == 2494 and d.id.is_unique
    d['positive'] = d.Label.gt(1)
    for prefix, family in [('old', 'raw_specter2_ridge'), ('v2', 'selection_procedure')]:
        selected = d[f'{family}_priority'].gt(0)
        d[f'{prefix}_selected'] = selected
        d[f'{prefix}_status'] = np.select(
            [selected & d.positive, selected & ~d.positive, ~selected & d.positive],
            ['TP', 'FP', 'FN'], default='TN')
        # Every selected outer-fold family here is a direct rating regressor.
        d[f'{prefix}_rating'] = 1 + 4 * d[f'{family}_gain']
    d['transition'] = d.old_status + ' -> ' + d.v2_status
    old_error = d.old_status.isin(['FP', 'FN'])
    new_error = d.v2_status.isin(['FP', 'FN'])
    d['change'] = np.select([old_error & ~new_error, ~old_error & new_error,
                             old_error & new_error],
                            ['fixed', 'new_error', 'still_error'], default='still_correct')
    manifest = json.loads((OUT / 'embedding_manifest.json').read_text())
    changed_ids = {int(k.split(':')[1]) for k in manifest['reencoded_keys'] if k.startswith('train:')}
    d['input_repaired'] = d.id.isin(changed_ids)
    d['abstract_group'] = np.where(d.abstract_usable_v2.astype(bool), 'usable_abstract', 'title_only')
    d['v2_rank_in_fold'] = pd.Series(pd.NA, index=d.index, dtype='Int64')
    for _, group in d.groupby('outer_fold'):
        ix = group[group.v2_selected].assign(id_sort=group.id.astype(str)).sort_values(
            ['selection_procedure_priority', 'id_sort'], ascending=[False, True]).index
        d.loc[ix, 'v2_rank_in_fold'] = range(1, len(ix) + 1)
    d['FP'] = d.v2_status.eq('FP')
    d['FN'] = d.v2_status.eq('FN')
    d.to_csv(OUT / 'residual_failure_rows.csv', index=False)
    transitions = pd.crosstab(d.old_status, d.v2_status).reindex(
        index=['TP', 'TN', 'FP', 'FN'], columns=['TP', 'TN', 'FP', 'FN'], fill_value=0)
    transitions.to_csv(OUT / 'residual_transitions.csv')
    venue = d.groupby('venue').agg(n=('id', 'size'), positives=('positive', 'sum'),
                                  FP=('FP', 'sum'), FN=('FN', 'sum'))
    venue['FPR_percent'] = (100 * venue.FP / (venue.n - venue.positives)).round(1)
    venue['FNR_percent'] = (100 * venue.FN / venue.positives).round(1)
    label = d.groupby('Label').agg(n=('id', 'size'), selected=('v2_selected', 'sum'),
                                  FN=('FN', 'sum'), mean_rating=('v2_rating', 'mean'),
                                  std_rating=('v2_rating', 'std')).round(3)
    quality = d.groupby('abstract_group').agg(n=('id', 'size'), positives=('positive', 'sum'),
                                            FP=('FP', 'sum'), FN=('FN', 'sum'))
    for name, frame in [('venue', venue), ('label', label), ('abstract_group', quality)]:
        frame.to_csv(OUT / f'residual_by_{name}.csv')
    fixed = int((old_error & ~new_error).sum())
    introduced = int((~old_error & new_error).sum())
    assert fixed == 80 and introduced == 81
    assert d.FP.sum() == 397 and d.FN.sum() == 321
    assert quality.loc['title_only', 'FN'] + quality.loc['usable_abstract', 'FN'] == 321
    top_fp = d[d.FP].sort_values(['v2_rank_in_fold', 'outer_fold']).head(6)
    high_fn = d[d.FN & d.Label.ge(4)].sort_values(['Label', 'v2_rating'], ascending=[False, True]).head(12)
    examples = pd.concat([top_fp, high_fn])
    example_columns = ['id', 'title', 'venue', 'Label', 'outer_fold', 'v2_status',
                       'old_rating', 'v2_rating', 'v2_rank_in_fold']
    examples[example_columns].to_csv(OUT / 'residual_examples.csv', index=False)
    repairs = d[d.input_repaired][example_columns + ['transition']].copy()
    repairs.to_csv(OUT / 'residual_repaired_inputs.csv', index=False)
    summary = {
        'n': len(d), 'old_errors': int(old_error.sum()), 'v2_errors': int(new_error.sum()),
        'fixed': fixed, 'introduced': introduced, 'persisting_errors': int((old_error & new_error).sum()),
        'FP': int(d.FP.sum()), 'FN': int(d.FN.sum()), 'repaired_training_inputs': len(changed_ids),
        'false_negatives_label_4_5': int((d.FN & d.Label.ge(4)).sum()),
        'false_negatives_cav_lics': int((d.FN & d.venue.isin(['cav', 'lics'])).sum()),
    }
    (OUT / 'residual_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = f'''# Vì sao v2 tăng ít, và các lỗi còn ở đâu?

Phân tích dùng 2.494 dự đoán **outer OOF** đã lưu: mỗi record được dự đoán khi không nằm trong tập train của model đó. Không refit, đổi ngưỡng hay chọn mô hình dựa trên các lỗi dưới đây. V2 ở đây là toàn bộ `selection_procedure`, không phải một model cuối duy nhất. So sánh với baseline trên đúng các outer folds; không so với holdout 500 mục cũ.

## Mức tăng nhỏ phản ánh một đánh đổi thực sự

Điểm trung bình 5 folds: **0.61777 → 0.61987**, tăng **0.00210**, khoảng **0.34% tương đối**. SD giữa folds của v2 là 0.01410; mức tăng này chưa chứng minh cải thiện đáng tin cậy ngoài tập dữ liệu này. SD không phải confidence interval.

- FP: **428 → 397** (giảm 31); FN: **289 → 321** (tăng 32).
- Sửa được **{fixed}** quyết định: 55 FP thành TN và 25 FN thành TP.
- Phát sinh **{introduced}** quyết định sai mới: 24 TN thành FP và 57 TP thành FN.
- Có **{summary['persisting_errors']}** quyết định vẫn sai. Tổng lỗi chọn/bỏ: **717 → 718**.
- Q trung bình tăng 0.49667 → 0.50717; R giảm 0.81883 → 0.79822. Metric còn phụ thuộc mức nhãn và thứ tự, nên tổng lỗi không giảm mà điểm vẫn có thể tăng nhẹ.

{markdown(transitions.reset_index())}

## Lỗi dữ liệu được sửa chỉ chiếm phần nhỏ

Chỉ 16/3.090 input thay đổi, gồm **15/2.494 mẫu train (~0.60%)**. Mô hình vẫn dùng encoder SPECTER2 đóng băng và các mô hình tuyến tính; lượt này không fine-tune encoder theo sở thích người dùng. Các mẫu được sửa ảnh hưởng cả model được fit và những dự đoán khác, nhưng không cung cấp một lượng lớn tín hiệu sở thích mới.

Phép ablation giữ cùng Ridge alpha=0.1, splits và cách tune ngưỡng: 0.61777 → 0.61958; FP 428 → 413, FN 289 → 294. Không quy chênh lệch toàn bộ v2 cho riêng việc làm sạch.

ID 1674 được chọn lại đúng; ID 1883 tăng expected rating rõ rệt nhưng vẫn bị bỏ. Đây là bằng chứng sửa input có ích ở một số ca nhưng không đủ giải quyết quyết định cuối. Các thay đổi dự đoán này gồm cả ảnh hưởng của thay model/ngưỡng, không phải hiệu ứng nhân quả riêng của từng sửa abstract. Chi tiết: `outputs/residual_repaired_inputs.csv`.

## Các lỗi còn lại

### 1. Chọn sai 397 bài nhãn 1, có bài đứng rất cao

Rank dưới đây tính **riêng trong outer fold** (khoảng 500 ứng viên/fold), không ghép các fold thành một ranking. Expected rating là 1+4*predicted gain, không phải nhãn đã quan sát của model.

{markdown(top_fp[example_columns].round({'old_rating': 3, 'v2_rating': 3}))}

Ví dụ ID 217 nhãn 1 nhưng đứng thứ 2 trong fold. Việc chỉ dịch ngưỡng chọn thông thường không giải quyết được một bài đã bị chấm cao như vậy. Cần cải thiện tín hiệu hoặc cách học sở thích, thay vì chỉ tăng coverage abstract.

### 2. Bỏ sót 321 bài, gồm 40 bài nhãn 4–5

{markdown(label.reset_index())}

Nhãn 1 có mean prediction khoảng 2.06, nhãn 5 khoảng 3.20. Dự đoán giữa các mức vẫn chồng lấn và co về giữa; đây là triệu chứng quan sát được của mô hình hiện tại, không chứng minh nhãn gốc sai.

{markdown(high_fn[example_columns].round({'old_rating': 3, 'v2_rating': 3}))}

### 3. Bỏ sót tập trung ở CAV/LICS; ICLP/KR còn chọn nhiều bài nhãn 1

{markdown(venue.reset_index())}

**277/321 FN (~86.3%)** ở CAV và LICS. FPR tính trên số nhãn 1 từng venue; FNR tính trên số nhãn >1 từng venue. Không suy ra cứ thuộc một venue là nên chọn hoặc bỏ: mỗi venue vẫn chứa cả hai loại nhãn. Có thể dùng thông tin này để thiết kế mô hình bổ sung đặc trưng/interaction rồi kiểm tra lại, không gán thủ công nhãn theo venue.

### 4. Thiếu abstract vẫn tồn tại nhưng không giải thích phần lớn lỗi

{markdown(quality.reset_index())}

`usable_abstract` nghĩa là vượt các quy tắc kiểm tra đang có, không phải tất cả đã được đọc đối chiếu thủ công. Còn 80 mục thiếu văn bản, 35 mục cần xác minh danh tính nguồn và 1 abstract chỉ dẫn sang công trình khác trên cả train+test; 9 input vẫn dài hơn 512 tokens. Có thể còn lỗi nguồn ngoài các ca đã xác nhận. Bảng trên kiểm tra trực tiếp các lỗi OOF, tránh quy tất cả lỗi cho coverage.

## Cần sửa gì tiếp theo?

Ưu tiên **học phân biệt sở thích ở các chủ đề gần nhau** và kiểm soát đánh đổi chọn/bỏ. V2 hai tầng đạt FP=424, FN=275 và điểm 0.62043: bỏ sót ít hơn quy trình đang chọn, nhưng FP cao hơn; chưa có họ nào giải quyết được cả hai ở mức lớn. Không thể chỉ hạ ngưỡng rồi coi là đã sửa vì sẽ đưa thêm bài nhãn 1 vào.

Các thí nghiệm kế tiếp có cơ sở: bổ sung author/venue/year cùng text dưới regularization; thử học ordinal hoặc ranking phù hợp các mức nhãn; so sánh encoder retrieval và cân nhắc fine-tuning sau khi có validation nghiêm ngặt. Đây là các giả thuyết cần đo, không phải bảo đảm tăng điểm. Giữ một tập đánh giá mới hoặc dùng đánh giá ngoài phù hợp vì tập OOF này đã được xem để thiết kế các thay đổi.

Chênh lệch inner CV của hai cấu hình đầu chỉ khoảng 0.00029; lựa chọn cấu hình còn không ổn định. Không gọi họ hybrid đạt 0.62242 là điểm của riêng model 50/50 cuối cùng, hoặc đổi sang họ thắng outer CV rồi báo cùng điểm như một đánh giá độc lập.

## Artifacts

- `outputs/residual_failure_rows.csv`: đủ 2.494 mục, trạng thái cũ/mới, transition, rating và rank trong fold.
- `outputs/residual_examples.csv`: các ví dụ sai nặng ở trên.
- `outputs/residual_transitions.csv`, `residual_by_venue.csv`, `residual_by_label.csv`, `residual_by_abstract_group.csv`.
- `outputs/residual_repaired_inputs.csv`: kết quả 15 input train đã sửa.
- `outputs/residual_summary.json`: các số đếm kiểm tra được.

Chạy lại: `python experiments/specter2_v2/analyze_residuals.py`.
'''
    (HERE / 'RESIDUAL_FAILURES.md').write_text(report)
    print(json.dumps(summary, indent=2))
    print('\n' + quality.to_string())
    print('\n' + venue.to_string())


if __name__ == '__main__':
    main()
