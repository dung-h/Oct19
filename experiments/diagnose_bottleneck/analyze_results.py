#!/usr/bin/env python3
"""Summarize paired diagnostic contrasts; never fit/select a final model."""
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from run_diagnostics import HERE, OUT, V2, author_dict, evaluate, load_clean_data, priority


CONTRASTS = [
    ('head_regression', 'regression_ridge', 'regression_rbf'),
    ('head_retrieval', 'retrieval_ridge', 'retrieval_rbf'),
    ('head_minilm', 'minilm_ridge', 'minilm_rbf'),
    ('head_tfidf', 'tfidf_ridge', 'tfidf_rbf'),
    ('embedding_retrieval_linear', 'regression_ridge', 'retrieval_ridge'),
    ('embedding_minilm_linear', 'regression_ridge', 'minilm_ridge'),
    ('embedding_retrieval_nonlinear', 'regression_rbf', 'retrieval_rbf'),
    ('embedding_minilm_nonlinear', 'regression_rbf', 'minilm_rbf'),
    ('conditioning', 'regression_ridge', 'regression_centered_ridge'),
    ('add_metadata', 'regression_ridge', 'regression_plus_metadata_ridge'),
    ('add_venue', 'regression_ridge', 'regression_plus_venue_ridge'),
    ('add_authors', 'regression_ridge', 'regression_plus_authors_ridge'),
    ('add_year', 'regression_ridge', 'regression_plus_year_ridge'),
]


def markdown(frame):
    frame = frame.copy()
    for col in frame.select_dtypes(include='floating').columns:
        frame[col] = frame[col].map(lambda value: f'{value:.5f}')
    header = '| ' + ' | '.join(frame.columns) + ' |'
    rows = ['| ' + ' | '.join(str(x).replace('|','\\|').replace('\n',' ') for x in row) + ' |'
            for row in frame.itertuples(index=False, name=None)]
    return '\n'.join([header, '| ' + ' | '.join(['---']*len(frame.columns)) + ' |'] + rows)


def main():
    metrics = pd.read_csv(OUT/'fold_metrics.csv')
    summary = pd.read_csv(OUT/'summary.csv').set_index('family')
    oof = pd.read_csv(OUT/'oof_predictions.csv', dtype={'id':str})
    labels = oof.Label.astype(int)
    baseline_selected = oof.regression_ridge_priority.gt(0)
    baseline_error = baseline_selected.ne(labels.gt(1))
    contrasts, changes = [], []
    for name, first, second in CONTRASTS:
        left = metrics[metrics.family.eq(first)].set_index('outer_fold')
        right = metrics[metrics.family.eq(second)].set_index('outer_fold')
        delta = right.score-left.score
        contrasts.append(dict(contrast=name, baseline=first, changed=second,
                              delta_H=float(delta.mean()), delta_SD=float(delta.std()),
                              folds_improved=int(delta.gt(0).sum()),
                              delta_AUC=float((right.read_auc-left.read_auc).mean()),
                              delta_MAE=float((right.validation_mae-left.validation_mae).mean()),
                              **{f'fold_{k}_delta':float(v) for k,v in delta.items()}))
    paired = pd.DataFrame(contrasts)
    paired.to_csv(OUT/'paired_contrasts.csv', index=False)
    text_families = [f'{space}_{head}' for space in ['regression','retrieval','minilm','tfidf']
                     for head in ['ridge','rbf']]
    common_error = pd.concat([oof[family+'_priority'].gt(0).ne(labels.gt(1))
                              for family in text_families],axis=1).all(axis=1)
    common_rows = oof.loc[common_error].copy()
    common_rows.to_csv(OUT/'errors_shared_by_all_eight_text_families.csv', index=False)
    venue_rows = []
    for family in summary.index:
        selected = oof[family+'_priority'].gt(0)
        error = selected.ne(labels.gt(1))
        status = np.select([selected & labels.gt(1), selected & labels.eq(1), ~selected & labels.gt(1)],
                            ['TP','FP','FN'], default='TN')
        frame = oof[['id','title','venue','Label','outer_fold']].copy()
        frame['status'] = status
        frame['family'] = family
        frame['rating'] = oof[family+'_rating']
        frame['baseline_error'] = baseline_error
        frame['error'] = error
        changes.append(dict(family=family, errors=int(error.sum()),
                            fixed_vs_regression_ridge=int((baseline_error & ~error).sum()),
                            introduced_vs_regression_ridge=int((~baseline_error & error).sum()),
                            persisting=int((baseline_error & error).sum())))
        for venue, group in frame.groupby('venue'):
            venue_rows.append(dict(family=family, venue=venue, n=len(group),
                                   FP=int(group.status.eq('FP').sum()), FN=int(group.status.eq('FN').sum())))
    pd.DataFrame(changes).to_csv(OUT/'error_changes.csv', index=False)
    pd.DataFrame(venue_rows).to_csv(OUT/'errors_by_venue.csv', index=False)
    data, _ = load_clean_data()
    train = data[data.split.eq('train')]
    splits = pd.read_csv(V2/'nested_splits.csv', dtype={'id':str})
    coverage_rows = []
    for fold in range(5):
        split = splits[splits.outer_fold.eq(fold)]
        fit = train[train.id.isin(split.loc[split.partition.eq('outer_train'),'id'])]
        val = train[train.id.isin(split.loc[split.partition.eq('outer_validation'),'id'])]
        known = set().union(*(author_dict(x) for x in fit.authors))
        has_author = val.authors.map(lambda x:bool(author_dict(x)))
        seen_author = val.authors.map(lambda x:bool(known.intersection(author_dict(x))))
        coverage_rows.append(dict(outer_fold=fold, n=len(val), no_author=int((~has_author).sum()),
                                  known_author=int(seen_author.sum()),
                                  only_unseen_authors=int((has_author & ~seen_author).sum())))
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUT/'author_coverage_by_fold.csv', index=False)
    example_ids = ['217','184','519','1344','3062','1883','1674']
    examples = oof[oof.id.isin(example_ids)].copy()
    example_families = ['regression_ridge','regression_rbf','retrieval_ridge','minilm_ridge',
                        'regression_plus_metadata_ridge']
    columns = ['id','title','Label','outer_fold']
    for family in example_families:
        examples[family+'_selected'] = examples[family+'_priority'].gt(0)
        columns.extend([family+'_rating', family+'_selected'])
    examples[columns].to_csv(OUT/'example_comparison.csv', index=False)

    # A fixed-alpha anchor provides comparability to the earlier clean-data ablation.
    searches = pd.read_csv(OUT/'inner_search.csv')
    anchor_rows = []
    for fold in range(5):
        row = searches[searches.outer_fold.eq(fold) & searches.key.eq('c002')].iloc[0]
        cache = np.load(OUT/f'outer_{fold}_predictions.npz')
        val = cache['val']
        ratings = cache['outer_c002']
        result = evaluate(labels.iloc[val], priority(ratings, row.threshold), oof.id.iloc[val])
        anchor_rows.append(dict(outer_fold=fold, threshold=row.threshold, **result))
    anchor = pd.DataFrame(anchor_rows)
    anchor.to_csv(OUT/'fixed_alpha_anchor.csv', index=False)

    # Geometry is descriptive, without labels, and is not proof of information loss.
    geom = []
    rng = np.random.default_rng(20261015)
    left, right = rng.integers(0,len(oof),size=(2,30000))
    mask = left != right
    for space, path in [('regression',V2/'embeddings_clean.npz'),
                         ('retrieval',OUT/'embeddings_retrieval.npz'),
                         ('minilm',OUT/'embeddings_minilm.npz')]:
        vectors = normalize(np.load(path)['embeddings'][:len(oof)].astype(np.float64))
        cosine = (vectors[left[mask]]*vectors[right[mask]]).sum(axis=1)
        geom.append(dict(space=space, cosine_p05=float(np.quantile(cosine,.05)),
                         cosine_median=float(np.median(cosine)), cosine_p95=float(np.quantile(cosine,.95))))
    pd.DataFrame(geom).to_csv(OUT/'embedding_geometry.csv',index=False)

    # Use a standalone plot for comparison across the five folds; no CI interpretation.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_names = ['regression_ridge','regression_rbf','retrieval_ridge','retrieval_rbf',
                  'minilm_ridge','minilm_rbf','tfidf_ridge','tfidf_rbf',
                  'regression_centered_ridge','regression_plus_metadata_ridge']
    fig, ax = plt.subplots(figsize=(10,6))
    for i, name in enumerate(plot_names):
        values = metrics[metrics.family.eq(name)].sort_values('outer_fold').score.to_numpy()
        ax.scatter(values, np.full(len(values),i), s=25, alpha=.65, color='#377eb8')
        ax.scatter([values.mean()],[i], marker='D', s=55, color='#e4572e')
    ax.set_yticks(range(len(plot_names)), plot_names)
    ax.invert_yaxis()
    ax.axvline(summary.loc['regression_ridge','score_mean'],color='gray',linestyle='--',alpha=.7)
    ax.set_xlabel('Official H on saved outer validation folds')
    ax.set_title('Blue: individual folds; orange diamond: mean (no confidence interval)')
    ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/'diagnostic_comparison.png',dpi=180)
    plt.close(fig)
    tables = summary.reset_index()
    focus = paired[paired.contrast.isin(['head_regression','embedding_retrieval_linear',
                   'embedding_minilm_linear','conditioning','add_metadata','add_venue','add_authors','add_year'])]
    matrix = tables[tables.family.isin(plot_names[:8])][['family','score_mean','score_std','FP','FN','read_auc','validation_mae']]
    metadata = tables[tables.family.isin(['regression_ridge','regression_plus_metadata_ridge',
                    'regression_plus_venue_ridge','regression_plus_authors_ridge','regression_plus_year_ridge',
                    'metadata_ridge','venue_ridge','authors_ridge','year_ridge'])]
    report = f'''# Thí nghiệm xác định nút thắt

## Kết luận từ các phép thử đã chạy

**Chưa có bằng chứng embedding hiện tại là nguyên nhân chính, cũng chưa có bằng chứng đổi head phi tuyến hay bổ sung metadata sẽ giải quyết phần lớn lỗi.** Kết luận này chỉ áp dụng các mô hình và cách biểu diễn được đo ở đây; không chứng minh encoder/head hiện tại là tối ưu.

- Đổi SPECTER2 regression sang retrieval với cùng Ridge: H {summary.loc['regression_ridge','score_mean']:.5f} → {summary.loc['retrieval_ridge','score_mean']:.5f}, giảm ở cả 5 folds. MiniLM + Ridge {summary.loc['minilm_ridge','score_mean']:.5f}, cũng giảm ở cả 5 folds. Các phép đổi embedding generic này không sửa nút thắt.
- Giữ embedding và thay Ridge bằng RBF: H {summary.loc['regression_rbf','score_mean']:.5f}, chỉ tăng trung bình 0.00057 và tốt hơn 2/5 folds. Train MAE giảm rõ hơn validation MAE; AUC chọn/bỏ gần như không tăng. Mô hình linh hoạt hơn chủ yếu khớp train tốt hơn, chưa cải thiện khả năng phân biệt sở thích ngoài train một cách lớn.
- Thêm cả author/venue/year: H {summary.loc['regression_plus_metadata_ridge','score_mean']:.5f}; chỉ thêm venue {summary.loc['regression_plus_venue_ridge','score_mean']:.5f}. Chưa thấy thiếu metadata là nguyên nhân chi phối dưới phép mã hóa/ghép đã thử. Metadata có thể cứu vài bài riêng mà tổng thể không tốt hơn.
- Center vector theo mean train rồi L2: H {summary.loc['regression_centered_ridge','score_mean']:.5f}. Cosine cao của embedding regression chưa đủ để gọi là lỗi; đổi hình học đơn giản này không giúp.
- Có **{int(common_error.sum())} mục cùng sai chọn/bỏ ở cả 8 họ text**: {int((common_error & labels.eq(1)).sum())} FP và {int((common_error & labels.gt(1)).sum())} FN. Đây là nhóm khó chung để điều tra tiếp, không phải bằng chứng nhãn gốc sai.

Phải điều chỉnh giả thuyết trước đó: không có cơ sở quy nguyên nhân chính cho việc head còn tuyến tính hoặc thiếu author/venue/year. **Giả thuyết còn lại cần kiểm chứng** là pretrained features và tín hiệu hiện có chưa phân biệt đủ sở thích cá nhân; cũng có thể còn giới hạn từ mục tiêu học, chất lượng input hoặc lượng dữ liệu. Thí nghiệm này chưa tách được các khả năng đó. Muốn kiểm chứng tiếp cần so sánh fine-tuning theo nhãn sở thích với frozen encoder dưới validation phù hợp, hoặc kiểm tra thêm chất lượng/tín hiệu dữ liệu; không thể khẳng định nguyên nhân chỉ từ việc nhiều model cùng sai.

## Thiết kế và giới hạn

Đã chạy **17 họ / 138 cấu hình** được khai báo trong `outputs/protocol.json` trước đánh giá. Dùng đúng 5 outer folds và 4 inner folds từ v2. Mỗi outer train tự chọn tham số/ngưỡng bằng inner CV. Không dùng nhãn outer validation để chọn cấu hình trong từng họ. Toàn bộ nhãn giữ nguyên từ ZIP; nhóm trùng title/identifier không giao qua train/validation.

Đối chiếu theo cặp giữ cùng các yếu tố còn lại: đổi representation với cùng Ridge hoặc RBF; đổi head trên cùng representation; thêm metadata vào SPECTER2 regression; thêm phép chuẩn hóa center-train-mean + L2. Hyperparameter grid của cùng head giống nhau giữa representations.

Head tuyến tính: Ridge alpha `[.001,.01,.1,1,10,100]`. Head phi tuyến: kernel ridge RBF alpha `[.01,.1,1]`, gamma multiplier `[.25,1,4]` trên median squared distance **tính chỉ trong training fold**, target trừ mean train rồi cộng lại. Vocab author/venue/TF-IDF, scaler năm và mean vector đều fit trong từng training fold. Không fit TF-IDF hoặc chuẩn hóa thống kê bằng cả dataset.

Các score là **trung bình H của 5 outer folds**, FP/FN là cộng 2.494 OOF. Model trong một họ có thể chọn tham số khác nhau mỗi fold. Đây không phải điểm đã xác nhận cho một model cuối fit toàn bộ train. Các fold dùng training sets chồng lấn; không gọi fold wins hoặc SD là kiểm định ý nghĩa thống kê.

Dataset và OOF cũ đã được xem khi xây dựng thí nghiệm. Nested CV hạn chế overfit lúc tune, không loại bỏ hoàn toàn ảnh hưởng của quá trình nghiên cứu thích nghi trên cùng dữ liệu.

## Đổi embedding và đổi head

{markdown(matrix)}

SPECTER2 retrieval dùng cùng base và cùng CLS/max512 như regression, chỉ đổi adapter; đây là phép so sánh gần kiểm soát nhất về embedding. MiniLM đổi cả encoder/tokenizer/pooling, là lựa chọn biểu diễn khác chứ không chỉ khác adapter. MiniLM dùng mean pooling theo attention mask, max512 để cùng ngân sách token (khác default256 của sentence-transformers); do tokenizer khác, số token và phần văn bản bị cắt không hoàn toàn trùng nhau.

Nguồn triển khai: [SPECTER2 model card](https://huggingface.co/allenai/specter2) mô tả adapter retrieval/regression và CLS; [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) mô tả mean pooling/default256. SHA model và input có trong protocol/embedding manifests; không fine-tune encoder trong thí nghiệm này.

## Các chênh lệch có kiểm soát

Delta = model thay đổi trừ model đối chiếu. H/AUC cao hơn tốt hơn, MAE thấp hơn tốt hơn. `folds_improved` chỉ mô tả số fold có H tăng.

{markdown(focus[['contrast','delta_H','delta_SD','folds_improved','delta_AUC','delta_MAE']])}

Toàn bộ các cặp và delta từng fold nằm trong `outputs/paired_contrasts.csv`; bảng này không lựa chọn thêm họ model để báo một điểm tổng thắng cuộc.

## Metadata riêng và khi kết hợp text

{markdown(metadata[['family','score_mean','FP','FN','read_auc','validation_mae']])}

Author là tên đầy đủ chuẩn hóa Unicode/case/whitespace rồi tách dấu phẩy/chấm phẩy, multi-hot + L2; không dùng nhãn tác giả được tính từ cả dataset. Venue one-hot; year chuẩn hóa bằng train, nhân0.25. Metadata tổng hợp được L2; khi ghép embedding, trọng số `[.25,1]` được chọn bên trong. Một phép ghép đơn giản không cải thiện không chứng minh metadata không có ích dưới mọi mô hình.

Author còn hạn chế coverage: cộng outer validation có {int(coverage.known_author.sum())}/2.494 mục có ít nhất một author đã gặp trong outer train, {int(coverage.no_author.sum())} mục không có author và {int(coverage.only_unseen_authors.sum())} mục chỉ có author chưa gặp. Không gán điểm thấp cho author chưa gặp; bộ mã hóa bỏ qua các tên ngoài vocab train.

## Dấu hiệu quá khớp và hình học

{markdown(tables[tables.family.isin(plot_names)][['family','train_mae','validation_mae']])}

Train MAE thấp hơn validation MAE là chênh lệch khớp mẫu/khái quát hóa, không tự chứng minh tất cả lỗi là overfit. Lựa chọn head tối ưu theo H chứ không theo MAE, nên hai thước đo có thể không cùng tăng.

{markdown(pd.DataFrame(geom))}

Cosine cao giữa các vector chưa chứng minh embedding hỏng: Ridge có intercept nên đã loại ảnh hưởng mean khi fit; RBF được scale theo khoảng cách train thay vì dùng gamma mặc định không phù hợp. Phép center+L2 được đo riêng để xem thay hình học có giúp hay không, không dựa vào histogram cosine để kết luận.

## Ví dụ và kết quả kiểm tra

Xem `outputs/example_comparison.csv` cho ID 217,184,519,1344,3062,1883,1674: rating và chọn/bỏ riêng cho mỗi biểu diễn/head. `error_changes.csv` đếm số lỗi sửa được và phát sinh so với SPECTER2 regression+Ridge trong cùng thí nghiệm; `errors_by_venue.csv` tách nhóm venue.

Anchor alpha=0.1 (chọn ngưỡng inner, cùng splits) đạt H trung bình **{anchor.score.mean():.5f}**, dùng để đối chiếu phép ablation sạch v2 (~0.61958). Lượt này chuẩn hóa dense features bằng float64 và dùng Ridge LSQR tol=1e-6 thay vì default1e-4 trước đó; có sai khác số học nhỏ. Các phép so sánh trong bảng đều dùng cùng thiết lập mới. Baseline Ridge trong bảng chính tìm alpha trên grid rộng hơn v2 (cả 5 fold vẫn chọn alpha=0.1); không coi tác động mở grid hoặc sai khác số học là tác động thay embedding/head.

![Kết quả từng fold](outputs/diagnostic_comparison.png)

Chạy kiểm tra: `python experiments/diagnose_bottleneck/verify_results.py`. File `outputs/verification.json` chỉ xuất sau khi kiểm tra hash input/splits, nhãn, group separation, tái chọn toàn bộ cấu hình bằng inner predictions và tính lại từng outer score từ OOF priority thành công.

## Chạy lại

```bash
python experiments/diagnose_bottleneck/run_diagnostics.py all
python -m unittest discover -s experiments/diagnose_bottleneck -p 'test_*.py' -v
python experiments/diagnose_bottleneck/verify_results.py
python experiments/diagnose_bottleneck/analyze_results.py
```

Môi trường hiện tại: `/tmp/assignment-specter2-venv/bin/python`, dependencies theo `requirements.txt`. Cache embeddings và prediction từng config/fold cho phép kiểm tra mà không encode/fit lại. Không tạo submission hoặc thay model v2.
'''
    (HERE/'RESULTS.md').write_text(report)
    print(paired[['contrast','delta_H','folds_improved','delta_AUC','delta_MAE']].to_string(index=False))
    print('\nGeometry\n', pd.DataFrame(geom).to_string(index=False))


if __name__ == '__main__':
    main()
