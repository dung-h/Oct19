#!/usr/bin/env python3
"""Summarize sample efficiency and the limits of observable content evidence."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import run_diagnosis as run


def main():
    out=run.OUT
    check=json.loads((out/'verification.json').read_text())
    metrics=pd.read_csv(out/'learning_curve_metrics.csv')
    columns=['n_fit','n_eval','score','read_auc','rating_mae','rating_rmse','multiclass_logloss','read_logloss']
    # Average subset seeds within each fold FIRST; full-size fits count once.
    folded=metrics.groupby(['outer_fold','fraction','model','partition'],as_index=False)[columns].mean()
    folded.to_csv(out/'fold_averages.csv',index=False)
    summary=folded.groupby(['fraction','model','partition'],as_index=False).agg(
        n_fit=('n_fit','mean'),H=('score','mean'),H_fold_sd=('score','std'),
        AUC=('read_auc','mean'),MAE=('rating_mae','mean'),RMSE=('rating_rmse','mean'),
        logloss=('multiclass_logloss','mean'),read_logloss=('read_logloss','mean'))
    summary.to_csv(out/'curve_summary.csv',index=False)
    endpoint_rows=[]
    for model in ['ridge','catboost']:
        for fold in range(5):
            subset=folded[folded.model.eq(model)&folded.outer_fold.eq(fold)]
            val=subset[subset.partition.eq('validation')].set_index('fraction')
            train=subset[subset.partition.eq('train')].set_index('fraction')
            endpoint_rows.append(dict(model=model,outer_fold=fold,
                delta_AUC_25_100=val.loc[1.,'read_auc']-val.loc[.25,'read_auc'],
                delta_AUC_75_100=val.loc[1.,'read_auc']-val.loc[.75,'read_auc'],
                delta_MAE_25_100=val.loc[1.,'rating_mae']-val.loc[.25,'rating_mae'],
                delta_MAE_75_100=val.loc[1.,'rating_mae']-val.loc[.75,'rating_mae'],
                delta_H_25_100=val.loc[1.,'score']-val.loc[.25,'score'],
                delta_H_75_100=val.loc[1.,'score']-val.loc[.75,'score'],
                AUC_gap_full=train.loc[1.,'read_auc']-val.loc[1.,'read_auc'],
                MAE_gap_full=val.loc[1.,'rating_mae']-train.loc[1.,'rating_mae']))
    endpoints=pd.DataFrame(endpoint_rows);endpoints.to_csv(out/'endpoint_fold_deltas.csv',index=False)
    effect=endpoints.groupby('model',as_index=False).mean(numeric_only=True).drop(columns='outer_fold')
    effect['folds_AUC_improve_25_100']=[int((endpoints.loc[endpoints.model.eq(m),'delta_AUC_25_100']>0).sum()) for m in effect.model]
    effect['folds_AUC_improve_75_100']=[int((endpoints.loc[endpoints.model.eq(m),'delta_AUC_75_100']>0).sum()) for m in effect.model]
    effect.to_csv(out/'endpoint_summary.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    for model,color in [('ridge','#2563eb'),('catboost','#c2410c')]:
        for part,style in [('train','--'),('validation','-')]:
            rows=summary[summary.model.eq(model)&summary.partition.eq(part)].sort_values('fraction')
            for ax,column,label in zip(axes,['AUC','MAE','H'],['Read / do-not-read AUC','Rating MAE','Official H (fixed cutoffs)']):
                ax.plot(rows.n_fit,rows[column],style,color=color,marker='o',label=f'{model} {part}')
                ax.set_xlabel('Training papers (mean across folds)');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Same held-out groups at every size; subset seeds averaged within each fold',fontsize=11)
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=180);fig.savefig(out/'learning_curves.pdf');plt.close(fig)
    reviews=pd.read_csv(out/'reviewed_pairs.csv',keep_default_na=False)
    pairs=pd.read_csv(out/'content_pairs.csv',keep_default_na=False)
    sampled=pairs[pairs.selection.eq('sample')]
    claims=pd.read_csv(out/'claim_spans.csv',keep_default_na=False)
    sample_claims=claims[claims.pair_id.isin(sampled.pair_id)]
    covered=sample_claims[['pair_id','role']].drop_duplicates().shape[0]
    zero=int(sampled.category_flag_differences.eq(0).sum())
    audit=dict(sampled_pairs=20,available_abstracts=int(sampled[['a_abstract','b_abstract']].ne('').sum().sum()),
        clear_observable_content_differences=20,preference_rationale_known=0,
        pairs_with_identical_category_flags=zero,records_with_candidate_claim_cues=covered,
        records_reviewed=40,reviewer_partial_flag_capture=int(reviews.flags_capture.eq('partial').sum()),
        reviewer_detail_not_explicit=int(reviews.flags_capture.eq('not_explicit').sum()),
        selected_sample_not_population_estimate=True)
    (out/'audit_summary.json').write_text(json.dumps(audit,indent=2)+'\n')
    r=effect.set_index('model').loc['ridge'];c=effect.set_index('model').loc['catboost']
    if c.delta_AUC_25_100>0 and c.delta_MAE_25_100<0:
        evidence='Trong phạm vi số mẫu đã thử, thêm các nhãn train thật giúp CatBoost phân biệt đọc/không đọc tốt hơn và giảm sai số rating. Điều này phù hợp với hạn chế về lượng ví dụ sở thích, chưa chứng minh tăng dữ liệu ngoài phạm vi này sẽ tiếp tục cho cùng mức lợi ích.'
    else:
        evidence='Learning curve chưa cho bằng chứng nhất quán rằng chỉ tăng số mẫu trong phạm vi đã thử sẽ giúp CatBoost. Cần xem từng fold và các metric trước khi quy nút thắt cho thiếu dữ liệu.'
    text=f'''# Chẩn đoán khi tối ưu bị chững — kết quả

## Điều đã xác định được

**{evidence}**

- CatBoost, 25% → 100% outer train: validation AUC **{c.delta_AUC_25_100:+.5f}**, MAE **{c.delta_MAE_25_100:+.5f}**, AUC tăng ở **{int(c.folds_AUC_improve_25_100)}/5 folds**.
- 75% → 100%: validation AUC **{c.delta_AUC_75_100:+.5f}**, MAE **{c.delta_MAE_75_100:+.5f}**, AUC tăng ở **{int(c.folds_AUC_improve_75_100)}/5 folds**.
- Ở full size, train AUC cao hơn validation **{c.AUC_gap_full:.5f}** với CatBoost và **{r.AUC_gap_full:.5f}** với Ridge. Đây là khoảng cách tổng quát hóa; có thể gồm overfitting của head và khác biệt kích thước nguồn tham chiếu. Không suy ra một nguyên nhân duy nhất từ khoảng cách này.
- Khảo sát 20 cặp cho thấy văn bản có những khác biệt phương pháp/đóng góp quan sát được, nhưng **chưa biết lý do thật của nhãn sở thích trong cả 20 cặp**. Hai vấn đề này phải tách biệt.

## Learning curves

Giữ nguyên năm outer folds. Trong từng outer train, tạo nested subsets ở 25/50/75/100% theo nhóm, hai seed lấy mẫu. Có 35 bộ fit khác nhau, mỗi bộ fit hai model cố định. Full-size fit chỉ tính một lần; seed lấy mẫu được trung bình trong từng outer fold trước khi trung bình năm folds. Không tìm hyperparameter hay cutoff mới.

{summary[['fraction','model','partition','n_fit','AUC','MAE','H','logloss']].to_markdown(index=False,floatfmt='.5f')}

![Learning curves](outputs/learning_curves.png)

[PDF để xuất](outputs/learning_curves.pdf). Dùng AUC/MAE/log-loss để đọc khoảng cách học và tổng quát hóa. Log-loss chỉ áp dụng cho CatBoost; Ridge không cung cấp phân phối xác suất năm nhãn. H có cutoff chẩn đoán cố định: Ridge gain>0.25; CatBoost P(read)>0.50, thứ tự theo expected gain. H này không phải điểm của submission đã tune ngưỡng; train H cũng thay đổi theo độ dài danh sách và không được so với 1 như một mức tối ưu mặc định.

{effect.to_markdown(index=False,floatfmt='.5f')}

Chỉ có một model seed cố định; hai seed ở đây thay **cách lấy subset**, không phải hai benchmark độc lập. Các train folds dùng chung nhiều dữ liệu. Các chênh lệch là mô tả paired folds, không có khẳng định ý nghĩa thống kê từ giả định độc lập. Kiến trúc đã được chọn trong nghiên cứu trước trên cùng dữ liệu, nên đây vẫn là chẩn đoán exploratory.

Tăng số mẫu đồng thời tăng số ví dụ huấn luyện, độ phủ history và dữ liệu để fit vocab/SVD. Vì thế learning curve đo lợi ích của thêm mẫu cho **toàn pipeline**, chưa tách riêng tác động của số nhãn cho classifier với tác động của nguồn láng giềng/phần biểu diễn.

## Khảo sát nội dung

Mẫu 20 cặp, bốn cặp mỗi venue, được chọn bằng similarity retrieval cao nhất giữa một bài nhãn 1 và một bài nhãn 4/5; không chọn theo model error. Các bài không trùng nhau trong mẫu và khác duplicate group trong mỗi cặp. Các ví dụ cũ được xuất riêng, không cộng vào mẫu 20 nếu không tự xuất hiện theo quy tắc. ID 217/355 tự xuất hiện dưới quy tắc này.

- Có đủ usable abstract cho **{audit['available_abstracts']}/40** bài; **{covered}/40** bài có ít nhất một candidate sentence theo discourse cues. Cues chỉ là cách tìm đoạn văn để xem, không phải phân tích ngữ nghĩa đã được xác minh và chưa được đưa vào model.
- Người đọc thấy khác biệt cụ thể về nội dung ở 20/20 cặp đã chọn; tất cả có hai đoạn bằng chứng nguyên văn và offsets. Đây không phải 20 quyết định rằng nhãn 1/4/5 đúng hay sai, hoặc rằng khác biệt vừa tìm được chính là lý do gán nhãn.
- **{zero}/20** cặp có toàn bộ category flags giống nhau: P05 (epistemic semantics) và P16 (infinite-domain CSP). Các vector/title features khác vẫn có thể giữ thông tin; không suy ra toàn bộ model đầu vào giống nhau.
- Theo nhận xét nội dung, {audit['reviewer_partial_flag_capture']}/20 cặp được bộ cờ thể hiện một phần khác biệt đã mô tả, {audit['reviewer_detail_not_explicit']}/20 không thể hiện tường minh khác biệt đó. Đây là rubric đọc của một người trên mẫu khó, không phải accuracy tổng quát của extractor.
- P07: `student` xuất hiện trong thông tin Ph.D. của tác giả, nhưng kích hoạt cờ domain education. P08: `tutorial` có rõ trong văn bản nhưng taxonomy document type chưa nhận diện. Literal phrase match đã được lưu đúng; việc diễn giải nó thành ngữ nghĩa tài liệu vẫn có thể sai hoặc thiếu.

[Đọc đủ 20 nhận xét và bằng chứng](outputs/CONTENT_REVIEW.md), [bảng CSV có offsets](outputs/reviewed_pairs.csv), [toàn văn các cặp để kiểm tra](outputs/PAIR_PACKET.md).

## Áp dụng vào hướng đi tiếp theo

1. **Ưu tiên kiểm soát variance và ổn định việc chọn model.** Giữ Ridge và CatBoost hiện tại làm mốc. Learning curve cho biết thêm bao nhiêu lợi ích khi tăng mẫu trong phạm vi có sẵn; không có cơ sở để gọi H≈0.64 là trần bài toán hoặc lập tức tăng model capacity.
2. **Nếu làm extraction tiếp, phải giữ nội dung của mệnh đề cụ thể thay vì thêm nhiều cờ chủ đề.** Ví dụ cần phân biệt “epistemic atoms ở rule heads” với “world views ba giá trị”, hoặc CSP reduction với local/global height-1 identities. Đánh giá chất lượng trích xuất bằng bằng chứng trước khi train lại. Chẩn đoán này chỉ xác định thông tin nào có thể đọc được, chưa kiểm chứng rằng extraction mới tăng H.
3. **Còn thiếu thông tin về quy trình gán nhãn.** Cần biết người gán nhãn đã thấy title, abstract, full text hay thông tin nào khác, và labels có cùng một tiêu chí/thời điểm hay không. Không suy đoán nhãn sai hoặc tự sửa labels. Không tạo thêm nhãn sở thích giả để giải quyết tình trạng ít mẫu.
4. **Đánh giá ứng viên hiện có theo lợi ích và độ ổn định**, không loại chỉ vì chưa tăng 0.02 H. Mốc 0.02 trước đây là mục tiêu nghiên cứu tự đặt, không phải yêu cầu assignment. Chưa có dữ liệu ngoài nghiên cứu này để khẳng định thứ hạng Kaggle hoặc khả năng cải thiện khi thêm dữ liệu tương lai.

## Kiểm chứng và tái lập

Đã kiểm tra nhãn gốc, 35 subset partitions, tính lồng nhau và tách duplicate groups, {check['grouped_crossfit_partitions_verified']} cross-fitting partitions; tính lại {check['metrics_recomputed']} metric rows từ raw predictions. Dự đoán full size khớp thí nghiệm cũ tới 1e-10. Kiểm tra 40 đoạn bằng chứng thủ công và {check['candidate_claim_spans_verified']} candidate cue spans. Ba unit tests về subset label isolation, cutoff và offsets đều qua. Không tạo submission/model family mới.

```bash
/tmp/assignment-specter2-venv/bin/python experiments/plateau_diagnosis/run_plan.py
```

Đọc [PLAN.md](PLAN.md), [outputs/verification.json](outputs/verification.json), [outputs/curve_summary.csv](outputs/curve_summary.csv), [outputs/endpoint_fold_deltas.csv](outputs/endpoint_fold_deltas.csv). Nguồn phương pháp: [scikit-learn learning curves](https://scikit-learn.org/stable/modules/learning_curve.html), [Cawley–Talbot về model-selection overfitting](https://jmlr.org/papers/v11/cawley10a.html), [Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml).
'''
    (run.HERE/'RESULTS.md').write_text(text)
    print(summary[['fraction','model','partition','AUC','MAE','H']].to_string(index=False))
    print(effect.to_string(index=False));print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
