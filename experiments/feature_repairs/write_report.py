#!/usr/bin/env python3
"""Report all locked feature repairs after independent verification."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
ARMS=['reference','counts','title128','abstract64','rules','matched','combined','selection_procedure']


def main():
    verification=json.loads((OUT/'verification.json').read_text())
    assert verification['status']=='PASSED'
    from diagnose_failures import diagnose
    accounting=diagnose()
    folds=pd.read_csv(OUT/'fold_metrics.csv')
    summary=folds.groupby('policy',sort=False).agg(H=('score','mean'),Q=('relevance','mean'),
        R=('coverage','mean'),AUC=('AUC','mean'),train_AUC=('train_AUC','mean'),
        gain_MAE=('gain_MAE','mean'),FP=('FP','sum'),FN=('FN','sum'),missed_high=('missed_high','sum')).reindex(ARMS)
    summary['AUC_gap']=summary.train_AUC-summary.AUC
    summary.to_csv(OUT/'summary.csv')
    reference=folds[folds.policy.eq('reference')].set_index('outer_fold')
    effects=[];by_fold=[]
    for arm in ARMS[1:]:
        part=folds[folds.policy.eq(arm)].set_index('outer_fold')
        delta=part.score-reference.score
        effects.append(dict(arm=arm,delta_H=float(delta.mean()),folds_improved=int((delta>0).sum()),
            delta_Q=float((part.relevance-reference.relevance).mean()),
            delta_R=float((part.coverage-reference.coverage).mean()),
            delta_AUC=float((part.AUC-reference.AUC).mean()),
            delta_FP=int(part.FP.sum()-reference.FP.sum()),
            delta_FN=int(part.FN.sum()-reference.FN.sum()),
            delta_missed_high=int(part.missed_high.sum()-reference.missed_high.sum())))
        for fold in delta.index:
            by_fold.append(dict(arm=arm,outer_fold=int(fold),delta_H=float(delta.loc[fold]),
                delta_FP=int(part.loc[fold,'FP']-reference.loc[fold,'FP']),
                delta_FN=int(part.loc[fold,'FN']-reference.loc[fold,'FN'])))
    effects=pd.DataFrame(effects);effects.to_csv(OUT/'paired_effects.csv',index=False)
    pd.DataFrame(by_fold).to_csv(OUT/'paired_effects_by_fold.csv',index=False)
    oof=pd.read_csv(OUT/'oof.csv',dtype={'id':str})
    y=oof.Label.to_numpy();positive=y>1;old=oof.reference_priority.to_numpy()>0
    transitions=[];breakdown=[]
    for arm in ARMS:
        selected=oof[arm+'_priority'].to_numpy()>0
        before=old!=positive;after=selected!=positive
        for i in range(len(oof)):
            transitions.append(dict(id=oof.id.iloc[i],arm=arm,Label=int(y[i]),venue=oof.venue.iloc[i],
                old_status=('TP' if positive[i] else 'FP') if old[i] else ('FN' if positive[i] else 'TN'),
                new_status=('TP' if positive[i] else 'FP') if selected[i] else ('FN' if positive[i] else 'TN'),
                fixed=int(before[i] and not after[i]),introduced=int(after[i] and not before[i])))
        for group in ['venue','Label']:
            for value,indices in oof.groupby(group).groups.items():
                ix=np.asarray(list(indices));fp=selected[ix]&~positive[ix];fn=~selected[ix]&positive[ix]
                breakdown.append(dict(arm=arm,group=group,value=value,n=len(ix),FP=int(fp.sum()),FN=int(fn.sum())))
    pd.DataFrame(transitions).to_csv(OUT/'failure_transitions.csv',index=False)
    pd.DataFrame(breakdown).to_csv(OUT/'failure_breakdown.csv',index=False)
    audit=pd.read_csv(OUT/'feature_audit.csv')
    outer=audit[audit.partition.str.endswith('_validation')]
    choice=json.loads((OUT/'final_choices.json').read_text())['selection_procedure']
    recipe=joblib.load(OUT/'final_recipe.joblib')
    schema=[]
    for arm in ARMS[:-1]:
        model=CatBoostClassifier()
        model.load_model(str(OUT/f"final_{'reference' if arm=='matched' else arm}.cbm"))
        for position,column in enumerate(model.feature_names_):
            if column.startswith('title128_'):
                group='word_title_SVD128'
            elif column.startswith('abstract64_'):
                group='word_abstract_SVD64'
            elif column.startswith('title_svd'):
                group='inherited_title_SVD32'
            elif column.startswith('content_'):
                group='literal_content_evidence'
            elif column.startswith('neighbor_'):
                group='reference_history'
            else:
                group='metadata_or_Ridge'
            schema.append(dict(arm=arm,position=position,feature=column,group=group,
                dtype='categorical' if column=='venue' else 'numeric'))
    schema=pd.DataFrame(schema);schema.to_csv(OUT/'feature_catalog.csv',index=False)
    feature_counts=schema.groupby('arm').size().reindex(ARMS[:-1])
    policy_delta=float(summary.loc['selection_procedure','H']-summary.loc['reference','H'])
    best=summary.loc[ARMS[:-1],'H'].idxmax()
    decision=('Candidate has a positive exploratory policy delta; keep it isolated for external validation.' if policy_delta>0 else
              'The selection policy did not improve mean H; retain the incumbent as the main deliverable.')
    change=pd.read_csv(OUT/'rule_changes.csv',dtype={'id':str})
    changes=change.groupby('feature').agg(changed=('id','size'),before_sum=('before','sum'),after_sum=('after','sum')).reset_index()
    changes.to_csv(OUT/'rule_change_summary.csv',index=False)
    body=f'''# Kết quả sửa feature

**{decision}**

Giữ cùng CatBoost 300 trees/depth4/L2=8, seed và 5outer/4inner groups. Threshold
và arm chọn trong inner folds. Toàn selection policy delta H **{policy_delta:+.5f}**;
arm outer cao nhất là `{best}`, nhưng không dùng outer winner để chọn final.
Đây là các fold đã nghiên cứu nhiều lần, nên score là exploratory, không là
validation độc lập hay điểm Kaggle. Mốc0.7 chưa được suy ra từ feature retention.

## Kết quả toàn bộ các arm

{summary.to_markdown(floatfmt='.5f')}

Mean H tính trên từng fold; không lấy harmonic mean của Q/R đã trung bình. FP/FN
và missed-high là tổng2.494 OOF records. Binary errors không thay thế H.

## Hiệu ứng so với reference cùng model

{effects.to_markdown(index=False,floatfmt='.5f')}

`counts` chỉ sửa raw reference/support fields. `title128` thay SVD32 bằng128;
`abstract64` thêm64 chiều lexical riêng và giữ presence flag đã có; `rules` sửa
wording education/neural và thêm tutorial; `matched` giữ nguyên model/training
frame reference, đổi inference sang ba reference subsets rồi average probability.
`combined` đo kết hợp trực tiếp. Không cộng delta từng arm để dự đoán combined.
Số feature từng arm: {feature_counts.to_dict()}. Catalog theo đúng column order của
saved models ở `outputs/feature_catalog.csv`.

## Feature đã được sửa và giới hạn còn lại

- Full input/cache guard mới hash cả venue/year/authors/doi/Label, ZIP, sạch
  abstract, embedding caches, parent feature packs, splits và source.
- Có **{len(change)}** thay đổi flag trên **{change[['split','id']].drop_duplicates().shape[0]}**
  records. Ledger và nguyên văn bằng chứng nằm trong `rule_changes.csv` và
  `rule_evidence.csv`; đây là lexical evidence, không phải nhãn sở thích mới.
- Trong năm outer training fits, SVD title32 giữ mean variance
  **{outer.title_svd32_variance.mean():.4f}**, title128 **{outer.title_svd128_variance.mean():.4f}**.
  Abstract64 giữ **{outer.abstract_svd64_variance.mean():.4f}**. Không diễn giải các
  tỷ lệ này thành phần trăm thông tin nhãn bị mất hoặc hiệu quả dự đoán chắc chắn.
- Mean OOV occurrence rate ở outer prediction: title word1/2
  **{outer.title_prediction_oov.mean():.4f}**, abstract **{outer.abstract_prediction_oov.mean():.4f}**.
  Bigrams/min_df/vocabulary cap góp vào OOV; đây không phải tỷ lệ toàn paper không
  có representation.
- Matched inference dùng đúng reference subset sizes của crossfit train; các
  query overlap groups được loại khỏi reference. Thay đổi này đồng thời ảnh
  hưởng Ridge phụ, vocab references, neighbor statistics và averaging; không
  quy delta riêng cho reference count.
- SVD vẫn không giám sát và nén thông tin; cờ vẫn không hiểu đóng góp cụ thể;
  authors chưa được thêm; chưa có encoder abstract học theo preference trong
  thí nghiệm này. Không tự gán nhãn sai cho các paper có preference khác nhau.

{changes.to_markdown(index=False,floatfmt='.0f')}

## Vì sao tăng feature chưa kéo được H?

Validation AUC của các arm sửa nằm trong **{summary.loc[ARMS[1:-1],'AUC'].min():.5f}..
{summary.loc[ARMS[1:-1],'AUC'].max():.5f}**, so với reference
**{summary.loc['reference','AUC']:.5f}**. Tăng title dimensions đưa train AUC từ
**{summary.loc['reference','train_AUC']:.5f}** lên **{summary.loc['title128','train_AUC']:.5f}**,
nhưng validation AUC là **{summary.loc['title128','AUC']:.5f}**. Combined train AUC
**{summary.loc['combined','train_AUC']:.5f}**, validation
**{summary.loc['combined','AUC']:.5f}**. Đây là bằng chứng rằng các thay đổi này fit
train tốt hơn mà chưa bổ sung khả năng phân biệt preference ngoài fit tương ứng;
không chứng minh encoder khác hoặc mọi biểu diễn abstract đều vô ích.

Combined so với reference: Q đổi **{summary.loc['combined','Q']-summary.loc['reference','Q']:+.5f}**,
recall đổi **{summary.loc['combined','R']-summary.loc['reference','R']:+.5f}**, FP đổi
**{int(summary.loc['combined','FP']-summary.loc['reference','FP']):+d}** và FN đổi
**{int(summary.loc['combined','FN']-summary.loc['reference','FN']):+d}**. Phần gain
recall đi cùng nhiều false positives hơn, nên H không tăng. Không quy kết chung
preprocessing gây plateau chỉ từ việc SVD giữ ít variance hoặc rule có lỗi từ ngữ.

Đối chiếu hậu nghiệm giữ read gate của một model và gain ranker của model kia,
giữ đúng thresholds đã chọn trong inner, cho phân rã sau:

{accounting.to_markdown(floatfmt='.5f')}

Gate contribution và rank contribution dùng trung bình hai thứ tự thay component
(Shapley accounting), cộng lại đúng delta H. Đây là phân rã metric đã quan sát,
không là chứng minh nguyên nhân gốc và không là các candidate được chọn/triển khai.
Các hybrid chỉ dùng chẩn đoán, không dùng outer labels để chọn final recipe.

## Final candidate và kiểm chứng

Final inner chọn `{choice['arm']}`, cutoff **{choice['threshold']:.2f}**,
inner H **{choice['inner_score']:.5f}**. Candidate ở
`outputs/submission_candidate.csv`, model ở `outputs/{recipe['model_file']}`,
recipe ở `outputs/final_recipe.joblib`. Giữ nguyên incumbent; không upload.

Verifier `{verification['status']}`: replay **{verification['control_partitions_replayed']}**
partitions của control; kiểm tra **{verification['grouped_reference_lineages_checked']}**
crossfits, **{verification['matched_reference_views_checked']}** matched views và
**{verification['projection_transforms_checked']}** projections; tính lại
**{verification['outer_policy_rows_recomputed']}** outer metric rows và
**{verification['inner_choices_recomputed']}** inner choices. Đã kiểm tra original
labels, missing abstract vectors, metadata-sensitive hashes, evidence spans,
heldout-label invariance, rebuild final features và model reload; CSV đủ596 ID.

Từng case/fold ở `outputs/failure_transitions.csv`, `failure_breakdown.csv` và
`paired_effects_by_fold.csv`. Xem [PLAN.md](PLAN.md),
[verification.json](outputs/verification.json), [feature audit](outputs/feature_audit.csv).
'''
    (HERE/'RESULTS.md').write_text(body)
    (HERE/'DECISION.md').write_text('# Quyết định sau sửa feature\n\n'+decision+'\n\n'+
        f"Full inner-selected policy H={summary.loc['selection_procedure','H']:.5f}, reference H={summary.loc['reference','H']:.5f}, delta={policy_delta:+.5f}. Final arm={choice['arm']}, cutoff={choice['threshold']:.2f}.\n\n"+
        'Không dùng arm outer cao nhất làm điểm final policy. Fold cũ/exploratory; chưa upload hay thay incumbent. Chi tiết: [RESULTS.md](RESULTS.md).\n')
    print(summary.to_string())
    print(decision)
    print('FINAL',json.dumps(choice))


if __name__=='__main__':
    main()
