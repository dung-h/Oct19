#!/usr/bin/env python3
"""Report predeclared contrasts, paired gate changes and count-range diagnostics."""
import json

import joblib
import numpy as np
import pandas as pd

import run_combined as r


def decomposition():
    ctx=r.base.context();r.base.protocol(ctx);records=[]
    for fold,fit,val,_ in r.base.previous.outer_members(ctx['data']):
        vs,_=r.collect(ctx,fit,val,f'outer{fold}_validation')
        choices=json.loads((r.OUT/f'outer{fold}_choices.json').read_text())
        gates={f:vs[f]['read']>choices[f]['threshold'] for f in r.FAMILIES}
        def h(gate,rank):
            return r.base.ab.evaluate(ctx['y'][val],np.where(gates[gate],10*vs[rank]['gain'],0.),ctx['data'].id.iloc[val])['score']
        h00=h('MC_full','MC_full')
        for f in r.FAMILIES:
            h10=h(f,'MC_full');h01=h('MC_full',f);h11=h(f,f)
            gate=.5*((h10-h00)+(h11-h01));rank=.5*((h01-h00)+(h11-h10))
            assert abs(gate+rank-(h11-h00))<1e-12
            records.append(dict(outer_fold=fold,family=f,delta_H=h11-h00,gate_effect=gate,rank_effect=rank))
    pd.DataFrame(records).to_csv(r.OUT/'gate_rank_decomposition.csv',index=False)


def main():
    f=pd.read_csv(r.OUT/'fold_metrics.csv');summary=pd.read_csv(r.OUT/'summary.csv').set_index('family')
    oof=pd.read_csv(r.OUT/'oof.csv',dtype={'id':str});y=oof.Label.to_numpy(int)
    pairs=[('Combined minus defaults only','ORD_combined','ORD_defaults_full'),
           ('Combined minus neighbor removal only','ORD_combined','ORD_without_neighbors'),
           ('Stronger regularization after combined','ORD_combined_regularized','ORD_combined'),
           ('Count fractions, ordinal','ORD_count_fraction','ORD_defaults_full'),
           ('Count fractions, multiclass','MC_count_fraction','MC_full')]
    contrasts=[]
    for name,a,b in pairs:
        left=f[f.family.eq(a)].set_index('outer_fold');right=f[f.family.eq(b)].set_index('outer_fold')
        delta=left.score-right.score
        contrasts.append(dict(contrast=name,treated=a,control=b,delta_H=float(delta.mean()),
            improved_folds=int((delta>0).sum()),min_fold_delta=float(delta.min()),max_fold_delta=float(delta.max()),
            delta_AUC=float((left.AUC-right.AUC).mean()),delta_train_AUC=float((left.train_AUC-right.train_AUC).mean())))
    pd.DataFrame(contrasts).to_csv(r.OUT/'contrasts.csv',index=False)
    changes=[]
    for family in r.FAMILIES:
        a=oof[family+'_priority'].to_numpy()>0;b=oof.MC_full_priority.to_numpy()>0
        changes.append(dict(family=family,recovered_positive=int(((y>1)&~b&a).sum()),
            newly_missed_positive=int(((y>1)&b&~a).sum()),removed_false_positive=int(((y==1)&b&~a).sum()),
            added_false_positive=int(((y==1)&~b&a).sum()),recovered_high=int(((y>=4)&~b&a).sum()),
            newly_missed_high=int(((y>=4)&b&~a).sum()),FP=int(((y==1)&a).sum()),FN=int(((y>1)&~a).sum()),
            missed_high=int(((y>=4)&~a).sum())))
    pd.DataFrame(changes).to_csv(r.OUT/'gate_changes.csv',index=False)
    drift=[]
    for fold in range(5):
        p=joblib.load(r.base.PRIOR/f'outer{fold}_validation_features.joblib')
        a,b=p['a'],p['b'];aa=r.transform(a,'ORD_count_fraction');bb=r.transform(b,'ORD_count_fraction')
        for c in a:
            if not(c.startswith('neighbor_') and c.endswith('_support')):continue
            for stage,x,z in [('raw_count',a[c],b[c]),('support_fraction',aa[c],bb[c])]:
                drift.append(dict(outer_fold=fold,column=c,stage=stage,train_mean=float(x.mean()),
                    validation_mean=float(z.mean()),train_min=float(x.min()),train_max=float(x.max()),
                    validation_outside_train_range=float(((z<x.min())|(z>x.max())).mean())))
    pd.DataFrame(drift).to_csv(r.OUT/'count_drift.csv',index=False)
    decomposition()
    choices=json.loads((r.OUT/'final_choices.json').read_text())
    display={
        'MC_full':'Multiclass hiện tại',
        'ORD_defaults_full':'Ordinal: khóa mặc định',
        'MC_without_neighbors':'Multiclass: bỏ láng giềng',
        'ORD_without_neighbors':'Ordinal: chỉ bỏ láng giềng',
        'ORD_combined':'Ordinal: kết hợp hai sửa đổi',
        'ORD_combined_regularized':'Ordinal: kết hợp + regularization',
        'ORD_count_fraction':'Ordinal: chuẩn hóa cột đếm',
        'MC_count_fraction':'Multiclass: chuẩn hóa cột đếm',
        'selection_procedure':'Quy trình chọn bằng inner CV'}
    table=summary[['score','relevance','coverage','AUC','train_AUC']].rename(index=display,
        columns={'score':'H','relevance':'Q','coverage':'R'})
    parts=pd.read_csv(r.OUT/'gate_rank_decomposition.csv').groupby('family',sort=False)[['delta_H','gate_effect','rank_effect']].mean()
    ranges=pd.DataFrame(drift).pivot_table(index='column',columns='stage',values='validation_outside_train_range',aggfunc='mean')
    # Parent fixed ordinal is context, not a member of this selection policy.
    original=pd.read_csv(r.diag.OUT/'control_summary.csv')
    oldscore=float(original.loc[original.family.eq('ORD_full'),'H'].iloc[0])
    lines=['# Kiểm tra kết hợp các sửa đổi','',
        'Đây là kết quả thăm dò trên cùng năm outer folds đã được xem nhiều lần. Không phải điểm Kaggle hay đánh giá trên holdout độc lập. Dùng H theo phần Evaluation chi tiết của đề bài; mâu thuẫn mô tả ngắn 60/40 vẫn chưa được xác minh.','',
        '## Kết quả','',table.to_markdown(floatfmt='.6f'),'',
        f'Ordinal gốc ở thí nghiệm trước: H={oldscore:.6f}. Mọi ngưỡng và lựa chọn mô hình trong bảng được quyết định bằng bốn inner folds; không chọn bằng nhãn outer. AUC train của multiclass bỏ láng giềng không có trong cache cũ.','',
        '## Các đối chứng đã định trước','',pd.DataFrame(contrasts).to_markdown(index=False,floatfmt='.6f'),'',
        '## Các lỗi chọn/bỏ bài so với multiclass hiện tại','',pd.DataFrame(changes).rename(columns={'family':'configuration'}).to_markdown(index=False),'',
        'Các số ở trên là tổng trên 2.494 dự đoán ngoài fold. FN là bài nhãn >1 bị bỏ; missed_high là bài nhãn 4/5 bị bỏ. H trong bảng đầu là trung bình năm fold, không tính lại trên một danh sách gộp.','',
        '## Tách thay đổi chọn bài và xếp hạng','',parts.to_markdown(floatfmt='.6f'),'',
        'Hoán đổi gate/ranker với multiclass hiện tại rồi chia đều phần tương tác theo Shapley. Đây là phân rã thay đổi H trên dự đoán đã có, không phải bằng chứng riêng về nguyên nhân trong biểu diễn hoặc dữ liệu. Hai thành phần cộng đúng bằng delta H.','',
        '## Cột đếm trước và sau chuẩn hóa','',ranges.to_markdown(floatfmt='.6f'),'',
        'Mỗi số là tỷ lệ query ngoài khoảng min/max của train, lấy trung bình năm fold. Chỉ số này mô tả phạm vi giá trị, không đo đầy đủ distribution shift.','',
        '## Giới hạn suy luận','',
        'Phép chuẩn hóa chỉ loại ảnh hưởng của đơn vị đếm ở sáu cột support và bỏ sáu cột references. Những thay đổi về hàng xóm gần nhất, phân bố nhãn hàng xóm và số ví dụ tham chiếu thực tế vẫn còn. Vì vậy không thể dùng kết quả này để khẳng định đã giải quyết mọi chênh lệch train/predict của history features.','',
        'Khóa mặc định là thay đổi đồng thời bootstrap và số lần cập nhật leaf. Bỏ láng giềng là thay đổi cả nhóm 150 feature. Đây là tác động của từng gói sửa đổi trong cấu hình cố định, chưa tách riêng từng nguyên nhân.','',
        '## Candidate cuối','',
        f"Quy trình inner CV cuối chọn `{choices['selection_procedure']['family']}`, cutoff={choices['selection_procedure']['threshold']:.2f}, inner H={choices['selection_procedure']['inner_score']:.6f}. `outputs/submission_candidate.csv` lưu đúng lựa chọn này; việc lưu candidate không chứng minh nó tốt hơn incumbent trên dữ liệu mới.",'',
        'Giữ nguyên incumbent ở `../preference_augmentation/outputs/submission_candidate.csv`. Không upload Kaggle. Xem `outputs/verification.json` cho kết quả kiểm tra độc lập, `count_drift.csv` cho thống kê trước/sau chuẩn hóa và `gate_changes.csv` cho lỗi thay đổi.','']
    (r.HERE/'RESULTS.md').write_text('\n'.join(lines))
    print(pd.DataFrame(contrasts).to_string(index=False))
    print(pd.DataFrame(changes).to_string(index=False))


if __name__=='__main__':main()
