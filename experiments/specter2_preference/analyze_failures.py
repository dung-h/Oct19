#!/usr/bin/env python3
"""Post-hoc OOF error accounting only; never consumed by training or selection."""
import json
import numpy as np
import pandas as pd

import run_preference as run


def main():
    d=pd.read_csv(run.OUT/'oof_predictions.csv',dtype={'id':str})
    summaries=[];transitions=[];groups=[]
    for arm,reference in [('adapted','frozen'),('adapted','incumbent'),('selection_procedure','incumbent')]:
        truth=d.Label.gt(1)
        old=d[reference+'_priority'].gt(0);new=d[arm+'_priority'].gt(0)
        old_wrong=old.ne(truth);new_wrong=new.ne(truth)
        fixed=old_wrong & ~new_wrong;introduced=~old_wrong & new_wrong
        summaries.append(dict(arm=arm,reference=reference,fixed=int(fixed.sum()),introduced=int(introduced.sum()),
                              net_errors_reduced=int(old_wrong.sum()-new_wrong.sum()),
                              errors_before=int(old_wrong.sum()),errors_after=int(new_wrong.sum())))
        part=d[['id','title','venue','Label','outer_fold']].copy()
        part['arm']=arm;part['reference']=reference
        part['old_status']=np.where(old,np.where(truth,'TP','FP'),np.where(truth,'FN','TN'))
        part['new_status']=np.where(new,np.where(truth,'TP','FP'),np.where(truth,'FN','TN'))
        part['fixed']=fixed;part['introduced']=introduced
        part['old_priority']=d[reference+'_priority'];part['new_priority']=d[arm+'_priority']
        transitions.append(part)
    for arm in ['incumbent','frozen','adapted','neural_selection','selection_procedure']:
        chosen=d[arm+'_priority'].gt(0);truth=d.Label.gt(1)
        for key in ['venue','Label']:
            for value,ix in d.groupby(key).groups.items():
                selected=chosen.loc[ix];positive=truth.loc[ix]
                groups.append(dict(family=arm,grouping=key,group=str(value),n=len(ix),
                    FP=int((selected&~positive).sum()),FN=int((~selected&positive).sum()),
                    TP=int((selected&positive).sum()),TN=int((~selected&~positive).sum())))
    pd.concat(transitions,ignore_index=True).to_csv(run.OUT/'failure_transitions.csv',index=False)
    pd.DataFrame(groups).to_csv(run.OUT/'failure_breakdown.csv',index=False)
    (run.OUT/'failure_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    inner=pd.read_csv(run.OUT/'inner_search.csv')
    profile=inner.sort_values('inner_score',ascending=False).drop_duplicates(['outer_fold','family','epoch'])
    profile.sort_values(['outer_fold','family','epoch']).to_csv(run.OUT/'inner_epoch_profile.csv',index=False)
    epoch_means=profile.groupby(['family','epoch']).agg(mean_inner_H=('inner_score','mean'),mean_inner_R=('inner_recall','mean')).reset_index()
    known=d[d.id.isin(['206','217','355','1000','1001','1045','1184','2383','3016','2758'])]
    known.to_csv(run.OUT/'known_cases.csv',index=False)
    summary=pd.read_csv(run.OUT/'summary.csv').set_index('family')
    delta=summary.loc['adapted','H']-summary.loc['incumbent','H']
    counts=pd.DataFrame(summaries)
    choices=[json.loads((run.OUT/f'outer{k}_choices.json').read_text())['primary']['adapted'] for k in range(5)]
    boundary=sum(x['epoch']==8 for x in choices)
    text=f'''# Chẩn đoán sau thí nghiệm

Adapted so với incumbent cố định: delta H **{delta:+.5f}**. Toàn policy chọn trong
inner có H **{summary.loc['selection_procedure','H']:.5f}**; không coi family thắng
outer là điểm của policy.

## Lỗi đọc/không đọc

{counts.to_markdown(index=False)}

Đây là thay đổi quyết định nhị phân, không thay thế H (còn phụ thuộc nhãn2..5 và
thứ tự). Chi tiết từng ID ở `failure_transitions.csv`, theo venue/label ở
`failure_breakdown.csv`; các trường hợp đã bàn trước ở `known_cases.csv`.

## Giới hạn lịch huấn luyện

Inner CV chọn checkpoint cuối8epochs cho adapted ở **{boundary}/5 outer folds**.
Bảng dưới chọn cutoff tốt nhất trong inner cho từng epoch, rồi lấy trung bình
qua outer training partitions. Đây là dữ liệu chẩn đoán có overlap, không là
validation độc lập hay lý do thay checkpoint theo outer.

{epoch_means.to_markdown(index=False,floatfmt='.5f')}

Nếu checkpoint cuối được chọn thường xuyên thì kết quả chỉ kiểm tra lịch1/2/4/8;
không chứng minh encoder đã hội tụ hoặc training lâu hơn vô ích. Đồng thời không
có cơ sở ngoại suy rằng tăng epochs sẽ tự đạt0.7. Chưa thử full transformer fine-tune.

So sánh FP ở ngưỡng inner recall gần0.8 phải xem cùng outer recall thực tế trong
`recall80_summary.csv`; hai nhánh không bị ép có cùng outer recall. Không gán chênh
lệch H hoàn toàn cho việc giảm FP.
'''
    (run.HERE/'DIAGNOSIS.md').write_text(text)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
