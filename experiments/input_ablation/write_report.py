#!/usr/bin/env python3
"""Write the completed plan's evidence, model provenance and remaining failures."""
import json
import numpy as np
import pandas as pd

import run_ablation as ab
import run_metadata as meta
from finalize_workflow import OUT as FINAL


def table(frame):
    frame=frame.copy()
    for col in frame.select_dtypes(include='floating'):
        frame[col]=frame[col].map(lambda x:f'{x:.5f}')
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    lines.extend('| '+' | '.join(str(x).replace('|','\\|').replace('\n',' ') for x in row)+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(lines)


def main():
    results=json.loads((FINAL/'results.json').read_text())
    inputs=pd.read_csv(ab.OUT/'summary.csv');metadata=pd.read_csv(meta.OUT/'summary.csv')
    baseline=pd.read_csv(ab.HERE.parent/'diagnose_bottleneck/outputs/summary.csv').set_index('family').loc['regression_ridge']
    source=json.loads((ab.OUT/'metadata_audit_summary.json').read_text())
    delta=pd.read_csv(ab.OUT/'paired_deltas.csv')
    workflow=pd.read_csv(FINAL/'workflow_oof.csv',dtype={'id':str})
    old=pd.read_csv(ab.HERE.parent/'diagnose_bottleneck/outputs/oof_predictions.csv',dtype={'id':str})
    assert np.array_equal(workflow.id,old.id)
    positive=workflow.Label.gt(1);old_selected=old.regression_ridge_priority.gt(0);new_selected=workflow.priority.gt(0)
    old_error=old_selected.ne(positive);new_error=new_selected.ne(positive)
    status=lambda selected:np.select([selected&positive,selected&~positive,~selected&positive],['TP','FP','FN'],default='TN')
    workflow['old_status']=status(old_selected);workflow['new_status']=status(new_selected)
    workflow['transition']=workflow.old_status+' -> '+workflow.new_status
    workflow['change']=np.select([old_error&~new_error,~old_error&new_error,old_error&new_error],
                                 ['fixed','new_error','still_error'],default='still_correct')
    workflow.to_csv(FINAL/'failure_transitions.csv',index=False)
    counts=pd.crosstab(workflow.old_status,workflow.new_status);counts.to_csv(FINAL/'failure_transition_counts.csv')
    workflow['FP']=workflow.new_status.eq('FP');workflow['FN']=workflow.new_status.eq('FN')
    byvenue=workflow.groupby('venue').agg(n=('id','size'),FP=('FP','sum'),FN=('FN','sum'))
    byvenue.to_csv(FINAL/'errors_by_venue.csv')
    bylabel=workflow.groupby('Label').agg(n=('id','size'),FP=('FP','sum'),FN=('FN','sum'));bylabel.to_csv(FINAL/'errors_by_label.csv')
    neural_note='Không cần huấn luyện adapter vì đủ tất cả điều kiện chuyển bước đã công bố.'
    neural_table='';stability_table=''
    if results['stage_decision']['advance_to_adapter']:
        root=ab.HERE.parent/'preference_adapter/outputs'
        n=pd.read_csv(root/'summary.csv');s=pd.read_csv(root/'stability_metrics.csv')
        neural_table=table(n)
        stable=s.groupby('family').agg(H=('score','mean'),H_std=('score','std'),FP=('FP','sum'),FN=('FN','sum')).reset_index()
        stability_table=table(stable)
        neural_note='Đã chạy đủ ba đối chứng trên 5 outer folds. Lịch train cố định 3 epochs, dùng inner_fold0 làm calibration để chọn threshold; không search epoch/lr trên outer. Nhãn outer không đi vào mining hoặc optimizer. Seed thứ hai fit lại các outer train, giữ cutoff của seed chính, là kiểm tra độ nhạy khởi tạo chứ không phải một lần tune độc lập.'
    stats=results['workflow_outer_metrics']
    headline=pd.DataFrame([
        dict(method='Baseline joint SPECTER2 + Ridge',H=baseline.score_mean,Q=baseline.Q,R=baseline.R,FP=int(baseline.FP),FN=int(baseline.FN),missed_high=40),
        dict(method='Full staged selection workflow',**{k:stats[k] for k in ['H','Q','R','FP','FN','missed_high']})])
    stageplots=[]
    im=pd.read_csv(ab.OUT/'fold_metrics.csv');mm=pd.read_csv(meta.OUT/'fold_metrics.csv')
    for label,fam,frame in [('SPECTER joint','specter_joint',im),('SPECTER title','specter_title',im),
                             ('SPECTER fields','specter_separate',im),('Input policy','selection_procedure',im),
                             ('Metadata policy','selection_procedure',mm)]:
        stageplots.append((label,frame[frame.family.eq(fam)].score.to_numpy()))
    if results['stage_decision']['advance_to_adapter']:
        nm=pd.read_csv(ab.HERE.parent/'preference_adapter/outputs/fold_metrics.csv')
        for fam in ['frozen','adapter_random','adapter_hard']:stageplots.append((fam,nm[nm.family.eq(fam)].score.to_numpy()))
    stageplots.append(('Full workflow',pd.read_csv(FINAL/'workflow_fold_metrics.csv').score.to_numpy()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,6))
    for i,(label,values) in enumerate(stageplots):
        ax.scatter(values,np.full(len(values),i),alpha=.65,color='#377eb8',s=24)
        ax.scatter([values.mean()],[i],color='#e4572e',marker='D',s=52)
    ax.set_yticks(range(len(stageplots)),[x[0] for x in stageplots]);ax.invert_yaxis()
    ax.axvline(baseline.score_mean,color='gray',linestyle='--',alpha=.6)
    ax.set_xlabel('Outer H: blue individual folds, orange mean; no confidence interval')
    ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(FINAL/'workflow_comparison.png',dpi=180);plt.close(fig)
    fixed=int((old_error&~new_error).sum());introduced=int((~old_error&new_error).sum())
    report=f'''# Kết quả thực hiện kế hoạch title/abstract → metadata → adapter → model cuối

## Kết quả toàn bộ quy trình

{table(headline)}

Chênh lệch H so với baseline: **{stats['H']-baseline.score_mean:+.5f}**. Số quyết định cũ được sửa **{fixed}**, lỗi mới phát sinh **{introduced}**; tổng lỗi chọn/bỏ **{int(old_error.sum())} → {int(new_error.sum())}**. Điểm phụ thuộc mức nhãn và thứ tự nên tổng lỗi không thay thế H. Đánh giá đúng cả quy trình: mỗi outer train chọn nguồn/model bằng điểm inner của chính outer train, rồi mới tính điểm outer. Không lấy họ có outer score cao nhất làm điểm của model cuối.

Đây là 2.494 dự đoán ngoài tập fit trên cùng grouped outer folds đã dùng trước đây. Dataset/outer results đã được quan sát để thiết kế nghiên cứu nên không gọi đây là benchmark hoàn toàn mới hoặc khẳng định tăng điểm Kaggle.

Điểm metadata-only policy0.63249 và điểm full policy ở bảng trên là hai đại lượng khác nhau. Full policy còn cho phép chọn các nhánh neural bằng inner evaluation; việc mở rộng lựa chọn có thể chọn nhầm cấu hình gần điểm nhau và làm outer score giảm. Neural dùng một calibration group còn mô hình truyền thống dùng mean4inner, nên các ước lượng lựa chọn có độ nhiễu khác nhau. Điểm toàn bộ workflow đã tính cả ảnh hưởng này; không dùng score của họ tốt nhất để che phần giảm do model selection.

## 1. Title/abstract ablation

{table(inputs[['family','H','H_std','Q','R','FP','FN','missed_high','AUC']])}

Chạy đúng sáu tổ hợp đã hứa, 72 cấu hình alpha/trọng số; mỗi họ tự chọn tham số/cutoff trong 4 inner folds. Title/abstract riêng được mã hóa thành hai khối rồi L2; trọng số abstract `[.25,.5,1,2]`. Abstract thiếu/withheld thành vector0. Joint SPECTER2 vectors dùng nguyên cache v2; title-only/abstract-only encode bằng cùng base/adapter SHA. TF-IDF fit vocab riêng trong mỗi training partition.

Title-only SPECTER2 đạt H 0.63010 so với joint 0.61949, tăng ở cả 5 folds; AUC tăng khoảng 0.0285. Phép đối chứng giữ cùng alpha=0.1 cũng cho title-only H khoảng 0.63012, joint 0.61949; không chỉ là hiệu ứng mở grid alpha. Baseline joint được tái tạo khớp prediction/priority cũ tới1e-10.

Kết luận giới hạn: **cách dùng abstract trong các pipeline đang thử làm giảm khả năng dự đoán nhãn so với title-only hoặc tách khối**. Không suy ra abstract vô ích trong mọi mô hình, không suy ra người gán nhãn chỉ đọc title. Title-only giảm FP nhưng có thể giảm recall. Quy trình tự chọn input trong inner đạt 0.62929; không đánh đồng nó với 0.63010 của họ title-only.

## 2. Audit và phép thử metadata

Kiểm tra đăng ký DOI và mục lục CEUR không sử dụng Label: xác minh author cho **{source['author_verified']}/3.090** mục, có **{source['authors_changed']}** khác biệt bộ họ tác giả (56 bổ sung cho trường gốc rỗng, 183 thay đổi danh sách không rỗng), **{source['years_changed']}** thay đổi năm trong phạm vi1 năm. Có {source['year_review']} ca năm khác nhiều hơn được gắn cờ, {source['exact_identity_mismatches']} ca không đạt kiểm tra danh tính nguồn. Tất cả có ledger, nguồn và bản gốc. Chưa xác minh toàn bộ dataset: DOI API thành công {source['DOI_metadata_success']}/{source['DOI_requested']} DOI duy nhất; các mục chưa xác minh giữ metadata gốc. Metadata source/audit có thể còn thiếu hoặc chưa hoàn chỉnh.

{table(metadata[['family','H','H_std','Q','R','FP','FN','missed_high','AUC']])}

Original/clean author dùng cùng surname encoding để tránh so sánh tên đầy đủ với họ rút gọn. Vocab author, venue onehot và scaler năm đều fit trong train fold. Text input của mỗi outer fold được chọn chỉ bằng stage1 inner CV. Bộ giải Ridge metadata dùng centered linear-kernel eigensolve để tái sử dụng factorization giữa các alpha; cùng mục tiêu Ridge với intercept, đã kiểm tra đối chiếu primal Ridge. Cache partial từ bộ giải chậm được lưu riêng, không trộn vào kết quả.

Metadata sạch không tạo cải thiện tổng H rõ so với metadata gốc trong cách ghép này. Venue giúp ở một số folds nhưng chủ yếu đổi đánh đổi Q/R. Global metadata policy H 0.63249, FP323/FN346, bỏ45 bài nhãn4–5. Vì recall giảm quá0.01 và missed-high tăng, quy trình tự chuyển sang bước adapter theo gate đã công bố, dù H tăng.

## 3. Đối chứng representation học theo sở thích

{neural_note}

{neural_table}

Kiểm tra seed thứ hai (cutoff giữ cố định từ seed chính):

{stability_table}

Nếu bước adapter được chạy: ba nhánh dùng MiniLM cùng SHA, max256, cùng khởi tạo bottleneck adapter seq_bn reduction16, cùng MLP/loss. Nhánh frozen đóng băng cả adapter; hai nhánh còn lại chỉ train113.040 tham số adapter và head, giữ base cố định. Input mode/weight lấy từ stage1 inner của từng outer; do backbone/budget token khác SPECTER2 nên suy luận nhân quả về cập nhật representation chỉ áp dụng giữa ba nhánh neural có đối chứng.

Loss gồm BCE đọc/không đọc, MSE gain trên mẫu dương và pairwise softplus. Hard negatives nằm trong top5 bài nhãn1 gần nhất thuộc cùng venue nếu có, tìm chỉ trong training partition. Không lấy danh sách 280 lỗi outer đã xem để gán trọng số fit. Preflight có chạy một số inner folds thừa; trước khi xem bất kỳ neural outer score nào đã chốt một calibration group cho mỗi outer, archive các preflight thừa và không dùng để chọn model.

Đây là một lịch huấn luyện nhỏ, cố định; kết quả âm không loại trừ mọi phương án fine-tune hoặc fine-tune SPECTER2. Đọc kết quả seed để đánh giá độ nhạy, không chọn seed cao điểm hơn để báo cáo.

## 4. Chọn model cuối và submission

Chọn lại input trên 4 grouped inner folds của toàn bộ train, seed20261117; chọn metadata trên các folds đó. Nếu có neural, cutoff/nhánh được đối chiếu trên calibration group0 của cùng final inner split. Kết quả outer không chọn trực tiếp model cuối.

- Nguồn được chọn: **{results['source']}**.
- Text config: `{json.dumps(results['base'],ensure_ascii=False)}`.
- Cấu hình cuối: `{json.dumps(results['chosen'],ensure_ascii=False)}`.
- CSV đủ596 ID theo template; **{results['test_selected']}** priority>0, **{596-results['test_selected']}** priority=0.
- Artifact nằm tại [final_outputs](final_outputs), gồm model/recipe, dự đoán test, kết quả full-train inner search và OOF của toàn bộ workflow.
- Chưa submit lên Kaggle. Nhãn gốc không bị sửa; raw abstracts, baseline và model v2 được giữ để đối chiếu.

![So sánh từng fold](final_outputs/workflow_comparison.png)

## Lỗi còn lại của workflow

{table(byvenue.reset_index())}

{table(bylabel.reset_index())}

Từng ID và chuyển đổi lỗi cũ/mới: [failure_transitions.csv](final_outputs/failure_transitions.csv). Chất lượng phải được xem cùng H, recall và mức nhãn; không coi cải thiện của một vài ca đã biết là bằng chứng tổng quát hóa.

## Tái tạo và kiểm tra

```bash
python experiments/input_ablation/run_ablation.py all
python experiments/input_ablation/audit_metadata.py
python experiments/input_ablation/run_metadata.py
# Khi stage_decision yêu cầu adapter:
python experiments/preference_adapter/run_adapter.py
python experiments/input_ablation/finalize_workflow.py
python experiments/input_ablation/verify_and_summarize.py
python experiments/input_ablation/verify_metadata.py
python experiments/input_ablation/verify_final.py
python experiments/input_ablation/write_report.py
```

Môi trường: `/tmp/assignment-specter2-venv/bin/python`; requirements kế thừa `../diagnose_bottleneck/requirements.txt`. Mỗi stage có protocol, cache prediction, hash dữ liệu và nguồn. Các verification files chỉ được xuất sau khi tính lại quyết định/score thành công. `audit_metadata.py` có giới hạn thời gian lấy DOI để nguồn chưa phản hồi được ghi chưa xác minh; cache làm các lượt sau có thể tăng coverage. Một snapshot audit khác cần thư mục thí nghiệm mới vì protocol kiểm tra hash.
'''
    (ab.HERE/'RESULTS.md').write_text(report)
    (FINAL/'failure_summary.json').write_text(json.dumps(dict(fixed=fixed,introduced=introduced,
        old_errors=int(old_error.sum()),new_errors=int(new_error.sum())),indent=2)+'\n')
    print(table(headline));print('Final model',results['source'],results['chosen'])


if __name__=='__main__':main()
