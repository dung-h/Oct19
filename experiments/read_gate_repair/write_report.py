#!/usr/bin/env python3
"""Write count-first report after verification."""
from __future__ import annotations
import json
import pandas as pd
import gate_lib as g


def main():
    verified=json.loads((g.OUT/'verification.json').read_text());assert verified['status']=='PASSED'
    summary=pd.read_csv(g.OUT/'summary.csv').set_index('branch')
    transitions=pd.read_csv(g.OUT/'transition_summary.csv').set_index('branch')
    folds=pd.read_csv(g.OUT/'fold_metrics.csv')
    primary=summary.loc['author_venue']; base=summary.loc['incumbent_deployed'];delta=transitions.loc['author_venue']
    per_fold=folds.pivot(index='outer_fold',columns='branch',values='total_error')
    lowered=int((per_fold.author_venue < per_fold.incumbent_deployed).sum())
    promotion=(primary.FP<=base.FP and primary.FN<=base.FN and (primary.FP<base.FP or primary.FN<base.FN)
               and primary.missed_high<=base.missed_high and lowered>=4)
    if promotion: status='promising';decision='Primary `author_venue` giảm lỗi chọn/bỏ theo rule đã khóa; có thể cân nhắc làm candidate experimental.'
    elif primary.total_error < base.total_error: status='tradeoff_or_mixed';decision='Primary giảm tổng lỗi nhưng không đạt đồng thời toàn bộ điều kiện FP/FN/fold đã khóa; giữ incumbent làm bản chính.'
    else: status='negative';decision='Primary không giảm tổng lỗi theo tiêu chí đã khóa; giữ incumbent làm bản chính và dừng hướng author/calibration này.'
    (g.OUT/'decision.json').write_text(json.dumps(dict(status=status,decision=decision,primary='author_venue',outer_folds_with_lower_total_error=lowered,uploaded=False),indent=2)+'\n')
    selected=pd.read_csv(g.OUT/'test_counts.csv').set_index('branch')
    report='# Sửa read gate bằng binary target, author history và calibration\n\n'
    report+=decision+'\n\n'
    report+='## Kết quả OOF: số mẫu nhận/rớt và lỗi\n\n'+summary[['accepted','dropped','TP','FP','FN','TN','total_error','missed_high','H','Q','R']].to_markdown(floatfmt='.5f')+'\n\n'
    report+='`author_venue` so với incumbent: cứu '+str(int(delta.FN_rescued))+' FN, bỏ '+str(int(delta.FP_removed))+' FP, tạo '+str(int(delta.new_FP))+' FP mới và làm mất '+str(int(delta.lost_TP))+' TP. Delta tổng lỗi='+str(int(delta.delta_error))+'.\n\n'
    report+='Tổng OOF `author_venue`: FP='+str(int(primary.FP))+', FN='+str(int(primary.FN))+', high-label FN='+str(int(primary.missed_high))+'. Incumbent: FP='+str(int(base.FP))+', FN='+str(int(base.FN))+', high-label FN='+str(int(base.missed_high))+'. Folds giảm total error: '+str(lowered)+'/5.\n\n'
    report+='## Test artifact\n\n'+selected.to_markdown()+'\n\n'
    report+='Test không có nhãn nên bảng này chỉ là accepted/dropped, không phải TP/FP/FN. `submission_author_venue.csv` là candidate experimental; incumbent cũ không bị ghi đè và không upload Kaggle.\n\n'
    report+='Verifier **PASSED**: replay toàn bộ decision outer, rebuild author features final và reload 2 CatBoost + calibrators cho 8 branch. Outer folds đã được xem nhiều lần nên đây là kết quả exploratory.\n'
    (g.HERE/'RESULTS.md').write_text(report)
    (g.HERE/'DECISION.md').write_text('# Quyết định read-gate repair\n\n'+decision+'\n\nXem [RESULTS.md](RESULTS.md).\n')
    (g.HERE/'README.md').write_text('# Read-gate repair\n\nChạy `python tests/test_gate.py`, `python run_gate.py all`, `python verify_gate.py`, rồi `python write_report.py`. Recipe khóa ở `PLAN.md`; không có threshold hay hyperparameter search.\n')
    print(decision)


if __name__=='__main__':main()
