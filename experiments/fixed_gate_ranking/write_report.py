#!/usr/bin/env python3
"""Write the Vietnamese report after fixed-gate ranking verification passes."""
import json
import pandas as pd
import ranking_lib as r


def main():
    verified = json.loads((r.OUT / 'verification.json').read_text())
    assert verified['status'] == 'PASSED'
    metrics = pd.read_csv(r.OUT / 'fold_metrics.csv')
    order = ['incumbent_rank', 'title_rank', 'half_rank', 'inner_selected_rank']
    summary = metrics.groupby('policy', sort=False).agg(H=('score', 'mean'), H_std=('score', 'std'), Q=('relevance', 'mean'), R=('coverage', 'mean'), FP=('FP', 'sum'), FN=('FN', 'sum'), selected=('selected', 'sum'), missed_high=('missed_high', 'sum')).reindex(order)
    summary.to_csv(r.OUT / 'summary.csv')
    paired_rows = []
    baseline = metrics[metrics.policy.eq('incumbent_rank')].set_index('outer_fold')
    for policy in order[1:]:
        current = metrics[metrics.policy.eq(policy)].set_index('outer_fold')
        delta = current.score - baseline.score
        paired_rows.append(dict(policy=policy, delta_H=float(delta.mean()), delta_Q=float((current.relevance - baseline.relevance).mean()), delta_R=float((current.coverage - baseline.coverage).mean()), improved_folds=int((delta > 0).sum()), per_fold_delta_H=json.dumps({str(int(k)): float(v) for k, v in delta.items()})))
    paired = pd.DataFrame(paired_rows); paired.to_csv(r.OUT / 'paired_effects.csv', index=False)
    ranks = pd.read_csv(r.OUT / 'rank_changes.csv', dtype={'id': str})
    rank_by_label = ranks.groupby(['policy', 'Label']).agg(n=('id', 'size'), mean_rank_change=('rank_change', 'mean'), median_rank_change=('rank_change', 'median')).reset_index()
    rank_by_label.to_csv(r.OUT / 'rank_by_label.csv', index=False)
    recipe = json.loads((r.OUT / 'final_recipe.json').read_text())
    policy = paired[paired.policy.eq('inner_selected_rank')].iloc[0]
    if policy.delta_H >= .003 and policy.improved_folds >= 4:
        decision, status = 'Promising: đề xuất candidate ranking phối hợp để cân nhắc submit sau một kiểm tra độc lập.', 'promising'
    elif policy.delta_H > 0:
        decision, status = 'Weak/mixed: candidate được lưu để tái tạo, nhưng incumbent vẫn là bản chính.', 'weak_mixed'
    else:
        decision, status = 'Negative: dừng hướng thay ranking bằng hai tín hiệu cache này; incumbent vẫn là bản chính.', 'negative'
    decisions = dict(status=status, decision=decision, policy_delta_H=float(policy.delta_H), policy_improved_folds=int(policy.improved_folds), final_recipe=recipe, incumbent_preserved=True, uploaded=False, limitation='Five repeatedly inspected grouped outer folds; exploratory only.')
    (r.OUT / 'decision.json').write_text(json.dumps(decisions, indent=2) + '\n')
    text = '# Giữ gate CatBoost, thay riêng ranking bằng title SPECTER2\n\n'
    text += '**' + decision + '**\n\n'
    text += 'Mỗi nhánh dùng đúng gate incumbent nên selected IDs, TP/FP/FN, coverage và missed-high phải giống nhau. Khác biệt H chỉ đến từ thứ tự của các bài đã chọn. Weight title được chọn bằng mean H của bốn inner folds, không chọn theo outer score. Đây là thí nghiệm cache-only, không train thêm.\n\n'
    text += '## Kết quả outer 5 folds\n\n' + summary.to_markdown(floatfmt='.5f') + '\n\n'
    text += paired[['policy','delta_H','delta_Q','delta_R','improved_folds']].to_markdown(index=False,floatfmt='.5f') + '\n\n'
    text += '`incumbent_rank` tái tạo metric incumbent nguồn. Các nhánh còn lại giữ cùng R/FP/FN/missed-high; Q và H phản ánh riêng đổi ranking. H là mean của năm fold H.\n\n'
    text += '## Final artifact\n\n```json\n' + json.dumps(recipe, indent=2) + '\n```\n\n'
    text += 'Candidate ở `outputs/submission_candidate.csv`, đủ 596 ID và không ghi đè incumbent. Không upload Kaggle. `rank_changes.csv` và `rank_by_label.csv` ghi thay đổi vị trí theo ID/nhãn để giải thích, không dùng chúng để sửa prediction.\n\n'
    text += f"Verifier **{verified['status']}**: {verified['outer_metric_rows_recomputed']} outer metric rows, {verified['inner_choices_recomputed']} inner choices, gate của {verified['gate_invariants_checked']} folds, reload CatBoost/neural final và kiểm tra 596 IDs. Outer folds đã được xem nhiều lần nên kết quả exploratory.\n"
    (r.HERE / 'RESULTS.md').write_text(text)
    (r.HERE / 'DECISION.md').write_text('# Quyết định fixed-gate ranking\n\n' + decision + '\n\n' + f"Policy delta H={policy.delta_H:+.5f}, wins={int(policy.improved_folds)}/5. Incumbent giữ nguyên; chưa upload Kaggle. Xem [RESULTS.md](RESULTS.md).\n")
    print(summary.to_string()); print(paired.to_string(index=False)); print(json.dumps(decisions, indent=2))


if __name__ == '__main__':
    main()
