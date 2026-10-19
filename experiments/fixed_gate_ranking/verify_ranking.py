#!/usr/bin/env python3
"""Independent replay for fixed-gate ranking outputs."""
import json

from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd

import ranking_lib as r


def synthetic_metric_check():
    y = np.array([5, 1, 3]); ids = np.array(['b', 'a', 'c'])
    value = r.official(y, np.array([9., 8., 7.]), ids)
    w = 1 / np.log2(np.arange(3) + 2)
    expected_q = (w[0] + w[2] * .5) / w.sum()
    assert abs(value['relevance'] - expected_q) < 1e-15
    assert value['coverage'] == 1 and value['TP'] == 2 and value['FP'] == 1
    tie = r.official(np.array([5, 1]), np.array([1., 1.]), np.array(['20', '3']))
    assert tie['relevance'] == 1 / (1 + 1 / np.log2(3))


def main():
    synthetic_metric_check()
    ctx = r.frame_context(); r.protocol(ctx)
    choices = json.loads((r.OUT / 'outer_choices.json').read_text())
    frame = pd.read_csv(r.OUT / 'fold_metrics.csv')
    oof = pd.read_csv(r.OUT / 'oof_predictions.csv', dtype={'id': str})
    assert len(frame) == 20 and len(oof) == 2494 * 4
    outer_metrics = choices_checked = invariant_checks = 0
    original = pd.read_csv(r.TITLE / 'fold_metrics.csv')
    for fold, fit, pred, nums in r.outer_members(ctx):
        key = f'outer{fold}'; saved = choices[key]
        incumbent_choice, adapted_choice = r.replay_choices(ctx, fit, nums, key)
        assert saved['incumbent'] == incumbent_choice and saved['adapted'] == adapted_choice
        inc_inner, title_inner, _ = r.inner_probabilities(fit, nums, key, adapted_choice['epoch'])
        selected, _ = r.choose_weight(ctx, fit, nums, inc_inner, title_inner, incumbent_choice['threshold'])
        assert saved['rank_policy'] == selected; choices_checked += 3
        inc, title, _ = r.outer_probabilities(fit, pred, key, adapted_choice['epoch'])
        ids = ctx['data'].id.iloc[pred].to_numpy(); y = ctx['y'][pred].astype(int)
        expected_selected = 1 - inc[:, 0] > incumbent_choice['threshold']
        for policy, weight in {**r.FIXED, 'inner_selected_rank': selected['weight']}.items():
            priority = r.priority(inc, title, incumbent_choice['threshold'], weight)
            part = oof[(oof.outer_fold == fold) & (oof.policy == policy)].reset_index(drop=True)
            assert part.id.tolist() == ids.tolist()
            np.testing.assert_allclose(part.priority, priority, atol=1e-12)
            np.testing.assert_array_equal(part.selected.to_numpy(bool), expected_selected)
            computed = r.official(y, priority, ids)
            row = frame[(frame.outer_fold == fold) & (frame.policy == policy)].iloc[0]
            for field, value in computed.items():
                assert abs(row[field] - value) < 1e-12, (fold, policy, field)
            outer_metrics += 1
        base = frame[(frame.outer_fold == fold) & (frame.policy == 'incumbent_rank')].iloc[0]
        old = original[(original.outer_fold == fold) & (original.family == 'incumbent')].iloc[0]
        for field in ['score', 'relevance', 'coverage', 'TP', 'FP', 'FN', 'selected', 'missed_high']:
            assert abs(base[field] - old[field]) < 1e-10
        same = frame[frame.outer_fold == fold]
        for field in ['TP', 'FP', 'FN', 'selected', 'missed_high', 'coverage']:
            assert same[field].nunique() == 1
        invariant_checks += 1
    recipe = json.loads((r.OUT / 'final_recipe.json').read_text())
    fit, pred, nums = r.final_partition(ctx); final = choices['final']
    assert recipe['choice'] == final and recipe['protocol_sha256'] == r.digest(r.OUT / 'protocol.json')
    inc_inner, title_inner, _ = r.inner_probabilities(fit, nums, 'final', final['adapted']['epoch'])
    selected, _ = r.choose_weight(ctx, fit, nums, inc_inner, title_inner, final['incumbent']['threshold'])
    assert selected == final['rank_policy']; choices_checked += 1
    inc, title, _ = r.final_probabilities(ctx, fit, pred, final['adapted']['epoch'])
    pack = joblib.load(r.INC / 'final_test_features.joblib')
    source_recipe = joblib.load(r.INC / 'final_recipe.joblib')
    model = CatBoostClassifier(); model.load_model(str(r.INC / 'final_classifier.cbm'))
    rebuilt = model.predict_proba(pack['b'][source_recipe['columns']])
    np.testing.assert_allclose(rebuilt, inc, atol=1e-10, rtol=1e-10)
    r.title_model.prepare(ctx, r.TITLE)
    neural = r.title_model.reload_predict(ctx, pred, r.TITLE / 'neural_model')
    np.testing.assert_allclose(neural, title, atol=1e-7, rtol=1e-7)
    priority = r.priority(inc, title, final['incumbent']['threshold'], final['rank_policy']['weight'])
    submission = pd.read_csv(r.OUT / 'submission_candidate.csv', dtype={'id': str})
    expected = ctx['template'][['id']].merge(pd.DataFrame({'id': ctx['data'].id.iloc[pred], 'priority': priority}), on='id', validate='one_to_one')
    pd.testing.assert_frame_equal(submission, expected)
    assert len(submission) == 596 and submission.id.is_unique and np.isfinite(submission.priority).all() and submission.priority.ge(0).all()
    incumbent_sha = '8161a8d0c3ce0245bfef0dc069577caa6b0e0cf961ebef5fc1cbbdce2dbcc0cf'
    assert r.digest(r.INC / 'submission_candidate.csv') == incumbent_sha
    result = dict(status='PASSED', synthetic_metric_cases=2, outer_metric_rows_recomputed=outer_metrics,
                  inner_choices_recomputed=choices_checked, gate_invariants_checked=invariant_checks,
                  catboost_final_reloaded=True, neural_final_reloaded=True, final_submission_rows=596,
                  incumbent_unchanged=True, uploaded=False)
    (r.OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
