#!/usr/bin/env python3
"""Post-hoc ranking/headroom diagnostic; never selects a final model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'abstract_preference_margin' / 'outputs'
OUT = HERE / 'outputs'
ARMS = ['reference', 'retrieval_margin', 'title_margin', 'abstract_margin']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(y, selected, rank_score, ids):
    order = np.array(sorted(np.flatnonzero(selected),
                            key=lambda i: (-float(rank_score[i]), str(ids[i]))), dtype=int)
    gain = (y[order] - 1) / 4
    weight = 1 / np.log2(np.arange(len(order)) + 2)
    q = float(np.dot(gain, weight) / weight.sum()) if len(order) else 0.0
    r = float(np.sum(selected & (y > 1)) / np.sum(y > 1))
    h = float(2 * q * r / (q + r)) if q + r else 0.0
    return dict(H=h, Q=q, R=r, accepted=int(selected.sum()),
                FP=int(np.sum(selected & (y == 1))), FN=int(np.sum(~selected & (y > 1)))), order


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_files = ['oof_predictions.csv', 'fold_metrics.csv', 'verification.json', 'protocol.json']
    inputs = {name: digest(SOURCE / name) for name in source_files}
    assert json.loads((SOURCE / 'verification.json').read_text())['status'] == 'PASSED'
    manifest = dict(status='post_hoc_diagnostic', source_sha256=inputs,
                    script_sha256=digest(Path(__file__).resolve()), arms=ARMS,
                    fixed_selection='reference selected at strict read probability >0.55',
                    rank_score='sum(p_k * (k-1)/4), k=1..5; tie by ascending string ID',
                    oracle_policies='Use true labels for headroom only; not deployable predictions',
                    training_fits=0, final_model_selection=False, uploaded=False)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    data = pd.read_csv(SOURCE / 'oof_predictions.csv', dtype={'id': str})
    old_metrics = pd.read_csv(SOURCE / 'fold_metrics.csv')
    assert len(data) == 2494 * 4
    assert set(data.arm) == set(ARMS)
    rows, movements = [], []
    for fold in sorted(data.outer_fold.unique()):
        partition = data[data.outer_fold == fold]
        base = partition[partition.arm == 'reference'].set_index('id').sort_index()
        assert base.index.is_unique
        ids = base.index.to_numpy()
        y = base.Label.to_numpy(int)
        selected = base.selected.to_numpy(bool)
        assert np.array_equal(selected, base.read_probability.to_numpy() > 0.55)
        old, base_order = score(y, selected, base.priority.to_numpy(), ids)
        expected = old_metrics[(old_metrics.arm == 'reference') & (old_metrics.outer_fold == fold)].iloc[0]
        for field in ['H', 'Q', 'R', 'accepted', 'FP', 'FN']:
            assert np.isclose(old[field], expected[field], rtol=0, atol=1e-12)
        base_rank = np.full(len(base), np.nan)
        base_rank[base_order] = np.arange(1, len(base_order) + 1)
        for arm in ARMS:
            model = partition[partition.arm == arm].set_index('id').loc[ids]
            assert np.array_equal(model.Label.to_numpy(), y)
            p = model[[f'p{k}' for k in range(1, 6)]].to_numpy()
            assert np.isfinite(p).all() and (p >= 0).all()
            np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-9)
            rank_score = p @ (np.arange(5) / 4)
            metrics, order = score(y, selected, rank_score, ids)
            # Independent pandas order and arithmetic validate the score implementation.
            table = pd.DataFrame({'id': ids, 'y': y, 'score': rank_score})[selected]
            table = table.sort_values(['score', 'id'], ascending=[False, True])
            assert table.id.tolist() == ids[order].tolist()
            independent_weight = np.array([1 / np.log2(r + 1) for r in range(1, len(table) + 1)])
            independent_q = sum(independent_weight * (table.y.to_numpy() - 1) / 4) / sum(independent_weight)
            assert abs(independent_q - metrics['Q']) < 1e-12
            if arm == 'reference':
                assert np.array_equal(order, base_order)
            rows.append(dict(outer_fold=int(fold), policy=arm, kind='prediction',
                             delta_H=metrics['H'] - old['H'], delta_Q=metrics['Q'] - old['Q'], **metrics))
            for field in ['R', 'accepted', 'FP', 'FN']:
                assert metrics[field] == old[field]
            for rank, i in enumerate(order, 1):
                movements.append(dict(outer_fold=int(fold), policy=arm, id=ids[i], Label=int(y[i]),
                                      title=base.title.iloc[i], venue=base.venue.iloc[i],
                                      incumbent_rank=int(base_rank[i]), new_rank=rank,
                                      movement=int(base_rank[i]) - rank))

        # Label-based counterfactuals are explanatory bounds, never candidate predictions.
        oracles = {
            'oracle_order_fixed_selection': (selected, y),
            'oracle_remove_selected_FP_existing_order': (selected & (y > 1), base.priority.to_numpy()),
            'oracle_positive_selection_existing_order': (y > 1, base[[f'p{k}' for k in range(1, 6)]].to_numpy() @ (np.arange(5) / 4)),
        }
        for name, (mask, values) in oracles.items():
            metrics, _ = score(y, mask, values, ids)
            rows.append(dict(outer_fold=int(fold), policy=name, kind='oracle',
                             delta_H=metrics['H'] - old['H'], delta_Q=metrics['Q'] - old['Q'], **metrics))
    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / 'fold_metrics.csv', index=False)
    summary = folds.groupby(['policy', 'kind'], sort=False).agg(
        H=('H', 'mean'), Q=('Q', 'mean'), R=('R', 'mean'), delta_H=('delta_H', 'mean'),
        improved_folds=('delta_H', lambda x: int((x > 1e-12).sum())),
        accepted=('accepted', 'sum'), FP=('FP', 'sum'), FN=('FN', 'sum'))
    summary.to_csv(OUT / 'summary.csv')
    pd.DataFrame(movements).to_csv(OUT / 'rank_changes.csv', index=False)
    for name, value in inputs.items():
        assert digest(SOURCE / name) == value, name
    validation = dict(status='PASSED', fold_policy_rows=len(folds),
                      reference_metric_replays=5, fixed_selection_checks=20,
                      independent_order_metric_checks=20, input_hashes_unchanged=True,
                      oracle_rows_are_not_predictions=True, uploaded=False)
    (OUT / 'verification.json').write_text(json.dumps(validation, indent=2) + '\n')
    report = ['# Fixed-selection rank diagnostic for abstract preference margin', '',
              'Post-hoc exploratory diagnostic on already examined OOF rows. No new fit, tuning, final-model selection, or Kaggle upload.', '',
              'All prediction arms use exactly the incumbent selected set and their own expected rating gain for order. '
              'Counts are summed; H/Q/R are five-fold means. Oracles use labels and cannot be deployed.', '',
              summary.reset_index().to_markdown(index=False, floatfmt='.6f'), '',
              'Oracle order is the maximum possible Q/H from reordering the current selected set. '
              'Oracle FP removal keeps the current true positives and order while eliminating known rating-1 items. '
              'Oracle positive selection includes all known rating>1 items with incumbent gain for order.', '',
              'Reference metrics reproduced; selection invariants, independent ranking/metric calculations and source hashes passed.']
    (HERE / 'RESULTS.md').write_text('\n'.join(report) + '\n')
    print(summary.to_string(float_format=lambda x: f'{x:.6f}'))
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
