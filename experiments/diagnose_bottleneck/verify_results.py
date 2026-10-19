#!/usr/bin/env python3
"""Verify fold integrity, inner-only selection, and saved OOF metrics."""
import hashlib
import json

import numpy as np
import pandas as pd

from run_diagnostics import OUT, V2, baseline, configurations, evaluate, load_clean_data, priority, select_inner


def main():
    data, _ = load_clean_data()
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert protocol['input_sha256'] == baseline.input_digest(data)
    assert protocol['split_sha256'] == hashlib.sha256((V2/'nested_splits.csv').read_bytes()).hexdigest()
    original_embedding_manifest = json.loads((V2/'embedding_manifest.json').read_text())
    assert original_embedding_manifest['input_sha256'] == protocol['input_sha256']
    assert original_embedding_manifest['adapter_revision'] == protocol['encoders']['regression']
    train = data[data.split.eq('train')].reset_index(drop=True)
    labels = train.Label.astype(int).to_numpy()
    groups = baseline.grouping(train)
    oof = pd.read_csv(OUT/'oof_predictions.csv', dtype={'id':str})
    metrics = pd.read_csv(OUT/'fold_metrics.csv')
    splits = pd.read_csv(V2/'nested_splits.csv', dtype={'id':str})
    assert np.array_equal(oof.id, train.id)
    assert np.array_equal(oof.Label.astype(int), labels)
    assert len(oof) == 2494 and oof.id.is_unique and (oof.outer_fold >= 0).all()
    id_index = dict(zip(train.id, train.index))
    configs = configurations()
    count = 0
    for fold in range(5):
        split = splits[splits.outer_fold.eq(fold)]
        fit_rows = split[split.partition.eq('outer_train')]
        val_rows = split[split.partition.eq('outer_validation')]
        fit = np.array([id_index[x] for x in fit_rows.id])
        val = np.array([id_index[x] for x in val_rows.id])
        assert set(groups[fit]).isdisjoint(groups[val])
        for inner_fold in range(4):
            assert set(groups[fit[fit_rows.inner_fold.ne(inner_fold)]]).isdisjoint(
                groups[fit[fit_rows.inner_fold.eq(inner_fold)]])
        cache = np.load(OUT/f'outer_{fold}_predictions.npz')
        inner = {c['key']:cache['inner_'+c['key']] for c in configs}
        chosen, _ = select_inner(data, labels, fit, fit_rows.inner_fold.to_numpy(), inner, configs)
        for row in metrics[metrics.outer_fold.eq(fold)].to_dict('records'):
            config = chosen[row['family']]
            assert row['config'] == config['key']
            assert abs(row['threshold']-config['threshold']) < 1e-12
            ratings = cache['outer_'+config['key']]
            priorities = priority(ratings, config['threshold'])
            np.testing.assert_allclose(oof.loc[val, row['family']+'_rating'], ratings, atol=1e-12)
            np.testing.assert_allclose(oof.loc[val, row['family']+'_priority'], priorities, atol=1e-12)
            result = evaluate(labels[val], priorities, data.id.iloc[val])
            for key, value in result.items():
                assert abs(value-row[key]) < 1e-9, (fold, row['family'], key)
            count += 1
    for name in ['retrieval', 'minilm']:
        cache = np.load(OUT/f'embeddings_{name}.npz')
        manifest = json.loads((OUT/f'embeddings_{name}.json').read_text())
        assert np.array_equal(cache['keys'], (data.split+':'+data.id).to_numpy())
        assert manifest['input_sha256'] == protocol['input_sha256']
        assert manifest['revision'] == protocol['encoders'][name]
        assert np.isfinite(cache['embeddings']).all()
    verified = dict(fold_metrics_recomputed=count, original_labels_preserved=True,
                    exact_saved_splits=True, outer_and_inner_groups_disjoint=True,
                    all_choices_recomputed_from_inner_only=True,
                    embedding_keys_hashes_and_revisions_validated=True)
    (OUT/'verification.json').write_text(json.dumps(verified, indent=2)+'\n')
    print(json.dumps(verified, indent=2))


if __name__ == '__main__':
    main()
