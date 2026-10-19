#!/usr/bin/env python3
"""Read-only model diagnosis: export existing train FN texts and fit-fold neighbors.

No model is fitted and no prediction or preference label is modified. Neighbor
labels are used only to describe existing errors, never as new model features.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'outputs'
sys.path.insert(0, str(ROOT / 'read_gate_repair'))
import gate_lib as g


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = g.context()
    data = ctx['data'].reset_index(drop=True)
    oof_path = g.TITLE / 'oof_predictions.csv'
    oof = pd.read_csv(oof_path, dtype={'id': str})
    assert oof.id.tolist() == data.id.iloc[:2494].tolist()
    records = data.iloc[:2494][['split', 'id', 'title', 'abstract', 'venue', 'year']].copy()
    records['row_index'] = np.arange(2494)
    records['Label'] = ctx['y'][:2494].astype(int)
    records['outer_fold'] = oof.outer_fold.to_numpy(int)
    records['p_read'] = 1 - oof.incumbent_p1.to_numpy(float)
    records['selected'] = oof.incumbent_priority.gt(0).to_numpy()
    records['title_selected'] = oof.adapted_priority.gt(0).to_numpy()
    records['has_abstract'] = records.abstract.str.strip().ne('')
    records['outcome'] = np.select(
        [records.selected & records.Label.gt(1), records.selected & records.Label.eq(1),
         ~records.selected & records.Label.gt(1)], ['TP', 'FP', 'FN'], default='TN')
    new = pd.read_csv(g.OUT / 'outcomes_by_id.csv', dtype={'id': str})
    new = new[new.branch.eq('author_venue')].set_index('id').loc[records.id]
    records['author_venue_selected'] = new.selected.to_numpy(bool)
    newly_lost = records[records.Label.gt(1) & records.selected & ~records.author_venue_selected]
    assert len(newly_lost) == 15
    newly_lost.to_csv(OUT / 'newly_lost_true_positives.csv', index=False)
    fn = records[records.outcome.eq('FN')].copy()
    assert len(fn) == 313 and len(fn[fn.Label.ge(4)]) == 38
    fn.to_csv(OUT / 'false_negatives.csv', index=False)
    fn[fn.Label.ge(4)].to_csv(OUT / 'high_false_negatives.csv', index=False)

    # These are descriptive, deterministic case selections, not a validation set.
    sample_indices = set(fn[fn.Label.ge(4)].row_index)
    for _, part in fn[fn.Label.lt(4)].groupby('venue'):
        sorted_part = part.sort_values(['p_read', 'id'])
        for k in [0, len(part) // 2, len(part) - 1]:
            sample_indices.add(int(sorted_part.iloc[k].row_index))
    reviewed = fn[fn.row_index.isin(sample_indices)].sort_values(['venue', 'Label', 'p_read'])
    reviewed.to_csv(OUT / 'review_sample.csv', index=False)

    embed_path = ROOT / 'diagnose_bottleneck' / 'outputs' / 'embeddings_retrieval.npz'
    z = np.load(embed_path)
    keys = (data.split + ':' + data.id).to_numpy(str)
    np.testing.assert_array_equal(z['keys'], keys)
    vectors = z['embeddings'].astype(np.float64)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    neighbors = []
    neighborhood_status = []
    for fold, fit, pred, _ in g.outer_members(ctx):
        positive_queries = pred[ctx['y'][pred] > 1]
        for q in positive_queries:
            refs = fit[(ctx['groups'][fit] != ctx['groups'][q]) &
                       (data.venue.iloc[fit].to_numpy() == data.venue.iloc[q])]
            sims = vectors[refs] @ vectors[q]
            positive = ctx['y'][refs] > 1
            best_pos = float(sims[positive].max())
            best_neg = float(sims[~positive].max())
            neighborhood_status.append(dict(id=data.id.iloc[q], Label=int(ctx['y'][q]),
                outcome=records.outcome.iloc[q], outer_fold=int(fold),
                nearest_label1_similarity=best_neg, nearest_positive_similarity=best_pos,
                nearest_label1_closer=bool(best_neg > best_pos)))
        query_rows = fn[fn.outer_fold.eq(fold)]
        assert set(query_rows.row_index).issubset(set(pred))
        for _, query in query_rows.iterrows():
            q = int(query.row_index)
            refs = fit[ctx['groups'][fit] != ctx['groups'][q]]
            sim = vectors[refs] @ vectors[q]
            for scope in ['same_venue', 'all_venues']:
                scope_mask = (data.venue.iloc[refs].to_numpy() == query.venue) if scope == 'same_venue' else np.ones(len(refs), bool)
                for polarity in ['label1', 'label2plus']:
                    label_mask = ctx['y'][refs] == 1 if polarity == 'label1' else ctx['y'][refs] > 1
                    eligible = np.flatnonzero(scope_mask & label_mask)
                    ordered = eligible[np.argsort(-sim[eligible], kind='stable')][:3]
                    for rank, pos in enumerate(ordered, 1):
                        ref = int(refs[pos]); r = data.iloc[ref]
                        neighbors.append(dict(query_id=query.id, query_label=int(query.Label),
                            outer_fold=int(fold), scope=scope, polarity=polarity, rank=rank,
                            cosine_similarity=float(sim[pos]), neighbor_id=r.id,
                            neighbor_label=int(ctx['y'][ref]), neighbor_venue=r.venue,
                            neighbor_title=r.title, neighbor_abstract=r.abstract))
    nn = pd.DataFrame(neighbors)
    nn.to_csv(OUT / 'neighbors.csv', index=False)
    ns = pd.DataFrame(neighborhood_status)
    ns.to_csv(OUT / 'positive_neighborhoods.csv', index=False)
    summary = []
    for key, part in records.groupby(['outcome', 'Label', 'venue']):
        summary.append(dict(outcome=key[0], Label=int(key[1]), venue=key[2], n=len(part),
            missing_abstract=int((~part.has_abstract).sum()),
            mean_p_read=float(part.p_read.mean())))
    pd.DataFrame(summary).to_csv(OUT / 'counts.csv', index=False)
    near = nn[nn.scope.eq('same_venue') & nn['rank'].eq(1)].pivot(
        index='query_id', columns='polarity', values='cosine_similarity')
    counts = dict(
        incumbent_fn=len(fn), incumbent_high_fn=int(fn.Label.ge(4).sum()),
        fn_missing_abstract=int((~fn.has_abstract).sum()),
        high_fn_missing_abstract=int((~fn[fn.Label.ge(4)].has_abstract).sum()),
        fn_below_045=int(fn.p_read.lt(.45).sum()),
        fn_from_050_to_055=int(fn.p_read.ge(.50).sum()),
        fn_title_also_drops=int((~fn.title_selected).sum()),
        high_fn_title_also_drops=int((~fn[fn.Label.ge(4)].title_selected).sum()),
        fn_nearest_label1_closer=int(near.label1.gt(near.label2plus).sum()),
        tp_nearest_label1_closer=int(ns[ns.outcome.eq('TP')].nearest_label1_closer.sum()),
        tp_total=int(ns.outcome.eq('TP').sum()),
        newly_lost_tp=len(newly_lost),
        newly_lost_high_tp=int(newly_lost.Label.ge(4).sum()),
        reviewed_high=int(reviewed.Label.ge(4).sum()),
        reviewed_lower=int(reviewed.Label.lt(4).sum()),
    )
    (OUT / 'summary.json').write_text(json.dumps(counts, indent=2) + '\n')
    manifest = dict(source_sha256=sha(Path(__file__)),
        oof_sha256=sha(oof_path), embedding_sha256=sha(embed_path),
        role='Diagnostic existing training errors; not a new validation benchmark.',
        neighbor_references='Respective outer fit only; entire query duplicate group excluded.',
        selection='All 38 label4/5 FN, plus low/median/high probability label2/3 FN per venue, deduplicated.',
        model_fits=0, prediction_changes=0, test_labeling=False)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    packet = '# Existing false negatives: source texts and fit-fold counterexamples\n\n'
    packet += 'Manual content review, not new preference labels. Cosine proximity is topical evidence, not a causal attribution.\n\n'
    for _, row in reviewed.iterrows():
        packet += f'## ID {row.id}; Label {row.Label}; venue {row.venue}; P(read) {row.p_read:.4f}\n\n'
        packet += f'{row.title}\n\nAbstract: {row.abstract or "[MISSING]"}\n\n'
        for polarity in ['label1', 'label2plus']:
            part = nn[(nn.query_id == row.id) & nn.scope.eq('same_venue') & nn.polarity.eq(polarity) & nn['rank'].eq(1)]
            for _, ref in part.iterrows():
                packet += f'### Nearest {polarity}: ID {ref.neighbor_id}; Label {ref.neighbor_label}; cosine {ref.cosine_similarity:.3f}\n\n'
                packet += f'{ref.neighbor_title}\n\n{ref.neighbor_abstract or "[MISSING]"}\n\n'
    (OUT / 'REVIEW_PACKET.md').write_text(packet)
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(4):
        main()
