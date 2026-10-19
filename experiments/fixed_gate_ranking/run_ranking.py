#!/usr/bin/env python3
"""Run the cache-only fixed-gate ranking study locked in PLAN.md."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ranking_lib as r


def audit(ctx):
    protocol = r.protocol(ctx)
    original, _ = r.title_run.ab.baseline.load_data()
    np.testing.assert_array_equal(original.Label.iloc[:2494].astype(int), ctx['y'][:2494].astype(int))
    lineage = []
    for fold, fit, pred, nums in r.outer_members(ctx):
        assert set(ctx['groups'][fit]).isdisjoint(ctx['groups'][pred])
        incumbent, adapted = r.replay_choices(ctx, fit, nums, f'outer{fold}')
        assert incumbent['epoch'] == 0 and adapted['family'] == 'adapted'
        for k in range(4):
            mask = nums == k
            a, b = fit[~mask], fit[mask]
            assert set(ctx['groups'][a]).isdisjoint(ctx['groups'][b])
            for role, field in [('incumbent_inner', 'reference'), ('title_inner', f"epoch{adapted['epoch']}")]:
                path = r.cache_path(f'outer{fold}', role, k)
                value = r.load_cache(path, a, b, field)
                lineage.append(dict(partition=f'outer{fold}_inner{k}', source=role, path=str(path.relative_to(r.ROOT.parent)),
                                    n_fit=len(a), n_pred=len(b), sha256=r.digest(path), epoch=adapted['epoch'], finite=bool(np.isfinite(value).all())))
        for role, field in [('incumbent_outer', 'reference'), ('title_outer', f"epoch{adapted['epoch']}")]:
            path = r.cache_path(f'outer{fold}', role)
            value = r.load_cache(path, fit, pred, field)
            lineage.append(dict(partition=f'outer{fold}_validation', source=role, path=str(path.relative_to(r.ROOT.parent)),
                                n_fit=len(fit), n_pred=len(pred), sha256=r.digest(path), epoch=adapted['epoch'], finite=bool(np.isfinite(value).all())))
    fit, pred, nums = r.final_partition(ctx)
    incumbent, adapted = r.replay_choices(ctx, fit, nums, 'final')
    manifest = json.loads((r.TITLE / 'neural_model' / 'manifest.json').read_text())
    assert manifest['arm'] == 'adapted' and manifest['epochs'] == [adapted['epoch']]
    for k in range(4):
        mask = nums == k
        for role, field in [('incumbent_inner', 'reference'), ('title_inner', f"epoch{adapted['epoch']}")]:
            path = r.cache_path('final', role, k)
            value = r.load_cache(path, fit[~mask], fit[mask], field)
            lineage.append(dict(partition=f'final_inner{k}', source=role, path=str(path.relative_to(r.ROOT.parent)),
                                n_fit=int((~mask).sum()), n_pred=int(mask.sum()), sha256=r.digest(path), epoch=adapted['epoch'], finite=bool(np.isfinite(value).all())))
    r.OUT.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(lineage).to_csv(r.OUT / 'lineage.csv', index=False)
    result = dict(status='PASSED', cache_records=len(lineage), outer_groups_isolated=True,
                  labels_unchanged=True, final_neural_manifest_matches=True,
                  protocol_sha256=r.digest(r.OUT / 'protocol.json'))
    (r.OUT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def inner_choices(ctx):
    choices = {}; searches = []
    for fold, fit, pred, nums in r.outer_members(ctx):
        prefix = f'outer{fold}'
        incumbent_choice, title_choice = r.replay_choices(ctx, fit, nums, prefix)
        incumbent, title, _ = r.inner_probabilities(fit, nums, prefix, title_choice['epoch'])
        selected, rows = r.choose_weight(ctx, fit, nums, incumbent, title, incumbent_choice['threshold'])
        choices[prefix] = dict(incumbent=incumbent_choice, adapted=title_choice, rank_policy=selected)
        searches.extend(dict(outer_fold=fold, incumbent_cutoff=incumbent_choice['threshold'], title_epoch=title_choice['epoch'], **row) for row in rows)
    fit, pred, nums = r.final_partition(ctx)
    incumbent_choice, title_choice = r.replay_choices(ctx, fit, nums, 'final')
    incumbent, title, _ = r.inner_probabilities(fit, nums, 'final', title_choice['epoch'])
    selected, rows = r.choose_weight(ctx, fit, nums, incumbent, title, incumbent_choice['threshold'])
    choices['final'] = dict(incumbent=incumbent_choice, adapted=title_choice, rank_policy=selected)
    searches.extend(dict(outer_fold='final', incumbent_cutoff=incumbent_choice['threshold'], title_epoch=title_choice['epoch'], **row) for row in rows)
    (r.OUT / 'outer_choices.json').write_text(json.dumps(choices, indent=2) + '\n')
    pd.DataFrame(searches).to_csv(r.OUT / 'inner_search.csv', index=False)
    return choices


def score_outer(ctx, choices):
    metrics = []; oof = []; rank_changes = []
    baseline_rows = pd.read_csv(r.TITLE / 'fold_metrics.csv')
    for fold, fit, pred, nums in r.outer_members(ctx):
        key = f'outer{fold}'; choice = choices[key]
        incumbent, title, _ = r.outer_probabilities(fit, pred, key, choice['adapted']['epoch'])
        ids = ctx['data'].id.iloc[pred].to_numpy(); y = ctx['y'][pred].astype(int)
        branches = dict(r.FIXED, inner_selected_rank=choice['rank_policy']['weight'])
        parts = {}; fold_metrics = {}
        for policy, weight in branches.items():
            table = r.rank_frame(ids, y, incumbent, title, choice['incumbent']['threshold'], weight, fold, policy)
            result = r.official(y, table.priority.to_numpy(), ids)
            row = dict(outer_fold=fold, policy=policy, weight=weight, incumbent_cutoff=choice['incumbent']['threshold'],
                                title_epoch=choice['adapted']['epoch'], inner_score=choice['rank_policy']['inner_score'] if policy == 'inner_selected_rank' else np.nan,
                                **result)
            metrics.append(row); fold_metrics[policy] = row
            oof.append(table); parts[policy] = table
        base = parts['incumbent_rank'].set_index('id')
        for policy, table in parts.items():
            if policy == 'incumbent_rank':
                continue
            now = table.set_index('id')
            assert base.selected.equals(now.selected)
            for field in ['TP', 'FP', 'FN', 'selected', 'missed_high', 'coverage']:
                assert fold_metrics[policy][field] == fold_metrics['incumbent_rank'][field]
            changed = now.loc[base.index, 'rank'] - base.loc[now.index, 'rank']
            rank_changes.append(pd.DataFrame(dict(id=base.index, outer_fold=fold, policy=policy, Label=base.Label.to_numpy(),
                                                  incumbent_rank=base['rank'].to_numpy(), new_rank=now['rank'].to_numpy(), rank_change=changed.to_numpy(),
                                                  incumbent_gain=base.incumbent_gain.to_numpy(), title_gain=now.title_gain.to_numpy())))
        expected = baseline_rows[(baseline_rows.family == 'incumbent') & (baseline_rows.outer_fold == fold)].iloc[0]
        row = fold_metrics['incumbent_rank']
        assert abs(row['score'] - expected.score) < 1e-10
        assert abs(row['relevance'] - expected.relevance) < 1e-10 and abs(row['coverage'] - expected.coverage) < 1e-12
    frame = pd.DataFrame(metrics)
    # All policies have the same gate, so every fold's selection outcomes must equal the baseline.
    for fold, group in frame.groupby('outer_fold'):
        for field in ['TP', 'FP', 'FN', 'selected', 'missed_high', 'coverage']:
            assert group[field].nunique() == 1, (fold, field)
    frame.to_csv(r.OUT / 'fold_metrics.csv', index=False)
    pd.concat(oof, ignore_index=True).to_csv(r.OUT / 'oof_predictions.csv', index=False)
    pd.concat(rank_changes, ignore_index=True).to_csv(r.OUT / 'rank_changes.csv', index=False)
    return frame


def final_candidate(ctx, choices):
    fit, pred, nums = r.final_partition(ctx); choice = choices['final']
    incumbent, title, _ = r.final_probabilities(ctx, fit, pred, choice['adapted']['epoch'])
    ids = ctx['data'].id.iloc[pred].to_numpy()
    table = r.rank_frame(ids, np.zeros(len(pred), dtype=int), incumbent, title, choice['incumbent']['threshold'], choice['rank_policy']['weight'], 'final', 'inner_selected_rank')
    details = table[['id', 'priority', 'incumbent_gain', 'title_gain']].copy()
    for k in range(5):
        details[f'incumbent_p{k+1}'] = incumbent[:, k]
        details[f'title_p{k+1}'] = title[:, k]
    details.to_csv(r.OUT / 'test_predictions.csv', index=False)
    submission = ctx['template'][['id']].merge(details[['id', 'priority']], on='id', validate='one_to_one')
    assert len(submission) == 596 and submission.id.is_unique and np.isfinite(submission.priority).all() and submission.priority.ge(0).all()
    submission.to_csv(r.OUT / 'submission_candidate.csv', index=False)
    recipe = dict(choice=choice, protocol_sha256=r.digest(r.OUT / 'protocol.json'), selected=int((submission.priority > 0).sum()), uploaded=False)
    (r.OUT / 'final_recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
    return recipe


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['audit', 'choices', 'outer', 'final', 'all'])
    stage = parser.parse_args().stage
    ctx = r.frame_context()
    if stage in ['audit', 'all']:
        print(json.dumps(audit(ctx), indent=2), flush=True)
    if stage in ['choices', 'outer', 'final', 'all']:
        r.protocol(ctx)
        choices = inner_choices(ctx)
    if stage in ['outer', 'all']:
        summary = score_outer(ctx, choices)
        print(summary.groupby('policy').score.mean().to_string(), flush=True)
    if stage in ['final', 'all']:
        print(json.dumps(final_candidate(ctx, choices), indent=2), flush=True)


if __name__ == '__main__':
    main()
