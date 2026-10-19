"""Shared, cache-only utilities for the fixed-incumbent-gate ranking study."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TITLE = ROOT / 'specter2_preference' / 'outputs'
CLAIM = ROOT / 'claim_stability' / 'outputs'
INC = ROOT / 'preference_augmentation' / 'outputs'
OUT = HERE / 'outputs'

sys.path.insert(0, str(ROOT / 'specter2_preference'))
import modeling as title_model
import run_preference as title_run

WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
FIXED = {'incumbent_rank': 0.0, 'half_rank': 0.5, 'title_rank': 1.0}
TOL = 1e-12
GAIN = np.arange(5, dtype=float) / 4


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probability_check(p, n):
    assert p.shape == (n, 5)
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)


def official(y, priority, ids):
    order = np.asarray(sorted(np.flatnonzero(priority > 0), key=lambda i: (-priority[i], str(ids[i]))), dtype=int)
    weights = 1 / np.log2(np.arange(len(order)) + 2)
    q = float(np.average((y[order] - 1) / 4, weights=weights)) if len(order) else 0.0
    tp = int((y[order] > 1).sum())
    positives = int((y > 1).sum())
    r = tp / positives if positives else 0.0
    return dict(score=2 * q * r / (q + r) if q + r else 0.0, relevance=q, coverage=r,
                TP=tp, FP=int(len(order) - tp), FN=int(positives - tp), selected=int(len(order)),
                missed_high=int(((y >= 4) & (priority == 0)).sum()))


def expected_gain(probability):
    return probability @ GAIN


def priority(incumbent, rank_probability, cutoff, weight):
    selected = 1 - incumbent[:, 0] > cutoff
    rank = (1 - weight) * expected_gain(incumbent) + weight * expected_gain(rank_probability)
    # Positive affine score preserves rank and keeps every selected paper selected.
    return np.where(selected, 1 + 9 * rank, 0.0)


def frame_context():
    ctx = title_run.context()
    data, y, groups = ctx['data'], ctx['y'], ctx['groups']
    assert len(data) == 3090 and len(y) == 3090
    assert np.isnan(y[2494:]).all()
    return ctx


def outer_members(ctx):
    return list(title_run.outer_members(ctx))


def final_partition(ctx):
    fit = np.arange(2494)
    pred = np.arange(2494, len(ctx['data']))
    memberships = pd.read_csv(INC / 'final_inner_splits.csv', dtype={'id': str})
    assert memberships.id.tolist() == ctx['data'].id.iloc[fit].tolist()
    return fit, pred, memberships.inner_fold.to_numpy()


def saved_source_choices(prefix):
    source = json.loads((TITLE / f'{prefix}_choices.json').read_text())
    return source['primary'] if prefix.startswith('outer') else source


def cache_path(prefix, role, fold=None):
    if role == 'incumbent_inner':
        return CLAIM / f'{prefix}_inner{fold}_predictions.npz'
    if role == 'title_inner':
        return TITLE / f'{prefix}_inner{fold}_adapted.npz'
    if role == 'incumbent_outer':
        return CLAIM / f'{prefix}_validation_predictions.npz'
    if role == 'title_outer':
        return TITLE / f'{prefix}_validation_adapted.npz'
    raise ValueError(role)


def load_cache(path, fit, pred, field):
    with np.load(path) as z:
        np.testing.assert_array_equal(z['fit'], fit)
        np.testing.assert_array_equal(z['pred'], pred)
        value = z[field].copy()
    probability_check(value, len(pred))
    return value


def inner_probabilities(fit, nums, prefix, title_epoch):
    incumbent = np.full((len(fit), 5), np.nan)
    title = np.full_like(incumbent, np.nan)
    lineage = []
    for k in range(4):
        mask = nums == k
        a, b = fit[~mask], fit[mask]
        pi = cache_path(prefix, 'incumbent_inner', k)
        pt = cache_path(prefix, 'title_inner', k)
        incumbent[mask] = load_cache(pi, a, b, 'reference')
        title[mask] = load_cache(pt, a, b, f'epoch{title_epoch}')
        lineage.extend([pi, pt])
    probability_check(incumbent, len(fit)); probability_check(title, len(fit))
    return incumbent, title, lineage


def replay_family_choice(ctx, fit, nums, prefix, family):
    """Replay the old four-inner choice, retaining source enumeration tie order."""
    source = saved_source_choices(prefix)[family]
    y = ctx['y'][fit]
    ids = ctx['data'].id.iloc[fit].to_numpy()
    candidates = []
    if family == 'incumbent':
        epochs = [0]
    elif family == 'adapted':
        epochs = title_model.CHECKPOINTS
    else:
        raise ValueError(family)
    for epoch in epochs:
        if family == 'incumbent':
            adapted_epoch = saved_source_choices(prefix)['adapted']['epoch']
            values, _, _ = inner_probabilities(fit, nums, prefix, adapted_epoch)
            probability = values
        else:
            _, values, _ = inner_probabilities(fit, nums, prefix, epoch)
            probability = values
        for threshold in title_model.THRESHOLDS:
            p = title_model.priority(probability, threshold)
            scores = [official(y[nums == k], p[nums == k], ids[nums == k])['score'] for k in range(4)]
            candidates.append(dict(family=family, epoch=int(epoch), threshold=float(threshold),
                                   inner_score=float(np.mean(scores))))
    chosen = max(candidates, key=lambda row: row['inner_score']).copy()
    assert chosen['family'] == source['family'] and chosen['epoch'] == source['epoch']
    assert abs(chosen['threshold'] - source['threshold']) < TOL
    assert abs(chosen['inner_score'] - source['inner_score']) < TOL
    return chosen, candidates


def replay_choices(ctx, fit, nums, prefix):
    incumbent, _ = replay_family_choice(ctx, fit, nums, prefix, 'incumbent')
    adapted, _ = replay_family_choice(ctx, fit, nums, prefix, 'adapted')
    return incumbent, adapted


def choose_weight(ctx, fit, nums, incumbent, title, cutoff):
    y = ctx['y'][fit]
    ids = ctx['data'].id.iloc[fit].to_numpy()
    rows = []
    for weight in WEIGHTS:
        p = priority(incumbent, title, cutoff, weight)
        score = float(np.mean([official(y[nums == k], p[nums == k], ids[nums == k])['score'] for k in range(4)]))
        rows.append(dict(weight=float(weight), inner_score=score))
    best_score = max(row['inner_score'] for row in rows)
    selected = next(row.copy() for row in rows if best_score - row['inner_score'] <= TOL)
    return selected, rows


def outer_probabilities(fit, pred, prefix, title_epoch):
    incumbent = load_cache(cache_path(prefix, 'incumbent_outer'), fit, pred, 'reference')
    title = load_cache(cache_path(prefix, 'title_outer'), fit, pred, f'epoch{title_epoch}')
    return incumbent, title, [cache_path(prefix, 'incumbent_outer'), cache_path(prefix, 'title_outer')]


def final_probabilities(ctx, fit, pred, title_epoch):
    details = pd.read_csv(INC / 'test_prediction_details.csv', dtype={'id': str})
    assert details.id.tolist() == ctx['data'].id.iloc[pred].tolist()
    incumbent = details[[f'p{k}' for k in range(1, 6)]].to_numpy(float)
    probability_check(incumbent, len(pred))
    path = TITLE / 'final_neural.npz'
    with np.load(path) as z:
        np.testing.assert_array_equal(z['fit'], fit); np.testing.assert_array_equal(z['pred'], pred)
        title = z[f'epoch{title_epoch}'].copy()
    probability_check(title, len(pred))
    return incumbent, title, [INC / 'test_prediction_details.csv', path]


def all_cache_paths(ctx):
    paths = [INC / 'final_inner_splits.csv', INC / 'test_prediction_details.csv',
             INC / 'final_classifier.cbm', INC / 'submission_candidate.csv',
             TITLE / 'final_neural.npz', TITLE / 'neural_model' / 'manifest.json',
             TITLE / 'final_choices.json']
    for fold, fit, pred, nums in outer_members(ctx):
        prefix = f'outer{fold}'; paths.append(TITLE / f'{prefix}_choices.json')
        for k in range(4):
            paths += [cache_path(prefix, 'incumbent_inner', k), cache_path(prefix, 'title_inner', k)]
        paths += [cache_path(prefix, 'incumbent_outer'), cache_path(prefix, 'title_outer')]
    for k in range(4):
        paths += [cache_path('final', 'incumbent_inner', k), cache_path('final', 'title_inner', k)]
    assert all(path.is_file() for path in paths)
    return paths


def protocol(ctx):
    OUT.mkdir(exist_ok=True, parents=True)
    files = [HERE / 'PLAN.md', HERE / 'experiment_spec.json', HERE / 'ranking_lib.py',
             HERE / 'run_ranking.py', HERE / 'verify_ranking.py', HERE / 'write_report.py',
             HERE / 'tests' / 'test_metric.py',
             ROOT / 'specter2_preference' / 'modeling.py', ROOT / 'specter2_preference' / 'run_preference.py']
    record = dict(
        source_sha256={str(path.relative_to(ROOT.parent)): digest(path) for path in files},
        cache_sha256={str(path.relative_to(ROOT.parent)): digest(path) for path in all_cache_paths(ctx)},
        input_sha256=title_run.ab.baseline.input_digest(ctx['data']),
        labels_sha256=hashlib.sha256(ctx['y'][:2494].astype(np.int64).tobytes()).hexdigest(),
        grouping_sha256=hashlib.sha256(ctx['groups'].astype(np.int64).tobytes()).hexdigest(),
        source_title_protocol_sha256=digest(TITLE / 'protocol.json'),
        source_incumbent_sha256=digest(INC / 'submission_candidate.csv'),
        specification=json.loads((HERE / 'experiment_spec.json').read_text()),
        weights=WEIGHTS, fixed_branches=FIXED, tolerance=TOL,
    )
    target = OUT / 'protocol.json'
    if target.exists():
        assert json.loads(target.read_text()) == record, 'Locked protocol/source/cache changed'
    else:
        target.write_text(json.dumps(record, indent=2) + '\n')
    return record


def rank_frame(ids, labels, incumbent, title, cutoff, weight, fold, policy):
    p = priority(incumbent, title, cutoff, weight)
    selected = p > 0
    order = np.full(len(p), np.nan)
    rank = sorted(np.flatnonzero(selected), key=lambda i: (-p[i], str(ids[i])))
    order[rank] = np.arange(1, len(rank) + 1)
    contribution = np.zeros(len(p), float)
    if rank:
        weights = 1 / np.log2(np.arange(len(rank)) + 2)
        contribution[np.asarray(rank)] = weights * (labels[np.asarray(rank)] - 1) / 4
    return pd.DataFrame(dict(id=ids, outer_fold=fold, policy=policy, Label=labels,
                             selected=selected, priority=p, rank=order,
                             incumbent_gain=expected_gain(incumbent), title_gain=expected_gain(title),
                             discounted_gain=contribution))
