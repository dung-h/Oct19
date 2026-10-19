"""Shared deterministic helpers for the fixed read-gate experiment."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'outputs'
INC = ROOT / 'preference_augmentation' / 'outputs'
CLAIM = ROOT / 'claim_stability' / 'outputs'
TITLE = ROOT / 'specter2_preference' / 'outputs'
META = ROOT / 'input_ablation' / 'outputs' / 'metadata_audit.csv'
sys.path.insert(0, str(ROOT / 'specter2_preference'))
import run_preference as title_run

BRANCHES = ['binary_raw', 'binary_global', 'binary_venue', 'author_raw', 'author_global', 'author_venue']
CONTROLS = ['incumbent_deployed', 'incumbent_at_050']
SEED = 20261018


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def context():
    ctx = title_run.context()
    assert len(ctx['data']) == 3090 and np.isnan(ctx['y'][2494:]).all()
    return ctx


def outer_members(ctx):
    return list(title_run.outer_members(ctx))


def final_partition(ctx):
    membership = pd.read_csv(INC / 'final_inner_splits.csv', dtype={'id': str})
    assert membership.id.tolist() == ctx['data'].id.iloc[:2494].tolist()
    return np.arange(2494), np.arange(2494, 3090), membership.inner_fold.to_numpy(int)


def pack_path(name: str) -> Path:
    return INC / f'{name}_features.joblib'


def load_pack(name: str, fit: np.ndarray, pred: np.ndarray):
    pack = joblib.load(pack_path(name))
    np.testing.assert_array_equal(pack['fit'], fit); np.testing.assert_array_equal(pack['pred'], pred)
    a, b = pack['a'].copy(), pack['b'].copy()
    assert a.shape[1] == b.shape[1] == 277 and a.columns.equals(b.columns)
    assert a.columns[0] == 'venue' and a.venue.notna().all() and b.venue.notna().all()
    assert np.isfinite(a.drop(columns='venue').to_numpy(float)).all()
    assert np.isfinite(b.drop(columns='venue').to_numpy(float)).all()
    return pack, a, b


def sigmoid_logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(probability, venue, target, kind: str):
    scalar = StandardScaler().fit(sigmoid_logit(probability))
    x = scalar.transform(sigmoid_logit(probability))
    encoder = None
    if kind == 'venue':
        encoder = OneHotEncoder(handle_unknown='ignore')
        v = encoder.fit_transform(np.asarray(venue, str).reshape(-1, 1))
        x = sparse.hstack([sparse.csr_matrix(x), v], format='csr')
    model = LogisticRegression(C=1, solver='lbfgs', max_iter=1000, fit_intercept=True,
                               class_weight=None, random_state=SEED).fit(x, target)
    return {'kind': kind, 'scalar': scalar, 'encoder': encoder, 'model': model}


def apply_calibrator(recipe, probability, venue):
    x = recipe['scalar'].transform(sigmoid_logit(probability))
    if recipe['encoder'] is not None:
        x = sparse.hstack([sparse.csr_matrix(x), recipe['encoder'].transform(np.asarray(venue, str).reshape(-1, 1))], format='csr')
    return recipe['model'].predict_proba(x)[:, 1]


def selection_metrics(y, selected):
    y = np.asarray(y, int); selected = np.asarray(selected, bool); positive = y > 1
    tp = int((selected & positive).sum()); fp = int((selected & ~positive).sum())
    fn = int((~selected & positive).sum()); tn = int((~selected & ~positive).sum())
    return dict(total=len(y), accepted=int(selected.sum()), dropped=int((~selected).sum()),
                TP=tp, FP=fp, FN=fn, TN=tn, total_error=fp + fn,
                precision=tp / (tp + fp) if tp + fp else 0., recall=tp / (tp + fn) if tp + fn else 0.,
                specificity=tn / (tn + fp) if tn + fp else 0.,
                balanced_accuracy=((tp / (tp + fn) if tp + fn else 0.) + (tn / (tn + fp) if tn + fp else 0.)) / 2,
                missed_high=int(((y >= 4) & ~selected).sum()))


def official(y, priority, ids):
    order = np.asarray(sorted(np.flatnonzero(priority > 0), key=lambda i: (-priority[i], str(ids[i]))), int)
    weights = 1 / np.log2(np.arange(len(order)) + 2)
    q = float(np.average((y[order] - 1) / 4, weights=weights)) if len(order) else 0.
    recall = float((y[order] > 1).sum() / (y > 1).sum()) if (y > 1).any() else 0.
    return dict(H=2*q*recall/(q+recall) if q+recall else 0., Q=q, R=recall)


def reliability(y, p):
    target = np.asarray(y, int) > 1; p = np.asarray(p, float)
    rows = []
    for lo in np.arange(0, 1, .1):
        hi = lo + .1; mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        rows.append(dict(bin=f'[{lo:.1f},{hi:.1f}{")" if hi < 1 else "]"}', n=int(mask.sum()),
                         mean_probability=float(p[mask].mean()) if mask.any() else np.nan,
                         positive_fraction=float(target[mask].mean()) if mask.any() else np.nan))
    eps = 1e-15; pclip=np.clip(p, eps, 1-eps)
    return rows, float(np.mean((p-target)**2)), float(-np.mean(target*np.log(pclip)+(1-target)*np.log(1-pclip)))


def load_incumbent_outer(fold, fit, pred):
    z = np.load(CLAIM / f'outer{fold}_validation_predictions.npz')
    np.testing.assert_array_equal(z['fit'], fit); np.testing.assert_array_equal(z['pred'], pred)
    return z['reference'].copy()


def load_incumbent_final(ctx, fit, pred):
    d = pd.read_csv(INC / 'test_prediction_details.csv', dtype={'id': str})
    assert d.id.tolist() == ctx['data'].id.iloc[pred].tolist()
    return d[[f'p{i}' for i in range(1,6)]].to_numpy(float)


def title_probability(prefix: str, fit, pred, epoch: int):
    path = TITLE / (f'{prefix}_validation_adapted.npz' if prefix.startswith('outer') else 'final_neural.npz')
    z = np.load(path); np.testing.assert_array_equal(z['fit'], fit); np.testing.assert_array_equal(z['pred'], pred)
    return z[f'epoch{epoch}'].copy()


def priority(selected, incumbent_probability, title_probability):
    rank = .5 * (incumbent_probability @ (np.arange(5)/4)) + .5 * (title_probability @ (np.arange(5)/4))
    return np.where(selected, 1 + 9 * rank, 0.)
