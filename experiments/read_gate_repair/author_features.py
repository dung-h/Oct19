"""Author parsing and leakage-safe smoothed author-history features."""
from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

AUTHOR_COLUMNS = [
    'author_count', 'author_missing', 'author_source_verified',
    'author_format_uncertain', 'known_author_count', 'known_author_fraction',
    'author_cold_start', 'author_mean_log_support', 'author_max_log_support',
    'author_unique_reference_papers_fraction', 'author_read_mean',
    'author_read_min', 'author_read_max', 'author_read_support_weighted_mean',
    'author_read_std', 'author_gain_mean', 'author_gain_min', 'author_gain_max',
    'author_gain_support_weighted_mean',
]
SMOOTHING = 10.0


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ''
    return html.unescape(str(value)).strip()


def normalise_name(value: object) -> str:
    value = unicodedata.normalize('NFC', _text(value)).casefold()
    return re.sub(r'\s+', ' ', value).strip(' ,;')


def parse_authors(value: object) -> tuple[list[str], bool]:
    """Return conservative full-name tokens and an uncertainty flag.

    The audit source uses comma/semicolon separated full names.  We never make a
    surname-only or fuzzy identity match.  Ambiguous ``et al.`` text is retained
    as an uncertainty flag rather than being invented as an author.
    """
    raw = _text(value)
    if not raw:
        return [], False
    uncertain = bool(re.search(r'\bet\s+al\.?\b|\bunknown\b|\bvarious\b', raw, re.I))
    cleaned = re.sub(r'\bet\s+al\.?\b', '', raw, flags=re.I).strip(' ,;')
    parts = re.split(r'\s*;\s*|\s+\band\b\s+|\s*,\s*', cleaned, flags=re.I)
    tokens = []
    for part in parts:
        token = normalise_name(part)
        if token and len(token) >= 2 and not re.fullmatch(r'[\W_]+', token):
            tokens.append(token)
    return list(dict.fromkeys(tokens)), uncertain


def source_ledger(data: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Join metadata by split/id and preserve every raw/verified provenance field."""
    keys = data[['split', 'id']].copy()
    merged = keys.merge(audit[['split', 'id', 'authors_original', 'authors_verified',
                               'author_verified', 'status']], on=['split', 'id'], how='left',
                        validate='one_to_one')
    assert len(merged) == len(data)
    verified = merged.author_verified.fillna(False).astype(str).str.casefold().eq('true')
    verified &= merged.authors_verified.notna() & merged.authors_verified.astype(str).str.strip().ne('')
    used = np.where(verified, merged.authors_verified, merged.authors_original)
    parsed = [parse_authors(x) for x in used]
    merged['authors_used'] = used
    merged['author_source_verified'] = verified.astype(int)
    merged['author_tokens'] = [x[0] for x in parsed]
    merged['author_format_uncertain'] = [int(x[1]) for x in parsed]
    merged['author_missing'] = [int(not x[0]) for x in parsed]
    return merged


def _reference_state(tokens: list[list[str]], labels: np.ndarray, groups: np.ndarray,
                     refs: np.ndarray) -> tuple[dict, dict, Counter, Counter, Counter, Counter, int, float, float]:
    """Aggregate enough statistics to remove an entire duplicate group per query."""
    refs = np.asarray(refs, dtype=int)
    author_all = defaultdict(lambda: [0, 0.0, 0.0])
    author_group = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0]))
    group_n, group_read, group_gain = Counter(), Counter(), Counter()
    for i in refs:
        g = groups[i]; read = float(labels[i] > 1); gain = float((labels[i] - 1) / 4)
        group_n[g] += 1; group_read[g] += read; group_gain[g] += gain
        for author in tokens[i]:
            author_all[author][0] += 1; author_all[author][1] += read; author_all[author][2] += gain
            state = author_group[author][g]
            state[0] += 1; state[1] += read; state[2] += gain
    return author_all, author_group, group_n, group_read, group_gain, Counter(groups[refs]), len(refs), float((labels[refs] > 1).sum()), float(((labels[refs]-1)/4).sum())


def history_features(tokens: list[list[str]], verified: np.ndarray, uncertain: np.ndarray,
                     labels: np.ndarray, groups: np.ndarray, refs: np.ndarray,
                     queries: np.ndarray) -> pd.DataFrame:
    """Features for queries, using only labels in refs and excluding query group."""
    state = _reference_state(tokens, labels, groups, refs)
    author_all, author_group, group_n, group_read, group_gain, _, total_n, total_read, total_gain = state
    rows = []
    for query in np.asarray(queries, dtype=int):
        g = groups[query]; n_ref = total_n - group_n[g]
        sum_read = total_read - group_read[g]; sum_gain = total_gain - group_gain[g]
        prior_read = sum_read / n_ref if n_ref else .5
        prior_gain = sum_gain / n_ref if n_ref else .0
        names = tokens[query]
        supports, reads, gains = [], [], []
        for name in names:
            n, s_read, s_gain = author_all[name]
            removed = author_group[name][g]
            n -= removed[0]; s_read -= removed[1]; s_gain -= removed[2]
            if n > 0:
                supports.append(float(n))
                reads.append((s_read + SMOOTHING * prior_read) / (n + SMOOTHING))
                gains.append((s_gain + SMOOTHING * prior_gain) / (n + SMOOTHING))
        known = len(supports)
        if known:
            weights = np.log1p(np.asarray(supports))
            read_weighted = float(np.average(reads, weights=weights))
            gain_weighted = float(np.average(gains, weights=weights))
            unique_fraction = float(sum(supports) / n_ref) if n_ref else 0.0
        else:
            reads, gains, supports = [prior_read], [prior_gain], []
            read_weighted, gain_weighted, unique_fraction = prior_read, prior_gain, 0.0
        rows.append(dict(
            author_count=len(names), author_missing=int(not names),
            author_source_verified=int(verified[query]), author_format_uncertain=int(uncertain[query]),
            known_author_count=known, known_author_fraction=known / len(names) if names else 0.0,
            author_cold_start=int(bool(names) and not known),
            author_mean_log_support=float(np.mean(np.log1p(supports))) if supports else 0.0,
            author_max_log_support=float(np.max(np.log1p(supports))) if supports else 0.0,
            author_unique_reference_papers_fraction=unique_fraction,
            author_read_mean=float(np.mean(reads)), author_read_min=float(np.min(reads)),
            author_read_max=float(np.max(reads)), author_read_support_weighted_mean=read_weighted,
            author_read_std=float(np.std(reads)) if known else 0.0,
            author_gain_mean=float(np.mean(gains)), author_gain_min=float(np.min(gains)),
            author_gain_max=float(np.max(gains)), author_gain_support_weighted_mean=gain_weighted,
        ))
    return pd.DataFrame(rows, columns=AUTHOR_COLUMNS, dtype=float)


def pair_features(ledger: pd.DataFrame, labels: np.ndarray, groups: np.ndarray,
                  fit: np.ndarray, pred: np.ndarray, lineage: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-fit fit rows using cache lineage, then use all fit rows for predictions."""
    tokens = ledger.author_tokens.tolist()
    verified = ledger.author_source_verified.to_numpy(bool)
    uncertain = ledger.author_format_uncertain.to_numpy(bool)
    fit, pred = np.asarray(fit, int), np.asarray(pred, int)
    lookup = {int(row): pos for pos, row in enumerate(fit)}
    train = pd.DataFrame(np.nan, index=np.arange(len(fit)), columns=AUTHOR_COLUMNS)
    seen = np.zeros(len(fit), bool)
    for item in lineage:
        refs = np.asarray(item['refs'], int); queries = np.asarray(item['queries'], int)
        assert set(refs).issubset(set(fit)) and set(queries).issubset(set(fit))
        assert not (set(groups[refs]) & set(groups[queries]))
        value = history_features(tokens, verified, uncertain, labels, groups, refs, queries)
        positions = [lookup[int(i)] for i in queries]
        train.iloc[positions] = value.to_numpy()
        seen[positions] = True
    assert seen.all() and train.notna().all().all()
    test = history_features(tokens, verified, uncertain, labels, groups, fit, pred)
    return train.astype(float), test.astype(float)
