"""Label-independent evidence extraction and strictly cross-fitted history features."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, normalize

SEED = 20261018
# General scientific vocabulary, frozen before any augmentation outer evaluation.
# A match asserts only that the phrase occurs, not the paper's true semantic class.
RULES = {
 'type': {
  'collection': r'\b(?:proceedings|revised selected papers|international workshops?)\b',
  'survey': r'\b(?:survey|systematic review|literature review|overview)\b',
  'invited': r'\b(?:invited talk|keynote|invited paper)\b',
  'short': r'\b(?:extended abstract|short paper|poster|demonstration)\b',
 },
 'problem': {
  'verification': r'\b(?:verif(?:y|ication)|model checking|soundness|correctness|termination|reachability)\b',
  'inference': r'\b(?:inference|entailment|reasoning|query answering|deduction)\b',
  'learning': r'\b(?:learning|classification|prediction|clustering|recognition)\b',
  'planning': r'\b(?:planning|scheduling|coordination|decision making|control synthesis)\b',
  'explanation': r'\b(?:explainab\w*|interpretab\w*|explanations?|transparency)\b',
  'optimization': r'\b(?:optimization|optimisation|maxsat|minimi[sz]ation|constraint solving)\b',
  'semantics': r'\b(?:semantics|expressiveness|equivalence|bisimulation|definability)\b',
  'complexity': r'\b(?:complexity|decidability|undecidability|lower bounds?|upper bounds?)\b',
 },
 'method': {
  'logic_programming': r'\b(?:logic programming|logic programs?|answer set(?:s| programming)?|datalog|prolog|ASP)\b',
  'inductive_logic': r'\b(?:inductive logic programming|ILP)\b',
  'epistemic': r'\b(?:epistemic|world views?|possible worlds?)\b',
  'argumentation': r'\b(?:argumentation|assumption.based arguments?)\b',
  'temporal_logic': r'\b(?:temporal logic|linear temporal|LTL|CTL|metric temporal)\b',
  'sat_smt': r'\b(?:SAT|SMT|satisfiability|satisfiability modulo theories)\b',
  'automata_nets': r'\b(?:automata|automaton|Petri nets?|workflow nets?|reset nets?)\b',
  'type_theory': r'\b(?:type theory|type systems?|dependent types?|lambda calculus|typed)\b',
  'neural': r'\b(?:neural|deep learning|transformers?|large language models?|LLMs?)\b',
  'probabilistic': r'\b(?:probabilistic|Bayesian|stochastic|Markov)\b',
  'abstract_interpretation': r'\b(?:abstract interpretation|static analysis|program analysis)\b',
  'causal': r'\b(?:causal|causality|counterfactuals?)\b',
 },
 'contribution': {
  'algorithm': r'\b(?:algorithms?|procedures?|computational methods?)\b',
  'tool': r'\b(?:tools?|solvers?|implementation|implemented|prototype|software|system description)\b',
  'theory': r'\b(?:theorems?|proofs?|prove|proven|characteri[sz]ation|formal foundations?)\b',
  'empirical': r'\b(?:experiments?|experimental|empirical|benchmarks?|evaluation|case stud(?:y|ies))\b',
  'dataset': r'\b(?:datasets?|corpus|corpora|data collection)\b',
 },
 'domain': {
  'health': r'\b(?:medic\w*|health\w*|clinical|patients?|diagnos\w*|diseases?|ICU|antimicrobial|bacterial|viral)\b',
  'transport': r'\b(?:vehicles?|driving|traffic|transport\w*|aerial|autonomous)\b',
  'biology': r'\b(?:biolog\w*|metabolic|genes?|genomic\w*|proteins?|bioinformatics)\b',
  'security': r'\b(?:security|privacy|cryptograph\w*|attacks?|vulnerabilit\w*|authentication)\b',
  'law': r'\b(?:legal|law|normative|regulations?|compliance|legislation)\b',
  'education': r'\b(?:education\w*|schools?|students?|teaching|curricul\w*)\b',
  'hardware': r'\b(?:hardware|circuits?|processors?|embedded|digital systems?)\b',
  'multiagent': r'\b(?:multi.agent|agents?|games?|strategies|strategic)\b',
  'databases': r'\b(?:databases?|knowledge graphs?|ontolog\w*|relational data)\b',
  'finance': r'\b(?:financ\w*|banking|trading|economic\w*|auctions?)\b',
 },
}


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_features(data):
    """Only title/abstract are read. No label, ID, venue or hand-selected case rules."""
    rows, evidence = [], []
    for _, row in data.iterrows():
        values = {}
        for field in ['title', 'abstract']:
            text = str(row[field])
            values[field+'_words'] = len(text.split())
            values[field+'_available'] = int(bool(text.strip()))
            for aspect, rules in RULES.items():
                hits = []
                for name, pattern in rules.items():
                    match = re.search(pattern, text, flags=re.I)
                    values[f'{field}_{aspect}_{name}'] = int(match is not None)
                    if match:
                        hits.append(name)
                        evidence.append(dict(split=row['split'], id=row['id'], field=field,
                            aspect=aspect, category=name, start=match.start(), end=match.end(),
                            evidence=text[match.start():match.end()]))
                values[f'{field}_{aspect}_unknown'] = int(not hits)
        rows.append(values)
    return pd.DataFrame(rows, dtype=float), pd.DataFrame(evidence)


def lexical_space(data, refs, queries, field):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
        max_df=.98, max_features=20000, sublinear_tf=True)
    try:
        reference = vectorizer.fit_transform(data[field].iloc[refs])
        query = vectorizer.transform(data[field].iloc[queries])
    except ValueError as exc:
        if 'vocabulary' not in str(exc) and 'terms remain' not in str(exc):
            raise
        reference = sparse.csr_matrix((len(refs), 1))
        query = sparse.csr_matrix((len(queries), 1))
    return reference, query


def ridge_space(data, title_vectors, refs, queries):
    venue = OneHotEncoder(handle_unknown='ignore')
    a = venue.fit_transform(data[['venue']].iloc[refs])
    b = venue.transform(data[['venue']].iloc[queries])
    return (normalize(sparse.hstack([sparse.csr_matrix(title_vectors[refs]), a])),
            normalize(sparse.hstack([sparse.csr_matrix(title_vectors[queries]), b])), venue)


def history(data, labels, groups, title_vectors, retrieval, refs, queries, include_neighbors=True):
    """Feature targets exclusively indexed by refs. Query labels are never accessed."""
    refs, queries = np.asarray(refs), np.asarray(queries)
    overlap = set(groups[refs]) & set(groups[queries])
    if overlap:
        # Test has two duplicate groups shared with train. Exclude that group
        # for each affected query, including its Ridge score and class prior.
        pieces = []
        for group in [None]+sorted(overlap):
            mask = ~np.isin(groups[queries], list(overlap)) if group is None else groups[queries]==group
            if not mask.any():
                continue
            allowed = refs if group is None else refs[groups[refs]!=group]
            part = history(data, labels, groups, title_vectors, retrieval, allowed, queries[mask], include_neighbors)
            part.index = np.flatnonzero(mask)
            pieces.append(part)
        return pd.concat(pieces).sort_index().reset_index(drop=True)
    assert set(groups[refs]).isdisjoint(groups[queries])
    a, b, _ = ridge_space(data, title_vectors, refs, queries)
    ridge = Ridge(alpha=.01, solver='lsqr', tol=1e-6).fit(a, labels[refs])
    result = {'ridge_rating': ridge.predict(b)}
    if not include_neighbors:
        return pd.DataFrame(result)
    sim_channels = {'retrieval': retrieval[queries] @ retrieval[refs].T}
    for field in ['title', 'abstract']:
        a, b = lexical_space(data, refs, queries, field)
        sim_channels[field] = (b @ a.T).toarray()
    ref_y = labels[refs].astype(int)
    prior = np.bincount(ref_y, minlength=6)[1:6].astype(float) / len(refs)
    rv, qv = data.venue.iloc[refs].to_numpy(), data.venue.iloc[queries].to_numpy()
    for channel, similarities in sim_channels.items():
        for scope in ['all', 'venue']:
            stats = []
            for i, sims in enumerate(similarities):
                allowed = np.flatnonzero((sims > 1e-8) & ((rv == qv[i]) if scope == 'venue' else True))
                ordered = allowed[np.argsort(-sims[allowed], kind='stable')]
                item = {'references': len(refs), 'support': len(allowed)}
                # Class-wise proximity helps distinguish positive and negative near neighbors.
                for label in range(1, 6):
                    ix = allowed[ref_y[allowed] == label]
                    item[f'class{label}_nearest'] = float(sims[ix].max()) if len(ix) else 0.
                for k in [5, 15]:
                    ix = ordered[:k]
                    weights = np.exp((sims[ix]-sims[ix].max())/.1) if len(ix) else np.array([])
                    counts = np.bincount(ref_y[ix], weights=weights, minlength=6)[1:6]
                    dist = (counts+3*prior)/(weights.sum()+3)
                    for label in range(1, 6):
                        item[f'k{k}_p{label}'] = dist[label-1]
                    item[f'k{k}_gain'] = float(dist @ (np.arange(5)/4))
                    item[f'k{k}_entropy'] = float(-(dist*np.log(dist+1e-12)).sum())
                    item[f'k{k}_similarity'] = float(np.average(sims[ix], weights=weights)) if len(ix) else 0.
                    item[f'k{k}_count'] = len(ix)
                stats.append(item)
            frame = pd.DataFrame(stats).add_prefix(f'{channel}_{scope}_')
            result.update({c: frame[c].to_numpy() for c in frame})
    return pd.DataFrame(result)


def crossfit_history(data, labels, groups, title, retrieval, fit, seed=SEED):
    fit = np.asarray(fit)
    splitter = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
    result, lineage = None, []
    for fold, (r, q) in enumerate(splitter.split(fit, labels[fit], groups[fit])):
        refs, query = fit[r], fit[q]
        frame = history(data, labels, groups, title, retrieval, refs, query)
        if result is None:
            result = pd.DataFrame(np.nan, index=np.arange(len(fit)), columns=frame.columns)
        result.iloc[q] = frame.to_numpy()
        lineage.append(dict(fold=fold, refs=refs.tolist(), queries=query.tolist()))
    assert result.notna().all().all()
    return result, lineage


def build_features(data, labels, groups, title, retrieval, content, fit, pred):
    """All target-derived training features cross-fitted; target-free SVD fit on fit only."""
    fit, pred = np.asarray(fit), np.asarray(pred)
    hf, lineage = crossfit_history(data, labels, groups, title, retrieval, fit)
    hp = history(data, labels, groups, title, retrieval, fit, pred)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=.98,
        max_features=20000, sublinear_tf=True)
    a = vectorizer.fit_transform(data.title.iloc[fit]); b = vectorizer.transform(data.title.iloc[pred])
    svd = TruncatedSVD(n_components=min(32, a.shape[1]-1), random_state=SEED)
    sf, sp = svd.fit_transform(a), svd.transform(b)
    frames = []
    for ix, hist, sv in [(fit, hf, sf), (pred, hp, sp)]:
        base = data[['venue', 'year']].iloc[ix].reset_index(drop=True).copy()
        base['venue'] = base.venue.fillna('unknown').astype(str)
        base['year'] = base.year.astype(float).fillna(0)
        for j in range(sv.shape[1]):
            base[f'title_svd{j:02}'] = sv[:, j]
        base['ridge_rating'] = hist.ridge_rating.to_numpy()
        c = content.iloc[ix].reset_index(drop=True).add_prefix('content_')
        n = hist.drop(columns='ridge_rating').add_prefix('neighbor_')
        frames.append(pd.concat([base, c, n], axis=1))
    # Ridge control is fitted directly to all fit; cannot reuse its crossfit training scores.
    a, b, venue = ridge_space(data, title, fit, pred)
    ridge = Ridge(alpha=.01, solver='lsqr', tol=1e-6).fit(a, labels[fit])
    recipe = dict(vectorizer=vectorizer, svd=svd, ridge=ridge, venue=venue)
    return frames[0], frames[1], ridge.predict(b), lineage, recipe


def columns_for(frame, family):
    return [c for c in frame if
        (not c.startswith(('content_', 'neighbor_'))) or
        (c.startswith('content_') and family in ['content', 'both']) or
        (c.startswith('neighbor_') and family in ['neighbors', 'both'])]


def rebuild_prediction(data, labels, groups, title, retrieval, content, fit, pred, recipe):
    h = history(data, labels, groups, title, retrieval, fit, pred)
    sv = recipe['svd'].transform(recipe['vectorizer'].transform(data.title.iloc[pred]))
    base = data[['venue', 'year']].iloc[pred].reset_index(drop=True).copy()
    base['venue'] = base.venue.fillna('unknown').astype(str)
    base['year'] = base.year.astype(float).fillna(0)
    for j in range(sv.shape[1]):
        base[f'title_svd{j:02}'] = sv[:, j]
    base['ridge_rating'] = h.ridge_rating.to_numpy()
    return pd.concat([base, content.iloc[pred].reset_index(drop=True).add_prefix('content_'),
        h.drop(columns='ridge_rating').add_prefix('neighbor_')], axis=1)
