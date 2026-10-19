#!/usr/bin/env python3
"""Describe the saved incumbent's actual inputs without training or modifying it."""
import hashlib
import json
from pathlib import Path
import re
import sys

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'outputs'
sys.path.insert(0, str(HERE.parent / 'preference_augmentation'))
import run_augmentation as run


def describe(name):
    if name == 'venue':
        return ('base', 'venue', 'String category; missing -> unknown; CatBoost native categorical input',
                'Original metadata; categorical processing learned by model inside fit', '')
    if name == 'year':
        return ('base', 'year', 'Float year; missing fallback 0; no scaling in CatBoost feature frame',
                'Original metadata; no learned numeric preprocessing', '')
    if name.startswith('title_svd'):
        return ('base', 'title', 'Word TF-IDF unigrams/bigrams -> unnormalized SVD coordinate, 32 total',
                'TF-IDF vocabulary/IDF and SVD fitted only on the current fit partition', '')
    if name == 'ridge_rating':
        return ('base', 'title + venue + permitted reference labels',
                'L2 SPECTER2 regression title embedding + venue one-hot; concat/L2; Ridge alpha .01 predicts rating',
                'Train: 3-fold grouped crossfit; prediction: fit references, excluding query duplicate group', '')
    if name.startswith('content_'):
        field, rest = name[len('content_'):].split('_', 1)
        if rest == 'words':
            operation, pattern = 'Whitespace word count', ''
        elif rest == 'available':
            operation, pattern = '1 if cleaned/usable field is nonempty, else 0', ''
        else:
            aspect, category = rest.split('_', 1)
            if category == 'unknown':
                operation, pattern = f'1 if no regex for aspect {aspect} matched; absence of evidence', ''
            else:
                pattern = run.feat.RULES[aspect][category]
                operation = 'Case-insensitive regex existence indicator; not a verified semantic category'
        return ('content', field, operation, 'Fixed rules; no label lookup or statistical fitting', pattern)
    match = re.fullmatch(r'neighbor_(retrieval|title|abstract)_(all|venue)_(.+)', name)
    if not match:
        raise ValueError(name)
    channel, scope, statistic = match.groups()
    source = ('Frozen SPECTER2 retrieval title+abstract cosine' if channel == 'retrieval'
              else f'{channel} reference-fitted word TF-IDF cosine')
    source += '; same venue references' if scope == 'venue' else '; all permitted references'
    if statistic == 'references':
        operation = 'Total permitted reference count before similarity/venue filtering'
    elif statistic == 'support':
        operation = 'Count with cosine > 1e-8 satisfying the scope'
    elif statistic.startswith('class'):
        operation = 'Maximum eligible cosine for that true reference class; 0 if none'
    else:
        count, kind = statistic.split('_', 1)
        suffix = dict(gain='Expected gain [0,.25,.5,.75,1] under smoothed label frequencies',
                      entropy='Natural-log entropy of smoothed label frequencies',
                      similarity='Similarity mean weighted by exp((cosine-max)/.1)',
                      count='Number of available references among the top k')
        operation = f'Top {count[1:]}: ' + (suffix[kind] if kind in suffix else
            'Reference class frequency weighted by exp((cosine-max)/.1), smoothed with 3 class-prior pseudo counts; not a calibrated probability')
    return ('neighbors', source, operation,
            'Train: 3-fold grouped crossfit; prediction: fit references; no query/heldout labels; duplicate groups excluded', '')


def main():
    OUT.mkdir(exist_ok=True)
    data, _, _, _, _, _, _, _ = run.context()
    pack = joblib.load(run.OUT / 'final_test_features.joblib')
    a, b = pack['a'], pack['b']
    model = CatBoostClassifier()
    model.load_model(str(run.OUT / 'final_classifier.cbm'))
    columns = model.feature_names_
    assert columns == a.columns.tolist() == b.columns.tolist()
    importance = pd.read_csv(run.OUT/'final_feature_importance.csv').set_index('feature').importance
    assert set(importance.index) == set(columns)
    rows = []
    for position, name in enumerate(columns):
        group, source, transform, fit_scope, pattern = describe(name)
        numeric = pd.api.types.is_numeric_dtype(a[name])
        rows.append(dict(position=position, feature=name, group=group, source=source,
            transform=transform, fit_scope=fit_scope, regex=pattern, dtype=str(a[name].dtype),
            train_missing=int(a[name].isna().sum()), test_missing=int(b[name].isna().sum()),
            train_min=float(a[name].min()) if numeric else None,
            train_max=float(a[name].max()) if numeric else None,
            test_min=float(b[name].min()) if numeric else None,
            test_max=float(b[name].max()) if numeric else None,
            training_feature_importance=float(importance.loc[name])))
    catalog = pd.DataFrame(rows)
    catalog.to_csv(OUT / 'feature_catalog.csv', index=False)
    numeric = pd.concat([a, b]).select_dtypes('number')
    params = model.get_all_params()
    vectorizer, svd = pack['recipe']['vectorizer'], pack['recipe']['svd']
    expected = data.iloc[np.r_[pack['fit'], pack['pred']]].reset_index(drop=True)
    joined = pd.concat([a,b],ignore_index=True)
    summary = dict(train_shape=list(a.shape), test_shape=list(b.shape),
        group_counts=catalog.group.value_counts().to_dict(),
        model_column_order_matches_saved_frames=True,
        numeric_nan=int(numeric.isna().sum().sum()), numeric_inf=int(np.isinf(numeric).sum().sum()),
        original_metadata_matches_saved_features=bool(joined.venue.tolist()==expected.venue.tolist() and np.array_equal(joined.year, expected.year)),
        missing_original_authors=int(data.authors.eq('').sum()),
        missing_model_input={field:int(data[field].eq('').sum()) for field in ['title','abstract','venue']},
        usable_abstracts_by_split=data.groupby('split').abstract.apply(lambda s:int(s.ne('').sum())).to_dict(),
        title_tfidf_vocabulary_size=len(vectorizer.vocabulary_),
        title_tfidf_params={k:vectorizer.get_params()[k] for k in ['lowercase','strip_accents','token_pattern','ngram_range','min_df','max_df','max_features','sublinear_tf','norm','stop_words']},
        final_title_svd_explained_variance=float(svd.explained_variance_ratio_.sum()),
        train_reference_counts=sorted(a.neighbor_retrieval_all_references.unique().tolist()),
        test_reference_counts=sorted(b.neighbor_retrieval_all_references.unique().tolist()),
        catboost_params={k:params.get(k) for k in ['iterations','depth','learning_rate','l2_leaf_reg','loss_function','bootstrap_type','bagging_temperature','leaf_estimation_iterations','random_seed']},
        catboost_final_choice=json.loads((run.OUT/'final_choice.json').read_text()),
        specter_title_input_stats=json.loads((HERE.parent/'specter2_preference_longer/outputs/input_stats.json').read_text()),
        current_submission_sha256=hashlib.sha256((run.OUT/'submission_candidate.csv').read_bytes()).hexdigest(),
        limitations=['Feature importance is descriptive, not evidence of causal benefit.',
            'Input digest covers split/id/title/abstract, but not every feature-bearing raw field; current metadata was compared directly above.',
            'Zero missing/infinite numeric values does not prove sufficient semantic representation.',
            'Crossfit training references are smaller than the prediction references.'])
    (OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
