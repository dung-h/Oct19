#!/usr/bin/env python3
"""Controlled representation/head/metadata experiments on the saved nested splits."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

import numpy as np
import pandas as pd
from scipy import linalg, sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
sys.path.insert(0, str(HERE.parent / 'specter2_v2'))
from prepare_data import load_clean_data, baseline
from run_v2 import evaluate

V2 = HERE.parent / 'specter2_v2' / 'outputs'
RETRIEVAL_REV = '2081559630a80fc5851d8f798a05ba81e9468089'
MINILM_REV = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
ALPHAS = [.001, .01, .1, 1., 10., 100.]
KERNEL_ALPHAS = [.01, .1, 1.]
GAMMAS = [.25, 1., 4.]
THRESHOLDS = np.arange(0, .626, .025)


def configurations():
    configs = []
    for space in ['regression', 'retrieval', 'minilm', 'tfidf']:
        for alpha in ALPHAS:
            configs.append(dict(family=f'{space}_ridge', space=space, head='ridge', alpha=alpha))
        for gamma in GAMMAS:
            for alpha in KERNEL_ALPHAS:
                configs.append(dict(family=f'{space}_rbf', space=space, head='rbf', alpha=alpha, gamma=gamma))
    for space in ['regression_centered', 'venue', 'authors', 'year', 'metadata']:
        for alpha in ALPHAS:
            configs.append(dict(family=f'{space}_ridge', space=space, head='ridge', alpha=alpha))
    for space in ['venue', 'authors', 'year', 'metadata']:
        for weight in [.25, 1.]:
            for alpha in ALPHAS:
                configs.append(dict(family=f'regression_plus_{space}_ridge',
                                    space=f'plus_{space}', head='ridge', alpha=alpha, weight=weight))
    for index, config in enumerate(configs):
        config['key'] = f'c{index:03d}'
    return configs


def write_protocol(data):
    split_path = V2 / 'nested_splits.csv'
    protocol = dict(
        purpose='diagnose representation vs head vs missing metadata; no final model selection/submission',
        configurations=configurations(), thresholds=THRESHOLDS.tolist(),
        inner_selection='maximize mean inner-fold official H; deterministic first config/threshold on ties',
        input_sha256=baseline.input_digest(data), split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        splits='exact 5 outer / 4 inner membership reused from specter2_v2',
        labels='original labels from source ZIP, never replaced or inferred',
        encoders=dict(regression=baseline.ADAPTER_REV, retrieval=RETRIEVAL_REV, minilm=MINILM_REV),
        encoder_input='clean title + tokenizer SEP + clean usable abstract; max 512 tokens for all encoders',
        minilm_note='512 positions intentionally used to match token budget; official sentence-transformers default is 256',
        preprocessing='L2 embeddings; centered variant fits mean on training only then L2; all vocab/scalers fit within fold',
        kernel='RBF exp(-gamma_multiplier * squared_distance / training_median_squared_distance); centered target',
        metadata='venue one-hot; exact normalized comma/semicolon-separated author names multi-hot then row L2; training-standardized year times 0.25; combined metadata row L2',
        limitations='Dataset and previous OOF already inspected; nested CV controls tuning, not adaptive research reuse. No definitive causal proof of optimal encoder/head.',
    )
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Existing protocol differs; use a new experiment directory.'
    else:
        path.write_text(json.dumps(protocol, indent=2) + '\n')


def encode(data):
    import torch
    from adapters import AutoAdapterModel
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(20261015)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    digest = baseline.input_digest(data)
    keys = (data.split + ':' + data.id).to_numpy(dtype=str)
    for name, repo, revision in [('retrieval', 'allenai/specter2', RETRIEVAL_REV),
                                  ('minilm', 'sentence-transformers/all-MiniLM-L6-v2', MINILM_REV)]:
        cache = OUT / f'embeddings_{name}.npz'
        manifest_path = OUT / f'embeddings_{name}.json'
        if cache.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            assert manifest['input_sha256'] == digest and manifest['revision'] == revision
            print(f'Using cached {name} embeddings', flush=True)
            continue
        started = time.monotonic()
        if name == 'retrieval':
            base_path = snapshot_download(baseline.BASE, revision=baseline.BASE_REV,
                                          allow_patterns=['*.bin', '*.json', 'vocab.txt'])
            path = snapshot_download(repo, revision=revision, allow_patterns=['*.bin', '*.json'])
            tokenizer = AutoTokenizer.from_pretrained(base_path)
            model = AutoAdapterModel.from_pretrained(base_path)
            model.load_adapter(path, load_as='retrieval', set_active=True)
        else:
            path = snapshot_download(repo, revision=revision,
                                     allow_patterns=['*.json', 'vocab.txt', 'pytorch_model.bin', 'model.safetensors'])
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModel.from_pretrained(path)
        model.to(device).eval().requires_grad_(False)
        texts = (data.title + tokenizer.sep_token + data.abstract).tolist()
        lengths = np.array([len(x) for x in tokenizer(texts, truncation=False)['input_ids']])
        order = np.argsort(lengths, kind='stable')
        vectors = np.zeros((len(data), model.config.hidden_size), dtype=np.float32)
        batch_size = 12 if name == 'retrieval' else 32
        for start in range(0, len(data), batch_size):
            ix = order[start:start + batch_size]
            batch = tokenizer([texts[i] for i in ix], max_length=512, truncation=True,
                              padding=True, return_tensors='pt', return_token_type_ids=False)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.inference_mode(), torch.autocast(device_type=device, enabled=device == 'cuda', dtype=torch.float16):
                hidden = model(**batch).last_hidden_state
                if name == 'retrieval':
                    result = hidden[:, 0, :].float()
                else:
                    mask = batch['attention_mask'].unsqueeze(-1).float()
                    result = (hidden.float()*mask).sum(1) / mask.sum(1).clamp(min=1)
            vectors[ix] = result.cpu().numpy()
            if start % (batch_size * 50) == 0:
                print(f'{name}: {min(start+batch_size,len(data))}/{len(data)}', flush=True)
        assert np.isfinite(vectors).all() and (np.linalg.norm(vectors, axis=1) > 0).all()
        np.savez_compressed(cache, embeddings=vectors, keys=keys)
        manifest = dict(repo=repo, revision=revision, base_revision=baseline.BASE_REV if name == 'retrieval' else None,
                        input_sha256=digest, shape=list(vectors.shape), max_length=512,
                        pooling='CLS' if name == 'retrieval' else 'attention_mask_mean',
                        truncated_rows=int((lengths > 512).sum()), elapsed_seconds=time.monotonic()-started,
                        device=device, mixed_precision=device == 'cuda')
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'Completed {name}: {manifest}', flush=True)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()


def author_dict(text):
    parts = re.split(r'[,;]', str(text))
    names = [re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', x).strip().casefold()) for x in parts]
    return {name: 1. for name in names if name}


def metadata_spaces(data, fit, pred):
    venue = DictVectorizer(sparse=True)
    vf = venue.fit_transform([{'venue='+x: 1.} for x in data.venue.iloc[fit]])
    vp = venue.transform([{'venue='+x: 1.} for x in data.venue.iloc[pred]])
    authors = DictVectorizer(sparse=True)
    af = normalize(authors.fit_transform([author_dict(x) for x in data.authors.iloc[fit]]))
    ap = normalize(authors.transform([author_dict(x) for x in data.authors.iloc[pred]]))
    scaler = StandardScaler()
    yf = sparse.csr_matrix(scaler.fit_transform(data[['year']].iloc[fit]) * .25)
    yp = sparse.csr_matrix(scaler.transform(data[['year']].iloc[pred]) * .25)
    return dict(venue=(vf, vp), authors=(af, ap), year=(yf, yp),
                metadata=(normalize(sparse.hstack([vf, af, yf], format='csr')),
                          normalize(sparse.hstack([vp, ap, yp], format='csr'))))


def build_spaces(data, features, fit, pred):
    spaces = {key: (value[fit], value[pred]) for key, value in features.items()}
    mean = features['regression'][fit].mean(axis=0, keepdims=True)
    spaces['regression_centered'] = (normalize(features['regression'][fit]-mean),
                                      normalize(features['regression'][pred]-mean))
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=.98,
                                max_features=50000, sublinear_tf=True)
    spaces['tfidf'] = (vectorizer.fit_transform(data.text.iloc[fit]), vectorizer.transform(data.text.iloc[pred]))
    spaces.update(metadata_spaces(data, fit, pred))
    for kind in ['venue', 'authors', 'year', 'metadata']:
        for weight in [.25, 1.]:
            spaces[(f'plus_{kind}', weight)] = tuple(normalize(sparse.hstack(
                [sparse.csr_matrix(spaces['regression'][i]), weight*spaces[kind][i]], format='csr')) for i in [0,1])
    return spaces


def kernel_predictions(kf, kp, y, alpha):
    regularized = kf.copy()
    regularized.flat[::len(y)+1] += alpha
    solution = linalg.cho_solve(linalg.cho_factor(regularized, lower=True, check_finite=False),
                                y-y.mean(), check_finite=False)
    return kp @ solution+y.mean(), kf @ solution+y.mean()


def fit_predict(data, features, labels, fit, pred, configs):
    spaces = build_spaces(data, features, fit, pred)
    predictions, diagnostics = {}, {}
    y = labels[fit].astype(float)
    for config in configs:
        if config['head'] != 'ridge':
            continue
        key = (config['space'], config['weight']) if 'weight' in config else config['space']
        xf, xp = spaces[key]
        model = Ridge(alpha=config['alpha'], solver='lsqr', tol=1e-6).fit(xf, y)
        predictions[config['key']] = model.predict(xp)
        diagnostics[config['key']] = dict(train_mae=float(mean_absolute_error(y, model.predict(xf))))
    for space in ['regression', 'retrieval', 'minilm', 'tfidf']:
        xf, xp = spaces[space]
        # Float64 distances avoid subtractive error for near-parallel regression vectors.
        df = euclidean_distances(xf.astype(np.float64), squared=True)
        dp = euclidean_distances(xp.astype(np.float64), xf.astype(np.float64), squared=True)
        median = float(np.median(df[np.triu_indices(len(fit), 1)]))
        assert median > 0 and np.isfinite(median)
        for gamma in GAMMAS:
            kf = np.exp(-gamma*df/median)
            kp = np.exp(-gamma*dp/median)
            for config in configs:
                if config['head'] != 'rbf' or config['space'] != space or config['gamma'] != gamma:
                    continue
                pred_rating, fit_rating = kernel_predictions(kf, kp, y, config['alpha'])
                predictions[config['key']] = pred_rating
                diagnostics[config['key']] = dict(train_mae=float(mean_absolute_error(y, fit_rating)),
                                                  median_squared_distance=median)
    assert len(predictions) == len(configs)
    assert all(np.isfinite(x).all() for x in predictions.values())
    return predictions, diagnostics


def priority(rating, threshold):
    gain = (rating-1)/4
    return np.where((gain > threshold) & (gain > 0), 10*gain, 0.)


def select_inner(data, labels, fit_indices, fold_numbers, predictions, configs):
    chosen = {}
    rows = []
    for config in configs:
        ratings = predictions[config['key']]
        best = None
        for threshold in THRESHOLDS:
            priorities = priority(ratings, threshold)
            scores = []
            for fold in range(4):
                mask = fold_numbers == fold
                scores.append(evaluate(labels[fit_indices[mask]], priorities[mask], data.id.iloc[fit_indices[mask]])['score'])
            row = dict(config, threshold=float(threshold), inner_score=float(np.mean(scores)))
            if best is None or row['inner_score'] > best['inner_score']:
                best = row
        rows.append(best)
        if config['family'] not in chosen or best['inner_score'] > chosen[config['family']]['inner_score']:
            chosen[config['family']] = best
    return chosen, rows


def load_features(data):
    paths = dict(regression=V2/'embeddings_clean.npz', retrieval=OUT/'embeddings_retrieval.npz',
                 minilm=OUT/'embeddings_minilm.npz')
    keys = (data.split + ':' + data.id).to_numpy(dtype=str)
    features = {}
    for name, path in paths.items():
        cache = np.load(path)
        assert np.array_equal(cache['keys'], keys)
        features[name] = normalize(cache['embeddings'].astype(np.float64))
    return features


def run(data):
    started = time.monotonic()
    features = load_features(data)
    labels = np.full(len(data), np.nan)
    train = np.flatnonzero(data.split.eq('train'))
    labels[train] = data.Label.iloc[train].astype(int)
    groups = baseline.grouping(data.iloc[train].reset_index(drop=True))
    splits = pd.read_csv(V2/'nested_splits.csv', dtype={'id':str})
    id_to_index = dict(zip(data.id.iloc[train], train))
    configs = configurations()
    oof = data.iloc[train][['id','title','venue','year','Label']].copy().reset_index(drop=True)
    oof['outer_fold'] = -1
    metrics, searches = [], []
    for outer_fold in range(5):
        part = splits[splits.outer_fold.eq(outer_fold)]
        fit_rows = part[part.partition.eq('outer_train')]
        val_rows = part[part.partition.eq('outer_validation')]
        fit = np.array([id_to_index[x] for x in fit_rows.id])
        val = np.array([id_to_index[x] for x in val_rows.id])
        inner_numbers = fit_rows.inner_fold.to_numpy()
        assert set(groups[fit]).isdisjoint(groups[val])
        cache_path = OUT/f'outer_{outer_fold}_predictions.npz'
        if cache_path.exists():
            cached = np.load(cache_path)
            assert np.array_equal(cached['fit'], fit) and np.array_equal(cached['val'], val)
            inner = {c['key']:cached['inner_'+c['key']] for c in configs}
            predictions = {c['key']:cached['outer_'+c['key']] for c in configs}
            diagnostics = json.loads((OUT/f'outer_{outer_fold}_fit_diagnostics.json').read_text())
        else:
            inner = {c['key']:np.full(len(fit), np.nan) for c in configs}
            for fold in range(4):
                fi, vi = fit[inner_numbers != fold], fit[inner_numbers == fold]
                assert set(groups[fi]).isdisjoint(groups[vi])
                pred, _ = fit_predict(data, features, labels, fi, vi, configs)
                for key in inner:
                    inner[key][inner_numbers == fold] = pred[key]
                print(f'outer {outer_fold+1}/5 inner {fold+1}/4 done; {time.monotonic()-started:.1f}s', flush=True)
            predictions, diagnostics = fit_predict(data, features, labels, fit, val, configs)
            np.savez_compressed(cache_path, fit=fit, val=val,
                                **{'inner_'+k:v for k,v in inner.items()},
                                **{'outer_'+k:v for k,v in predictions.items()})
            (OUT/f'outer_{outer_fold}_fit_diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
        chosen, rows = select_inner(data, labels, fit, inner_numbers, inner, configs)
        searches.extend(dict(outer_fold=outer_fold, **row) for row in rows)
        for family, config in chosen.items():
            ratings = predictions[config['key']]
            priorities = priority(ratings, config['threshold'])
            result = evaluate(labels[val], priorities, data.id.iloc[val])
            metrics.append(dict(outer_fold=outer_fold, family=family, config=config['key'],
                                alpha=config['alpha'], gamma=config.get('gamma'), weight=config.get('weight'),
                                threshold=config['threshold'], inner_score=config['inner_score'],
                                **result, **diagnostics[config['key']],
                                validation_mae=float(mean_absolute_error(labels[val], ratings)),
                                read_auc=float(roc_auc_score(labels[val]>1, ratings))))
            # Train records occupy the original first 2494 positions (verified below).
            oof.loc[val, family+'_rating'] = ratings
            oof.loc[val, family+'_priority'] = priorities
        oof.loc[val, 'outer_fold'] = outer_fold
        pd.DataFrame(metrics).to_csv(OUT/'fold_metrics.csv', index=False)
        pd.DataFrame(searches).to_csv(OUT/'inner_search.csv', index=False)
        oof.to_csv(OUT/'oof_predictions.csv', index=False)
        print(f'OUTER {outer_fold+1} completed; elapsed={time.monotonic()-started:.1f}s', flush=True)
        print(pd.DataFrame(metrics).query('outer_fold == @outer_fold')[['family','score','FP','FN','validation_mae']].to_string(index=False), flush=True)
    assert (oof.outer_fold >= 0).all() and oof.id.is_unique and len(oof) == 2494
    frame = pd.DataFrame(metrics)
    summary = frame.groupby('family').agg(score_mean=('score','mean'), score_std=('score','std'),
                Q=('relevance','mean'), R=('coverage','mean'), FP=('FP','sum'), FN=('FN','sum'),
                train_mae=('train_mae','mean'), validation_mae=('validation_mae','mean'), read_auc=('read_auc','mean'))
    summary.to_csv(OUT/'summary.csv')
    (OUT/'run_manifest.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-started,
                families=len(summary), configurations=len(configs), train_rows=len(train),
                protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
                no_submission_created=True), indent=2)+'\n')
    print(summary.sort_values('score_mean', ascending=False).to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['all', 'embed', 'train'], default='all', nargs='?')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data, _ = load_clean_data()
    assert np.array_equal(np.flatnonzero(data.split.eq('train')), np.arange(2494))
    write_protocol(data)
    if args.stage in ['all','embed']:
        encode(data)
    if args.stage in ['all','train']:
        with threadpool_limits(limits=4):
            run(data)


if __name__ == '__main__':
    main()
