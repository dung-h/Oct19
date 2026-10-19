"""Independent feature repairs; never modify inherited feature modules or labels."""
import copy
import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 20261018
ARMS = ['reference', 'counts', 'title128', 'abstract64', 'rules', 'matched', 'combined']
EDUCATION = r'\b(?:educat\w*|teach\w*|schools?|curricul\w*|classrooms?|pedagog\w*)\b'
NEURAL = r'\b(?:neural|deep learning|large language models?|LLMs?|BERT|GPT(?:-\w+)?|attention[- ]based transformers?|transformer[- ]based (?:language|neural) models?)\b'
TUTORIAL = r'\btutorials?\b'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_digest(data):
    # Unlike the inherited text-only digest, include metadata and train labels.
    fields = ['split', 'id', 'title', 'abstract', 'venue', 'year', 'authors', 'doi', 'Label']
    payload = data[fields].fillna('').to_dict('records')
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def fixed_content(data, old_content, old_rules):
    rules = copy.deepcopy(old_rules)
    rules['domain']['education'] = EDUCATION
    rules['method']['neural'] = NEURAL
    rules['type']['tutorial'] = TUTORIAL
    rows, evidence = [], []
    for row in data[['split','id','title','abstract']].itertuples(index=False):
        values = {}
        for field in ['title', 'abstract']:
            text = getattr(row, field)
            values[field+'_words'] = len(text.split())
            values[field+'_available'] = int(bool(text.strip()))
            for aspect, definitions in rules.items():
                hits = []
                for category, pattern in definitions.items():
                    match = re.search(pattern, text, re.I)
                    values[f'{field}_{aspect}_{category}'] = int(match is not None)
                    if match:
                        hits.append(category)
                        evidence.append(dict(split=row.split, id=row.id, field=field,
                            aspect=aspect, category=category, start=match.start(),
                            end=match.end(), evidence=text[match.start():match.end()]))
                values[f'{field}_{aspect}_unknown'] = int(not hits)
        rows.append(values)
    fixed = pd.DataFrame(rows, dtype=float)
    fixed = fixed[list(old_content)+[c for c in fixed if c not in old_content]]
    changes = []
    for col in fixed:
        before = old_content[col].to_numpy() if col in old_content else np.zeros(len(data))
        after = fixed[col].to_numpy()
        for i in np.flatnonzero(before != after):
            changes.append(dict(split=data.split.iloc[i], id=data.id.iloc[i], feature=col,
                before=float(before[i]), after=float(after[i]),
                reason='require explicit education/neural wording; tutorial evidence; recompute unknown'))
    return fixed, pd.DataFrame(evidence), pd.DataFrame(changes), rules


def normalize_counts(frame):
    frame = frame.copy()
    for column in list(frame):
        if column.startswith('neighbor_') and column.endswith('_support'):
            references = column[:-len('support')]+'references'
            frame[column] = frame[column] / frame[references].clip(lower=1)
            frame.rename(columns={column:column+'_fraction'}, inplace=True)
    return frame.drop(columns=[c for c in frame if c.startswith('neighbor_') and c.endswith('_references')])


def fit_svd(a, b, dimensions):
    components = min(dimensions, a.shape[0]-1, a.shape[1]-1)
    if components < 1:
        return np.zeros((a.shape[0],dimensions)), np.zeros((b.shape[0],dimensions)), None
    model = TruncatedSVD(n_components=components, random_state=SEED)
    train = model.fit_transform(a)
    pred = model.transform(b)
    return (np.pad(train, ((0,0),(0,dimensions-components))),
            np.pad(pred, ((0,0),(0,dimensions-components))), model)


def transform_svd(model, matrix, dimensions):
    if model is None:
        return np.zeros((matrix.shape[0],dimensions))
    value = model.transform(matrix)
    return np.pad(value, ((0,0),(0,dimensions-value.shape[1])))


def oov_rate(vectorizer, texts):
    analyze = vectorizer.build_analyzer()
    total = missing = 0
    for text in texts:
        terms = analyze(text)
        total += len(terms)
        missing += sum(t not in vectorizer.vocabulary_ for t in terms)
    return missing/total if total else 0.


def matched_histories(ctx, lineage, pred, base_features):
    histories = []
    for split in lineage:
        refs = np.asarray(split['refs'], dtype=int)
        value = base_features.history(ctx['data'],ctx['y'],ctx['groups'],ctx['title'],
                                      ctx['retrieval'],refs,pred)
        value = value.rename(columns={c:'neighbor_'+c for c in value if c!='ridge_rating'})
        histories.append(value)
    return histories


def build_rich(ctx, old, fit, pred, base_features):
    data = ctx['data']
    title_vectorizer = old['recipe']['vectorizer']
    a = title_vectorizer.transform(data.title.iloc[fit])
    b = title_vectorizer.transform(data.title.iloc[pred])
    title_a, title_b, title_svd = fit_svd(a,b,128)
    abstract_vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.98,
        max_features=20000,sublinear_tf=True)
    a = abstract_vectorizer.fit_transform(data.abstract.iloc[fit])
    b = abstract_vectorizer.transform(data.abstract.iloc[pred])
    abstract_a, abstract_b, abstract_svd = fit_svd(a,b,64)
    assert np.all(abstract_a[data.abstract.iloc[fit].eq('')]==0)
    assert np.all(abstract_b[data.abstract.iloc[pred].eq('')]==0)
    histories = matched_histories(ctx,old['lineage'],pred,base_features)
    audit = dict(title_vocabulary=len(title_vectorizer.vocabulary_),
        abstract_vocabulary=len(abstract_vectorizer.vocabulary_),
        title_svd32_variance=float(old['recipe']['svd'].explained_variance_ratio_.sum()),
        title_svd128_variance=float(title_svd.explained_variance_ratio_.sum()),
        abstract_svd64_variance=float(abstract_svd.explained_variance_ratio_.sum()),
        title_prediction_oov=oov_rate(title_vectorizer,data.title.iloc[pred]),
        abstract_prediction_oov=oov_rate(abstract_vectorizer,data.abstract.iloc[pred]),
        train_reference_counts=sorted(old['a'].neighbor_retrieval_all_references.unique().tolist()),
        old_prediction_reference_counts=sorted(old['b'].neighbor_retrieval_all_references.unique().tolist()),
        matched_prediction_reference_counts=[sorted(v.neighbor_retrieval_all_references.unique().tolist()) for v in histories])
    return dict(fit=fit,pred=pred,title_a=title_a,title_b=title_b,abstract_a=abstract_a,
        abstract_b=abstract_b,histories=histories,audit=audit,
        projection=dict(title_vectorizer=title_vectorizer,title_svd=title_svd,
                        abstract_vectorizer=abstract_vectorizer,abstract_svd=abstract_svd))


def frame(ctx, old, rich, arm, partition, view=0):
    result = old[partition].copy()
    indices = old['fit'] if partition=='a' else old['pred']
    if partition=='b' and arm in ['matched','combined']:
        history = rich['histories'][view]
        for column in history:
            result[column] = history[column].to_numpy()
    if arm in ['rules','combined']:
        repaired = ctx['fixed_content'].iloc[indices].reset_index(drop=True)
        for column in repaired:
            result['content_'+column] = repaired[column].to_numpy()
    if arm in ['title128','combined']:
        result = result.drop(columns=[c for c in result if c.startswith('title_svd')])
        extra = pd.DataFrame(rich['title_'+partition],columns=[f'title128_{j:03}' for j in range(128)])
        result = pd.concat([result,extra],axis=1)
    if arm in ['abstract64','combined']:
        extra = pd.DataFrame(rich['abstract_'+partition],columns=[f'abstract64_{j:03}' for j in range(64)])
        result = pd.concat([result,extra],axis=1)
    if arm in ['counts','combined']:
        result = normalize_counts(result)
    numeric = result.select_dtypes('number').to_numpy()
    assert np.isfinite(numeric).all()
    return result


def rebuild_rich(ctx, bundle, base_features):
    """Inference from saved projection objects; no projection fitting on queries."""
    fit,pred = bundle['fit'],bundle['pred']
    recipe = bundle['projection']
    data = ctx['data']
    rich = dict(title_b=transform_svd(recipe['title_svd'],recipe['title_vectorizer'].transform(data.title.iloc[pred]),128),
        abstract_b=transform_svd(recipe['abstract_svd'],recipe['abstract_vectorizer'].transform(data.abstract.iloc[pred]),64))
    old = dict(fit=fit,pred=pred)
    old['b'] = base_features.rebuild_prediction(data,ctx['y'],ctx['groups'],ctx['title'],
        ctx['retrieval'],ctx['old_content'],fit,pred,bundle['base_recipe'])
    rich['histories'] = matched_histories(ctx,bundle['lineage'],pred,base_features)
    return old,rich
