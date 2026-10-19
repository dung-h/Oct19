"""Full-size text and evidence-role blocks; no supervised history or SVD."""
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from roles import ROLES,KINDS


def fit_spaces(texts,fit):
    specs=[('title','word',30000),('abstract','word',30000),('title','char_wb',30000)]
    specs += [(role,'word',10000) for role in ROLES]
    spaces=[]
    for field,analyzer,max_features in specs:
        model=TfidfVectorizer(analyzer=analyzer,ngram_range=(3,5) if analyzer=='char_wb' else (1,2),
            min_df=3 if analyzer=='char_wb' else 2,max_df=.98,max_features=max_features,
            sublinear_tf=True,dtype=np.float64)
        try:model.fit(texts[field].iloc[fit])
        except ValueError as exc:
            if not any(x in str(exc) for x in ['vocabulary','terms remain','max_df corresponds']):raise
            model=None
        spaces.append((field,analyzer,model))
    return spaces


def transform(texts,embedding,indices,spaces):
    fields={}
    for k,(field,analyzer,model) in enumerate(spaces):
        fields[k]=model.transform(texts[field].iloc[indices]) if model is not None else sparse.csr_matrix((len(indices),1))
    raw=normalize(sparse.hstack([fields[k] for k in range(3)]+
        [sparse.csr_matrix(embedding[field][indices].astype(np.float64)) for field in ['title','abstract']],format='csr'))
    presence=np.column_stack([texts[role].iloc[indices].str.len().gt(0).to_numpy(float) for role in ROLES])
    kind=np.column_stack([texts.kind.iloc[indices].eq(k).to_numpy(float) for k in KINDS])
    flags=np.column_stack([presence,kind,texts.invalid.iloc[indices].to_numpy(float)])
    semantic=normalize(sparse.hstack([fields[k] for k in range(3,6)]+
        [sparse.csr_matrix(embedding[role][indices].astype(np.float64)) for role in ROLES]+
        [sparse.csr_matrix(flags)],format='csr'))
    augmented=normalize(sparse.hstack([raw,semantic],format='csr'))
    assert np.isfinite(raw.data).all() and np.isfinite(augmented.data).all()
    return {'text':raw,'roles':augmented}
