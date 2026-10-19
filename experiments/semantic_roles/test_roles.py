import numpy as np
import pandas as pd
from roles import spans,parse


def main():
    title='A title.';abstract='We study a problem. We construct a method. Our result is exact.'
    sent=spans(title,abstract)
    assert len(sent)==4
    for s in sent:assert (title if s['field']=='title' else abstract)[s['start']:s['end']]==s['text']
    good='{"object":1,"method":2,"result":3,"kind":"research"}'
    obj,invalid=parse(good,sent);assert not invalid and obj['result']==3
    for bad in ['not JSON',good.replace('3','30'),good.replace('1','true'),good.replace('research','imagined')]:
        obj,invalid=parse(bad,sent);assert invalid and obj['result'] is None
    _,invalid=parse(good,spans(title,''));assert invalid
    obj,invalid=parse('{"object":0,"method":null,"result":null,"kind":"collection"}',spans(title,''))
    assert not invalid and obj['object']==0
    obj,invalid=parse('[1,2,3,0]',sent);assert not invalid and obj['result']==3 and obj['kind']=='research'
    _,invalid=parse('[1,2,3,99]',sent);assert invalid
    from experiment import choose
    # Strict cutoff with tied probabilities must keep both .5 rows together.
    y=np.array([1,2,4,5]);p=np.array([.2,.5,.5,.9]);inc=np.array([.1,.9,.9,.9])
    selected=choose(y,p,inc)
    assert selected['FP']==0 and selected['FN']==0 and selected['threshold']==.2
    assert np.array_equal(p>selected['threshold'],[False,True,True,True])
    from features import fit_spaces,transform
    frame=pd.DataFrame(dict(title=['alpha mesh','alpha signal','beta mesh','forbidden marker'],
        abstract=['alpha mesh','alpha signal','beta mesh','forbidden marker'],
        object=['alpha mesh','alpha signal','beta mesh','forbidden marker'],
        method=['']*4,result=['']*4,kind=['research']*4,invalid=[0]*4))
    spaces=fit_spaces(frame,np.array([0,1,2]))
    for _,_,vectorizer in spaces:
        if vectorizer is not None:
            assert 'forbidden' not in vectorizer.vocabulary_ and 'marker' not in vectorizer.vocabulary_
    vectors={name:np.eye(4,384,dtype=np.float32) for name in ['title','abstract','object','method','result']}
    built=transform(frame,vectors,np.array([3]),spaces)
    assert built['text'].shape[1]>=768 and built['roles'].shape[1]>=built['text'].shape[1]+1152
    assert np.isfinite(built['roles'].data).all()
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('semantic_verifier',Path(__file__).with_name('verify.py'))
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    independent_metric=verifier.independent_metric
    got=independent_metric(np.array([5,1,3]),np.array([9.,8.,7.]),np.array(['a','b','c']))
    weights=1/np.log2(np.arange(3)+2)
    assert abs(got['Q']-(weights[0]+.5*weights[2])/weights.sum())<1e-14
    assert got['FP']==1 and got['FN']==0
    print('PASS: evidence/schema, missing abstract, strict threshold/ties, fit-only vocab, full dimensions, metric.')


if __name__=='__main__':main()
