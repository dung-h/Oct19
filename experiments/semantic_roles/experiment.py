"""Locked experiment IO, cached source probabilities, counts and threshold rule."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from roles import HERE,OUT,ROOT,source,digest

BRANCHES=['text_raw','roles_raw','text_blend','roles_blend']
PRIMARY='roles_blend'


def context():
    ctx=source.context()
    texts=pd.read_csv(OUT/'texts.csv',dtype={'id':str},keep_default_na=False)
    z=np.load(OUT/'embeddings.npz')
    keys=(ctx['data'].split+':'+ctx['data'].id).to_numpy(str)
    assert texts.id.tolist()==ctx['data'].id.tolist() and texts.split.tolist()==ctx['data'].split.tolist()
    np.testing.assert_array_equal(z['keys'],keys)
    assert texts.title.tolist()==ctx['data'].title.tolist() and texts.abstract.tolist()==ctx['data'].abstract.tolist()
    ctx['texts']=texts;ctx['embedding']={k:z[k] for k in z.files if k!='keys'}
    manifest=json.loads((OUT/'embedding_manifest.json').read_text())
    assert manifest['input_sha256']==digest(OUT/'texts.csv') and manifest['extraction_sha256']==digest(OUT/'extraction.jsonl')
    return ctx


def jobs(ctx):
    result=[]
    for fold,fit,pred,nums in source.outer_members(ctx):
        for k in range(4):
            mask=nums==k;result.append((f'outer{fold}_inner{k}',fit[~mask],fit[mask]))
        result.append((f'outer{fold}_validation',fit,pred))
    fit,pred,nums=source.final_partition(ctx)
    for k in range(4):result.append((f'final_inner{k}',fit[nums!=k],fit[nums==k]))
    result.append(('final_test',fit,pred))
    return result


def incumbent(name,fit,pred,ctx):
    if name=='final_test':return source.load_incumbent_final(ctx,fit,pred)
    path=source.CLAIM/(name+'_predictions.npz')
    with np.load(path) as z:
        np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['pred'],pred)
        p=z['reference'].copy()
    assert p.shape==(len(pred),5) and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
    return p


def protocol(ctx):
    files=[HERE/f for f in ['PLAN.md','extraction_prompt.txt','roles.py','fast_decode.py','features.py','experiment.py',
                           'run.py','verify.py','report.py','test_roles.py','workflow.py']]
    files += [ROOT/'read_gate_repair/gate_lib.py',ROOT/'specter2_preference/run_preference.py',
              ROOT/'specter2_preference/modeling.py',ROOT/'input_ablation/run_ablation.py',
              ROOT/'diagnose_bottleneck/run_diagnostics.py',ROOT/'specter2/run_experiment.py',
              ROOT/'specter2_v2/prepare_data.py']
    inputs=[OUT/'extraction_protocol.json',OUT/'extraction.jsonl',OUT/'texts.csv',OUT/'evidence.csv',
            OUT/'decoder_verification.json',OUT/'preflight_v4_extraction.jsonl',OUT/'preflight_extraction.jsonl',
            OUT/'embeddings.npz',OUT/'embedding_manifest.json',ROOT/'specter2_v2/outputs/nested_splits.csv',
            source.INC/'final_inner_splits.csv',source.INC/'submission_candidate.csv',
            source.INC/'test_prediction_details.csv',ROOT/'specter2_v2/outputs/abstracts_cleaned.csv',
            ROOT.parent/'assignment-data-mining-261.zip']
    inputs += [source.CLAIM/(name+'_predictions.npz') for name,_,_ in jobs(ctx) if name!='final_test']
    record=dict(sources={str(p.relative_to(ROOT.parent)):digest(p) for p in files},
        inputs={str(p.relative_to(ROOT.parent)):digest(p) for p in inputs},
        input_sha256=hashlib.sha256(ctx['data'].to_csv(index=False).encode()).hexdigest(),
        labels_sha256=hashlib.sha256(ctx['y'][:2494].astype(np.int64).tobytes()).hexdigest(),
        groups_sha256=hashlib.sha256(ctx['groups'].astype(np.int64).tobytes()).hexdigest(),
        primary=PRIMARY,branches=BRANCHES,classification='binary LogisticRegression C4 L2 liblinear tol1e-5 max_iter2000 seed20261018',
        operating_rule='min FP subject to inner FN and high FN <= source incumbent at0.55; ties FN/highFN/descending threshold',
        blend_weight=.5,ranking='incumbent expected gain',uploaded=False)
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==record,'Locked source/input changed'
    else:path.write_text(json.dumps(record,indent=2)+'\n')
    return record


def branch_values(text,roles,inc):
    return dict(text_raw=text,roles_raw=roles,text_blend=.5*(text+inc),roles_blend=.5*(roles+inc))


def choose(y,p,reference):
    y=np.asarray(y,int);p=np.asarray(p,float);old=np.asarray(reference,float)>.55
    budget_fn=int(((y>1)&~old).sum());budget_high=int(((y>=4)&~old).sum())
    rows=[]
    for cutoff in np.r_[0.,np.unique(p)]:
        selected=p>cutoff
        fn=int(((y>1)&~selected).sum());high=int(((y>=4)&~selected).sum())
        if fn<=budget_fn and high<=budget_high:
            rows.append(dict(threshold=float(cutoff),FP=int(((y==1)&selected).sum()),FN=fn,missed_high=high))
    assert rows
    best=min(rows,key=lambda r:(r['FP'],r['FN'],r['missed_high'],-r['threshold']))
    return dict(**best,reference_FN=budget_fn,reference_high_FN=budget_high)


def metrics(y,p,ids):
    # All selected priorities positive; tie breaks by lexicographic string ID.
    y=np.asarray(y,int);p=np.asarray(p,float);sel=p>0
    tp=int(((y>1)&sel).sum());fp=int(((y==1)&sel).sum());fn=int(((y>1)&~sel).sum());tn=int(((y==1)&~sel).sum())
    order=np.asarray(sorted(np.flatnonzero(sel),key=lambda i:(-p[i],str(ids[i]))),int)
    w=1/np.log2(np.arange(len(order))+2)
    q=float(np.average((y[order]-1)/4,weights=w)) if len(order) else 0.
    r=tp/int((y>1).sum()) if (y>1).any() else 0.
    return dict(total=len(y),accepted=int(sel.sum()),dropped=int((~sel).sum()),TP=tp,FP=fp,FN=fn,TN=tn,
        total_error=fp+fn,missed_high=int(((y>=4)&~sel).sum()),H=2*q*r/(q+r) if q+r else 0.,Q=q,R=r)


def priorities(selected,incumbent_probability):
    return np.where(selected,1+9*(incumbent_probability@(np.arange(5)/4)),0.)
