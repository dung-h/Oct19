"""Post-hoc diagnostic references fitted only from pilot fit labels."""
import argparse
import numpy as np

from analyze import HERE, PILOT, metrics, prediction, write_csv
from common import load_inputs, official
from pathlib import Path


def decomposition(y,p):
    p=np.asarray(p,float)
    p=p/p.sum(1,keepdims=True)
    read=1-p[:,0]
    positive=y>1
    eps=np.finfo(float).eps
    binary=float(-np.log(np.clip(np.where(positive,read,p[:,0]),eps,1)).mean())
    cond=p[positive,y[positive]-1]/read[positive]
    conditional=float(-np.log(np.clip(cond,eps,1)).mean())
    return dict(binary_read_logloss=binary,positive_rating_logloss=conditional,
                positive_fraction=float(positive.mean()),
                reconstructed_5class_logloss=binary+positive.mean()*conditional)


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--source-output',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    source,out=args.source_output.resolve(),args.output.resolve()
    if (out/'reference_metrics.csv').exists():
        raise RuntimeError('Reference analysis exists; use a new output namespace')
    docs,targets,parts,base=load_inputs()
    rows,losses=[] ,[]
    for tag,part in parts.items():
        y=np.array([targets[i] for i in part['pred']])
        fy=np.array([targets[i] for i in part['fit']])
        prior=np.bincount(fy,minlength=6)[1:]/len(fy)
        missing_priors={}
        for present in (False,True):
            fit_y=[targets[i] for i in part['fit'] if bool(docs[i]['abstract'].strip())==present]
            missing_priors[present]=(np.bincount(fit_y,minlength=6)[1:]+1)/(len(fit_y)+5)
        arms=dict(fit_prior=np.tile(prior,(len(y),1)),
                  fit_missingness_prior=np.array([missing_priors[bool(docs[i]['abstract'].strip())] for i in part['pred']]),
                  incumbent=np.array(base[tag]['probabilities']))
        arms.update({arm:prediction(source,tag,arm,part['pred'],docs) for arm in ('frozen','trained')})
        for arm,p in arms.items():
            m=metrics(y,p)
            d=decomposition(y,p)
            assert abs(d['reconstructed_5class_logloss']-m['rating_logloss'])<1e-6
            native=official(y,np.where(1-p[:,0]>.55,1+9*(p@(np.arange(5)/4)),0),
                            [docs[i]['id'] for i in part['pred']])
            rows.append(dict(partition=tag,arm=arm,**m,**native))
            losses.append(dict(partition=tag,arm=arm,**d))
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'reference_metrics.csv',rows)
    write_csv(out/'loss_decomposition.csv',losses)
    print(out)


if __name__=='__main__':
    main()
