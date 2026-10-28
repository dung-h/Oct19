"""Fit-only CPU screening of CE, ordinal RPS, and lightly weighted CE heads."""
from __future__ import annotations
import argparse, json, hashlib
import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

ROOT=Path(__file__).resolve().parents[2]
PILOT=ROOT/'experiments/qwen_rank_pilot_v1/outputs/ranking_pilot_v1_r2_torch212_cu130'
OUT=Path(os.environ.get('QWEN_OBJECTIVE_OUTPUT',Path(__file__).resolve().parent/'outputs/objective_screen_v2'))
PARTS=('outer1_inner0','outer1_inner1'); ARMS=('ce','ordinal_rps','weighted_ce'); LAMBDAS=(.001,.01,.1,1.,10.)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rps(p,y):
    c=np.cumsum(p,axis=1)[:,:-1]; t=(y[:,None] <= np.arange(1,5)[None,:]).astype(float)
    return ((c-t)**2).mean(axis=1)
def class_weights(y,cap=2.):
    n=np.bincount(y-1,minlength=5).astype(float)
    if (n==0).any(): raise ValueError('missing class in fit fold')
    u=np.minimum(cap,np.sqrt(len(y)/(5*n)))
    w=u/np.sum((n/len(y))*u)
    return w

def fit(x,y,lam,arm,max_iter):
    n,d=x.shape; yi=y-1; cnt=np.bincount(yi,minlength=5).astype(float)
    b0=np.log((cnt+1)/(n+5)); theta0=np.r_[np.zeros(5*d),b0]
    weights=class_weights(y) if arm=='weighted_ce' else np.ones(5)
    def obj(th):
        w=th[:5*d].reshape(5,d); b=th[5*d:]; z=x@w.T+b
        p=softmax(z,axis=1); ce=logsumexp(z,axis=1)-z[np.arange(n),yi]
        per=weights[yi]*ce
        gradp=p.copy(); gradp[np.arange(n),yi]-=1
        gradp*=weights[yi,None]
        loss=per.mean()
        if arm=='ordinal_rps':
            c=np.cumsum(p,axis=1)[:,:-1]; t=(y[:,None]<=np.arange(1,5)[None,:]).astype(float)
            loss+=np.mean(((c-t)**2).mean(axis=1))
            # d RPS / d p_j = sum_{k>=j}(2/4)(CDF_k-target_k)
            dc=(2./4.)*(c-t)/n
            dp=np.zeros_like(p)
            for j in range(5): dp[:,j]=dc[:,j:].sum(axis=1) if j<4 else 0.
            gradp+=p*(dp-(dp*p).sum(axis=1,keepdims=True))*n
        loss+=.5*lam*(w*w).sum()
        gradp/=n
        gw=gradp.T@x+lam*w; gb=gradp.sum(0)
        return float(loss),np.r_[gw.ravel(),gb]
    res=minimize(obj,theta0,jac=True,method='L-BFGS-B',options={'maxiter':max_iter,'ftol':1e-12,'gtol':1e-8,'maxls':40})
    if not res.success or not np.isfinite(res.fun): raise RuntimeError(f'{arm} lambda={lam}: {res.message}')
    return res.x[:5*d].reshape(5,d),res.x[5*d:]
def probabilities(x,sc,w,b): return softmax(sc.transform(x)@w.T+b,axis=1)
def main():
    cfg=json.loads((Path(__file__).parent/'config.json').read_text())
    OUT.mkdir(parents=True,exist_ok=False)
    f=np.load(PILOT/'frozen_features.npz'); feat={int(i):x.astype(np.float64) for i,x in zip(f['indices'],f['hidden'])}
    docs={int(row['index']):row for row in map(json.loads,(ROOT/'experiments/qwen_remote_v1/inputs/documents.jsonl').open())}
    parts=json.loads((ROOT/'experiments/qwen_remote_v1/inputs/partitions.json').read_text())
    rows={}
    for part in PARTS:
        ids=list(map(int,parts[part]['fit'])); ymap=json.loads((PILOT/part/'fit_labels.json').read_text()); y=np.array([int(ymap[str(i)]) for i in ids])
        x=np.stack([feat[i] for i in ids]); groups=np.array([docs[i]['group'] for i in ids])
        folds=list(GroupKFold(n_splits=3,shuffle=True,random_state=cfg['seed']).split(x,y,groups)); arm_results={}
        for arm in ARMS:
            lambda_rows=[]
            for lam in LAMBDAS:
                oof=np.zeros((len(y),5)); fold_ce=[]
                for tr,va in folds:
                    sc=StandardScaler().fit(x[tr]); w,b=fit(sc.transform(x[tr]),y[tr],lam,arm,2000)
                    p=probabilities(x[va],sc,w,b)
                    if arm=='weighted_ce': p=p/class_weights(y[tr])[None,:]; p/=p.sum(1,keepdims=True)
                    oof[va]=p; fold_ce.append(float(log_loss(y[va],p,labels=[1,2,3,4,5])))
                ce=float(log_loss(y,oof,labels=[1,2,3,4,5])); read=np.isin(y,[2,3,4,5]).astype(int)
                auc=float(roc_auc_score(read,1-oof[:,0])); row={'lambda':lam,'ce':ce,'read_auc':auc,'rps':float(rps(oof,y).mean()),'fold_ce':fold_ce}
                lambda_rows.append(row)
            best=min(r['ce'] for r in lambda_rows); selected=max(r['lambda'] for r in lambda_rows if r['ce']<=best+1e-6)
            arm_results[arm]={'lambda_rows':lambda_rows,'selected_lambda':selected,'selected':next(r for r in lambda_rows if r['lambda']==selected)}
        a=arm_results['ce']['selected']; eligible=[]
        for arm in ('ordinal_rps','weighted_ce'):
            c=arm_results[arm]['selected']; improvement=a['ce']-c['ce']
            fold_gain=sum(x-y>=.01 for x,y in zip(a['fold_ce'],c['fold_ce']))
            ok=improvement>=.01 and c['read_auc']>=a['read_auc'] and fold_gain>=2
            if arm=='ordinal_rps': ok=ok and c['rps']<=a['rps']
            arm_results[arm]['eligible']=ok; arm_results[arm]['ce_improvement_vs_ce']=improvement
            arm_results[arm]['cv_folds_ce_improved']=fold_gain
            if ok: eligible.append(arm)
        selected=min(eligible,key=lambda a:arm_results[a]['selected']['ce']) if eligible else 'ce'
        out={'partition':part,'n':len(ids),'cv_folds':[[len(a),len(b)] for a,b in folds],
             'group_disjoint':all(not(set(groups[a])&set(groups[b])) for a,b in folds),
             'arms':arm_results,'selected_family_fit_only':selected,
             'gpu_candidate':selected!='ce'}
        (OUT/f'{part}.json').write_text(json.dumps(out,indent=2,allow_nan=False)); rows[part]=out
        print(json.dumps({'partition':part,'selected':selected,'candidate_CV':{a:round(arm_results[a]['selected']['ce'],6) for a in ARMS},'eligibility':{a:arm_results[a].get('eligible',False) for a in ARMS}}),flush=True)
    decision={'status':'SCREEN_COMPLETE','gpu_candidates':{p:r['selected_family_fit_only'] for p,r in rows.items()},'run_gpu':any(r['gpu_candidate'] for r in rows.values()),'outputs':str(OUT)}
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2)); print(json.dumps(decision))
if __name__=='__main__': main()
