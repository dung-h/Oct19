#!/usr/bin/env python3
"""Secondary exploratory probability blend; no additional model fitting."""
import json
from pathlib import Path

from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,roc_auc_score
from threadpoolctl import threadpool_limits

import modeling as m
import run_preference as run

WEIGHTS=[0.,.25,.5,.75,1.]


def select(ctx,fit,nums,incumbent,neural):
    candidates=[]
    y=ctx['y'][fit];ids=ctx['data'].id.iloc[fit].to_numpy()
    for weight in WEIGHTS:
        p=(1-weight)*incumbent+weight*neural
        for threshold in m.THRESHOLDS:
            priority=m.priority(p,threshold)
            h=np.mean([run.official(y[nums==k],priority[nums==k],ids[nums==k])['score'] for k in range(4)])
            candidates.append(dict(weight=float(weight),threshold=float(threshold),inner_score=float(h)))
    return max(candidates,key=lambda r:r['inner_score']).copy(),candidates


def inner_values(fit,nums,prefix,choice):
    inc=np.full((len(fit),5),np.nan);neural=np.full_like(inc,np.nan)
    for k in range(4):
        mask=nums==k;a=fit[~mask];b=fit[mask]
        inc[mask],_=run.load_incumbent(f'{prefix}_inner{k}',a,b)
        z=np.load(run.OUT/f"{prefix}_inner{k}_{choice['family']}.npz")
        assert np.array_equal(z['fit'],a) and np.array_equal(z['pred'],b)
        neural[mask]=z[f"epoch{choice['epoch']}"]
    return inc,neural


def independent_choice(ctx,fit,nums,inc,neural):
    rows=[]
    for weight in WEIGHTS:
        values=inc*(1-weight)+neural*weight
        gain=values@np.array([0.,.25,.5,.75,1.])
        for cutoff in m.THRESHOLDS:
            priorities=np.where(1-values[:,0]>cutoff,gain*10,0.)
            scores=[]
            for k in range(4):
                ix=fit[nums==k]
                scores.append(run.ab.evaluate(ctx['y'][ix],priorities[nums==k],ctx['data'].id.iloc[ix])['score'])
            rows.append(dict(weight=weight,threshold=float(cutoff),inner_score=float(np.mean(scores))))
    return max(rows,key=lambda r:r['inner_score'])


def main():
    with threadpool_limits(4):
        ctx=run.context();run.protocol(ctx)
        out=run.OUT/'blend';out.mkdir(exist_ok=True)
        proto=dict(plan_sha256=m.digest(run.HERE/'BLEND_PLAN.md'),source_sha256=m.digest(__file__),
                   primary_protocol_sha256=m.digest(run.OUT/'protocol.json'),weights=WEIGHTS,
                   thresholds=m.THRESHOLDS.tolist(),adaptive_research=True)
        path=out/'protocol.json'
        if path.exists():assert json.loads(path.read_text())==proto
        else:path.write_text(json.dumps(proto,indent=2)+'\n')
        oof=ctx['data'][['id','title','venue','Label']].iloc[:2494].copy()
        metrics=[];searches=[];decisions=[]
        for fold,fit,val,nums in run.outer_members(ctx):
            choice=json.loads((run.OUT/f'outer{fold}_choices.json').read_text())['primary']['neural_selection']
            inc,neural=inner_values(fit,nums,f'outer{fold}',choice)
            selected,search=select(ctx,fit,nums,inc,neural)
            assert selected==independent_choice(ctx,fit,nums,inc,neural)
            searches.extend(dict(outer_fold=fold,**r) for r in search)
            decisions.append(dict(outer_fold=fold,neural=choice,**selected))
            ip,_=run.load_incumbent(f'outer{fold}_validation',fit,val)
            z=np.load(run.OUT/f"outer{fold}_validation_{choice['family']}.npz")
            npred=z[f"epoch{choice['epoch']}"]
            probability=(1-selected['weight'])*ip+selected['weight']*npred
            priority=m.priority(probability,selected['threshold'])
            actual=run.official(ctx['y'][val],priority,ctx['data'].id.iloc[val].to_numpy())
            independent=run.ab.evaluate(ctx['y'][val],priority,ctx['data'].id.iloc[val].to_numpy())
            for k,v in independent.items():assert abs(actual[k]-v)<1e-12
            actual.update(AUC=float(roc_auc_score(ctx['y'][val]>1,1-probability[:,0])),
                          logloss=float(log_loss(ctx['y'][val]-1,probability,labels=np.arange(5))))
            metrics.append(dict(outer_fold=fold,**selected,**actual))
            oof.loc[val,'outer_fold']=fold;oof.loc[val,'priority']=priority
            for k in range(5):oof.loc[val,f'p{k+1}']=probability[:,k]
        metrics=pd.DataFrame(metrics)
        metrics.to_csv(out/'fold_metrics.csv',index=False)
        pd.DataFrame(searches).to_csv(out/'inner_search.csv',index=False)
        oof.to_csv(out/'oof_predictions.csv',index=False)
        fit=np.arange(2494);test=np.arange(2494,len(ctx['data']))
        nums=pd.read_csv(run.OUT/'final_inner_splits.csv').inner_fold.to_numpy()
        neural_choice=json.loads((run.OUT/'final_choices.json').read_text())['neural_selection']
        inc,neural=inner_values(fit,nums,'final',neural_choice)
        final,search=select(ctx,fit,nums,inc,neural)
        assert final==independent_choice(ctx,fit,nums,inc,neural)
        pd.DataFrame(search).to_csv(out/'final_search.csv',index=False)
        pack=joblib.load(run.PRIOR/'final_test_features.joblib')
        assert np.array_equal(pack['fit'],fit) and np.array_equal(pack['pred'],test)
        recipe=joblib.load(run.PRIOR/'final_recipe.joblib')
        classifier=CatBoostClassifier();classifier.load_model(str(run.PRIOR/'final_classifier.cbm'))
        assert np.array_equal(classifier.classes_,np.arange(5))
        ip=classifier.predict_proba(pack['b'][recipe['columns']])
        old_choice=json.loads((run.PRIOR/'final_choice.json').read_text())
        old_csv=pd.read_csv(run.PRIOR/'submission_candidate.csv',dtype={'id':str}).set_index('id').loc[ctx['data'].id.iloc[test]]
        np.testing.assert_allclose(old_csv.priority,m.priority(ip,old_choice['threshold']),atol=1e-10)
        z=np.load(run.OUT/'final_neural.npz')
        assert np.array_equal(z['pred'],test)
        npred=z[f"epoch{neural_choice['epoch']}"]
        probability=(1-final['weight'])*ip+final['weight']*npred
        priority=m.priority(probability,final['threshold'])
        details=pd.DataFrame({'id':ctx['data'].id.iloc[test].to_numpy(),'priority':priority})
        for k in range(5):details[f'p{k+1}']=probability[:,k]
        details.to_csv(out/'test_predictions.csv',index=False)
        submission=ctx['template'][['id']].merge(details[['id','priority']],on='id',validate='one_to_one')
        assert len(submission)==596 and submission.id.is_unique and np.isfinite(submission.priority).all() and (submission.priority>=0).all()
        submission.to_csv(out/'submission_blend_candidate.csv',index=False)
        decision=dict(outer=decisions,final=dict(neural=neural_choice,**final),
                      model_sources=dict(neural='outputs/neural_model',incumbent='../preference_augmentation/outputs/final_classifier.cbm'),
                      selected_test=int((priority>0).sum()),uploaded=False)
        (out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
        primary=pd.read_csv(run.OUT/'fold_metrics.csv')
        paired=[]
        for reference in ['incumbent','selection_procedure']:
            old=primary[primary.family.eq(reference)].set_index('outer_fold')
            for row in metrics.itertuples():
                paired.append(dict(outer_fold=row.outer_fold,reference=reference,
                              delta_H=row.score-old.loc[row.outer_fold,'score'],
                              delta_FP=int(row.FP-old.loc[row.outer_fold,'FP']),delta_FN=int(row.FN-old.loc[row.outer_fold,'FN'])))
        paired=pd.DataFrame(paired);paired.to_csv(out/'paired_effects.csv',index=False)
        summary={k:float(metrics[c].mean()) for k,c in [('H','score'),('Q','relevance'),('R','coverage'),('AUC','AUC'),('logloss','logloss')]}
        summary.update({k:int(metrics[k].sum()) for k in ['FP','FN','missed_high']})
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        # Check persisted artifacts independently of in-memory CSV order.
        stored=pd.read_csv(out/'submission_blend_candidate.csv',dtype={'id':str}).set_index('id').loc[ctx['data'].id.iloc[test]]
        np.testing.assert_allclose(stored.priority,priority,atol=1e-10)
        assert set(stored.index)==set(ctx['template'].id)
        verify=dict(status='PASSED',inner_weight_choices_recomputed=True,official_metric_crosschecked=True,
                    incumbent_model_reloaded=True,neural_model_reload='verified by primary verify.py',
                    test_rows=596,selected=int((priority>0).sum()),sha256=m.digest(out/'submission_blend_candidate.csv'))
        assert json.loads((run.OUT/'verification.json').read_text())['status']=='PASSED'
        (out/'verification.json').write_text(json.dumps(verify,indent=2)+'\n')
        text=f'''# Phối hợp SPECTER2 với CatBoost — kiểm tra bổ sung

Đây là nghiên cứu bổ sung sau khi đã xem một phần outer của primary; không là
đánh giá độc lập. Weight và cutoff ở mỗi outer vẫn chỉ chọn bằng4 inner folds.
Giữ nguyên neural arm/epoch của primary. Không huấn luyện model bổ sung.

```json
{json.dumps(summary,indent=2)}
```

{metrics.to_markdown(index=False,floatfmt='.5f')}

{paired.to_markdown(index=False,floatfmt='.5f')}

Final inner selection:

```json
{json.dumps(decision['final'],indent=2)}
```

CSV `outputs/blend/submission_blend_candidate.csv`, đủ596 ID, {int((priority>0).sum())}
priority dương. Primary CSV/model cũ giữ riêng; chưa gửi Kaggle. Kiểm tra artifact:
`outputs/blend/verification.json`. Lựa chọn cuối không dùng outer labels; các
điểm local vẫn có giới hạn do dataset được xem lặp lại trong nghiên cứu.
'''
        (run.HERE/'BLEND_RESULTS.md').write_text(text)
        print(json.dumps(summary,indent=2));print('FINAL BLEND',final)


if __name__=='__main__':main()
