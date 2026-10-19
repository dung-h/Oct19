#!/usr/bin/env python3
"""Produce paired effects and honest deployment decision after complete verification."""
import json
import numpy as np
import pandas as pd
import run_augmentation as ex


def table(frame):
    return frame.to_markdown(index=False,floatfmt='.5f')


def main():
    out=ex.OUT
    verification=json.loads((out/'verification.json').read_text())
    m=pd.read_csv(out/'fold_metrics.csv');summary=pd.read_csv(out/'summary.csv')
    oof=pd.read_csv(out/'oof_predictions.csv',dtype={'id':str})
    old=pd.read_csv(ex.ab.OUT.parent/'final_outputs/workflow_oof.csv',dtype={'id':str})
    assert np.array_equal(old.id,oof.id) and np.array_equal(old.Label,oof.Label)
    oldmetrics=pd.read_csv(ex.ab.OUT.parent/'final_outputs/workflow_fold_metrics.csv').set_index('outer_fold')
    effects=[]
    for arm,reference in [('content','base'),('neighbors','base'),('both','base'),('both','content'),('both','neighbors'),
                          ('selection_procedure','ridge'),('selection_procedure','previous_workflow')]:
        a=m[m.family.eq(arm)].set_index('outer_fold')
        b=oldmetrics if reference=='previous_workflow' else m[m.family.eq(reference)].set_index('outer_fold')
        delta=a.score-b.score
        effects.append(dict(arm=arm,reference=reference,delta_H=delta.mean(),folds_improved=int((delta>0).sum()),
            delta_R=(a.coverage-b.coverage).mean(),delta_FP=int(a.FP.sum()-b.FP.sum()),
            delta_FN=int(a.FN.sum()-b.FN.sum()),delta_high_FN=int(a.missed_high.sum()-b.missed_high.sum()),
            **{f'fold{k}_delta':float(v) for k,v in delta.items()}))
    effects=pd.DataFrame(effects);effects.to_csv(out/'paired_effects.csv',index=False)
    y=oof.Label.to_numpy();oldsel=old.priority.to_numpy()>0;newsel=oof.selection_procedure_priority.to_numpy()>0
    def status(sel):return np.where(y>1,np.where(sel,'TP','FN'),np.where(sel,'FP','TN'))
    failures=oof[['id','title','venue','Label','outer_fold']].copy()
    failures['old_priority']=old.priority.to_numpy()
    failures['new_priority']=oof.selection_procedure_priority.to_numpy()
    failures['old_status']=status(oldsel);failures['new_status']=status(newsel)
    olderr=oldsel!=(y>1);newerr=newsel!=(y>1)
    failures['change']=np.select([olderr&~newerr,~olderr&newerr,olderr&newerr],['fixed','introduced','persisting'],default='correct')
    failures.to_csv(out/'failure_transitions.csv',index=False)
    breakdown=[]
    for field in ['venue','Label']:
        for value,rows in failures.groupby(field):
            breakdown.append(dict(field=field,value=value,n=len(rows),FP=int(rows.new_status.eq('FP').sum()),
                FN=int(rows.new_status.eq('FN').sum()),fixed=int(rows.change.eq('fixed').sum()),
                introduced=int(rows.change.eq('introduced').sum())))
    pd.DataFrame(breakdown).to_csv(out/'failure_breakdown.csv',index=False)
    known=failures[failures.id.isin(['217','355','1045','1184','1000','1001','2383','206'])].copy()
    priority_cols=[c for c in oof if c.endswith('_priority')]
    known=known.merge(oof[['id']+priority_cols],on='id',validate='one_to_one')
    known.to_csv(out/'known_cases.csv',index=False)
    cf=pd.read_csv(out/'content_features.csv',dtype={'id':str})
    cf=cf[cf.split.eq('train')].set_index('id')
    flags=[c for c in cf if c!='split' and not c.endswith(('_words','_unknown','_available'))]
    pair_audit=[]
    for a,b in [('217','355'),('1045','1184')]:
        different=[c for c in flags if cf.loc[a,c]!=cf.loc[b,c]]
        pair_audit.append(dict(id_a=a,id_b=b,different_content_flags=len(different),fields=';'.join(different)))
    pair_audit=pd.DataFrame(pair_audit)
    pair_audit.to_csv(out/'content_pair_audit.csv',index=False)
    effect=effects[(effects.arm=='selection_procedure')&(effects.reference=='previous_workflow')].iloc[0]
    success=bool(effect.delta_H>=.02 and effect.folds_improved>=4)
    choice=json.loads((out/'final_choice.json').read_text())
    decision=dict(success_target_met=success,delta_H_vs_previous=float(effect.delta_H),
        improved_folds=int(effect.folds_improved),candidate_choice=choice,
        recommendation='candidate eligible for further leaderboard validation' if success else 'retain previous deliverable; candidate experimental',
        previous_files_unchanged=True,kaggle_upload=False)
    (out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    cov=pd.read_csv(out/'content_coverage.csv');cv=cov[cov.split.eq('train')]
    text=f'''# Evidence-backed preference augmentation: completed experiment

## Result and decision

{decision['recommendation']}. Full inner-selected procedure delta H vs previous workflow: **{effect.delta_H:+.5f}**, improving **{int(effect.folds_improved)}/5** paired folds. The target +0.02 was a practical research stopping criterion, not an assignment requirement. Recall/high-label tradeoffs remain visible below. No Kaggle upload or previous artifact replacement.

{table(summary[['family','H','H_std','Q','R','FP','FN','missed_high','AUC']])}

Each family above independently chooses its depth and threshold using four inner folds. `selection_procedure` chooses among all arms and Ridge in those inner folds. Its outer score estimates the complete selection procedure; the retrospectively best family score is not the final candidate score. Existing outer labels were repeatedly inspected in earlier research: this is not a pristine independent benchmark or a claim about hidden Kaggle performance.

## Matched augmentation effects

{table(effects[['arm','reference','delta_H','folds_improved','delta_R','delta_FP','delta_FN','delta_high_FN']])}

All four CatBoost arms share the same learner/grid, base features and target. Content effects are measured against `base`, not only against a different Ridge learner. Neighbor effects likewise keep the classifier unchanged. Interactions are examined by both-versus-single comparisons; these are paired descriptive comparisons, not independent significance tests.

## What was augmented

No synthetic paper or preference label was created. The original 2,494 labels were preserved. Content rules record literal lexical evidence for document type, problem, method, contribution and domain, separately from title and usable abstract. Every indicator has a source field, exact text and offsets in `content_evidence.csv`; lack of a match becomes unknown, not a confirmed semantic negative. The fixed dictionary cannot resolve all subtle contribution differences between same-topic papers.

{table(cv[['field','aspect','n','with_evidence','unknown']])}

History features use SPECTER2 retrieval (not the regression adapter for nearest neighbors), title TF-IDF and abstract TF-IDF; top5/top15 globally and within venue. Features include five-label distributions, weighted gain, entropy/disagreement, support and nearest per-class similarity. Three extra grouped cross-fitting folds create every training history and Ridge score. Prediction references contain only the fit set, with the query's entire duplicate group excluded. Two groups overlap train and test; their label histories are excluded for affected test queries. The ordinary Ridge control is trained on all original fit labels.

Base features are venue, year, 32 title TF-IDF SVD coordinates and a cross-fitted title/venue Ridge rating. Target-free SVD/vocabulary are fitted inside each model training partition. History vocabularies are fitted separately on each reference subset. Training crossfit references are smaller than full-fit prediction references, a remaining distribution shift; reference counts are exposed to the learner.

## Failures and limitations

Old binary errors **{int(olderr.sum())}**, new errors **{int(newerr.sum())}**; fixed **{int((olderr&~newerr).sum())}**, introduced **{int((~olderr&newerr).sum())}**. Binary error counts supplement, rather than replace, official H.

{table(known[['id','title','Label','old_status','new_status']])}

Known examples were already inspected and are diagnostic only; they did not receive hand labels, special weights or ID-specific rules. Label disagreement between similar texts does not prove label corruption. A negative result for this limited lexical augmentation or history summary does not prove that full semantic extraction or every local preference model will fail. Conversely, positive aggregate effects cannot establish the researcher's true rationale.

Per-arm priorities are preserved in `known_cases.csv`. In this run, local history rescues ID1000 (antimicrobial prediction) and ID2383 (workflow-net soundness) in the neighbors arm, but the globally inner-selected arms still exclude them. ID217 remains a false positive across the four CatBoost arms; ID1045 remains excluded across them. This separates two issues: some useful local signals are present but not reliably retained by model selection, while other hard cases still lack useful discrimination in these augmented representations. No manual per-ID blending is used.

An extraction audit exposes a concrete limitation: IDs217/355 have **zero different content-category flags**, despite their different preference labels; the model can still distinguish their base title SVD/Ridge values, but this augmentation adds no category distinction for that pair. IDs1045/1184 differ in {int(pair_audit.iloc[1].different_content_flags)} content flags. This prevents claiming that the extractor captures the exact contribution differences needed for every hard case. See `content_pair_audit.csv`.

## Final candidate

Selected on four grouped inner folds of all training data (seed20261118), without selecting directly by outer results:

```json
{json.dumps(choice,indent=2)}
```

`outputs/submission_candidate.csv`: 596 template IDs, {verification['positive_priority']} positive priorities. Model/recipe and features are retained. The candidate is {'eligible for further validation' if success else 'experimental; retain the prior submission'} under the predeclared decision criterion.

## Verification and reproduction

Verified {verification['evidence_spans_verified']} evidence spans, {verification['grouped_crossfit_partitions_verified']} crossfit partitions, {verification['outer_metrics_recomputed']} outer metrics, all inner choices, label perturbation invariance, original labels, independently rebuilt final features and identical full ranking after model reload. Six leakage/evidence/metric unit tests also passed.

```bash
/tmp/assignment-specter2-venv/bin/python experiments/preference_augmentation/run_plan.py
```

See [PLAN.md](PLAN.md) for the pre-evaluation protocol, and [outputs/verification.json](outputs/verification.json), [outputs/paired_effects.csv](outputs/paired_effects.csv), [outputs/failure_transitions.csv](outputs/failure_transitions.csv). Classification/regression satisfy the assignment's substantive-method requirement. Report both successful and unsuccessful comparisons; use the detailed harmonic-mean Evaluation because the copied short Overview has a conflicting metric description.
'''
    (ex.HERE/'RESULTS.md').write_text(text)
    print(table(summary[['family','H','R','FP','FN','missed_high']]))
    print(json.dumps(decision,indent=2))


if __name__=='__main__':main()
