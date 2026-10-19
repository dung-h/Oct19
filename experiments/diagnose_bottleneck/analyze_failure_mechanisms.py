#!/usr/bin/env python3
"""Post-hoc diagnosis only: saved OOF predictions, oracle thought experiments, source audit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from run_diagnostics import HERE, OUT, V2, evaluate


DEST = OUT / 'failure_mechanisms'
FAMILIES = [f'{space}_{head}' for space in ['regression','retrieval','minilm','tfidf']
            for head in ['ridge','rbf']]


def main():
    DEST.mkdir(exist_ok=True)
    d = pd.read_csv(OUT/'oof_predictions.csv', dtype={'id':str})
    metrics = pd.read_csv(OUT/'fold_metrics.csv')
    abstracts = pd.read_csv(V2/'abstracts_cleaned.csv', dtype={'id':str})
    d = d.merge(abstracts[abstracts.split.eq('train')][['id','abstract_usable_v2']],
                on='id', validate='one_to_one')
    assert len(d) == 2494 and d.id.is_unique
    positive = d.Label.gt(1)
    d['positive'] = positive
    d['selected'] = d.regression_ridge_priority.gt(0)
    d['FP'] = d.selected & ~positive
    d['FN'] = ~d.selected & positive
    errors = pd.DataFrame({family:d[family+'_priority'].gt(0).ne(positive) for family in FAMILIES})
    d['wrong_model_count'] = errors.sum(axis=1)
    d['all_eight_wrong'] = errors.all(axis=1)
    d['rank_in_fold'] = pd.Series(pd.NA,index=d.index,dtype='Int64')
    d['threshold_rating'] = np.nan
    scenarios, venue_auc, top_rows = [], [], []
    for fold, group in d.groupby('outer_fold'):
        y = group.Label.to_numpy()
        ids = group.id.to_numpy()
        r = group.regression_ridge_rating.to_numpy()
        pr = group.regression_ridge_priority.to_numpy()
        selected = pr > 0
        cutoff = 1+4*metrics[(metrics.outer_fold==fold)&metrics.family.eq('regression_ridge')].threshold.iloc[0]
        d.loc[group.index,'threshold_rating'] = cutoff
        order = np.lexsort((ids, -r))
        weight = 1/np.log2(np.arange(len(order))+2)
        q = np.cumsum(weight*(y[order]-1)/4)/np.cumsum(weight)
        recall = np.cumsum(y[order]>1)/(y>1).sum()
        h = 2*q*recall/np.maximum(q+recall,1e-15)
        # A scalar threshold cannot split equal ratings; consider only tie-group endpoints.
        ends = np.r_[r[order][:-1] != r[order][1:], True]
        allowed = np.flatnonzero(ends)
        best = allowed[np.argmax(h[allowed])]
        chosen_order = np.flatnonzero(selected)
        chosen_order = chosen_order[np.lexsort((ids[chosen_order],-pr[chosen_order]))]
        d.loc[group.index[chosen_order],'rank_in_fold'] = range(1,len(chosen_order)+1)
        for k in [20,50,100]:
            top_rows.append(dict(outer_fold=fold, k=k, FP=int((y[chosen_order[:k]]==1).sum())))
        scenario = dict(
            outer_fold=fold,
            actual=evaluate(y,pr,ids)['score'],
            oracle_threshold_fixed_order=float(h[best]),
            oracle_rerank_same_selection=evaluate(y,np.where(selected,y,0),ids)['score'],
            oracle_remove_selected_label1=evaluate(y,np.where(selected&(y>1),pr,0),ids)['score'],
            oracle_all_positive_model_order=evaluate(y,np.where(y>1,r-r.min()+1,0),ids)['score'],
            oracle_all_positive_true_order=evaluate(y,np.where(y>1,y,0),ids)['score'],
            actual_selected=int(selected.sum()), oracle_prefix_size=int(best+1))
        assert scenario['oracle_threshold_fixed_order'] >= scenario['actual']-1e-10
        assert scenario['oracle_rerank_same_selection'] >= scenario['actual']-1e-10
        scenarios.append(scenario)
        for venue, subset in group.groupby('venue'):
            venue_auc.append(dict(outer_fold=fold, venue=venue, n=len(subset),
                                   auc=roc_auc_score(subset.Label.gt(1),subset.regression_ridge_rating)))
    d['rating_margin'] = d.regression_ridge_rating-d.threshold_rating
    d['within_0_2_of_threshold'] = d.rating_margin.abs().le(.2)
    d.to_csv(DEST/'all_rows.csv',index=False)
    d[d.all_eight_wrong].to_csv(DEST/'common_280_errors.csv',index=False)
    oracle = pd.DataFrame(scenarios)
    oracle.to_csv(DEST/'oracle_diagnostics_by_fold.csv',index=False)
    pd.DataFrame(top_rows).to_csv(DEST/'top_k_false_positives.csv',index=False)
    venue = d.groupby('venue').agg(n=('id','size'),positive=('positive','sum'),FP=('FP','sum'),FN=('FN','sum'))
    venue['FPR'] = venue.FP/(venue.n-venue.positive)
    venue['FNR'] = venue.FN/venue.positive
    venue['within_venue_mean_fold_auc'] = pd.DataFrame(venue_auc).groupby('venue').auc.mean()
    venue.to_csv(DEST/'by_venue.csv')
    label = d.groupby('Label').agg(n=('id','size'),FP=('FP','sum'),FN=('FN','sum'),
                                 all_eight_wrong=('all_eight_wrong','sum'))
    label.to_csv(DEST/'by_label.csv')
    quality = d.groupby('abstract_usable_v2').agg(n=('id','size'),FP=('FP','sum'),FN=('FN','sum'))
    quality.to_csv(DEST/'by_abstract_availability.csv')
    shared_venue = d[d.all_eight_wrong].groupby('venue').agg(n=('id','size'),FP=('FP','sum'),FN=('FN','sum'))
    shared_venue.to_csv(DEST/'common_errors_by_venue.csv')
    agreement = []
    for i,left in enumerate(FAMILIES):
        for right in FAMILIES[i+1:]:
            agreement.append(dict(left=left,right=right,
                error_jaccard=float((errors[left]&errors[right]).sum()/(errors[left]|errors[right]).sum())))
    pd.DataFrame(agreement).to_csv(DEST/'model_error_overlap.csv',index=False)
    summary = dict(
        baseline='diagnose_bottleneck regression_ridge, 2494 outer OOF rows',
        n=len(d), FP=int(d.FP.sum()), FN=int(d.FN.sum()),
        shared_errors=int(d.all_eight_wrong.sum()),
        shared_FP=int((d.all_eight_wrong&d.FP).sum()), shared_FN=int((d.all_eight_wrong&d.FN).sum()),
        FP_within_0_2=int((d.FP&d.within_0_2_of_threshold).sum()),
        FN_within_0_2=int((d.FN&d.within_0_2_of_threshold).sum()),
        mean_error_jaccard=float(pd.DataFrame(agreement).error_jaccard.mean()),
        oracle_means=oracle.drop(columns=['outer_fold','actual_selected','oracle_prefix_size']).mean().to_dict(),
        warning='All oracle scenarios use validation truth for diagnosis only, not model selection, expected improvement or deployable predictions. Scenario gains are not additive. True-order-all-positive is not a proven global optimum for H.')
    assert (summary['FP'],summary['FN'],summary['shared_errors']) == (414,293,280)
    assert int(quality.loc[0,'FN']) == 0
    assert int(shared_venue.FN.sum()) == 58
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    # Human source checks identify bibliographic discrepancies, never change target labels.
    audit = [
        dict(id='217',field='authors',dataset_value='SaÚl FernÁndez GonzÁlez',
             verified_value='Stefania Costantini; Andrea Formisano',
             source='https://ceur-ws.org/Vol-3193/paper1EELP.pdf',
             finding='Publisher PDF authors disagree with dataset. Collected abstract matches the title and publisher abstract; not evidence of a wrong abstract or wrong Label.'),
        dict(id='372',field='authors',dataset_value='Yong Shi',
             verified_value='Nicos Angelopoulos; Jan Wielemaker',
             source='https://arxiv.org/abs/1909.08254',
             finding='Author-deposited paper authors disagree with dataset. Collected abstract matches the title and source abstract; not evidence of a wrong abstract or wrong Label.'),
    ]
    pd.DataFrame(audit).to_csv(DEST/'verified_metadata_discrepancies.csv',index=False)
    print(json.dumps(summary,indent=2))
    print(venue.round(4).to_string())


if __name__ == '__main__':
    main()
