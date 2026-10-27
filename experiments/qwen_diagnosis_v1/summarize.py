"""Combine saved CPU analysis and fit probes; no additional GPU work."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze import fingerprint, load_inputs, metrics, write_csv, write_json


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--diagnosis-output',type=Path,required=True)
    args=ap.parse_args()
    root=args.diagnosis_output.resolve()
    dest=root/'summary_v1'
    if dest.exists():
        raise RuntimeError('Summary exists; retain artifacts and use a fresh output')
    probe=root/'fit_probe_v2_expandable'
    summaries=json.loads((probe/'summary.json').read_text())
    manifest=json.loads((probe/'protocol_lock.json').read_text())
    assert manifest['source_fingerprint']==fingerprint()
    assert json.loads((probe/'status.json').read_text())['status']=='DIAGNOSTIC_COMPLETE'
    docs,targets,parts,_=load_inputs()
    derived=[]
    for tag,summary in summaries.items():
        records=[json.loads(s) for s in (probe/tag/'probabilities.jsonl').read_text().splitlines()]
        trained={r['index']:r for r in records if r['arm']=='trained'}
        assert len(trained)==128 and set(trained)<=set(parts[tag]['fit'])
        assert set(trained).isdisjoint(parts[tag]['pred'])
        full_indices=list(trained)
        groups={'fit_full128':full_indices,
                'fit_abstract_present':[i for i in full_indices if docs[i]['abstract'].strip()]}
        for name,indices in groups.items():
            m=metrics(np.array([targets[i] for i in indices]),np.array([trained[i]['probabilities'] for i in indices]))
            derived.append(dict(partition=tag,probe=name,**m))
        ablations=[json.loads(s) for s in (probe/tag/'sensitivity.jsonl').read_text().splitlines()]
        for mode in ('title_only','abstract_only'):
            rows=[r for r in ablations if r['mode']==mode]
            assert len(rows)==32
            m=metrics(np.array([targets[r['index']] for r in rows]),np.array([r['probabilities'] for r in rows]))
            derived.append(dict(partition=tag,probe='fit_'+mode+'32',**m))
        indices=[r['index'] for r in ablations if r['mode']=='title_only']
        m=metrics(np.array([targets[i] for i in indices]),np.array([trained[i]['probabilities'] for i in indices]))
        derived.append(dict(partition=tag,probe='fit_full_same32',**m))
    refs=list(csv.DictReader((root/'references_v2/reference_metrics.csv').open()))
    losses=list(csv.DictReader((root/'references_v2/loss_decomposition.csv').open()))
    means={arm:{k:float(np.mean([float(r[k]) for r in refs if r['arm']==arm]))
                for k in ('rating_logloss','read_AUC','read_Brier','H','Q','R')}
           for arm in ('fit_prior','fit_missingness_prior','incumbent','frozen','trained')}
    loss_means={arm:{k:float(np.mean([float(r[k]) for r in losses if r['arm']==arm]))
                     for k in ('binary_read_logloss','positive_rating_logloss')}
                for arm in ('fit_prior','incumbent','trained')}
    improvement=means['fit_prior']['rating_logloss']-means['trained']['rating_logloss']
    binary_improvement=loss_means['fit_prior']['binary_read_logloss']-loss_means['trained']['binary_read_logloss']
    result=dict(source_fingerprint_unchanged=True,heldout_reference_means=means,
        heldout_CE_decomposition_means=loss_means,
        gain_vs_fit_prior_logloss=improvement,
        binary_read_share_of_gain_vs_prior=binary_improvement/improvement,
        saved_fit_parity_max=max(r['fit_saved_parity_max'] for r in summaries.values()),
        fit_gradient_checks={tag:r['gradient_probe'] for tag,r in summaries.items()},
        fit_probe_optimizer_steps=0,
        interpretation='Weak learned content association; most frozen-to-trained CE improvement is prior correction. Selection discrimination and ordinal discrimination remain weaker than incumbent. Calibration alone is insufficient to explain matched-budget failure.',
        quality_repair_proven=False)
    dest.mkdir()
    write_csv(dest/'derived_fit_metrics.csv',derived)
    write_json(dest/'findings.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
