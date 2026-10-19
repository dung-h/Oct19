#!/usr/bin/env python3
"""Audit fixed reference properties where contextual changes query-only decisions."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "diagnosis"
sys.path.insert(0, str(HERE.parent / "abstract_preference_margin"))
import apm_context as ac
import model_v3 as m


def effect(query_selected, contextual_selected, labels):
    return np.select(
        [~query_selected & contextual_selected & (labels > 1),
         query_selected & ~contextual_selected & (labels == 1),
         ~query_selected & contextual_selected & (labels == 1),
         query_selected & ~contextual_selected & (labels > 1)],
        ["contextual_rescues_positive", "contextual_removes_fp", "contextual_new_fp", "contextual_loses_positive"],
        default="unchanged")


def main():
    ctx = ac.context(); m.prepare(ctx)
    comparison = pd.read_csv(OUT / "oof_comparison.csv", dtype={"id": str})
    space = ctx["spaces"]["retrieval_margin"]
    rows = []
    for fold, fit, pred, _ in ac.outer_jobs(ctx):
        refs, audit = m.references(ctx, fit, pred)
        audit_by_query = {entry["query"]: entry for entry in audit}
        part = comparison[comparison.outer_fold.eq(fold)].set_index("row_index").loc[pred]
        labels = ctx["labels"][pred]
        effects = effect(part.query_only_selected.to_numpy(bool), part.contextual_selected.to_numpy(bool), labels)
        for offset, query in enumerate(pred):
            query_refs = refs[offset]
            ref_labels = ctx["labels"][query_refs]
            cosine = space[query_refs] @ space[query]
            item = audit_by_query[int(query)]
            rows.append(dict(
                id=str(ctx["ids"][query]), row_index=int(query), outer_fold=fold, Label=int(labels[offset]),
                title=str(ctx["data"].title.iloc[query]), venue=str(ctx["venues"][query]), effect=str(effects[offset]),
                query_only_selected=bool(part.query_only_selected.iloc[offset]),
                contextual_selected=bool(part.contextual_selected.iloc[offset]),
                query_only_read=float(part.query_only_read_probability.iloc[offset]),
                contextual_read=float(part.contextual_read_probability.iloc[offset]),
                contextual_minus_query_read=float(part.contextual_read_probability.iloc[offset] - part.query_only_read_probability.iloc[offset]),
                query_abstract_available=bool(pd.notna(ctx["data"].abstract.iloc[query]) and str(ctx["data"].abstract.iloc[query]).strip()),
                query_side_truncated=bool(ctx["raw_token_lengths"][query] > m.SIDE_TOKENS),
                negative_similarity_mean=float(cosine[:3].mean()), positive_similarity_mean=float(cosine[3:].mean()),
                similarity_gap=float(cosine[3:].mean() - cosine[:3].mean()),
                negative_same_venue_count=int((ctx["venues"][query_refs[:3]] == ctx["venues"][query]).sum()),
                positive_same_venue_count=int((ctx["venues"][query_refs[3:]] == ctx["venues"][query]).sum()),
                positive_reference_label_mean=float(ref_labels[3:].mean()), positive_reference_label_max=int(ref_labels[3:].max()),
                negative_reference_repeated=int(item["negative_repeated"]), positive_reference_repeated=int(item["positive_repeated"]),
                negative_venue_restricted=bool(item["negative_same_venue"]), positive_venue_restricted=bool(item["positive_same_venue"]),
                reference_ids="|".join(map(str, ctx["ids"][query_refs])),
                reference_titles=" || ".join(str(ctx["data"].title.iloc[r]) for r in query_refs),
            ))
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "reference_audit.csv", index=False)
    fields = ["negative_similarity_mean", "positive_similarity_mean", "similarity_gap", "negative_same_venue_count",
              "positive_same_venue_count", "positive_reference_label_mean", "positive_reference_label_max",
              "query_abstract_available", "query_side_truncated", "negative_venue_restricted", "positive_venue_restricted"]
    summary = audit.groupby("effect", sort=False).agg(n=("id", "size"), label_mean=("Label", "mean"),
        read_delta_mean=("contextual_minus_query_read", "mean"), **{field: (field, "mean") for field in fields})
    summary.to_csv(OUT / "reference_effect_summary.csv")
    cases = audit[audit.effect.ne("unchanged")].sort_values(["effect", "outer_fold", "id"]).groupby("effect", sort=False).head(12)
    cases.to_csv(OUT / "reference_case_sample.csv", index=False)
    text = ["# Contextual reference audit", "",
            "The reference rule is reconstructed from each outer fit pool. Cases are a fixed sample: first 12 by `(effect, outer_fold, id)`, not selected for favourable content.", "",
            summary.reset_index().to_markdown(index=False, floatfmt=".6f"), "",
            "`contextual_rescues_positive` and `contextual_removes_fp` are helpful changes. `contextual_new_fp` and `contextual_loses_positive` are harmful changes. Similarity is frozen retrieval cosine; `positive_*` refers to the three Label>1 references.", ""]
    (OUT / "REFERENCE_AUDIT.md").write_text("\n".join(text), encoding="utf8")
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
