#!/usr/bin/env python3
"""Describe existing holdout failures without selecting or retraining a model."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer

import run_experiment as experiment

OUT = experiment.OUT / "failure_analysis"
FAMILY = "specter2_ridge"


def evaluate_priorities(labels, priorities, ids):
    labels, priorities, ids = np.asarray(labels), np.asarray(priorities), np.asarray(ids, dtype=str)
    selected = np.flatnonzero(priorities > 0)
    order = selected[np.lexsort((ids[selected], -priorities[selected]))]
    if not len(order):
        return {"score": 0.0, "relevance": 0.0, "coverage": 0.0, "selected": 0}
    weights = 1 / np.log2(np.arange(len(order)) + 2)
    q = float(np.average((labels[order] - 1) / 4, weights=weights))
    r = float(sum(labels[order] > 1) / sum(labels > 1)) if sum(labels > 1) else 0.0
    return {"score": 2 * q * r / (q + r) if q + r else 0.0,
            "relevance": q, "coverage": r, "selected": len(order)}


def summarize(frame):
    pos, selected = frame.Label.gt(1), frame.selected
    tp, fp = int((pos & selected).sum()), int((~pos & selected).sum())
    fn, tn = int((pos & ~selected).sum()), int((~pos & ~selected).sum())
    return pd.Series({"n": len(frame), "worthwhile": int(pos.sum()), "irrelevant": int((~pos).sum()),
                      "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                      "recall": tp / (tp + fn) if tp + fn else np.nan,
                      "precision": tp / (tp + fp) if tp + fp else np.nan,
                      "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
                      "mean_prediction": frame.prediction.mean(),
                      "mean_label": frame.Label.mean(), "MAE": frame.absolute_error.mean()})


def plot_diagnostics(holdout, by_venue):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Optional matplotlib is absent; CSV diagnostics are still complete.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    labels = np.arange(1, 6)
    means = holdout.groupby("Label")[["prediction", "tfidf_ridge_prediction"]].mean()
    axes[0, 0].plot(labels, labels, "--", color="gray", label="Perfect rating prediction")
    axes[0, 0].plot(labels, means.prediction, "o-", label="SPECTER2 + Ridge")
    axes[0, 0].plot(labels, means.tfidf_ridge_prediction, "s-", label="TF-IDF + Ridge")
    axes[0, 0].axhline(2, color="red", alpha=.35, label="SPECTER2 selection threshold")
    axes[0, 0].set(xlabel="True preference label", ylabel="Mean predicted rating", xticks=labels,
                   title="A. Predictions shrink toward the middle")
    axes[0, 0].legend(fontsize=8)
    confusion = np.array([[sum(holdout.status == "TN"), sum(holdout.status == "FP")],
                          [sum(holdout.status == "FN"), sum(holdout.status == "TP")]])
    axes[0, 1].imshow(confusion, cmap="Blues")
    for (i, j), value in np.ndenumerate(confusion):
        axes[0, 1].text(j, i, str(value), ha="center", va="center", fontsize=17)
    axes[0, 1].set(xticks=[0, 1], xticklabels=["Exclude", "Recommend"], yticks=[0, 1],
                   yticklabels=["Label 1", "Labels 2-5"], title="B. Selection errors on 500 holdout items")
    x = np.arange(len(by_venue))
    axes[1, 0].bar(x - .18, by_venue.recall, .36, label="Recall (labels 2-5)")
    axes[1, 0].bar(x + .18, by_venue.false_positive_rate, .36, label="False positive rate (label 1)")
    axes[1, 0].set(xticks=x, xticklabels=by_venue.index, ylim=(0, 1.12),
                   ylabel="Rate", title="C. Errors vary by venue (different support sizes)")
    axes[1, 0].legend(fontsize=8)
    ranked = holdout[holdout.selected].sort_values("rank")
    cutoffs = [20, 50, 100, 200, len(ranked)]
    bottom = np.zeros(len(cutoffs))
    colors = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"]
    for label, color in zip(labels, colors):
        values = np.array([ranked.head(k).Label.eq(label).mean() for k in cutoffs])
        axes[1, 1].bar(np.arange(len(cutoffs)), values, bottom=bottom, color=color, label=str(label))
        bottom += values
    axes[1, 1].set(xticks=np.arange(len(cutoffs)), xticklabels=[f"Top {k}" for k in cutoffs],
                   ylabel="Fraction of selected items", title="D. True labels in the reading-list prefixes")
    axes[1, 1].legend(title="True label", fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.02))
    fig.tight_layout()
    fig.savefig(OUT / "failure_diagnostics.png", dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    data, _ = experiment.load_data()
    source = pd.read_csv(experiment.ROOT / "abstracts_collected.csv", dtype=str, keep_default_na=False)
    provenance = source[["split", "id", "abstract_source", "retrieval_confidence", "api_title", "retrieved_title"]]
    data = data.merge(provenance, on=["split", "id"], validate="one_to_one")
    tokenizer = AutoTokenizer.from_pretrained(experiment.BASE, revision=experiment.BASE_REV, local_files_only=True)
    inputs = (data.title + tokenizer.sep_token + data.abstract).tolist()
    lengths = tokenizer(inputs, truncation=False, return_token_type_ids=False)["input_ids"]
    data["input_tokens"] = [len(tokens) for tokens in lengths]
    data["truncated"] = data.input_tokens > 512
    data["latex_boilerplate_markers"] = data.abstract.str.count(r"\\documentclass|\\usepackage|\\setlength")
    data["abstract_quality"] = np.where(data.abstract_originally_missing, "missing",
                                       np.where(data.abstract_withheld, "withheld_for_review", "available"))
    raw_holdout = pd.read_csv(experiment.OUT / "holdout_predictions.csv", dtype={"id": str})
    extra = data[data.split == "train"].drop(columns=["title", "Label"])
    holdout = raw_holdout.merge(extra, on="id", validate="one_to_one")
    holdout["prediction"] = holdout[FAMILY + "_prediction"]
    holdout["selected"] = holdout[FAMILY + "_priority"] > 0
    holdout["residual"] = holdout.prediction - holdout.Label
    holdout["absolute_error"] = holdout.residual.abs()
    holdout["status"] = np.select([
        holdout.Label.gt(1) & holdout.selected, holdout.Label.eq(1) & holdout.selected,
        holdout.Label.gt(1) & ~holdout.selected], ["TP", "FP", "FN"], default="TN")
    holdout["rank"] = np.nan
    ranked = holdout[holdout.selected].sort_values([FAMILY + "_priority", "id"], ascending=[False, True])
    holdout.loc[ranked.index, "rank"] = np.arange(1, len(ranked) + 1)
    holdout["tfidf_selected"] = holdout.tfidf_ridge_priority > 0
    holdout["metadata_added_selected"] = holdout.specter2_metadata_ridge_priority > 0
    holdout["abstract_excerpt"] = holdout.abstract.str[:1200]

    # Token-budget diagnostic only: strip rendering boilerplate, not math content.
    polluted = data[data.latex_boilerplate_markers > 0].copy()
    boilerplate = (r"\\documentclass(?:\[[^\]]*\])?\{[^}]*\}|"
                   r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\}|"
                   r"\\setlength\{[^}]*\}\{[^}]*\}|\\(?:begin|end)\{document\}")
    cleaned = [re.sub(boilerplate, " ", text) for text in polluted.abstract]
    cleaned_inputs = [title + tokenizer.sep_token + text for title, text in zip(polluted.title, cleaned)]
    polluted["tokens_after_boilerplate_removal"] = [len(x) for x in tokenizer(cleaned_inputs, truncation=False)["input_ids"]]
    polluted[["split", "id", "title", "input_tokens", "latex_boilerplate_markers",
              "tokens_after_boilerplate_removal"]].to_csv(OUT / "latex_boilerplate_token_diagnostic.csv", index=False)

    summaries = {}
    for col in ["Label", "venue", "year", "abstract_quality", "truncated"]:
        result = holdout.groupby(col, observed=True)[holdout.columns].apply(summarize)
        result.to_csv(OUT / f"by_{col.lower()}.csv")
        summaries[col] = result.astype(object).where(pd.notna(result), None).reset_index().to_dict("records")
    by_venue = pd.read_csv(OUT / "by_venue.csv", index_col="venue")

    # All neighbors below are drawn from development, never from the same holdout.
    split = pd.read_csv(experiment.OUT / "validation_split.csv", dtype={"id": str})
    development = split[split.partition == "development"].reset_index(drop=True)
    cache = np.load(experiment.OUT / "embeddings.npz")
    keys = {key: i for i, key in enumerate(cache["keys"])}
    embedding = normalize(cache["embeddings"])
    dev_indices = np.array([keys["train:" + identity] for identity in development.id])
    hold_indices = np.array([keys["train:" + identity] for identity in holdout.id])
    mean = embedding[dev_indices].mean(axis=0)
    centered_dev = normalize(embedding[dev_indices] - mean)
    centered_hold = normalize(embedding[hold_indices] - mean)
    similarities = centered_hold @ centered_dev.T
    neighbors = np.argsort(-similarities, axis=1)[:, :5]
    neighbor_rows = []
    for row_index, order in enumerate(neighbors):
        holdout.loc[row_index, "nearest5_label_mean"] = pd.to_numeric(development.iloc[order].Label).mean()
        for position, neighbor_index in enumerate(order, start=1):
            neighbor = development.iloc[neighbor_index]
            neighbor_rows.append({"holdout_id": holdout.iloc[row_index].id, "neighbor_position": position,
                                  "neighbor_id": neighbor.id, "neighbor_title": neighbor.title,
                                  "neighbor_label": neighbor.Label,
                                  "centered_cosine": float(similarities[row_index, neighbor_index])})
    pd.DataFrame(neighbor_rows).to_csv(OUT / "development_neighbors.csv", index=False)
    holdout.drop(columns=["abstract", "text"]).to_csv(OUT / "holdout_error_rows.csv", index=False)
    holdout[holdout.status == "FP"].sort_values("rank").drop(columns=["abstract", "text"]).to_csv(
        OUT / "false_positives.csv", index=False)
    holdout[holdout.status == "FN"].sort_values(["Label", "prediction"], ascending=[False, True]).drop(
        columns=["abstract", "text"]).to_csv(OUT / "false_negatives.csv", index=False)
    holdout[(holdout.Label >= 4) & holdout.selected].sort_values("rank", ascending=False).drop(
        columns=["abstract", "text"]).to_csv(OUT / "high_label_low_rank.csv", index=False)

    prefix_rows = []
    for k in [10, 20, 50, 100, 200, len(ranked)]:
        prefix = ranked.head(k)
        prefix_rows.append({"k": k, "mean_label": prefix.Label.mean(),
                            **{f"label_{label}": int(prefix.Label.eq(label).sum()) for label in range(1, 6)}})
    pd.DataFrame(prefix_rows).to_csv(OUT / "rank_prefixes.csv", index=False)

    y, p, chosen, ids = (holdout.Label.to_numpy(), holdout[FAMILY + "_priority"].to_numpy(),
                         holdout.selected.to_numpy(), holdout.id.to_numpy())
    scenarios = {
        "actual": p,
        "oracle_rerank_same_selected_set": np.where(chosen, y, 0),
        "oracle_remove_selected_label1_keep_original_order": np.where(y > 1, p, 0),
        "oracle_select_all_label2_to5_keep_predicted_order": np.where(y > 1, holdout.prediction.to_numpy() + 10, 0),
        "oracle_select_all_label2_to5_order_by_true_label": np.where(y > 1, y, 0),
    }
    counterfactuals = [{"scenario": name, **evaluate_priorities(y, scores, ids)} for name, scores in scenarios.items()]
    original_score = pd.read_csv(experiment.OUT / "holdout_metrics.csv").set_index("family").loc[FAMILY, "score"]
    assert abs(counterfactuals[0]["score"] - original_score) < 1e-9
    pd.DataFrame(counterfactuals).to_csv(OUT / "oracle_diagnostics_not_model_results.csv", index=False)

    # Identify conflicting labels for exactly the input available to SPECTER2.
    labeled = data[data.split == "train"].copy()
    labeled["model_input"] = labeled.title + tokenizer.sep_token + labeled.abstract
    conflicts = []
    for _, group in labeled.groupby("model_input"):
        if len(group) > 1 and group.Label.nunique() > 1:
            conflicts.extend(group[["id", "title", "Label", "year", "doi"]].to_dict("records"))
    pd.DataFrame(conflicts).to_csv(OUT / "identical_input_conflicting_labels.csv", index=False)
    pos, tfidf = y > 1, holdout.tfidf_selected.to_numpy()
    complementary = {"both_miss_worthwhile": int((~chosen & ~tfidf & pos).sum()),
                     "specter_miss_tfidf_selects": int((~chosen & tfidf & pos).sum()),
                     "tfidf_miss_specter_selects": int((chosen & ~tfidf & pos).sum()),
                     "both_recommend_label1": int((chosen & tfidf & ~pos).sum()),
                     "specter_FP_tfidf_excludes": int((chosen & ~tfidf & ~pos).sum()),
                     "tfidf_FP_specter_excludes": int((~chosen & tfidf & ~pos).sum())}
    rng = np.random.default_rng(42)
    a, b = rng.integers(0, len(dev_indices), (2, 10000))
    random_cosines = np.sum(embedding[dev_indices[a]] * embedding[dev_indices[b]], axis=1)
    diagnostics = {"overall": summarize(holdout).to_dict(), "slices": summaries,
                   "counterfactuals": counterfactuals, "tfidf_complementarity": complementary,
                   "latex_polluted_rows": len(polluted),
                   "raw_cosine_development_random_pairs": {"p10": float(np.quantile(random_cosines, .1)),
                       "median": float(np.median(random_cosines)), "p90": float(np.quantile(random_cosines, .9))},
                   "false_negatives_by_label": holdout[holdout.status == "FN"].Label.value_counts().sort_index().to_dict(),
                   "identical_input_conflict_records": conflicts,
                   "notes": ["Analysis uses a holdout that has now been inspected; no model/submission has changed.",
                             "Oracle diagnostics use true labels and are not achievable model results.",
                             "Neighbor cosine uses development-mean-centered L2 embeddings for interpretation only; original model unchanged."]}
    (OUT / "summary.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    plot_diagnostics(holdout, by_venue)
    print(json.dumps({"overall": diagnostics["overall"], "counterfactuals": counterfactuals,
                      "tfidf_complementarity": complementary}, indent=2))
    print(f"Saved failure analysis to {OUT}")


if __name__ == "__main__":
    main()
