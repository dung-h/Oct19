#!/usr/bin/env python3
"""Post-hoc descriptive breakdown for the completed graph ablation."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def counts(frame: pd.DataFrame, priority: str) -> dict:
    selected = frame[priority].gt(0)
    positive = frame.Label.gt(1)
    return dict(n=len(frame), selected=int(selected.sum()), TP=int((selected & positive).sum()),
                FP=int((selected & ~positive).sum()), FN=int((~selected & positive).sum()),
                missed_label4=int(((frame.Label == 4) & ~selected).sum()),
                missed_label5=int(((frame.Label == 5) & ~selected).sum()))


def main() -> None:
    oof = pd.read_csv(OUT / "oof_predictions.csv", dtype={"id": str})
    rows = []
    for scope, part in [("all_oof", oof), ("graph_covered", oof[oof.graph_covered]), ("fallback_only", oof[~oof.graph_covered])]:
        for family, priority in [("incumbent", "incumbent_priority"), ("graph", "graph_priority")]:
            rows.append(dict(scope=scope, family=family, **counts(part, priority)))
    selection = pd.DataFrame(rows)
    selection.to_csv(OUT / "selection_by_coverage.csv", index=False)

    covered_by_fold = oof.groupby("outer_fold", sort=True).graph_covered.agg(graph_covered="sum", total="size").reset_index()
    covered_by_fold["coverage_fraction"] = covered_by_fold.graph_covered / covered_by_fold.total
    covered_by_fold.to_csv(OUT / "coverage_by_fold.csv", index=False)

    summary = pd.read_csv(OUT / "summary.csv")
    decision = json.loads((OUT / "decision.json").read_text())
    transitions = pd.read_csv(OUT / "transitions.csv").iloc[0].to_dict()
    text = "# Bibliographic-coupling ablation\n\n"
    text += summary.to_markdown(index=False, floatfmt=".6f") + "\n\n"
    text += "## Selection by graph availability\n\n"
    text += selection.to_markdown(index=False) + "\n\n"
    text += "`fallback_only` is exactly the incumbent probability by construction; identical counts verify the fallback.\n\n"
    text += "## Coverage by outer fold\n\n" + covered_by_fold.to_markdown(index=False, floatfmt=".4f") + "\n\n"
    text += "## Decision\n\n```json\n" + json.dumps(dict(decision=decision, transitions=transitions), indent=2) + "\n```\n"
    text += "\nThe incumbent comparator uses the graph arm's threshold selected from its own four inner folds, so each outer comparison holds the selection policy fixed. The known global incumbent score under its separately selected policy is reported elsewhere and is not substituted here.\n"
    (OUT / "RESULTS.md").write_text(text, encoding="utf8")
    print(selection.to_string(index=False))


if __name__ == "__main__":
    main()
