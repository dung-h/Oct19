#!/usr/bin/env python3
"""LOCAL packaging only. Remote users use the committed inputs, never old caches."""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import HERE, jsonl, sha, write_json

ROOT = HERE.parents[1]


def main():
    dest = HERE / "inputs"
    if (dest / "manifest.json").exists():
        raise RuntimeError("Inputs already locked; export a new experiment version")
    dest.mkdir(parents=True, exist_ok=True)
    setup = ROOT / "experiments/llm_preference/outputs/setup"
    paths = [setup / n for n in ("documents.jsonl", "targets.jsonl", "partitions.json")]
    paths.append(ROOT / "assignment-data-mining-261.zip")
    docs = jsonl(paths[0])
    with zipfile.ZipFile(paths[-1]) as z:
        train = pd.read_csv(io.BytesIO(z.read("train.csv")), dtype=str, keep_default_na=False)
    raw = train.set_index("id").to_dict("index")
    targets = jsonl(paths[1])
    assert len(docs) == len(raw) == len(targets) == 2494
    for row in targets:
        assert row["rating"] == int(raw[row["id"]]["Label"])
    for d in docs:
        original = raw[d["id"]]
        for field in ("authors", "venue", "year", "doi"):
            d[field] = original[field]
        d["author_provenance"] = "original_dataset_unverified"
    (dest / "documents.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs))
    (dest / "targets.jsonl").write_bytes(paths[1].read_bytes())
    all_parts = json.loads(paths[2].read_text())
    config = json.loads((HERE / "config.json").read_text())
    parts = {tag: all_parts[tag] for tag in config["partitions"]}
    write_json(dest / "partitions.json", parts)
    baselines = {}
    for tag, part in parts.items():
        path = ROOT / f"experiments/preference_augmentation/outputs/{tag}.npz"
        paths.append(path)
        with np.load(path) as z:
            assert list(z["fit"]) == part["fit"] and list(z["pred"]) == part["pred"]
            baselines[tag] = dict(fit=part["fit"], pred=part["pred"], probabilities=z["both_d4"].tolist())
    write_json(dest / "baseline.json", baselines)
    write_json(dest / "manifest.json", dict(
        description="Fixed cleaned incumbent inputs; no new abstract or author repairs; train only",
        documents=2494, test_documents=0,
        sources={str(p.relative_to(ROOT)): sha(p) for p in paths},
        files={name: sha(dest / name) for name in
               ("documents.jsonl", "targets.jsonl", "partitions.json", "baseline.json")}))
    print("Portable train bundle exported; all remote dependencies are inside inputs/.")


if __name__ == "__main__":
    main()
