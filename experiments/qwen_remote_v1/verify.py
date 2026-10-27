#!/usr/bin/env python3
"""CPU verification of packaged data, metric parity and optional real tokenization."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
import csv
import io

import numpy as np

from common import HERE, load_inputs, official, prompt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--tokenizer", action="store_true", help="Requires the pinned model already downloaded")
    args = parser.parse_args()
    config = json.loads((HERE / "config.json").read_text())
    docs, targets, parts, base = load_inputs()
    assert list(parts) == config["partitions"]
    assert set(parts["outer1_inner0"]["pred"]).isdisjoint(parts["outer1_inner1"]["pred"])
    expected_sizes = {"outer1_inner0": (1494, 502), "outer1_inner1": (1497, 499)}
    report = dict(data_hashes=True, original_labels=None, baseline={}, tokenizer=None)
    zip_path = HERE.parents[1] / "assignment-data-mining-261.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            labels = {r["id"]: int(r["Label"]) for r in csv.DictReader(io.StringIO(z.read("train.csv").decode("utf-8-sig")))}
        assert {docs[i]["id"]: y for i, y in targets.items()} == labels
        report["original_labels"] = True
    # Hand-computed cases: lexical text-ID tie break, empty list and irrelevant-only selection.
    toy = official([1, 5, 2], [0, 10, 5], ["a", "b", "c"])
    expected_q = (1 + 0.25/np.log2(3)) / (1 + 1/np.log2(3))
    assert abs(toy["Q"]-expected_q) < 1e-12 and toy["R"] == 1
    assert toy["TP"] == 2 and toy["FP"] == toy["FN"] == 0
    assert official([1, 5], [0, 0], ["2", "10"])["H"] == 0
    assert official([1, 5], [1, 0], ["2", "10"])["H"] == 0
    tied = official([1, 5], [1, 1], ["2", "10"])
    assert abs(tied["Q"] - 1/(1+1/np.log2(3))) < 1e-12
    expected_h = {"outer1_inner0": 0.6487764769, "outer1_inner1": 0.6463847723}
    for tag, part in parts.items():
        assert (len(part["fit"]), len(part["pred"])) == expected_sizes[tag]
        pred = part["pred"]
        p = np.array(base[tag]["probabilities"])
        m = official([targets[i] for i in pred],
                     np.where(1-p[:, 0] > .55, 1+9*(p @ (np.arange(5)/4)), 0),
                     [docs[i]["id"] for i in pred])
        assert abs(m["H"]-expected_h[tag]) < 1e-9
        report["baseline"][tag] = m
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["id"], revision=config["model"]["revision"],
                                                   local_files_only=True)
        answer_ids = []
        for text in config["input"]["answer_strings"]:
            answer = tokenizer.encode(text, add_special_tokens=False)
            assert len(answer) == 1 and tokenizer.decode(answer) == text
            answer_ids.extend(answer)
        assert len(set(answer_ids)) == 5
        lengths = []
        for doc in docs.values():
            ids, stats = prompt(tokenizer, doc, config)
            altered = dict(doc, rating=6-targets[doc["index"]], Label="LEAK_MARKER",
                           authors="LEAK_MARKER", venue="LEAK_MARKER", year="LEAK_MARKER", id="LEAK_MARKER")
            assert prompt(tokenizer, altered, config)[0] == ids
            lengths.append(stats["tokens"])
        report["tokenizer"] = dict(documents=len(lengths), answer_ids=answer_ids,
                                    min_tokens=min(lengths), max_tokens=max(lengths),
                                    median_tokens=float(np.median(lengths)), input_mutation_invariant=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
