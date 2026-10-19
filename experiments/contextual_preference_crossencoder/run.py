#!/usr/bin/env python3
"""Lock cross-encoder input contract and run its non-evaluative preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "abstract_preference_margin"))
import apm_context as ac
import model

OUT = HERE / "outputs"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol(ctx, input_stats):
    # Context is built from the cleaned immutable frame; serialising this exact frame
    # avoids assuming where an upstream experiment happened to keep its raw CSV.
    data_sha256 = hashlib.sha256(ctx["data"].to_csv(index=False).encode()).hexdigest()
    value = dict(plan_sha256=digest(HERE / "PLAN.md"), source_sha256={name: digest(HERE / name) for name in ["model.py", "run.py"]},
                 data_sha256=data_sha256, split_sha256=digest(ac.split_path()), seed=model.SEED,
                 base=model.old.BASE, base_revision=model.old.BASE_REV, adapter=model.old.ADAPTER, adapter_revision=model.old.ADAPTER_REV,
                 reference="3 nearest label1 + 3 nearest label>1; fit only; exclude publication group; venue only if >=3; retrieval_margin cosine then id",
                 tokens="separate title+abstract 126 wordpieces each, [CLS] query [SEP] reference [SEP], cap255", input_stats=input_stats,
                 contextual_loss="CE5 + .5 BCE(read) + .25 BCE(query_label > reference_label; ties omitted)",
                 control_loss="CE5 + .5 BCE(read)", optimizer="AdamW head .001 adapter .00005 wd .01 clip1; query batch4; 2 epochs",
                 evaluation="5 outer x4 inner grouped CV; threshold 0..0.9 step .05 selected by mean inner H")
    path = OUT / "protocol.json"
    if path.exists():
        # A failed setup before any preflight result is not an experimental run;
        # permit correcting its code/input manifest once. Afterwards it is immutable.
        if json.loads(path.read_text()) != value and (OUT / "preflight.json").exists():
            raise RuntimeError("Locked protocol/source/input has changed; create a new experiment directory.")
    path.write_text(json.dumps(value, indent=2) + "\n")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "preflight"], nargs="?", default="preflight")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    ctx = ac.context()
    input_stats = model.prepare(ctx)
    protocol(ctx, input_stats)
    (OUT / "input_stats.json").write_text(json.dumps(input_stats, indent=2) + "\n")
    if args.stage == "prepare":
        print(json.dumps(input_stats, indent=2)); return
    fold, fit, _, nums = ac.outer_jobs(ctx)[0]
    calibration = fit[nums == 0][:model.QUERY_BATCH]
    outcome = model.preflight(ctx, fit, calibration)
    outcome.update(outer_fold=int(fold), calibration_inner_fold=0,
                   protocol_sha256=digest(OUT / "protocol.json"))
    (OUT / "preflight.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
