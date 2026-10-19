"""Protocol lock, source audits and resumable outer/final CatBoost runs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from apm_context import (ARMS, ENCODER, ENCODER_REVISION, EXPERIMENTS, HERE,
                         N_FIT, N_TOTAL, NEW_ARMS, OUT, PRIMARY, PROJECT,
                         READ_THRESHOLD, SEED, context, digest, final_pack_path,
                         load_source_pack, outer_jobs, source,
                         source_probabilities, split_path)
from apm_features import build_partition_features, feature_names
from apm_metrics import official_metrics, priorities

SOURCE_FILES = [
    "PLAN.md", "apm_context.py", "apm_features.py", "apm_metrics.py",
    "apm_runner.py", "apm_report.py", "apm_verify.py", "run.py",
    "report.py", "verify.py", "test_apm_features.py",
]
DEPENDENCY_FILES = [
    EXPERIMENTS / "read_gate_repair" / "gate_lib.py",
    EXPERIMENTS / "specter2_preference" / "run_preference.py",
    EXPERIMENTS / "specter2_preference" / "modeling.py",
    EXPERIMENTS / "input_ablation" / "run_ablation.py",
    EXPERIMENTS / "diagnose_bottleneck" / "run_diagnostics.py",
    EXPERIMENTS / "specter2" / "run_experiment.py",
    EXPERIMENTS / "specter2_v2" / "prepare_data.py",
    EXPERIMENTS / "semantic_roles" / "roles.py",
]


def partition_jobs(ctx):
    return [(f"outer{fold}_validation", fit, pred, fold)
            for fold, fit, pred, _ in outer_jobs(ctx)]


def input_paths(ctx):
    paths = [
        PROJECT / "assignment-data-mining-261.zip",
        EXPERIMENTS / "specter2_v2" / "outputs" / "abstracts_cleaned.csv",
        EXPERIMENTS / "specter2_v2" / "outputs" / "nested_splits.csv",
        source.INC / "final_inner_splits.csv",
        source.INC / "submission_candidate.csv",
        source.INC / "test_prediction_details.csv",
        EXPERIMENTS / "semantic_roles" / "outputs" / "texts.csv",
        ctx["mini_embedding_path"], ctx["mini_manifest_path"],
        ctx["retrieval_embedding_path"], ctx["retrieval_manifest_path"],
        final_pack_path(),
    ]
    for name, _, _, fold in partition_jobs(ctx):
        paths.append(source.pack_path(name))
        paths.append(source.CLAIM / f"outer{fold}_validation_predictions.npz")
    paths.append(source.pack_path("final_test"))
    return paths


def write_protocol(ctx):
    sources = [HERE / name for name in SOURCE_FILES] + DEPENDENCY_FILES
    inputs = input_paths(ctx)
    assert all(path.is_file() for path in sources + inputs)
    data = ctx["data"]
    protocol = dict(
        plan_sha256=digest(HERE / "PLAN.md"),
        sources={str(path.relative_to(PROJECT)): digest(path) for path in sources},
        inputs={str(path.relative_to(PROJECT)): digest(path) for path in inputs},
        data_sha256=hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
        label_sha256=hashlib.sha256(ctx["labels"].astype(np.int64).tobytes()).hexdigest(),
        groups_sha256=hashlib.sha256(ctx["groups"].astype(np.int64).tobytes()).hexdigest(),
        abstract_available_sha256=hashlib.sha256(ctx["abstract_available"].tobytes()).hexdigest(),
        abstract_truncated_sha256=hashlib.sha256(ctx["abstract_truncated"].tobytes()).hexdigest(),
        feature_names=feature_names(),
        arms=ARMS, primary=PRIMARY, fixed_gate_strictly_greater_than=READ_THRESHOLD,
        feature_spec=dict(scopes=["all", "venue"], groups={"N":[1],"P":[2,3,4,5],"G":[4,5]},
                          per_class=["max","mean5","mean15","support15","present"],
                          margins=["PminusN","GminusN"], total=43),
        model=dict(loss_function="MultiClass", classes_count=5, iterations=300,
                   depth=4, learning_rate=0.05, l2_leaf_reg=8, seed=SEED,
                   thread_count=4, categorical=["venue"], target="Label-1"),
        prediction_policy="three matched source-history reference views; mean probabilities; use incumbent vector when query abstract unavailable",
        ranking="selected iff P(Label>1)>0.55; priority=1+9*incumbent expected gain; otherwise0; tie by string ID",
        embeddings=dict(minilm_model=ENCODER, minilm_revision=ENCODER_REVISION,
                        specter2_manifest_sha256=digest(ctx["retrieval_manifest_path"]),
                        minilm_manifest_sha256=digest(ctx["mini_manifest_path"])),
        versions={name: importlib.metadata.version(name) for name in
                  ["numpy","pandas","scikit-learn","catboost","joblib"]},
        outer_folds=5, history_crossfit_folds=3, max_fits=24,
        limitations=["outer folds have been inspected in earlier experiments",
                     "test labels are unavailable", "semantic-neighbor scores are not probabilities of preference"],
        uploaded=False,
    )
    path = OUT / "protocol.json"
    if path.exists():
        old = json.loads(path.read_text())
        assert old == protocol, "Locked protocol/source/input changed; use a documented amendment/new output directory"
    else:
        path.write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def audit_source_artifacts(ctx):
    assert ctx["data"].id.astype(str).is_unique
    assert len(ctx["labels"]) == N_FIT and set(np.unique(ctx["labels"])) == {1,2,3,4,5}
    assert ctx["abstract_available"][:N_FIT].sum() == 2396
    assert ctx["abstract_available"][N_FIT:].sum() == 578
    assert ctx["abstract_truncated"].sum() == 12
    rows = []
    for name, fit, pred, fold in partition_jobs(ctx):
        pack, a, b = load_source_pack(name, fit, pred)
        assert list(a.columns) == list(b.columns) and len(a.columns) == 277
        assert a.columns[0] == "venue"
        assert np.isfinite(a.drop(columns="venue").to_numpy(float)).all()
        assert np.isfinite(b.drop(columns="venue").to_numpy(float)).all()
        assert set(ctx["groups"][fit]).isdisjoint(set(ctx["groups"][pred])) or name == "final_test"
        seen = []
        for entry in pack["lineage"]:
            refs, queries = np.asarray(entry["refs"], int), np.asarray(entry["queries"], int)
            assert set(refs).issubset(set(fit)) and set(queries).issubset(set(fit))
            assert not (set(ctx["groups"][refs]) & set(ctx["groups"][queries]))
            seen.extend(queries.tolist())
        assert sorted(seen) == sorted(np.asarray(fit, int).tolist())
        base = source_probabilities(name, fit, pred, ctx)
        assert base.shape == (len(pred),5) and np.isfinite(base).all()
        np.testing.assert_allclose(base.sum(axis=1), 1.0, atol=1e-6)
        rows.append(dict(partition=name, n_fit=len(fit), n_pred=len(pred), n_features=len(a.columns),
                         crossfit_views=len(pack["lineage"]), baseline_probability_rows=len(base),
                         duplicate_groups_overlapping=int(len(set(ctx["groups"][fit]) & set(ctx["groups"][pred])))))
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "data_audit.csv", index=False)
    pd.DataFrame({"feature":feature_names(), "index":np.arange(43)}).to_csv(OUT/"feature_catalog.csv",index=False)
    return audit


def audit(ctx):
    OUT.mkdir(parents=True, exist_ok=True)
    # Run the independent hand-calculated edge-case check before locking hashes.
    subprocess.run([sys.executable, str(HERE / "test_apm_features.py")], cwd=HERE, check=True)
    audit_source_artifacts(ctx)
    protocol = write_protocol(ctx)
    summary = dict(status="PASSED", partitions=6, abstract_train=int(ctx["abstract_available"][:N_FIT].sum()),
                   abstract_test=int(ctx["abstract_available"][N_FIT:].sum()),
                   truncated_abstracts=int(ctx["abstract_truncated"].sum()),
                   source_features=277, new_features=43, augmented_features=320,
                   protocol_sha256=digest(OUT/"protocol.json"), incumbent_preserved=True, uploaded=False)
    (OUT / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return protocol


def estimator():
    return CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=8,
                              loss_function="MultiClass", classes_count=5, random_seed=SEED,
                              thread_count=4, verbose=False, allow_writing_files=False)


def add_features(base, values):
    result = base.copy().reset_index(drop=True)
    values = np.asarray(values, dtype=np.float32)
    assert values.shape == (len(result),43) and np.isfinite(values).all()
    for j, name in enumerate(feature_names()):
        if name in result.columns:
            raise AssertionError(f"new feature collides with source feature: {name}")
        result[name] = values[:,j]
    return result


def _prediction_cache(name, fit, pred, ctx, protocol_sha):
    cache = OUT / f"{name}_predictions.npz"
    if not cache.exists():
        return None
    with np.load(cache) as z:
        assert str(z["protocol_sha256"].item()) == protocol_sha
        np.testing.assert_array_equal(z["fit"], fit)
        np.testing.assert_array_equal(z["pred"], pred)
        result = {arm:z[arm].copy() for arm in ARMS}
    for arm, probabilities in result.items():
        assert probabilities.shape == (len(pred),5) and np.isfinite(probabilities).all()
        if arm == "reference":
            source_p = source_probabilities(name, fit, pred, ctx)
            np.testing.assert_array_equal(probabilities, source_p)
    return result


def run_partition(ctx, name, fit, pred, fold=None):
    protocol = write_protocol(ctx)
    protocol_sha = digest(OUT/"protocol.json")
    cached = _prediction_cache(name, fit, pred, ctx, protocol_sha)
    if cached is not None:
        print(name, "already complete", flush=True)
        return cached

    pack, x_fit, x_pred = load_source_pack(name, fit, pred)
    margin = build_partition_features(ctx, name, fit, pred, pack, protocol_sha)
    p_source = source_probabilities(name, fit, pred, ctx)
    y_fit = ctx["labels"][np.asarray(fit, int)] - 1
    predictions = {"reference":p_source}
    start = time.monotonic()
    reference_model = estimator()
    reference_model.fit(x_fit, y_fit, cat_features=["venue"])
    replay = reference_model.predict_proba(x_pred)
    max_diff = float(np.max(np.abs(replay - p_source)))
    if not np.allclose(replay, p_source, atol=1e-7, rtol=1e-6):
        raise AssertionError(f"{name}: CatBoost reference replay differs by {max_diff}")
    reference_model.save_model(str(OUT / f"{name}_reference.cbm"))
    print(name, "reference parity", max_diff, "seconds", round(time.monotonic()-start,1), flush=True)

    train_views = margin["train_new"]
    pred_views = margin["pred_new"]
    abstract_ok = ctx["abstract_available"][np.asarray(pred, int)]
    for arm_index, arm in enumerate(NEW_ARMS):
        train_frame = add_features(x_fit, train_views[arm_index])
        model = estimator()
        model.fit(train_frame, y_fit, cat_features=["venue"])
        per_view = []
        for view in range(pred_views.shape[1]):
            pred_frame = add_features(x_pred, pred_views[arm_index,view])
            per_view.append(model.predict_proba(pred_frame))
        probabilities = np.mean(per_view, axis=0)
        # Missing-query fallback exactly follows the incumbent source probability.
        probabilities[~abstract_ok] = p_source[~abstract_ok]
        if not np.isfinite(probabilities).all():
            raise AssertionError(f"{name}/{arm}: non-finite prediction")
        if not np.allclose(probabilities.sum(axis=1),1.0,atol=1e-6):
            raise AssertionError(f"{name}/{arm}: probability rows do not sum to one")
        predictions[arm] = probabilities
        model.save_model(str(OUT / f"{name}_{arm}.cbm"))
        print(name, arm, "fit", len(fit), "predict", len(pred),
              "seconds", round(time.monotonic()-start,1), flush=True)

    np.savez_compressed(OUT/f"{name}_predictions.npz", protocol_sha256=np.asarray(protocol_sha),
                        fit=np.asarray(fit,int),pred=np.asarray(pred,int),**predictions)
    pd.DataFrame(margin["pred_support"]).to_csv(OUT/f"{name}_support.csv",index=False)
    pd.DataFrame(margin["pred_evidence"]).to_csv(OUT/f"{name}_neighbors.csv",index=False)
    return predictions


def run_outer(ctx):
    protocol = write_protocol(ctx)
    protocol_sha = digest(OUT/"protocol.json")
    predictions = {}
    for fold, fit, pred, nums in outer_jobs(ctx):
        assert set(ctx["groups"][fit]).isdisjoint(set(ctx["groups"][pred]))
        name = f"outer{fold}_validation"
        predictions[fold] = run_partition(ctx,name,fit,pred,fold)
    (OUT/"outer_run.json").write_text(json.dumps(dict(status="COMPLETE",folds=5,
        arms=ARMS,protocol_sha256=protocol_sha,uploaded=False),indent=2)+"\n")
    return predictions


def run_final(ctx):
    decision_path = OUT/"decision.json"
    if not decision_path.is_file() or json.loads(decision_path.read_text()).get("status") != "promising":
        raise RuntimeError("Final refit is conditional on a verified primary passing the locked outer screening rule")
    fit = np.arange(N_FIT,dtype=int)
    pred = np.arange(N_FIT,N_TOTAL,dtype=int)
    name = "final_test"
    predictions = run_partition(ctx,name,fit,pred)
    template = ctx["template"][["id"]].copy()
    source_gain = predictions["reference"] @ (np.arange(5,dtype=float)/4)
    rows = []
    for arm in ARMS:
        read = 1.0-predictions[arm][:,0]
        selected = read > READ_THRESHOLD
        priority = np.where(selected,1.0+9.0*source_gain,0.0)
        details = pd.DataFrame(dict(id=ctx["ids"][pred], priority=priority,
                                    read_probability=read,selected=selected))
        template.merge(details[["id","priority"]],on="id",validate="one_to_one").to_csv(
            OUT/f"submission_{arm}.csv",index=False)
        rows.append(dict(arm=arm,accepted=int(selected.sum()),dropped=int((~selected).sum()),
                         changed_from_incumbent=int((selected != (1-predictions["reference"][:,0]>.55)).sum())))
    pd.DataFrame(rows).to_csv(OUT/"test_counts.csv",index=False)
    (OUT/"submission_candidate.csv").write_bytes((OUT/f"submission_{PRIMARY}.csv").read_bytes())
    (OUT/"final_recipe.json").write_text(json.dumps(dict(primary=PRIMARY,threshold=READ_THRESHOLD,
        feature_names=feature_names(),protocol_sha256=digest(OUT/"protocol.json"),
        uploaded=False,incumbent_preserved=True),indent=2)+"\n")
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


def main(stage):
    OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(4):
        ctx = context()
        if stage == "audit":
            audit(ctx)
        elif stage == "outer":
            audit_source_artifacts(ctx)
            write_protocol(ctx)
            run_outer(ctx)
        elif stage == "final":
            run_final(ctx)
        elif stage == "all":
            audit(ctx)
            run_outer(ctx)
        else:
            raise ValueError(stage)

